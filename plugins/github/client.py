"""GitHub API client for the bundled ``plugins/github`` connector.

Every action goes through a resolved credential chain:

1. **GitHub App installation token** (preferred when configured) — actions
   are attributed to the bot login ``<slug>[bot]`` (e.g. ``jarpis-bot[bot]``),
   never to the account owner. This is the whole point of the connector:
   agent comments/reviews/issues are visibly distinct from the human user's.
2. ``GITHUB_TOKEN`` / ``GH_TOKEN`` (PAT) — attributed to the account owner.
3. ``gh auth token`` (gh CLI) — attributed to the account owner.

Every tool result includes ``auth_method`` + ``actor`` so the agent and the
user can always tell WHO performed an action. ``github_identity`` proves the
chain end-to-end (bot slug or user login + accessible repos).

Credentials are profile-scoped secrets read via ``agent.secret_scope``, so
multiplexed gateway sessions never leak another profile's token.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from agent.github_auth import GitHubAppAuth

logger = logging.getLogger(__name__)

API_BASE = "https://api.github.com"
_DEFAULT_TIMEOUT = 15


class GitHubError(Exception):
    """Raised on API failures; carries status code + GitHub message."""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = response or {}


class GitHubClient:
    """Thin GitHub REST client with bot-first credential resolution.

    Stateless per call (construct fresh or reuse); token cache lives in
    ``GitHubAppAuth``. Methods return parsed JSON payloads and raise
    ``GitHubError`` on non-2xx.
    """

    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._method: Optional[str] = None
        self._actor: Optional[str] = None

    # ------------------------------------------------------------------
    # Auth resolution (bot first)
    # ------------------------------------------------------------------

    def _resolve_credentials(self) -> Tuple[str, str]:
        """Return (auth_method, token). Bot identity wins when configured."""
        if self._token:
            return self._method or "unknown", self._token

        # 1. GitHub App installation token — bot identity.
        app_auth = GitHubAppAuth()
        if app_auth.credentials_configured():
            token = app_auth.installation_token()
            if token:
                self._token = token
                self._method = "github-app"
                self._actor = app_auth.bot_login() or "github-app[bot]"
                return self._method, token

        # 2. PAT.
        from agent.secret_scope import get_secret

        pat = get_secret("GITHUB_TOKEN") or get_secret("GH_TOKEN")
        if pat:
            self._token = pat
            self._method = "pat"
            return self._method, pat

        # 3. gh CLI.
        gh_token = self._try_gh_cli()
        if gh_token:
            self._token = gh_token
            self._method = "gh-cli"
            return self._method, gh_token

        raise GitHubError(
            "No GitHub credentials configured. Set GITHUB_APP_ID + "
            "GITHUB_APP_PRIVATE_KEY_PATH + GITHUB_APP_INSTALLATION_ID (GitHub App "
            "bot identity — recommended) or GITHUB_TOKEN/GH_TOKEN, or run `gh auth login`."
        )

    def _try_gh_cli(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.debug("gh CLI token lookup failed: %s", e)
        return None

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    @property
    def auth_method(self) -> str:
        return self._method or "unknown"

    @property
    def actor(self) -> str:
        """Login the current token acts as (bot login for App auth)."""
        if self._actor:
            return self._actor
        # PAT / gh CLI: resolve via GET /user (only valid for user tokens).
        if self._method in ("pat", "gh-cli"):
            try:
                user = self._request("GET", "/user")
                self._actor = user.get("login") or "unknown-user"
            except GitHubError:
                self._actor = "unknown-user"
        return self._actor or "unknown"

    def attribution(self) -> Dict[str, str]:
        """Attribution block included in every tool result."""
        try:
            self._resolve_credentials()
        except GitHubError:
            pass  # error path handled by the caller
        return {"auth_method": self.auth_method, "actor": self.actor}

    # ------------------------------------------------------------------
    # HTTP core
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        retried: bool = False,
    ) -> Any:
        try:
            import httpx
        except ImportError as e:
            raise GitHubError(f"httpx is required for GitHub API calls: {e}")

        auth_method, token = self._resolve_credentials()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        url = f"{API_BASE}{path}"
        try:
            resp = httpx.request(
                method,
                url,
                headers=headers,
                params=params,
                json=payload,
                timeout=_DEFAULT_TIMEOUT,
            )
        except Exception as e:
            raise GitHubError(f"GitHub API request failed: {e}")

        # 401 with an installation token: the cached token expired or was
        # revoked — mint a fresh one and retry exactly once.
        if resp.status_code == 401 and not retried:
            logger.debug("GitHub API 401 with method=%s; refreshing token", auth_method)
            self._token = None
            self._method = None
            self._actor = None
            return self._request(method, path, params=params, payload=payload, retried=True)

        if resp.status_code >= 400:
            body = {}
            try:
                body = resp.json()
            except Exception:
                pass
            message = body.get("message") or f"GitHub API {resp.status_code}"
            raise GitHubError(message, status_code=resp.status_code, response=body)

        if not resp.content:
            return {}
        try:
            return resp.json()
        except Exception:
            return {"_raw": resp.text}

    # ------------------------------------------------------------------
    # Issues
    # ------------------------------------------------------------------

    def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str = "",
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"title": title}
        if body:
            payload["body"] = body
        if labels:
            payload["labels"] = labels
        if assignees:
            payload["assignees"] = assignees
        return self._request("POST", f"/repos/{owner}/{repo}/issues", payload=payload)

    def get_issue(self, owner: str, repo: str, number: int, include_comments: bool = False) -> Dict[str, Any]:
        issue = self._request("GET", f"/repos/{owner}/{repo}/issues/{number}")
        if include_comments:
            issue["comments_data"] = self._request(
                "GET", f"/repos/{owner}/{repo}/issues/{number}/comments", params={"per_page": 100}
            )
        return issue

    def list_issues(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        labels: Optional[str] = None,
        assignee: Optional[str] = None,
        creator: Optional[str] = None,
        sort: str = "created",
        direction: str = "desc",
        per_page: int = 30,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "state": state,
            "sort": sort,
            "direction": direction,
            "per_page": min(max(per_page, 1), 100),
        }
        if labels:
            params["labels"] = labels
        if assignee:
            params["assignee"] = assignee
        if creator:
            params["creator"] = creator
        return self._request("GET", f"/repos/{owner}/{repo}/issues", params=params)

    def comment_issue(self, owner: str, repo: str, number: int, body: str) -> Dict[str, Any]:
        return self._request(
            "POST", f"/repos/{owner}/{repo}/issues/{number}/comments", payload={"body": body}
        )

    # ------------------------------------------------------------------
    # Pull requests
    # ------------------------------------------------------------------

    def get_pull_request(self, owner: str, repo: str, number: int) -> Dict[str, Any]:
        return self._request("GET", f"/repos/{owner}/{repo}/pulls/{number}")

    def review_pull_request(
        self,
        owner: str,
        repo: str,
        number: int,
        event: str,
        body: str = "",
        comments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Submit a PR review. ``event``: APPROVE | REQUEST_CHANGES | COMMENT.

        ``comments`` is an optional list of inline review comments, each
        ``{"path": ..., "line": ..., "body": ...}``.
        """
        payload: Dict[str, Any] = {"event": event}
        if body:
            payload["body"] = body
        if comments:
            payload["comments"] = comments
        return self._request("POST", f"/repos/{owner}/{repo}/pulls/{number}/reviews", payload=payload)

    def merge_pull_request(
        self,
        owner: str,
        repo: str,
        number: int,
        method: str = "squash",
        commit_title: str = "",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"merge_method": method}
        if commit_title:
            payload["commit_title"] = commit_title
        return self._request("PUT", f"/repos/{owner}/{repo}/pulls/{number}/merge", payload=payload)

    # ------------------------------------------------------------------
    # Identity verification (read-only)
    # ------------------------------------------------------------------

    def verify_identity(self) -> Dict[str, Any]:
        """Prove which identity the resolved credential acts as.

        For App auth: bot login (``<slug>[bot]``) + accessible repos.
        For PAT/gh CLI: the account login via ``GET /user`` + accessible repos.
        """
        method, _ = self._resolve_credentials()
        info: Dict[str, Any] = {"auth_method": method, "actor": self.actor}

        if method == "github-app":
            from agent.github_auth import GitHubAppAuth

            app_auth = GitHubAppAuth()
            info["app_slug"] = app_auth.app_slug()
            info["installation_id"] = None  # never expose the raw id in results
        else:
            try:
                user = self._request("GET", "/user")
                info["account"] = user.get("login")
                info["user_type"] = user.get("type")
            except GitHubError as e:
                info["account_error"] = e.message

        try:
            repos = self._request(
                "GET", "/installation/repositories", params={"per_page": 100}
            )
            info["accessible_repos"] = [r.get("full_name") for r in repos.get("repositories", [])]
        except GitHubError:
            # /installation/repositories is App-only; PAT users get repos via /user.
            try:
                user_repos = self._request("GET", "/user/repos", params={"per_page": 100, "sort": "updated"})
                info["accessible_repos"] = [r.get("full_name") for r in user_repos]
            except GitHubError:
                info["accessible_repos"] = []
        return info


def parse_repo(repo: str) -> Tuple[str, str]:
    """Split ``owner/name`` into (owner, repo). Accepts bare repo names too."""
    repo = (repo or "").strip().strip("/")
    if not repo:
        raise GitHubError("repo is required (format: owner/name)")
    if "/" in repo:
        owner, _, name = repo.partition("/")
        if not owner or not name:
            raise GitHubError(f"invalid repo format: {repo!r} (expected owner/name)")
        return owner, name
    return repo, repo


def repo_full_name(repo: str) -> str:
    return "/".join(parse_repo(repo))
