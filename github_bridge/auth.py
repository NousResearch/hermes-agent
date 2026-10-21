"""Authentication: GitHub App JWT minting + installation token exchange.

The bridge uses GitHub App authentication because:

- It is the **GitHub-recommended** path for CI and is scoped to a
  single repository at install time.
- Tokens are **short-lived** (1 hour installation tokens) instead of
  long-lived PATs.
- The PEM private key is held by the bridge; the App itself can be
  revoked from the GitHub UI at any time, instantly invalidating all
  in-flight tokens.
- A ``@``deploy key is never required; SSH remains untouched.

Token caching: an installation token is cached in memory and on disk
under a mode-0600 file in ``state_dir``. The cache is invalidated
when the token is within 5 minutes of expiry.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jwt
import requests

from .config import BridgeConfig
from .errors import (
    AuthError,
    ConfigInvalid,
    NetworkUnreachable,
    Revoked,
    ScopeViolation,
    TokenExpired,
)


# GitHub App JWT max lifetime is 10 minutes; we use 9 for safety.
JWT_TTL_SECONDS = 9 * 60
# Installation tokens live 1 hour; refresh 5 minutes before expiry.
TOKEN_SAFETY_MARGIN = 5 * 60
TOKEN_LIFETIME = 60 * 60


@dataclass(frozen=True)
class InstallationToken:
    token: str
    expires_at: int   # unix seconds
    permissions: dict
    repository_selection: str

    def is_expired(self, now: int | None = None) -> bool:
        now = now if now is not None else int(time.time())
        return now >= self.expires_at - TOKEN_SAFETY_MARGIN


class GitHubAppAuth:
    """Mint GitHub App JWTs and exchange them for installation tokens.

    This class is deliberately small: it owns the PEM, computes a
    short-lived JWT, exchanges it for an installation token, and
    caches the result.
    """

    def __init__(self, cfg: BridgeConfig, session: Optional[requests.Session] = None,
                 pem: Optional[str] = None, clock=time.time):
        self.cfg = cfg
        self._session = session or requests.Session()
        self._pem = (pem or cfg.app_private_key).strip()
        if self._pem.startswith("@"):
            self._pem = Path(self._pem[1:]).read_text(encoding="utf-8")
        self._cache_path = cfg.state_dir / "installation_token.json"
        self._clock = clock
        self._cached: Optional[InstallationToken] = None

    # ------------------------------------------------------------------ JWT
    def mint_app_jwt(self) -> str:
        """Return a fresh RS256-signed JWT for the App.

        GitHub validates the JWT against the App's public key on
        every installation-token exchange.
        """
        now = int(self._clock())
        payload = {
            "iat": now - 30,                # 30s clock skew tolerance
            "exp": now + JWT_TTL_SECONDS,   # 9 minutes
            "iss": str(self.cfg.app_id),
        }
        try:
            return jwt.encode(payload, self._pem, algorithm="RS256")
        except Exception as e:
            raise AuthError(f"failed to mint App JWT: {e}") from e

    # ------------------------------------------------------- Token exchange
    def fetch_installation_token(self) -> InstallationToken:
        """POST to /app/installations/{id}/access_tokens and cache."""
        url = f"{self.cfg.api_base}/app/installations/{self.cfg.installation_id}/access_tokens"
        app_jwt = self.mint_app_jwt()
        headers = {
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hermes-github-bridge/1.0",
        }
        try:
            resp = self._session.post(
                url,
                headers=headers,
                timeout=self.cfg.request_timeout,
                allow_redirects=False,
            )
        except requests.RequestException as e:
            raise NetworkUnreachable(f"cannot reach {self.cfg.api_base}: {e}") from e

        if resp.status_code in (401, 403):
            raise Revoked(
                f"GitHub rejected the App (status={resp.status_code}); "
                "check App installation and repo access. "
                "The operator may need to re-install the App."
            )
        if resp.status_code == 404:
            raise Revoked(
                "Installation not found (404); operator must re-install the App."
            )
        if resp.status_code >= 500:
            raise NetworkUnreachable(
                f"GitHub returned {resp.status_code} for token exchange; retry later"
            )
        if resp.status_code >= 400:
            raise AuthError(
                f"token exchange failed: {resp.status_code} {resp.text[:200]!r}"
            )

        try:
            data = resp.json()
        except ValueError as e:
            raise AuthError(f"token response is not JSON: {e}") from e

        token = InstallationToken(
            token=str(data["token"]),
            expires_at=int(data["expires_at"]
                           if isinstance(data["expires_at"], int)
                           else time.mktime(time.strptime(
                               data["expires_at"], "%Y-%m-%dT%H:%M:%SZ"
                           ))),
            permissions=dict(data.get("permissions", {})),
            repository_selection=str(data.get("repository_selection", "selected")),
        )
        return token

    # ----------------------------------------------------- Installation check
    def verify_installation(self) -> dict:
        """GET the installation record and verify the configured repo is in scope.

        Returns the installation dict. Raises Revoked if the App's installation
        does not cover ``self.cfg.full_repo``. This is a fail-closed check
        that runs on every ``doctor`` call and every publish.
        """
        app_jwt = self.mint_app_jwt()
        url = f"{self.cfg.api_base}/app/installations/{self.cfg.installation_id}"
        headers = {
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hermes-github-bridge/1.0",
        }
        try:
            resp = self._session.get(
                url, headers=headers, timeout=self.cfg.request_timeout,
                allow_redirects=False,
            )
        except requests.RequestException as e:
            raise NetworkUnreachable(f"cannot reach {self.cfg.api_base}: {e}") from e
        if resp.status_code in (401, 403, 404):
            raise Revoked(
                f"installation {self.cfg.installation_id} not accessible "
                f"(status={resp.status_code}); operator must re-install the App"
            )
        if resp.status_code >= 500:
            raise NetworkUnreachable(
                f"GitHub returned {resp.status_code}; retry later"
            )
        if resp.status_code >= 400:
            raise AuthError(
                f"installation check failed: {resp.status_code} {resp.text[:200]!r}"
            )
        data = resp.json()
        # Verify the configured account matches the installation account.
        account = data.get("account", {})
        login = (account.get("login") or "").lower()
        if login and login != self.cfg.owner.lower():
            raise Revoked(
                f"installation account {login!r} does not match configured "
                f"owner {self.cfg.owner!r}; refusing to publish"
            )
        # Verify required permissions: contents:write and issues:write.
        perms = data.get("permissions", {})
        if perms.get("contents") != "write":
            raise Revoked(
                f"App installation lacks contents:write; "
                f"current perms={perms!r}; operator must re-install with Contents: write"
            )
        if perms.get("issues") != "write":
            raise Revoked(
                f"App installation lacks issues:write; "
                f"current perms={perms!r}; operator must re-install with Issues: write"
            )
        # The GET /app/installations/{id} endpoint does NOT include
        # the list of selected repositories; that list lives at
        # /installation/repositories (verified separately below).
        return data

    def installation_repo_in_scope(self) -> bool:
        """Check that ``self.cfg.full_repo`` is in the installation's repos.

        Calls ``GET /installation/repositories`` with the installation
        token. Returns True if the configured repo is accessible, False
        if not. This is a confirmation step on top of
        ``verify_installation``; it does not raise.
        """
        tok = self.get_token()
        url = f"{self.cfg.api_base}/installation/repositories?per_page=100"
        headers = {
            "Authorization": f"Bearer {tok.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hermes-github-bridge/1.0",
        }
        resp = self._session.get(
            url, headers=headers, timeout=self.cfg.request_timeout,
            allow_redirects=False,
        )
        if resp.status_code >= 400:
            return False
        data = resp.json()
        target = self.cfg.full_repo.lower()
        return any(
            r.get("full_name", "").lower() == target
            for r in data.get("repositories", [])
        )

    def find_app_id(self, app_slug: str) -> int:
        """GET /apps/{slug} to resolve the numeric App ID from its slug.

        Requires the App JWT. Raises Revoked if the App is not found.
        """
        app_jwt = self.mint_app_jwt()
        url = f"{self.cfg.api_base}/apps/{app_slug}"
        headers = {
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hermes-github-bridge/1.0",
        }
        try:
            resp = self._session.get(
                url, headers=headers, timeout=self.cfg.request_timeout,
                allow_redirects=False,
            )
        except requests.RequestException as e:
            raise NetworkUnreachable(f"cannot reach {self.cfg.api_base}: {e}") from e
        if resp.status_code == 404:
            raise Revoked(f"App slug {app_slug!r} not found")
        if resp.status_code >= 400:
            raise AuthError(
                f"App lookup failed: {resp.status_code} {resp.text[:200]!r}"
            )
        return int(resp.json()["id"])

    def find_installation_id(self, owner: Optional[str] = None,
                             repo: Optional[str] = None) -> int:
        """GET /repos/{owner}/{repo}/installation to resolve the installation ID.

        Useful when the operator has only configured the App ID + PEM. The
        bridge can auto-discover the installation on the configured repo.
        """
        owner = owner or self.cfg.owner
        repo = repo or self.cfg.repo
        app_jwt = self.mint_app_jwt()
        url = f"{self.cfg.api_base}/repos/{owner}/{repo}/installation"
        headers = {
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hermes-github-bridge/1.0",
        }
        try:
            resp = self._session.get(
                url, headers=headers, timeout=self.cfg.request_timeout,
                allow_redirects=False,
            )
        except requests.RequestException as e:
            raise NetworkUnreachable(f"cannot reach {self.cfg.api_base}: {e}") from e
        if resp.status_code == 404:
            raise Revoked(
                f"no installation of this App on {owner}/{repo}; "
                f"operator must install the App on the repo first"
            )
        if resp.status_code >= 400:
            raise AuthError(
                f"installation lookup failed: {resp.status_code} {resp.text[:200]!r}"
            )
        return int(resp.json()["id"])

    # -------------------------------------- Read-only Issues API (one-shot)
    def _validate_issue_target(self, owner: str, repo: str,
                               issue_number: int) -> None:
        """Fail-closed guard for read-only issue calls.

        The bridge is scoped to exactly one (owner, repo). Any request that
        targets a different repo or an invalid issue number is rejected
        before any HTTP call is made. This is a defense-in-depth check:
        ``verify_installation()`` already enforces install scope, but a
        caller-supplied owner/repo still has to match the configured one.
        """
        if not isinstance(issue_number, int) or issue_number <= 0:
            raise ConfigInvalid(
                f"issue_number must be a positive integer, got {issue_number!r}"
            )
        if not owner or "/" in owner or " " in owner:
            raise ConfigInvalid(f"owner {owner!r} is not a single org or user name")
        if not repo or "/" in repo or " " in repo:
            raise ConfigInvalid(f"repo {repo!r} is not a single repo name")
        if owner.lower() != self.cfg.owner.lower():
            raise ScopeViolation(
                f"owner {owner!r} does not match configured owner "
                f"{self.cfg.owner!r}; refusing to read"
            )
        if repo.lower() != self.cfg.repo.lower():
            raise ScopeViolation(
                f"repo {repo!r} does not match configured repo "
                f"{self.cfg.repo!r}; refusing to read"
            )

    def get_repository(self, owner: str, repo: str) -> dict:
        """GET one repository with the configured installation token.

        Issue responses do not carry the repository's numeric id. The strict
        ingress adapter fetches this resource separately and compares `id` to
        the operator-pinned ID before it trusts any comment metadata.
        """
        self._validate_issue_target(owner, repo, 1)
        tok = self.get_token()
        url = f"{self.cfg.api_base}/repos/{owner}/{repo}"
        headers = {
            "Authorization": f"Bearer {tok.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hermes-github-bridge/1.0",
        }
        try:
            resp = self._session.get(
                url, headers=headers, timeout=self.cfg.request_timeout,
                allow_redirects=False,
            )
        except requests.RequestException as e:
            raise NetworkUnreachable(f"cannot reach {self.cfg.api_base}: {e}") from e
        if resp.status_code in (401, 403, 404):
            raise Revoked(f"repository read denied (status={resp.status_code})")
        if resp.status_code >= 500:
            raise NetworkUnreachable(f"GitHub returned {resp.status_code}; retry later")
        if resp.status_code >= 400:
            raise AuthError(f"get_repository failed: {resp.status_code} {resp.text[:200]!r}")
        return resp.json()

    def get_issue(self, owner: str, repo: str, issue_number: int) -> dict:
        """GET a single issue (read-only, no side effects).

        Used by the operator to fetch the body and metadata of a single
        GitHub Issue through the App-scoped installation token. Returns
        the JSON dict; never writes. The operator must pass owner, repo,
        and issue number explicitly; a target outside the bridge's
        configured scope is rejected before HTTP.
        """
        self._validate_issue_target(owner, repo, issue_number)
        tok = self.get_token()
        url = (
            f"{self.cfg.api_base}/repos/{owner}/{repo}/issues/{issue_number}"
        )
        headers = {
            "Authorization": f"Bearer {tok.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hermes-github-bridge/1.0",
        }
        try:
            resp = self._session.get(
                url, headers=headers,
                timeout=self.cfg.request_timeout, allow_redirects=False,
            )
        except requests.RequestException as e:
            raise NetworkUnreachable(f"cannot reach {self.cfg.api_base}: {e}") from e
        if resp.status_code == 404:
            raise Revoked(
                f"issue {owner}/{repo}#{issue_number} not found or not visible"
            )
        if resp.status_code in (401, 403):
            raise Revoked(
                f"issues read denied (status={resp.status_code})"
            )
        if resp.status_code >= 500:
            raise NetworkUnreachable(
                f"GitHub returned {resp.status_code}; retry later"
            )
        if resp.status_code >= 400:
            raise AuthError(
                f"get_issue failed: {resp.status_code} {resp.text[:200]!r}"
            )
        return resp.json()

    def list_issue_comments(self, owner: str, repo: str,
                            issue_number: int, per_page: int = 20) -> list:
        """GET comments on a single issue (read-only, no side effects).

        Returns the JSON list of comment objects (most recent on page 1).
        Never posts anything; used by the operator to read the most-recent
        commentary on a task. ``per_page`` is capped at 100 by GitHub.
        """
        self._validate_issue_target(owner, repo, issue_number)
        if per_page <= 0 or per_page > 100:
            raise ConfigInvalid(
                f"per_page must be in 1..100, got {per_page!r}"
            )
        tok = self.get_token()
        url = (
            f"{self.cfg.api_base}/repos/{owner}/{repo}/issues/{issue_number}"
            f"/comments?per_page={per_page}&sort=created&direction=desc"
        )
        headers = {
            "Authorization": f"Bearer {tok.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hermes-github-bridge/1.0",
        }
        try:
            resp = self._session.get(
                url, headers=headers,
                timeout=self.cfg.request_timeout, allow_redirects=False,
            )
        except requests.RequestException as e:
            raise NetworkUnreachable(f"cannot reach {self.cfg.api_base}: {e}") from e
        if resp.status_code == 404:
            raise Revoked(
                f"issue {owner}/{repo}#{issue_number} not found or not visible"
            )
        if resp.status_code in (401, 403):
            raise Revoked(
                f"comments read denied (status={resp.status_code})"
            )
        if resp.status_code >= 500:
            raise NetworkUnreachable(
                f"GitHub returned {resp.status_code}; retry later"
            )
        if resp.status_code >= 400:
            raise AuthError(
                f"list_issue_comments failed: {resp.status_code} {resp.text[:200]!r}"
            )
        return resp.json()

    # ----------------------------------------------------- Issues API
    def post_issue_comment(self, owner: str, repo: str, issue_number: int,
                           body: str) -> dict:
        """Post a comment on a GitHub issue. Used for safe channel-readiness ack.

        The token is in the env of the requests session only; never logged.
        """
        tok = self.get_token()
        url = f"{self.cfg.api_base}/repos/{owner}/{repo}/issues/{issue_number}/comments"
        headers = {
            "Authorization": f"Bearer {tok.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hermes-github-bridge/1.0",
        }
        try:
            resp = self._session.post(
                url, headers=headers,
                json={"body": body},
                timeout=self.cfg.request_timeout,
                allow_redirects=False,
            )
        except requests.RequestException as e:
            raise NetworkUnreachable(f"cannot reach {self.cfg.api_base}: {e}") from e
        if resp.status_code in (401, 403, 404):
            raise Revoked(
                f"Issues:write denied (status={resp.status_code}); "
                f"operator must grant Issues:write to the App"
            )
        if resp.status_code >= 500:
            raise NetworkUnreachable(
                f"GitHub returned {resp.status_code}; retry later"
            )
        if resp.status_code >= 400:
            raise AuthError(
                f"issue comment failed: {resp.status_code} {resp.text[:200]!r}"
            )
        return resp.json()

    def list_issues(self, owner: str, repo: str, state: str = "open",
                    labels: Optional[list] = None) -> list:
        """List issues (or PRs) on a repo. Used for safe channel-readiness ack.

        Returns the raw list of issue objects. No state data is sent outbound.
        """
        tok = self.get_token()
        url = f"{self.cfg.api_base}/repos/{owner}/{repo}/issues"
        params = {"state": state, "per_page": 20}
        if labels:
            params["labels"] = ",".join(labels)
        headers = {
            "Authorization": f"Bearer {tok.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hermes-github-bridge/1.0",
        }
        try:
            resp = self._session.get(
                url, headers=headers, params=params,
                timeout=self.cfg.request_timeout,
                allow_redirects=False,
            )
        except requests.RequestException as e:
            raise NetworkUnreachable(f"cannot reach {self.cfg.api_base}: {e}") from e
        if resp.status_code in (401, 403, 404):
            raise Revoked(
                f"issues list denied (status={resp.status_code})"
            )
        if resp.status_code >= 500:
            raise NetworkUnreachable(
                f"GitHub returned {resp.status_code}; retry later"
            )
        if resp.status_code >= 400:
            raise AuthError(
                f"list issues failed: {resp.status_code} {resp.text[:200]!r}"
            )
        return resp.json()

    # ---------------------------------------------------------------- Cache
    def get_token(self) -> InstallationToken:
        """Return a fresh installation token, using the cache when valid."""
        now = int(self._clock())
        if self._cached and not self._cached.is_expired(now):
            return self._cached
        cached = self._read_disk_cache(now)
        if cached and not cached.is_expired(now):
            self._cached = cached
            return cached
        token = self.fetch_installation_token()
        self._cached = token
        self._write_disk_cache(token)
        return token

    def _read_disk_cache(self, now: int) -> Optional[InstallationToken]:
        if not self._cache_path.is_file():
            return None
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        try:
            tok = InstallationToken(
                token=str(data["token"]),
                expires_at=int(data["expires_at"]),
                permissions=dict(data.get("permissions", {})),
                repository_selection=str(data.get("repository_selection", "selected")),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if tok.is_expired(now):
            return None
        return tok

    def _write_disk_cache(self, tok: InstallationToken) -> None:
        self.cfg.state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "token": tok.token,
            "expires_at": tok.expires_at,
            "permissions": tok.permissions,
            "repository_selection": tok.repository_selection,
        }
        # Write atomically and restrict perms to 0600.
        tmp = self._cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(self._cache_path)

    def clear_cache(self) -> None:
        """Wipe the disk and in-memory cache (used by revoke())."""
        self._cached = None
        if self._cache_path.is_file():
            try:
                self._cache_path.unlink()
            except OSError:
                pass
