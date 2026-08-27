"""Repository-bound GitHub MCP broker for sanitized remote-model work.

The model never supplies an owner, repository, URL, GraphQL document, or shell
command.  Each public method compiles typed arguments into one fixed REST route.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Mapping
from urllib.parse import quote

import httpx


class BrokerError(RuntimeError):
    """A fail-closed staging broker denial or sanitized network failure."""


ALLOWED_TOOL_NAMES = {
    "staging_branch_get",
    "staging_branch_create",
    "staging_file_get",
    "staging_file_put",
    "staging_issue_create",
    "staging_pull_request_create",
    "staging_review_create",
}

_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")
_BRANCH_RE = re.compile(r"[A-Za-z0-9._/-]+")
_MAX_TEXT = 512_000
_PROPOSAL_BRANCH_PREFIX = "hermes-proposal/"


@dataclass(frozen=True)
class BrokerConfig:
    owner: str
    repository: str
    token: str = field(repr=False)
    audit_path: Path

    def __post_init__(self) -> None:
        if not _NAME_RE.fullmatch(self.owner or ""):
            raise BrokerError("invalid staging owner configuration")
        if not _NAME_RE.fullmatch(self.repository or ""):
            raise BrokerError("invalid staging repository configuration")
        if not self.token.strip():
            raise BrokerError("HERMES_STAGING_GITHUB_TOKEN is required")
        if not self.token.startswith(("github_pat_", "ghs_")):
            raise BrokerError(
                "staging credential must be a fine-grained token or GitHub App installation token"
            )
        if not self.audit_path.is_absolute():
            raise BrokerError("staging audit path must be absolute")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "BrokerConfig":
        env = os.environ if environ is None else environ
        token = env.get("HERMES_STAGING_GITHUB_TOKEN", "").strip()
        if not token:
            raise BrokerError(
                "HERMES_STAGING_GITHUB_TOKEN is required; GH_TOKEN and GITHUB_TOKEN are not used"
            )
        default_audit = (
            Path(env.get("HERMES_HOME", str(Path.home() / ".hermes")))
            / "secure-worker"
            / "github-broker-audit.jsonl"
        )
        return cls(
            owner=env.get("HERMES_STAGING_GITHUB_OWNER", "").strip(),
            repository=env.get("HERMES_STAGING_GITHUB_REPO", "").strip(),
            token=token,
            audit_path=Path(
                env.get("HERMES_STAGING_GITHUB_AUDIT_PATH", str(default_audit))
            ).expanduser().resolve(),
        )


def _branch(value: str) -> str:
    if (
        not value
        or len(value) > 255
        or not _BRANCH_RE.fullmatch(value)
        or value.startswith(("/", "."))
        or value.casefold().startswith("refs/")
        or value.endswith(("/", "."))
        or ".." in value
        or "//" in value
    ):
        raise BrokerError("invalid branch name")
    return value


def _repo_path(value: str) -> str:
    if not value or "\\" in value:
        raise BrokerError("invalid repository path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or any(part.casefold() == ".git" for part in path.parts)
    ):
        raise BrokerError("invalid repository path")
    return "/".join(quote(part, safe="-._~") for part in path.parts)


def _writable_repo_path(value: str) -> str:
    encoded = _repo_path(value)
    lowered = PurePosixPath(value).as_posix().casefold()
    if (
        lowered == ".github/dependabot.yml"
        or lowered.startswith(".github/workflows/")
        or lowered.startswith(".github/actions/")
    ):
        raise BrokerError("GitHub execution control path is not writable")
    return encoded


def _proposal_branch(value: str) -> str:
    branch = _branch(value)
    if not branch.startswith(_PROPOSAL_BRANCH_PREFIX):
        raise BrokerError("write target must be a dedicated proposal branch")
    return branch


def _bounded_text(value: str, label: str, *, allow_empty: bool = False) -> str:
    encoded = value.encode("utf-8")
    if (not allow_empty and not value.strip()) or len(encoded) > _MAX_TEXT:
        raise BrokerError(f"invalid {label}")
    return value


class GitHubStagingBroker:
    """Typed REST operations bound to exactly one staging repository."""

    def __init__(self, config: BrokerConfig, *, client: httpx.Client | None = None) -> None:
        self.config = config
        self._client = client or httpx.Client(
            base_url="https://api.github.com",
            follow_redirects=False,
            timeout=30.0,
        )
        self._repo_prefix = f"/repos/{quote(config.owner)}/{quote(config.repository)}"

    def _audit(self, operation: str, status: str) -> None:
        repository_id = hashlib.sha256(
            f"{self.config.owner}/{self.config.repository}".encode()
        ).hexdigest()[:24]
        row = {
            "schema": "hermes.secure-worker.github-broker-receipt.v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "status": status,
            "repository_id": repository_id,
        }
        try:
            self.config.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        except OSError as exc:
            raise BrokerError("unable to write broker audit receipt") from exc

    def _request(
        self,
        operation: str,
        method: str,
        route: str,
        *,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if route != self._repo_prefix and not route.startswith(self._repo_prefix + "/"):
            raise BrokerError("internal repository route escaped staging binding")
        try:
            response = self._client.request(
                method,
                route,
                json=payload,
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {self.config.token}",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
        except httpx.HTTPError as exc:
            self._audit(operation, "network_error")
            raise BrokerError("staging GitHub request failed") from exc
        if 300 <= response.status_code < 400:
            self._audit(operation, "redirect_denied")
            raise BrokerError("staging GitHub redirect denied")
        if response.status_code < 200 or response.status_code >= 300:
            self._audit(operation, f"http_{response.status_code}")
            raise BrokerError(f"staging GitHub request returned HTTP {response.status_code}")
        try:
            result = response.json()
        except ValueError as exc:
            self._audit(operation, "invalid_json")
            raise BrokerError("staging GitHub response was not valid JSON") from exc
        if not isinstance(result, dict):
            self._audit(operation, "invalid_shape")
            raise BrokerError("staging GitHub response shape was invalid")
        self._audit(operation, "ok")
        return result

    def verify_repository_boundary(self) -> None:
        """Verify exact private staging identity and non-admin write permissions."""

        result = self._request(
            "verify_repository_boundary", "GET", self._repo_prefix
        )
        permissions = result.get("permissions")
        valid_permissions = (
            isinstance(permissions, dict)
            and permissions.get("pull") is True
            and permissions.get("push") is True
            and permissions.get("admin") is not True
            and permissions.get("maintain") is not True
        )
        if (
            result.get("full_name") != f"{self.config.owner}/{self.config.repository}"
            or result.get("private") is not True
            or not valid_permissions
        ):
            self._audit("verify_repository_boundary", "identity_or_permission_denied")
            raise BrokerError("staging repository boundary verification failed")
        actions = self._request(
            "verify_actions_disabled", "GET", f"{self._repo_prefix}/actions/permissions"
        )
        if actions.get("enabled") is not False:
            self._audit("verify_repository_boundary", "actions_enabled_denied")
            raise BrokerError("GitHub Actions must be disabled in the staging repository")

    def get_branch(self, branch: str) -> dict[str, object]:
        branch = _branch(branch)
        return self._request("get_branch", "GET", f"{self._repo_prefix}/branches/{branch}")

    def create_branch(self, branch: str, from_branch: str) -> dict[str, object]:
        branch = _proposal_branch(branch)
        from_branch = _branch(from_branch)
        source = self._request(
            "get_source_ref",
            "GET",
            f"{self._repo_prefix}/git/ref/heads/{from_branch}",
        )
        obj = source.get("object")
        sha = obj.get("sha") if isinstance(obj, dict) else None
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", sha):
            raise BrokerError("source branch response did not contain a valid commit")
        return self._request(
            "create_branch",
            "POST",
            f"{self._repo_prefix}/git/refs",
            payload={"ref": f"refs/heads/{branch}", "sha": sha},
        )

    def get_file(self, path: str, ref: str) -> dict[str, object]:
        encoded_path = _repo_path(path)
        ref = _branch(ref)
        route = f"{self._repo_prefix}/contents/{encoded_path}?ref={quote(ref, safe='')}"
        return self._request("get_file", "GET", route)

    def put_file(
        self,
        path: str,
        content: str,
        message: str,
        branch: str,
        sha: str | None = None,
    ) -> dict[str, object]:
        encoded_path = _writable_repo_path(path)
        content = _bounded_text(content, "file content", allow_empty=True)
        message = _bounded_text(message, "commit message")
        branch = _proposal_branch(branch)
        payload: dict[str, object] = {
            "message": message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch,
        }
        if sha is not None:
            if not re.fullmatch(r"[0-9a-fA-F]{40,64}", sha):
                raise BrokerError("invalid existing file sha")
            payload["sha"] = sha
        return self._request(
            "put_file", "PUT", f"{self._repo_prefix}/contents/{encoded_path}", payload=payload
        )

    def create_issue(self, title: str, body: str) -> dict[str, object]:
        return self._request(
            "create_issue",
            "POST",
            f"{self._repo_prefix}/issues",
            payload={
                "title": _bounded_text(title, "issue title"),
                "body": _bounded_text(body, "issue body", allow_empty=True),
            },
        )

    def create_pull_request(
        self, title: str, body: str, head: str, base: str
    ) -> dict[str, object]:
        return self._request(
            "create_pull_request",
            "POST",
            f"{self._repo_prefix}/pulls",
            payload={
                "title": _bounded_text(title, "pull request title"),
                "body": _bounded_text(body, "pull request body", allow_empty=True),
                "head": _proposal_branch(head),
                "base": _branch(base),
            },
        )

    def create_review(self, pull_number: int, event: str, body: str) -> dict[str, object]:
        if not isinstance(pull_number, int) or isinstance(pull_number, bool) or pull_number <= 0:
            raise BrokerError("invalid pull request number")
        event = event.upper()
        if event not in {"APPROVE", "REQUEST_CHANGES", "COMMENT"}:
            raise BrokerError("invalid review event")
        return self._request(
            "create_review",
            "POST",
            f"{self._repo_prefix}/pulls/{pull_number}/reviews",
            payload={"event": event, "body": _bounded_text(body, "review body", allow_empty=True)},
        )


def create_mcp_server(broker: GitHubStagingBroker | None = None):
    """Create the fixed staging MCP surface."""

    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        raise BrokerError("the pinned MCP SDK is unavailable") from exc

    bound = broker or GitHubStagingBroker(BrokerConfig.from_env())
    server = MCPServer(
        "hermes-secure-github-staging",
        instructions=(
            "Operate only on the configured private staging repository. "
            "This server cannot address the production repository."
        ),
    )

    @server.tool()
    def staging_branch_get(branch: str) -> str:
        """Read one branch from the configured staging repository."""
        return json.dumps(bound.get_branch(branch))

    @server.tool()
    def staging_branch_create(branch: str, from_branch: str = "main") -> str:
        """Create a staging branch from another staging branch."""
        return json.dumps(bound.create_branch(branch, from_branch))

    @server.tool()
    def staging_file_get(path: str, ref: str = "main") -> str:
        """Read a file from the configured staging repository."""
        return json.dumps(bound.get_file(path, ref))

    @server.tool()
    def staging_file_put(
        path: str, content: str, message: str, branch: str, sha: str | None = None
    ) -> str:
        """Create or update a UTF-8 file in the configured staging repository."""
        return json.dumps(bound.put_file(path, content, message, branch, sha))

    @server.tool()
    def staging_issue_create(title: str, body: str = "") -> str:
        """Create an issue in the configured staging repository."""
        return json.dumps(bound.create_issue(title, body))

    @server.tool()
    def staging_pull_request_create(
        title: str, head: str, base: str = "main", body: str = ""
    ) -> str:
        """Create a pull request entirely inside the staging repository."""
        return json.dumps(bound.create_pull_request(title, body, head, base))

    @server.tool()
    def staging_review_create(pull_number: int, event: str, body: str = "") -> str:
        """Create a typed review on a staging pull request."""
        return json.dumps(bound.create_review(pull_number, event, body))

    return server


def main() -> int:
    try:
        broker = GitHubStagingBroker(BrokerConfig.from_env())
        broker.verify_repository_boundary()
        server = create_mcp_server(broker)
        asyncio.run(server.run_stdio_async())
    except (BrokerError, KeyboardInterrupt) as exc:
        if isinstance(exc, BrokerError):
            print(f"secure GitHub broker refused to start: {exc}", file=os.sys.stderr)
            return 1
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
