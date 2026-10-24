"""Provider-neutral GitHub workflow bootstrap for the CLI.

This module deliberately consumes the already-provisioned Hermes secret scope.
It does not know which secret provider supplied ``GITHUB_TOKEN`` and does not
implement provider precedence.  GitHub MCP is not required.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from agent.secret_scope import get_secret
from agent.secret_sources.base import ErrorKind
from hermes_cli._subprocess_compat import noninteractive_git_env

GITHUB_CREDENTIAL_PURPOSE = "github-workflow"
_DEFAULT_TIMEOUT = 10.0
_DEFAULT_CACHE_SECONDS = 900.0
_CACHE: dict[str, tuple[float, "AuthState", Optional[str], Optional[str]]] = {}


class AuthState(str, Enum):
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    VERIFIED = "verified"
    INVALID = "invalid"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMITED = "rate_limited"
    NETWORK_ERROR = "network_error"


@dataclass(frozen=True)
class GitHubRepository:
    root: Optional[str] = None
    remote: Optional[str] = None
    owner: Optional[str] = None
    name: Optional[str] = None
    branch: Optional[str] = None
    sha: Optional[str] = None


@dataclass(frozen=True)
class GitHubCapability:
    workflow_enabled: bool
    credential_available: bool
    auth_state: AuthState
    repository: GitHubRepository = field(default_factory=GitHubRepository)
    identity: Optional[str] = None
    public_read_available: bool = True
    remediation: str = ""
    error_kind: Optional[ErrorKind] = None
    _credential: Optional[str] = field(default=None, repr=False, compare=False)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "workflow_enabled": self.workflow_enabled,
            "credential_available": self.credential_available,
            "auth_state": self.auth_state.value,
            "repository": {
                "root": self.repository.root,
                "remote": self.repository.remote,
                "owner": self.repository.owner,
                "name": self.repository.name,
                "branch": self.repository.branch,
                "sha": self.repository.sha,
            },
            "identity": self.identity,
            "public_read_available": self.public_read_available,
            "remediation": self.remediation,
        }

    def __str__(self) -> str:
        return str(self.to_public_dict())


def _workflow_config(config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    cfg = (config or {}).get("github", {})
    workflow = cfg.get("workflow", {}) if isinstance(cfg, Mapping) else {}
    return dict(workflow) if isinstance(workflow, Mapping) else {}


def _git_stdout(cwd: str, args: list[str], timeout: float = _DEFAULT_TIMEOUT) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
            env=noninteractive_git_env(), timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def parse_github_remote(remote: str) -> Optional[tuple[str, str]]:
    value = (remote or "").strip()
    match = re.match(r"^(?:https?://github\.com/|git@github\.com:)([^/]+)/([^/]+?)(?:\.git)?/?$", value, re.I)
    if not match:
        return None
    return match.group(1), match.group(2)


def repository_context(cwd: str) -> GitHubRepository:
    root = _git_stdout(cwd, ["rev-parse", "--show-toplevel"]).strip() or None
    base = root or cwd
    remote = _git_stdout(base, ["config", "--get", "remote.origin.url"]).strip() or None
    parsed = parse_github_remote(remote or "")
    branch = _git_stdout(base, ["branch", "--show-current"]).strip() or None
    sha = _git_stdout(base, ["rev-parse", "HEAD"]).strip() or None
    owner, name = parsed if parsed else (None, None)
    return GitHubRepository(root=root, remote=remote, owner=owner, name=name, branch=branch, sha=sha)


def classify_github_response(status: int, headers: Mapping[str, str]) -> AuthState:
    if status == 200:
        return AuthState.VERIFIED
    if status == 401:
        return AuthState.INVALID
    if status == 403:
        lowered = {str(k).lower(): str(v) for k, v in headers.items()}
        if lowered.get("x-ratelimit-remaining") == "0" or "retry-after" in lowered:
            return AuthState.RATE_LIMITED
        return AuthState.PERMISSION_DENIED
    return AuthState.NETWORK_ERROR if status >= 500 else AuthState.PERMISSION_DENIED


def clear_preflight_cache() -> None:
    _CACHE.clear()


def _fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _preflight(token: str, timeout: float, cache_seconds: float) -> tuple[AuthState, Optional[str], Optional[str]]:
    key = _fingerprint(token)
    now = time.monotonic()
    cached = _CACHE.get(key)
    if cached and now - cached[0] < cache_seconds:
        return cached[1:]
    try:
        request = urllib.request.Request(
            "https://api.github.com/user",
            headers={"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            identity = None
            try:
                import json
                identity = json.loads(response.read().decode()).get("login")
            except Exception:
                pass
            state = classify_github_response(response.status, dict(response.headers.items()))
            result = (state, identity, None)
    except urllib.error.HTTPError as error:
        state = classify_github_response(error.code, dict(error.headers.items()))
        result = (state, None, "GitHub rejected the credential; check the configured GitHub secret.")
    except (OSError, urllib.error.URLError, TimeoutError):
        result = (AuthState.NETWORK_ERROR, None, "GitHub could not be reached; check connectivity and retry.")
    _CACHE[key] = (now, *result)
    return result


def resolve_github_capability(config: Optional[Mapping[str, Any]] = None, *, operation: str = "public-read", perform_preflight: bool = True) -> GitHubCapability:
    workflow = _workflow_config(config)
    enabled = bool(workflow.get("enabled", True))
    if not enabled:
        return GitHubCapability(False, False, AuthState.DISABLED, remediation="Enable github.workflow.enabled to activate GitHub workflow support.")
    token = get_secret("GITHUB_TOKEN", None)
    if not token:
        return GitHubCapability(True, False, AuthState.UNAVAILABLE, public_read_available=True, error_kind=ErrorKind.NOT_CONFIGURED, remediation="Add GITHUB_TOKEN to a configured Hermes secret source, then run `hermes secrets status`.")
    if not perform_preflight and operation == "public-read":
        return GitHubCapability(True, True, AuthState.UNAVAILABLE, public_read_available=True, remediation="Public-read operation does not require GitHub authentication.", _credential=token)
    timeout = float(workflow.get("api_timeout_seconds", _DEFAULT_TIMEOUT))
    cache_seconds = float(workflow.get("preflight_cache_seconds", _DEFAULT_CACHE_SECONDS))
    state, identity, detail = _preflight(token, timeout, cache_seconds)
    remediation = detail or ("GitHub authentication verified." if state is AuthState.VERIFIED else "Check the configured GitHub credential and retry.")
    kind = None if state is AuthState.VERIFIED else (ErrorKind.AUTH_FAILED if state is AuthState.INVALID else ErrorKind.NETWORK if state is AuthState.NETWORK_ERROR else None)
    return GitHubCapability(True, True, state, identity=identity, public_read_available=True, remediation=remediation, error_kind=kind, _credential=token)


def activation_relevant(prompt: str, repository: Optional[GitHubRepository] = None) -> bool:
    if repository and repository.owner and repository.name:
        return True
    return bool(re.search(r"\b(github|repository|repo|pull request|\bpr\b|issue|branch|fork|upstream|actions|commit|push)\b", prompt or "", re.I))


def workflow_git_env(base: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Return isolated noninteractive Git env for future ephemeral auth use."""
    env = noninteractive_git_env(base)
    env.pop("GIT_ASKPASS", None)
    env.pop("SSH_ASKPASS", None)
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env
