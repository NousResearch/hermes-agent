"""Configuration for the GitHub Bridge.

All values are loaded from environment variables OR from a single JSON
file referenced by ``GITHUB_BRIDGE_CONFIG`` (preferred: file, since
the App private key is multi-line PEM and difficult to keep in env).

Required keys (operator provides these ONCE):
    - app_id:            integer; GitHub App numeric ID
    - installation_id:   integer; installation ID on the target repo
    - app_private_key:   PEM, multi-line string
    - owner:             GitHub org or user (no @)
    - repo:              single repo name (not org-wide)
    - branch_prefix:     default ``hermes/review/``; never ``main``/``master``

Optional:
    - api_base:          default ``https://api.github.com`` (HARDENED;
                         any other value raises ConfigInvalid)
    - git_base:          default ``https://x-access-token:TOK@github.com``
                         (token substituted at runtime, never written)
    - state_dir:         default ``/home/isv/.local/share/github-bridge``
    - request_timeout:   seconds, default 15
    - max_retries:       default 3 (1/2/4s backoff)
    - operator_safety:   default ``fail-closed`` (cannot be downgraded)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, Optional

from .errors import ConfigInvalid, ConfigMissing


DEFAULT_API_BASE = "https://api.github.com"
DEFAULT_GIT_BASE_TEMPLATE = "https://x-access-token:{token}@github.com"
DEFAULT_STATE_DIR = "/home/isv/.local/share/github-bridge"
DEFAULT_BRANCH_PREFIX = "hermes/review/"
DEFAULT_TIMEOUT = 15
DEFAULT_MAX_RETRIES = 3

# Branches that are NEVER written to. Force-matching a protected branch
# raises ProtectedBranch before any HTTP call.
PROTECTED_BRANCHES = frozenset({
    "main", "master", "develop", "release", "releases",
})


@dataclass(frozen=True)
class BridgeConfig:
    app_id: int
    installation_id: int
    app_private_key: str
    owner: str
    repo: str
    branch_prefix: str
    api_base: str
    git_base_template: str
    state_dir: Path
    request_timeout: int
    max_retries: int
    operator_safety: str
    issue_ingress_repository_id: Optional[int] = None
    issue_ingress_app_id: Optional[int] = None
    issue_ingress_installation_id: Optional[int] = None
    issue_ingress_actor_id: Optional[int] = None
    trusted_commenters: frozenset[str] = field(default_factory=lambda: frozenset({"isvgithub"}))
    issue_ingress_issue_number: int = 22
    issue_ingress_max_body_bytes: int = 1024
    issue_ingress_executor_profile: Optional[str] = None
    issue_ingress_session_id: Optional[str] = None
    issue_ingress_result_files: tuple[str, ...] = ()
    issue_ingress_sources: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate config against the bridge's safety invariants.

        Runs on every load. Raises ConfigInvalid if any invariant fails.
        """
        self._validate_ingress_policy()
        if not isinstance(self.app_id, int) or self.app_id <= 0:
            raise ConfigInvalid("app_id must be a positive integer")
        if not isinstance(self.installation_id, int) or self.installation_id <= 0:
            raise ConfigInvalid("installation_id must be a positive integer")
        if not self.app_private_key or "BEGIN" not in self.app_private_key:
            if not self.app_private_key.startswith("@"):
                raise ConfigInvalid("app_private_key is not a PEM block")
            else:
                # @ prefix means "read PEM from file"; defer validation
                # to pem_path() / first use.
                pass
        if "PRIVATE KEY" not in self.app_private_key and not self.app_private_key.startswith("@"):
            raise ConfigInvalid("app_private_key missing 'PRIVATE KEY' marker")
        if not self.owner or "/" in self.owner or " " in self.owner:
            raise ConfigInvalid("owner must be a single org or user name")
        if not self.repo or "/" in self.repo or " " in self.repo:
            raise ConfigInvalid("repo must be a single repo name")
        if self.branch_prefix in PROTECTED_BRANCHES:
            raise ConfigInvalid(
                f"branch_prefix {self.branch_prefix!r} is a protected branch"
            )
        if self.branch_prefix.startswith(tuple(f"{b}/" for b in PROTECTED_BRANCHES)):
            raise ConfigInvalid(
                f"branch_prefix {self.branch_prefix!r} conflicts with a protected namespace"
            )
        if not self.api_base.startswith("https://api.github.com"):
            raise ConfigInvalid(
                f"api_base {self.api_base!r} must be HTTPS api.github.com"
            )
        if "@" in self.api_base or "?" in self.api_base:
            raise ConfigInvalid("api_base must not contain credentials or query string")
        if self.operator_safety != "fail-closed":
            raise ConfigInvalid(
                f"operator_safety={self.operator_safety!r}; only 'fail-closed' is allowed"
            )
        if self.request_timeout <= 0 or self.request_timeout > 60:
            raise ConfigInvalid("request_timeout must be 1..60 seconds")
        if self.max_retries < 0 or self.max_retries > 5:
            raise ConfigInvalid("max_retries must be 0..5")
        if not isinstance(self.trusted_commenters, frozenset) or not self.trusted_commenters:
            raise ConfigInvalid("trusted_commenters must be a non-empty frozenset")
        for login in self.trusted_commenters:
            if not isinstance(login, str) or not re.match(r"^[a-z0-9](?:[a-z0-9-]{0,37}[a-z0-9])?$", login):
                raise ConfigInvalid("trusted_commenters contains an invalid GitHub login")
        for key, value in (
            ("issue_ingress_repository_id", self.issue_ingress_repository_id),
            ("issue_ingress_app_id", self.issue_ingress_app_id),
            ("issue_ingress_installation_id", self.issue_ingress_installation_id),
            ("issue_ingress_actor_id", self.issue_ingress_actor_id),
        ):
            if value is not None and (type(value) is not int or value <= 0):
                raise ConfigInvalid(f"{key} must be a positive integer when configured")

    def _validate_ingress_policy(self) -> None:
        if type(self.issue_ingress_issue_number) is not int or self.issue_ingress_issue_number not in {22, 27}:
            raise ConfigInvalid("issue ingress requires one explicit Issue 22 or 27 source")
        if (not isinstance(self.issue_ingress_sources, dict)
                or any(key not in {"22", "27"} or not isinstance(value, str)
                       or not Path(value).is_absolute() for key, value in self.issue_ingress_sources.items())):
            raise ConfigInvalid("issue_ingress_sources requires explicit 22/27 absolute config paths")
        if type(self.issue_ingress_max_body_bytes) is not int or not 1024 <= self.issue_ingress_max_body_bytes <= 8192:
            raise ConfigInvalid("issue_ingress_max_body_bytes must be 1024..8192")
        if (not isinstance(self.issue_ingress_result_files, tuple)
                or len(self.issue_ingress_result_files) > 8
                or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+\.(?:md|json|mmd)", name)
                       for name in self.issue_ingress_result_files)):
            raise ConfigInvalid("issue_ingress_result_files must contain bounded plain artifact names")

    def git_base(self, token: str) -> str:
        """Return the git base URL with the token substituted.

        The token is inserted in-memory and never written to disk or logs.
        The URL is intended only for the in-process git subprocess env.
        """
        if not token:
            raise ConfigInvalid("git_base requires a non-empty token")
        return self.git_base_template.format(token=token)

    @property
    def full_repo(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def pem_path(self) -> Optional[Path]:
        """If app_private_key starts with @, treat it as a file path."""
        if self.app_private_key.startswith("@"):
            p = Path(self.app_private_key[1:])
            if not p.is_file():
                raise ConfigInvalid(f"PEM file not found: {p}")
            return p
        return None


def _from_env() -> dict:
    """Read config from env vars, with empty strings filtered out."""
    fields = {
        "app_id": os.environ.get("GITHUB_BRIDGE_APP_ID", "").strip(),
        "installation_id": os.environ.get("GITHUB_BRIDGE_INSTALLATION_ID", "").strip(),
        "app_private_key": os.environ.get("GITHUB_BRIDGE_APP_PEM", "").strip(),
        "owner": os.environ.get("GITHUB_BRIDGE_OWNER", "").strip(),
        "repo": os.environ.get("GITHUB_BRIDGE_REPO", "").strip(),
        "branch_prefix": os.environ.get("GITHUB_BRIDGE_BRANCH_PREFIX", DEFAULT_BRANCH_PREFIX).strip(),
        "api_base": os.environ.get("GITHUB_BRIDGE_API_BASE", DEFAULT_API_BASE).strip(),
        "state_dir": os.environ.get("GITHUB_BRIDGE_STATE_DIR", DEFAULT_STATE_DIR).strip(),
        "request_timeout": os.environ.get("GITHUB_BRIDGE_TIMEOUT", str(DEFAULT_TIMEOUT)).strip(),
        "max_retries": os.environ.get("GITHUB_BRIDGE_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)).strip(),
        "operator_safety": os.environ.get("GITHUB_BRIDGE_OPERATOR_SAFETY", "fail-closed").strip(),
        "issue_ingress_repository_id": os.environ.get("GITHUB_BRIDGE_ISSUE_REPOSITORY_ID", "").strip(),
        "issue_ingress_app_id": os.environ.get("GITHUB_BRIDGE_ISSUE_APP_ID", "").strip(),
        "issue_ingress_installation_id": os.environ.get("GITHUB_BRIDGE_ISSUE_INSTALLATION_ID", "").strip(),
        "issue_ingress_actor_id": os.environ.get("GITHUB_BRIDGE_ISSUE_ACTOR_ID", "").strip(),
        "trusted_commenters": [s.strip() for s in os.environ.get("GITHUB_BRIDGE_TRUSTED_COMMENTERS", "").split(",") if s.strip()],
    }
    return {k: v for k, v in fields.items() if v}


def _from_file(path: Path) -> dict:
    """Read config from JSON file. PEM can be inline or @file:path form."""
    if not path.is_file():
        raise ConfigMissing(f"config file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigInvalid(f"config file is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ConfigInvalid("config file must contain a JSON object")
    return data


def load_config(explicit_path: Optional[Path] = None,
                 partial: bool = False) -> BridgeConfig:
    """Load and validate the bridge config.

    Resolution order (later wins):
        1. Defaults (DEFAULT_*).
        2. Config file (GITHUB_BRIDGE_CONFIG or explicit path).
        3. Environment variables.

    Args:
        explicit_path: override the config file path.
        partial: when True, allow missing ``installation_id`` (and
                 ``app_id`` if the operator wants to provide only the
                 App slug). The returned config still has the defaults
                 (``0``) for missing fields. The caller is expected to
                 call ``resolve`` to fill them in before any other
                 operation. ``validate()`` is NOT called in partial
                 mode because placeholder values would fail it.
    """
    raw: dict = {}
    file_path = explicit_path or os.environ.get("GITHUB_BRIDGE_CONFIG")
    if file_path:
        raw.update(_from_file(Path(file_path)))
    raw.update(_from_env())  # env overrides file

    # Required keys.
    if partial:
        # Partial: app_id, app_private_key, owner, repo are required.
        # installation_id and app_id (the latter only when no slug path
        # is used) may be filled later.
        required = ("app_private_key", "owner", "repo")
    else:
        required = ("app_id", "installation_id", "app_private_key", "owner", "repo")
    missing = [k for k in required if k not in raw or not raw[k]]
    if missing:
        raise ConfigMissing(
            f"missing required config: {', '.join(missing)}. "
            "See docs/OPERATOR_SETUP.md for the one-time setup."
        )

    # Coerce types. Missing numeric IDs default to 0 (placeholder).
    def _int_or_zero(k):
        v = raw.get(k, 0)
        return int(v) if v not in (None, "") else 0

    def _optional_id(k):
        v = raw.get(k)
        return None if v in (None, "") else int(v)

    try:
        cfg = BridgeConfig(
            app_id=_int_or_zero("app_id"),
            installation_id=_int_or_zero("installation_id"),
            app_private_key=str(raw["app_private_key"]).strip(),
            owner=str(raw["owner"]).strip(),
            repo=str(raw["repo"]).strip(),
            branch_prefix=str(raw.get("branch_prefix", DEFAULT_BRANCH_PREFIX)).strip(),
            api_base=str(raw.get("api_base", DEFAULT_API_BASE)).strip(),
            git_base_template=DEFAULT_GIT_BASE_TEMPLATE,
            state_dir=Path(raw.get("state_dir", DEFAULT_STATE_DIR)),
            request_timeout=int(raw.get("request_timeout", DEFAULT_TIMEOUT)),
            max_retries=int(raw.get("max_retries", DEFAULT_MAX_RETRIES)),
            operator_safety=str(raw.get("operator_safety", "fail-closed")),
            issue_ingress_repository_id=_optional_id("issue_ingress_repository_id"),
            issue_ingress_app_id=_optional_id("issue_ingress_app_id"),
            issue_ingress_installation_id=_optional_id("issue_ingress_installation_id"),
            issue_ingress_actor_id=_optional_id("issue_ingress_actor_id"),
            issue_ingress_issue_number=raw.get("issue_ingress_issue_number", 22),
            issue_ingress_max_body_bytes=raw.get("issue_ingress_max_body_bytes", 1024),
            issue_ingress_executor_profile=raw.get("issue_ingress_executor_profile"),
            issue_ingress_session_id=raw.get("issue_ingress_session_id"),
            issue_ingress_result_files=tuple(raw.get("issue_ingress_result_files", ())),
            issue_ingress_sources=raw.get("issue_ingress_sources", {}),
            trusted_commenters=frozenset(str(x).strip().lower() for x in (raw.get("trusted_commenters") or ["isvgithub"]) if str(x).strip()),
        )
    except (ValueError, TypeError) as e:
        raise ConfigInvalid(f"config field has wrong type: {e}") from e

    if not partial:
        cfg.validate()
    return cfg
