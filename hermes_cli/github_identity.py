"""Central credential boundary for Hermes-owned GitHub automation."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Mapping, Sequence


_LOGIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
_TOKEN_ENV = re.compile(r"^HERMES_[A-Z0-9_]*GITHUB[A-Z0-9_]*TOKEN$")
_SHARED_TOKEN_ENVS = frozenset({"GH_TOKEN", "GITHUB_TOKEN"})


class GitHubIdentityError(RuntimeError):
    """The dedicated Hermes GitHub identity was absent or did not match."""


@dataclass(frozen=True, slots=True)
class GitHubAutomationIdentity:
    expected_login: str
    token_env: str = "HERMES_GITHUB_BOT_TOKEN"

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "GitHubAutomationIdentity":
        source = os.environ if environ is None else environ
        login = source.get("HERMES_GITHUB_BOT_LOGIN", "").strip()
        if not _LOGIN.fullmatch(login):
            raise GitHubIdentityError("HERMES_GITHUB_BOT_LOGIN is required")
        return cls(login)

    def command_environment(
        self, environ: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        if self.token_env in _SHARED_TOKEN_ENVS or not _TOKEN_ENV.fullmatch(self.token_env):
            raise GitHubIdentityError("a dedicated Hermes GitHub token variable is required")
        source = os.environ if environ is None else environ
        token = source.get(self.token_env, "")
        if not token:
            raise GitHubIdentityError(f"{self.token_env} is required")
        child = dict(source)
        child.pop("GH_TOKEN", None)
        child.pop("GITHUB_TOKEN", None)
        child["GH_TOKEN"] = token
        return child

    def git_command_environment(
        self, environ: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        """Bind Git HTTPS to the bot token without exposing it in argv."""

        child = self.command_environment(environ)
        encoded = base64.b64encode(f"x-access-token:{child['GH_TOKEN']}".encode()).decode()
        child.update(
            {
                "GIT_CONFIG_COUNT": "2",
                "GIT_CONFIG_KEY_0": "credential.helper",
                "GIT_CONFIG_VALUE_0": "",
                "GIT_CONFIG_KEY_1": "http.https://github.com/.extraheader",
                "GIT_CONFIG_VALUE_1": f"AUTHORIZATION: basic {encoded}",
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
        return child


def run_as_github_automation(
    argv: Sequence[str],
    *,
    identity: GitHubAutomationIdentity | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    timeout: float = 30,
) -> subprocess.CompletedProcess[str]:
    """Verify the bot viewer, then run one fixed-argv ``gh`` command."""

    selected = identity or GitHubAutomationIdentity.from_environment(environ)
    if not argv or os.path.basename(str(argv[0])).casefold() != "gh":
        raise GitHubIdentityError("GitHub automation may execute only gh")
    child_env = selected.command_environment(environ)
    viewer = subprocess.run(
        ["gh", "api", "user"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=child_env,
    )
    try:
        payload = json.loads(viewer.stdout) if viewer.returncode == 0 else None
    except json.JSONDecodeError:
        payload = None
    login = payload.get("login") if isinstance(payload, dict) else None
    if not isinstance(login, str) or login.casefold() != selected.expected_login.casefold():
        raise GitHubIdentityError("Hermes GitHub automation identity did not match")
    return subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=child_env,
        cwd=cwd,
    )
