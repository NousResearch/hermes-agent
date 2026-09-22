"""Credential boundary owned by the GitHub PR feedback plugin."""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from typing import Mapping


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
        if environ is None:
            from agent.secret_scope import get_github_automation_secret

            login = get_github_automation_secret("HERMES_GITHUB_BOT_LOGIN")
        else:
            login = source.get("HERMES_GITHUB_BOT_LOGIN", "")
        login = login.strip()
        if not _LOGIN.fullmatch(login):
            raise GitHubIdentityError("HERMES_GITHUB_BOT_LOGIN is required")
        return cls(login)

    def command_environment(
        self, environ: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        if self.token_env in _SHARED_TOKEN_ENVS or not _TOKEN_ENV.fullmatch(self.token_env):
            raise GitHubIdentityError("a dedicated Hermes GitHub token variable is required")
        source = os.environ if environ is None else environ
        if environ is None:
            from agent.secret_scope import get_github_automation_secret

            token = get_github_automation_secret(self.token_env)
        else:
            token = source.get(self.token_env, "")
        if not token:
            raise GitHubIdentityError(f"{self.token_env} is required")
        child = dict(source)
        child.pop("GH_TOKEN", None)
        child.pop("GITHUB_TOKEN", None)
        child[self.token_env] = token
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
