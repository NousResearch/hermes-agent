"""Resolve gateway requesters to Git/GitHub attribution identities.

The checked-in mapping is intentionally small and explicit.  It is used only
for interactive Slack-originated sessions; unattended jobs with no requester
continue to use the process/default bot identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional

from gateway.config import Platform

CONFIG_RELATIVE_PATH = "config/requester_identities.json"
CONFIG_PATH = Path(__file__).resolve().parents[1] / CONFIG_RELATIVE_PATH


@dataclass(frozen=True)
class RequesterIdentity:
    slack_user_id: str
    name: str
    email: str
    github_login: str

    def as_env(self, bot_email: str = "") -> dict[str, str]:
        env = {
            "GIT_AUTHOR_NAME": self.name,
            "GIT_AUTHOR_EMAIL": self.email,
            "GIT_COMMITTER_NAME": self.name,
            "GIT_COMMITTER_EMAIL": self.email,
            "HERMES_REQUESTER_GITHUB_LOGIN": self.github_login,
            "HERMES_REQUESTER_NAME": self.name,
            "HERMES_REQUESTER_EMAIL": self.email,
        }
        if bot_email:
            env["HERMES_BOT_GIT_EMAIL"] = bot_email
        return env


def _load_mapping(path: Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{CONFIG_RELATIVE_PATH} must contain a JSON object")
    return data


def resolve_slack_requester_identity(
    slack_user_id: Optional[str],
    *,
    path: Path = CONFIG_PATH,
) -> Optional[RequesterIdentity]:
    """Return mapped identity for *slack_user_id*, or None if unmapped/absent."""

    if not slack_user_id:
        return None
    mapping = _load_mapping(path)
    raw = mapping.get(str(slack_user_id))
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Identity mapping for {slack_user_id} must be an object")
    missing = [k for k in ("name", "email", "github_login") if not str(raw.get(k, "")).strip()]
    if missing:
        raise ValueError(
            f"Identity mapping for {slack_user_id} is missing: {', '.join(missing)}"
        )
    return RequesterIdentity(
        slack_user_id=str(slack_user_id),
        name=str(raw["name"]).strip(),
        email=str(raw["email"]).strip(),
        github_login=str(raw["github_login"]).strip().lstrip("@"),
    )


def should_require_requester_identity(platform: Any, user_id: Optional[str]) -> bool:
    """Only Slack human-triggered sessions require attribution mapping.

    Cron/unattended jobs have no Slack requester and must keep current bot-authored
    behavior. Non-Slack gateway platforms are intentionally out of phase-1 scope.
    """

    value = platform.value if hasattr(platform, "value") else str(platform or "")
    return value == Platform.SLACK.value and bool(user_id)


def format_missing_identity_message(slack_handle: str) -> str:
    handle = slack_handle or "Slack user"
    return (
        f"No GitHub identity mapped for {handle}; add yourself to "
        f"{CONFIG_RELATIVE_PATH} and retry."
    )


def requester_identity_prompt(identity: RequesterIdentity, bot_email: str) -> str:
    """System prompt fragment forcing commit/PR attribution behavior."""

    coauthor = f"citizen-wall-e <{bot_email}>" if bot_email else "citizen-wall-e <bot-email>"
    return (
        "## Slack requester GitHub attribution\n"
        f"This session was requested by @{identity.github_login} via Slack. "
        "When you create commits in this session, the terminal environment is "
        "preconfigured with the requester's Git author/committer identity.\n"
        "\n"
        "Required for any PR-producing work in this session:\n"
        f"- Every commit message must end with `Co-Authored-By: {coauthor}`.\n"
        f"- Every pull request body must start with `Requested by @{identity.github_login} via Slack`.\n"
        f"- Every `gh pr create` must assign @{identity.github_login} (for example `--assignee {identity.github_login}`).\n"
        "Do not silently fall back to bot-only attribution.\n"
    )
