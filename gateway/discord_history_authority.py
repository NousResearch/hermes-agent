"""In-process authority binding for bounded Discord guild history reads.

The model authors only the exact Discord target and pagination cursor.  The
authenticated requester (interactive Discord) or reviewed job id (cron) comes
from task-local runtime context and is therefore not a model-controlled field.
This module makes no semantic decision; it only binds an identity to an exact
deployment-reviewed target set.
"""

from __future__ import annotations

import re
from contextvars import ContextVar, Token
from types import MappingProxyType
from typing import Any, Final, Mapping

from gateway.discord_connector_protocol import DiscordConnectorHistoryAuthority
from gateway.session_context import get_session_env


CONTROL_TOWER_CHANNEL_ID: Final = "1504852355588423801"
VOICE_DIGEST_THREAD_ID: Final = "1524321461714681976"
CANARY_REQUESTER_USER_ID: Final = "1279454038731264061"
CANARY_HISTORY_READER_SERVICE_UNIT: Final = (
    "muncho-capability-producer-discord-edge.service"
)
CANARY_HISTORY_READER_SERVICE_USER: Final = "muncho-cap-discord"

# Code-owned scheduler authority.  Delivery/origin metadata is deliberately
# absent: it is a return address, not proof of who is requesting a read.
REVIEWED_PRODUCTION_CRON_HISTORY_TARGETS: Final[Mapping[str, frozenset[str]]] = (
    MappingProxyType(
        {
            "06ef64d72891": frozenset({CONTROL_TOWER_CHANNEL_ID}),
            "e62f55ca93ca": frozenset({VOICE_DIGEST_THREAD_ID}),
        }
    )
)

_CRON_JOB_RE = re.compile(r"^[0-9a-f]{12}$")
_SNOWFLAKE_RE = re.compile(r"^[1-9][0-9]{0,24}$")
_UNBOUND: Final = object()
_CURRENT_CRON_JOB_ID: ContextVar[Any] = ContextVar(
    "discord_history_current_cron_job_id",
    default=_UNBOUND,
)


class DiscordHistoryAuthorityError(RuntimeError):
    """Stable, content-free failure while deriving history authority."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def bind_cron_history_job(job_id: Any) -> Token[Any]:
    """Bind one scheduler-owned job id for the current execution context.

    Validation happens at consumption time so an unknown or malformed job can
    be represented and proven to fail closed by the real tool path.
    """

    return _CURRENT_CRON_JOB_ID.set(job_id)


def reset_cron_history_job(token: Token[Any]) -> None:
    """Restore the prior task-local cron binding."""

    _CURRENT_CRON_JOB_ID.reset(token)


def resolve_discord_history_authority(
    channel_id: Any,
) -> DiscordConnectorHistoryAuthority:
    """Derive immutable authority for one exact requested Discord target."""

    requested = str(channel_id or "").strip()
    if _SNOWFLAKE_RE.fullmatch(requested) is None:
        raise DiscordHistoryAuthorityError("discord_history_target_invalid")

    cron_job_id = _CURRENT_CRON_JOB_ID.get()
    if cron_job_id is not _UNBOUND:
        if not isinstance(cron_job_id, str) or _CRON_JOB_RE.fullmatch(cron_job_id) is None:
            raise DiscordHistoryAuthorityError("discord_history_cron_context_invalid")
        allowed_targets = REVIEWED_PRODUCTION_CRON_HISTORY_TARGETS.get(cron_job_id)
        if allowed_targets is None:
            raise DiscordHistoryAuthorityError("discord_history_cron_not_reviewed")
        if requested not in allowed_targets:
            raise DiscordHistoryAuthorityError(
                "discord_history_cron_target_not_reviewed"
            )
        return DiscordConnectorHistoryAuthority.reviewed_cron(cron_job_id)

    platform = get_session_env("HERMES_SESSION_PLATFORM", "").strip().casefold()
    requester_user_id = get_session_env("HERMES_SESSION_USER_ID", "").strip()
    if platform != "discord" or _SNOWFLAKE_RE.fullmatch(requester_user_id) is None:
        raise DiscordHistoryAuthorityError("discord_history_requester_context_missing")
    return DiscordConnectorHistoryAuthority.authenticated_user(requester_user_id)


def reviewed_cron_history_targets_json() -> dict[str, list[str]]:
    """Return the canonical non-secret projection used by sealed config tests."""

    return {
        job_id: sorted(targets)
        for job_id, targets in sorted(
            REVIEWED_PRODUCTION_CRON_HISTORY_TARGETS.items()
        )
    }


__all__ = [
    "CONTROL_TOWER_CHANNEL_ID",
    "CANARY_HISTORY_READER_SERVICE_UNIT",
    "CANARY_HISTORY_READER_SERVICE_USER",
    "CANARY_REQUESTER_USER_ID",
    "VOICE_DIGEST_THREAD_ID",
    "DiscordHistoryAuthorityError",
    "REVIEWED_PRODUCTION_CRON_HISTORY_TARGETS",
    "bind_cron_history_job",
    "reset_cron_history_job",
    "resolve_discord_history_authority",
    "reviewed_cron_history_targets_json",
]
