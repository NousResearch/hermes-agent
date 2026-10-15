"""Agent Sessions vs legacy assistant.threads status helpers.

``agents.sessions.setStatus`` (slack-sdk ≥ 3.44) accepts only the lifecycle
enum ``active|processing|suspended|closed``. Hermes still builds a human
phrase for the thread footer; this module maps that phrase to the enum and
falls back to ``assistant.threads.setStatus`` when the new API refuses.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Workspaces whose agents.sessions.setStatus rejected a valid enum. Heartbeats
# are ~2s; retrying a permanent app-capability miss would just spam Slack.
# Transient errors are not stored.
_AGENT_SESSIONS_STATUS_REJECTED: set[str] = set()
_AGENT_SESSIONS_STATUS_WARNED = False
_AGENT_SESSIONS_STATUS_HARD_FAILS = frozenset({
    "not_authorized", "invalid_arguments", "missing_scope",
})


def reset_status_runtime_state() -> None:
    """Test seam: clear per-process fallback flags without leaking across cases."""
    global _AGENT_SESSIONS_STATUS_WARNED
    _AGENT_SESSIONS_STATUS_REJECTED.clear()
    _AGENT_SESSIONS_STATUS_WARNED = False


def agent_session_status_enum(status: str) -> str:
    """Map the human status phrase (or empty clear) to the Agent Sessions enum.

    A non-empty working-state phrase becomes ``processing``; the empty clear
    (``stop_typing``) becomes ``active``.
    """
    return "processing" if (status or "").strip() else "active"


def _api_error_code(err: BaseException) -> str:
    response = getattr(err, "response", None)
    if not isinstance(response, dict):
        return ""
    code = response.get("error")
    return code if isinstance(code, str) else ""


def _log_agent_sessions_status_failure(fail_label: str, err: BaseException) -> None:
    """Surface the first Agent Sessions status rejection; later ones stay DEBUG."""
    global _AGENT_SESSIONS_STATUS_WARNED
    if _AGENT_SESSIONS_STATUS_WARNED:
        logger.debug(
            "[Slack] agents.sessions.setStatus %s: %s; retrying via assistant.threads",
            fail_label, err, exc_info=True)
        return
    _AGENT_SESSIONS_STATUS_WARNED = True
    logger.warning(
        "[Slack] agents.sessions.setStatus %s: %s; falling back to "
        "assistant.threads.setStatus (that API is enum-only: "
        "active|processing|suspended|closed; not_authorized means the app "
        "is not on Agent Sessions). Later failures stay at DEBUG.",
        fail_label, err, exc_info=True)


def _remember_agent_sessions_status_rejection(team_id: str, err: BaseException) -> None:
    """Skip Agent Sessions status for this workspace after a permanent API refusal."""
    if _api_error_code(err) in _AGENT_SESSIONS_STATUS_HARD_FAILS:
        _AGENT_SESSIONS_STATUS_REJECTED.add(team_id or "")


async def apply_thread_status(
    client: Any,
    chat_id: str,
    team_id: str,
    thread_ts: str,
    status: str,
    fail_label: str,
    *,
    sdk_supports: bool,
) -> None:
    """Set the thread status line on ``client``.

    Agent Sessions gets the lifecycle enum; a failure there falls back to
    ``assistant.threads.setStatus`` with the human phrase (empty string clears).
    """
    workspace = team_id or ""
    if sdk_supports and workspace not in _AGENT_SESSIONS_STATUS_REJECTED:
        method = getattr(client, "agents_sessions_setStatus", None)
        if method is not None:
            try:
                await method(
                    channel_id=chat_id, thread_ts=thread_ts,
                    status=agent_session_status_enum(status))
                return
            except Exception as e:  # health: allow BLE001 -- Slack/network; traceback logged; turn must continue
                _log_agent_sessions_status_failure(fail_label, e)
                _remember_agent_sessions_status_rejection(workspace, e)
    try:
        await client.assistant_threads_setStatus(
            channel_id=chat_id, thread_ts=thread_ts, status=status)
    except Exception as e:  # health: allow BLE001 -- Slack/network; traceback logged; turn must continue
        logger.debug("[Slack] assistant.threads.setStatus %s: %s", fail_label, e, exc_info=True)
