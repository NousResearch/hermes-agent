"""Slack working-status line: how a rejected ``assistant.threads.setStatus`` write is logged.

The typing refresh re-sends the status every few seconds for the whole turn, so a write Slack
rejects fails on every tick. The first rejection per (workspace, Slack error) is a WARNING that
names the method and Slack's error code; repeats stay at DEBUG.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_status_failure(
    warned: set[tuple[str, str]], team_id: str, fail_label: str, exc: Exception) -> None:
    """WARNING the first time ``team_id`` hits this error, DEBUG after; ``warned`` holds the (team,
    error) pairs already reported. The error is Slack's ``error`` field, else the exception type."""
    response = getattr(exc, "response", None)  # SlackApiError: a SlackResponse (``.data``) or dict
    payload = response if isinstance(response, dict) else getattr(response, "data", None)
    error = (payload.get("error") if isinstance(payload, dict) else None) or type(exc).__name__
    if (team_id, error) in warned:
        logger.debug("[Slack] assistant.threads.setStatus %s: %s", fail_label, exc)
        return
    warned.add((team_id, error))
    logger.warning(
        "[Slack] assistant.threads.setStatus %s (%s) in workspace %s, so the working-status line "
        "is not updated; repeats are logged at DEBUG: %s", fail_label, error, team_id or "-", exc)
