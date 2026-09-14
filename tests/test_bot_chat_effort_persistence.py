"""Bot-chat reasoning-effort pick persistence (config.set reasoning for Bot Chats).

Regression coverage for the two-part fix:
1. `_set_reasoning` writes the pick into the bot profile config when the
   session is a bot chat (the desktop picker sends no scope, so the pick
   would otherwise stay session-scoped and snap back on rebuild).
2. Resume paths re-derive the bot-chat marker from the stored row, because
   creation stamps it from params but resume never did.
"""

import contextlib
import json
from unittest import mock

import pytest

import tui_gateway.server as server


# ── _row_follow_profile_config ─────────────────────────────────────────

_MARKER_CASES = [
    ({}, False),
    (None, False),
    ({"title": "Bot Chat", "hidden": False}, True),
    ({"title": "Bot Chat", "hidden": False, "model_config": "{}"}, True),
    ({"title": "Some chat", "hidden": False,
      "model_config": json.dumps({"follow_profile_config": True})}, True),
    ({"title": "Some chat", "hidden": False, "model_config": "{}"}, False),
    ({"title": "Group: something", "hidden": True, "model_config": "{}"}, True),
    ({"title": "Group: something", "hidden": False, "model_config": "{}"}, False),
    ({"title": "Some chat", "hidden": False,
      "model_config": json.dumps({"room_plumbing": True})}, True),
]


@pytest.mark.parametrize("row,expected", _MARKER_CASES)
def test_row_follow_profile_config(row, expected):
    assert server._row_follow_profile_config(row) is expected


# ── _set_reasoning bot-chat persistence ────────────────────────────────

def _reasoning_params():
    return {"key": "reasoning", "value": "high", "scope": "", "session_id": ""}


def _run_set_reasoning(session):
    return server._set_reasoning("r1", _reasoning_params(), "reasoning", "high", session)


def test_set_reasoning_bot_chat_persists_to_profile_config():
    session = {"follow_profile_config": True, "agent": None}
    with mock.patch.object(server, "_session_profile_runtime_scope",
                           side_effect=lambda s: contextlib.nullcontext()), \
         mock.patch.object(server, "_write_config_key") as write:
        _run_set_reasoning(session)
    write.assert_called_once_with("agent.reasoning_effort", "high")
    assert session["create_reasoning_override"] is not None


def test_set_reasoning_bot_chat_title_detection():
    session = {"title": "Bot Chat", "agent": None}
    with mock.patch.object(server, "_session_profile_runtime_scope",
                           side_effect=lambda s: contextlib.nullcontext()), \
         mock.patch.object(server, "_write_config_key") as write:
        _run_set_reasoning(session)
    write.assert_called_once_with("agent.reasoning_effort", "high")


def test_set_reasoning_plain_session_stays_session_scoped():
    session = {"agent": None}
    with mock.patch.object(server, "_session_profile_runtime_scope",
                           side_effect=lambda s: contextlib.nullcontext()), \
         mock.patch.object(server, "_write_config_key") as write:
        _run_set_reasoning(session)
    write.assert_not_called()
    assert session["create_reasoning_override"] is not None
