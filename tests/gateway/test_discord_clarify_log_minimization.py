"""INFO log minimization for the Discord clarify choice callbacks.

The resolve-time INFO record must identify the clarify request and the chosen
option index without copying the operator's answer text or display name into
operational logs: choice text is arbitrary conversational content and may
include private business or personal details.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Repo root importable
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

# Triggers the shared discord mock from tests/gateway/conftest.py before
# importing the production module.
from plugins.platforms.discord.adapter import ClarifyChoiceView  # noqa: E402


_SECRET_CHOICE = "SYNTHETIC_PRIVATE_CHOICE_123"
_SECRET_NAME = "Private Tester Name"


def _clear_clarify_state():
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


def _make_interaction(*, user_id="42", display_name=_SECRET_NAME):
    user = MagicMock()
    user.id = user_id
    user.display_name = display_name
    response = MagicMock()
    response.edit_message = AsyncMock()
    response.send_message = AsyncMock()
    response.defer = AsyncMock()
    return MagicMock(user=user, response=response, message=None)


def _resolved_records(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if "clarify button resolved" in r.getMessage()
    ]


@pytest.mark.asyncio
async def test_resolved_info_log_omits_choice_text_and_display_name(caplog):
    """A successful numeric resolve logs id/index/ok only — the canonical
    choice text and the operator's display name must not reach INFO records."""
    from tools import clarify_gateway as cm

    _clear_clarify_state()
    cm.register("cidLogOk", "sk-LogOk", "Pick", [_SECRET_CHOICE, "other option"])
    view = ClarifyChoiceView(
        choices=[_SECRET_CHOICE, "other option"],
        clarify_id="cidLogOk",
        allowed_user_ids={"42"},
    )
    interaction = _make_interaction()

    with caplog.at_level("INFO", logger="plugins.platforms.discord.adapter"):
        await view._resolve_choice(interaction, index=0, choice=_SECRET_CHOICE)

    records = _resolved_records(caplog)
    assert records, "expected an INFO resolve record"
    logged = "\n".join(records)
    assert _SECRET_CHOICE not in logged
    assert _SECRET_NAME not in logged
    # Still identifies the request and the picked option, and reports success.
    assert "cidLogOk" in logged
    assert "choice_index=0" in logged
    assert "ok=True" in logged
    # The gateway received the exact canonical answer, not the log payload.
    with cm._lock:
        entry = cm._entries["cidLogOk"]
    assert entry.event.is_set()
    assert entry.response == _SECRET_CHOICE


@pytest.mark.asyncio
async def test_resolved_info_log_reports_unavailable_request_without_content(caplog):
    """A resolve against an expired/unknown entry reports failure without
    echoing the answer text or display name."""
    _clear_clarify_state()
    # No entry registered: resolve_gateway_clarify returns False.
    view = ClarifyChoiceView(
        choices=[_SECRET_CHOICE],
        clarify_id="cidLogMiss",
        allowed_user_ids={"42"},
    )
    interaction = _make_interaction()

    with caplog.at_level("INFO", logger="plugins.platforms.discord.adapter"):
        await view._resolve_choice(interaction, index=0, choice=_SECRET_CHOICE)

    records = _resolved_records(caplog)
    assert records, "expected an INFO resolve record"
    logged = "\n".join(records)
    assert _SECRET_CHOICE not in logged
    assert _SECRET_NAME not in logged
    assert "cidLogMiss" in logged
    assert "choice_index=0" in logged
    assert "ok=False" in logged
