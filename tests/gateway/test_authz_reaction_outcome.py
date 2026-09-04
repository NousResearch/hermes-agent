"""An authorization refusal must not paint the success reaction.

`_process_message_background` scored a turn with `not bool(response)`, which
cannot tell "the handler deliberately produced no output" from "the message
was refused before any handler ran". A rejected sender therefore saw the
in-progress reaction swapped for the ack reaction and no reply: it reads as
"the agent acknowledged me and then broke".

`ProcessingOutcome.CANCELLED` already means "leave the message unreacted",
which is exactly what a refusal should look like.
"""

from __future__ import annotations

import pytest

from gateway.platforms.base import ProcessingOutcome


class _Recorder:
    """Minimal stand-in for the shared reaction-ack flow."""

    _OK_EMOJI = "✅"
    _FAIL_EMOJI = "❌"

    def __init__(self):
        self.added: list[str] = []
        self.removed = 0

    async def _add_reaction(self, chat_id, message_id, emoji):
        self.added.append(emoji)

    async def _remove_reaction(self, chat_id, message_id):
        self.removed += 1


class _Source:
    chat_id = "c1"


class _Event:
    def __init__(self):
        self.source = _Source()
        self.message_id = "m1"


async def _run_hook(outcome: ProcessingOutcome) -> _Recorder:
    from gateway.platforms.base import BasePlatformAdapter

    rec = _Recorder()
    await BasePlatformAdapter.on_processing_complete(rec, _Event(), outcome)
    return rec


@pytest.mark.asyncio
async def test_cancelled_leaves_the_message_unreacted():
    """The contract this fix relies on: CANCELLED paints nothing."""
    rec = await _run_hook(ProcessingOutcome.CANCELLED)

    assert rec.added == [], "a cancelled turn must not paint any reaction"
    assert rec.removed == 1, "the in-progress reaction must still be cleared"


@pytest.mark.asyncio
async def test_success_still_paints_the_ack():
    """Guard: ordinary completed turns keep their ack reaction."""
    rec = await _run_hook(ProcessingOutcome.SUCCESS)

    assert rec.added == ["✅"]


@pytest.mark.asyncio
async def test_failure_still_paints_the_failure_reaction():
    """Guard: genuine failures keep their failure reaction."""
    rec = await _run_hook(ProcessingOutcome.FAILURE)

    assert rec.added == ["❌"]


def test_refused_event_maps_to_cancelled():
    """The mapping the background path applies, exercised directly.

    Mirrors the expression in _process_message_background: a refused event
    short-circuits to CANCELLED regardless of the response/delivery scoring
    that would otherwise have produced SUCCESS.
    """

    def _outcome(event, *, delivery_attempted=False, delivery_succeeded=False, response=None):
        authorization_refused = bool(
            getattr(event, "_hermes_authorization_refused", False)
        )
        processing_ok = delivery_succeeded if delivery_attempted else not bool(response)
        if authorization_refused:
            return ProcessingOutcome.CANCELLED
        return ProcessingOutcome.SUCCESS if processing_ok else ProcessingOutcome.FAILURE

    refused = _Event()
    refused._hermes_authorization_refused = True
    # The reported shape: no delivery, no response. Pre-fix this scored SUCCESS.
    assert _outcome(refused) is ProcessingOutcome.CANCELLED

    # An ordinary suppressed/no-op turn is unchanged.
    assert _outcome(_Event()) is ProcessingOutcome.SUCCESS
    # A delivered turn is unchanged.
    assert _outcome(
        _Event(), delivery_attempted=True, delivery_succeeded=True
    ) is ProcessingOutcome.SUCCESS
    # A failed delivery is unchanged.
    assert _outcome(
        _Event(), delivery_attempted=True, delivery_succeeded=False
    ) is ProcessingOutcome.FAILURE


@pytest.mark.asyncio
async def test_handle_message_marks_an_unauthorized_sender(monkeypatch, tmp_path):
    """End of the chain: the real refusal path sets the marker the hook reads.

    Drives ``GatewayRunner._handle_message`` with an unauthorized group sender
    (the reported shape: allowed channel, no allowlist entry for the user), so
    the marker is proven to come from production code rather than the test.
    """
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.base import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    runner = GatewayRunner(GatewayConfig())
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chan-1",
        chat_type="group",
        user_id="9999",
        user_name="stranger",
    )
    event = MessageEvent(text="hello", source=source, internal=False)

    assert runner._is_user_authorized(source) is False, "test needs an unauthorized sender"

    result = await runner._handle_message(event)

    assert result is None
    assert getattr(event, "_hermes_authorization_refused", False) is True, (
        "the refusal was not marked, so the reaction flow would still score it SUCCESS"
    )
