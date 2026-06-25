from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from hermes_cli.signal_coo import ActionLedger
from hermes_cli.signal_coo.gtm_radar_adapter import build_torben_gtm_radar_adapter


@pytest.mark.asyncio
async def test_signal_gtm_reply_router_sends_ack_and_persists(monkeypatch, tmp_path):
    from gateway import run as gateway_run
    from gateway.run import GatewayRunner

    monkeypatch.setenv("TORBEN_GTM_GROK_DRAFTING", "0")
    profile_home = tmp_path / "torben"
    ledger = ActionLedger(profile_home / "state" / "torben-action-ledger.json")
    now = datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc)
    build_torben_gtm_radar_adapter(
        {
            "generated_at": "2026-06-25T11:45:00Z",
            "scanned_count": 1,
            "findings": [
                {
                    "id": "gtm-1",
                    "fingerprint": "gtm-finding-1",
                    "title": "Poisoned Playbooks",
                    "summary": "RAG security agents can be steered by poisoned knowledge sources.",
                    "content_route": "longform_article",
                    "pillar": "security_ai",
                }
            ],
        },
        ledger=ledger,
        state_path=profile_home / "state" / "gtm-state.json",
        now=now,
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", profile_home)

    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.SIGNAL: adapter}
    runner.session_store = MagicMock()
    session_entry = SimpleNamespace(session_id="session-1", session_key="signal:user")
    source = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="+15105553337",
        user_id="+15105553337",
        user_name="Eric Freeman",
        chat_type="dm",
    )
    event = MessageEvent(
        text="draft 1",
        source=source,
        message_id="signal-message-1",
    )

    result = await runner._maybe_handle_torben_gtm_reply(
        event=event,
        source=source,
        session_entry=session_entry,
        history=[],
        persist_user_message="draft 1",
        persist_user_timestamp=(now + timedelta(minutes=1)).timestamp(),
    )

    assert result is not None
    assert result["api_calls"] == 0
    assert result["torben_gtm_reply_router"]["status"] == "content_package_staged"
    adapter.send.assert_awaited_once()
    sent_text = adapter.send.await_args.args[1]
    assert "approval-ready Magnus package" in sent_text
    assert "not as a generic chat reply" in sent_text
    assert "LinkedIn opener:" in sent_text
    assert "Reply with: approve article, approve linkedin, approve x thread, revise, or hold." in sent_text
    assert "Nothing has been posted" in sent_text
    payload = result["torben_gtm_reply_router"]["payload"]
    assert payload["content_package_status"] == "approval_required"
    assert payload["drafts"]["linkedin_post"]["body"]
    assert payload["drafts"]["x_thread"]["posts"]
    assert payload["visual_plan"]["image_prompt"]
    assert runner.session_store.append_to_transcript.call_count == 3
    assert runner.session_store.update_session.call_args.kwargs["last_prompt_tokens"] == 0
