"""An admitted Dashboard/TUI write to a Telegram topic must not be re-authorized as a bot message.

The viewer's identity is proven at SessionAuthority.submit and committed as local_operator_v1.
A Telegram group/topic has no single source.user_id; the ordinary native ingress gate would drop
the restored event unless execute_admission keeps that already-authorized turn internal.
"""

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_forwarded_operator_turn_survives_telegram_topic_ingress(owner):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run_inbound import GatewayInboundMixin
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession
    from gateway.session_contract import SessionActor, SessionRef, Submission
    from gateway.session_ingress import execute_admission
    from gateway.session_operator import check_local_input
    from hermes_state_runtime import claim_session_input

    sid = "telegram-group-topic"
    ref = SessionRef("default", sid)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="-100123", chat_type="group",
        thread_id="42", user_id=None, profile="default",
    )
    owner.db.create_session(sid, source="telegram")
    owner.sessions[sid] = LiveSession(source, "telegram:default:-100123:42")
    owner._publish_pending = lambda _ref: None
    owner._schedule = lambda _ref: None

    operator = SessionActor(
        "authenticated-dashboard", "default",
        frozenset({"session:submit", "session:operator"}), "desktop",
    )
    await owner.submit(
        operator, Submission("typed-on-topic", ref, {"text": "hello from Desktop"}, "queue"),
    )
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id=sid)
    assert row["payload"]["local_operator_v1"] == {
        "profile_id": "default", "session_id": sid, "principal_id": operator.subject,
    }
    check_local_input(owner, ref, row)  # real durable provenance verification

    async def identity(event, _source):
        return event

    # Model an unallowlisted anonymous group/topic source: ordinary Telegram messages must
    # still fail the native auth gate, while this authorized operator's turn must bypass it.
    gate = SimpleNamespace(
        _scale_to_zero_note_real_inbound=lambda: None,
        _hm_pre_gateway_dispatch_hook=identity,
        _is_user_authorized_for_source=lambda _source: False,
    )
    assert await GatewayInboundMixin._hm_admit_event(
        gate, MessageEvent(text="anonymous Telegram message", source=source),
    ) is None

    admitted = []

    async def handle(event):
        admitted.append(await GatewayInboundMixin._hm_admit_event(gate, event) is not None)
        return None

    owner.runner._handle_message = handle
    owner.runner._adapter_for_source = lambda _source: None
    await execute_admission(owner, ref, row)

    # Reverting gateway/session_ingress.py to the PR merge base makes this [False]:
    # local_operator_v1 is reconstructed with internal=False and dropped by the native gate.
    assert admitted == [True]
