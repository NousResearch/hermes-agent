"""Adapter-level regressions for request-scoped approvals (PR #125039, review F1/F2/F3).

F1 — Teams must not acknowledge a decision that never settled: the resolver's count, not the
preflight session predicate, authorizes the success label.
F2 — QQ legacy keys ending in a 32-hex identity must keep their full session; the request-scoped
encoding is versioned (v2) so the two wire formats can never collide.
F3 — Matrix must retain every live approval card for a session: presenting B for the same session
must not orphan A's registration; settling/expiring one card must not erase a sibling.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace as N

import pytest


# ── F1: Teams ──────────────────────────────────────────────────────────────────────────────


def _teams_adapter():
    from plugins.platforms.teams import adapter as teams

    obj = object.__new__(teams.TeamsAdapter)
    obj._card_action_denied = lambda from_account: None
    return teams, obj


def _teams_ctx(data: dict) -> N:
    return N(activity=N(value=N(action=N(data=data)), from_=N(id="authorized")))


class TestTeamsResolveThenAck:
    def test_stale_target_must_not_ack_success(self, monkeypatch):
        """A's card expired while B is still pending: has_blocking_approval is True (B), but the
        targeted resolve returns 0 — the card must show the expired notice, not "Allowed (once)"."""
        teams, obj = _teams_adapter()
        rendered: list = []
        monkeypatch.setattr(teams, "TextBlock", lambda **kw: kw)
        monkeypatch.setattr(teams, "InvokeResponse", lambda status, body: body, raising=False)
        monkeypatch.setattr(teams, "AdaptiveCard", lambda **kw: kw, raising=False)
        monkeypatch.setattr(
            teams, "AdaptiveCardActionCardResponse", lambda value: value, raising=False)
        obj._invoke_card = lambda body: body  # unwrap the SDK response for assertion
        monkeypatch.setattr(
            "tools.approval.has_blocking_approval", lambda _s: True)
        monkeypatch.setattr(
            "tools.approval.resolve_gateway_approval", lambda *a, **kw: 0)

        card = asyncio.run(obj._on_card_action(_teams_ctx(
            {"hermes_action": "approve_once", "session_key": "s", "request_id": "expired"})))
        rendered.append(card)
        text = card[-1]["text"]
        assert "expired" in text.lower() or "resolved" in text.lower(), text
        assert "Allowed" not in text and "Denied" not in text, text

    def test_live_target_acks_success(self, monkeypatch):
        """The exact-ID resolve returning 1 renders the decision label."""
        teams, obj = _teams_adapter()
        monkeypatch.setattr(teams, "TextBlock", lambda **kw: kw)
        obj._invoke_card = lambda body: body
        monkeypatch.setattr(
            "tools.approval.has_blocking_approval", lambda _s: True)
        monkeypatch.setattr(
            "tools.approval.resolve_gateway_approval", lambda *a, **kw: 1)

        card = asyncio.run(obj._on_card_action(_teams_ctx(
            {"hermes_action": "approve_once", "session_key": "s", "request_id": "live"})))
        assert "Allowed" in card[-1]["text"]

    def test_resolver_exception_renders_expired_not_success(self, monkeypatch):
        """A resolver failure must not render the success label either."""
        teams, obj = _teams_adapter()
        monkeypatch.setattr(teams, "TextBlock", lambda **kw: kw)
        obj._invoke_card = lambda body: body
        monkeypatch.setattr(
            "tools.approval.has_blocking_approval", lambda _s: True)

        def _boom(*a, **kw):
            raise RuntimeError("queue unavailable")

        monkeypatch.setattr("tools.approval.resolve_gateway_approval", _boom)
        card = asyncio.run(obj._on_card_action(_teams_ctx(
            {"hermes_action": "deny", "session_key": "s", "request_id": "x"})))
        assert "Denied" not in card[-1]["text"]

    def test_composed_stale_A_leaves_live_B_untouched(self, monkeypatch):
        """Stale A + live B against the real queue: B's entry and its waiter stay pending."""
        from tools import approval as _approval
        from tools.approval_gateway_wait import _ApprovalEntry

        entry_b = _ApprovalEntry({"command": "b-cmd", "request_id": "rid-b"})
        entry_a = _ApprovalEntry({"command": "a-cmd", "request_id": "rid-a"})
        SESSION = "teams-composed"
        with _approval._lock:
            _approval._gateway_queues[SESSION] = [entry_a, entry_b]
        try:
            teams, obj = _teams_adapter()
            monkeypatch.setattr(teams, "TextBlock", lambda **kw: kw)
            obj._invoke_card = lambda body: body
            monkeypatch.setattr(
                "tools.approval.has_blocking_approval", lambda s: True)

            card = asyncio.run(obj._on_card_action(_teams_ctx(
                {"hermes_action": "approve_once", "session_key": SESSION,
                 "request_id": "rid-a-stale"})))
            assert "Allowed" not in card[-1]["text"]
            with _approval._lock:
                remaining = [e.data["request_id"] for e in _approval._gateway_queues.get(SESSION, [])]
            assert remaining == ["rid-a", "rid-b"], remaining
            assert not entry_b.event.is_set(), "live B's waiter must stay blocked"
        finally:
            with _approval._lock:
                _approval._gateway_queues.pop(SESSION, None)


# ── F2: QQ keyboards ───────────────────────────────────────────────────────────────────────


class TestQQWireFormatDisambiguation:
    def test_legacy_hex_identity_is_not_a_request_id(self):
        from gateway.platforms.qqbot.keyboards import parse_approval_button_data

        user = "ab" * 16
        session = f"agent:main:qqbot:dm:{user}"
        assert parse_approval_button_data(f"approve:{session}:allow-once") == (
            session, "", "allow-once")

    def test_legacy_group_hex_identity_round_trip(self):
        from gateway.platforms.qqbot.keyboards import parse_approval_button_data

        session = f"agent:main:qqbot:group:groupopenid:{'cd' * 16}"
        assert parse_approval_button_data(f"approve:{session}:deny") == (session, "", "deny")

    def test_v2_payload_parses_session_and_request_id(self):
        from gateway.platforms.qqbot.keyboards import parse_approval_button_data

        rid = "ef" * 16
        session = f"agent:main:qqbot:dm:{'ab' * 16}"
        data = f"approve:v2:{session}:{rid}:allow-once"
        assert parse_approval_button_data(data) == (session, rid, "allow-once")

    def test_v2_with_adjacent_hex_components_round_trip(self):
        """New payload whose session ALSO ends in 32-hex: unambiguous via the v2 marker."""
        from gateway.platforms.qqbot.keyboards import parse_approval_button_data

        rid = "ef" * 16
        session = f"agent:main:qqbot:dm:{'ab' * 16}"
        data = f"approve:v2:{session}:{rid}:allow-always"
        assert parse_approval_button_data(data) == (session, rid, "allow-always")

    def test_build_and_parse_round_trip_with_request_id(self):
        from gateway.platforms.qqbot.keyboards import build_approval_keyboard, parse_approval_button_data

        rid = "ef" * 16
        session = f"agent:main:qqbot:c2c:{'ab' * 16}"
        kb = build_approval_keyboard(session, request_id=rid)
        for btn in kb.content.rows[0].buttons:
            parsed = parse_approval_button_data(btn.action.data)
            assert parsed == (session, rid, parsed[2]), btn.action.data

    def test_build_and_parse_round_trip_without_request_id(self):
        from gateway.platforms.qqbot.keyboards import build_approval_keyboard, parse_approval_button_data

        session = f"agent:main:qqbot:c2c:{'ab' * 16}"
        kb = build_approval_keyboard(session)
        for btn in kb.content.rows[0].buttons:
            parsed = parse_approval_button_data(btn.action.data)
            assert parsed == (session, "", parsed[2]), btn.action.data

    def test_malformed_versioned_inputs_rejected(self):
        from gateway.platforms.qqbot.keyboards import parse_approval_button_data

        assert parse_approval_button_data("approve:v2:sess:not-hex-32:allow-once") is None
        # v2 with a missing decision is not a valid payload of either format
        assert parse_approval_button_data("approve:v2:sess:" + "ef" * 16) is None


# ── F3: Matrix retention ───────────────────────────────────────────────────────────────────


def _matrix_adapter_for_prompt():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    obj = object.__new__(MatrixAdapter)
    obj._client = object()
    obj._approval_prompts_by_event = {}
    obj._approval_prompt_by_session = {}
    obj._approval_timeout_seconds = 300
    obj._approval_reaction_map = {
        "✅": "once", "🌀": "session", "♾️": "always", "♾": "always",
        "❌": "deny", "❎": "deny"}
    return obj


class TestMatrixMultiCardRetention:
    @pytest.mark.asyncio
    async def test_two_live_cards_survive_presentation(self, monkeypatch):
        """Present A then B for one session: BOTH stay registered and answerable."""
        from gateway.platforms.base import ExecApprovalPrompt

        obj = _matrix_adapter_for_prompt()
        sent: list = []

        async def register(chat, text, metadata, make, registry, emojis, label):
            event = "event-" + str(len(sent) + 1)
            sent.append(event)
            registry[event] = make(event, "authorized", time.monotonic() + 10 ** 6)
            return N(success=True, message_id=event)

        monkeypatch.setattr(obj, "_send_reaction_prompt", register)
        for rid in ("a" * 32, "b" * 32):
            prompt = ExecApprovalPrompt(
                chat_id="room", session_key="s", text="Synthetic approval",
                actions=[("Allow", "once", "primary"), ("Deny", "deny", "danger")],
                command="synthetic-cmd", description="d", smart_denied=False,
                metadata={}, request_id=rid)
            await obj._send_exec_approval_prompt(prompt)
        assert set(obj._approval_prompts_by_event) == set(sent)
        assert set(obj._approval_prompt_by_session["s"]) == set(sent)

    @pytest.mark.asyncio
    async def test_settling_one_card_keeps_the_sibling_answerable(self, monkeypatch):
        """Resolve card A (count=1): B's registration must survive; A's is retired."""
        from plugins.platforms.matrix.adapter import _MatrixApprovalPrompt

        obj = _matrix_adapter_for_prompt()
        obj._approval_prompts_by_event["$A"] = _MatrixApprovalPrompt(
            session_key="s", chat_id="room", message_id="$A", request_id="a" * 32)
        obj._approval_prompts_by_event["$B"] = _MatrixApprovalPrompt(
            session_key="s", chat_id="room", message_id="$B", request_id="b" * 32)
        obj._approval_prompt_by_session["s"] = {"$A", "$B"}

        monkeypatch.setattr("tools.approval.resolve_gateway_approval", lambda *a, **kw: 1)
        monkeypatch.setattr(obj, "_redact_bot_approval_reactions", self._noop_redact())
        monkeypatch.setattr(obj, "_send_invalid_reaction_feedback", self._noop_feedback())
        obj._is_authorized_user = lambda user_id: True
        obj._approval_require_sender = False

        await obj._handle_approval_reaction("room", "$A", "✅", "@user:example.org")
        assert "$A" not in obj._approval_prompts_by_event
        assert "$B" in obj._approval_prompts_by_event, "sibling card must stay answerable"
        assert obj._approval_prompt_by_session.get("s") == {"$B"}

    @pytest.mark.asyncio
    async def test_expiring_one_card_keeps_the_sibling_answerable(self, monkeypatch):
        """Expiry of card A retires only A; B survives."""
        from plugins.platforms.matrix.adapter import _MatrixApprovalPrompt

        obj = _matrix_adapter_for_prompt()
        obj._approval_prompts_by_event["$A"] = _MatrixApprovalPrompt(
            session_key="s", chat_id="room", message_id="$A", request_id="a" * 32,
            expires_at=time.monotonic() - 1)
        obj._approval_prompts_by_event["$B"] = _MatrixApprovalPrompt(
            session_key="s", chat_id="room", message_id="$B", request_id="b" * 32,
            expires_at=time.monotonic() + 10 ** 6)
        obj._approval_prompt_by_session["s"] = {"$A", "$B"}

        monkeypatch.setattr(obj, "_redact_bot_approval_reactions", self._noop_redact())
        monkeypatch.setattr(obj, "_send_invalid_reaction_feedback", self._noop_feedback())
        obj._is_authorized_user = lambda _u: True
        obj._approval_require_sender = False

        await obj._handle_approval_reaction("room", "$A", "✅", "@user:example.org")
        assert "$A" not in obj._approval_prompts_by_event
        assert "$B" in obj._approval_prompts_by_event
        assert obj._approval_prompt_by_session.get("s") == {"$B"}

    def _noop_redact(self):
        async def _redact(_room, _prompt):
            return None
        return _redact

    def _noop_feedback(self):
        async def _feedback(*_a):
            return None
        return _feedback
