"""Tests for Matrix reaction-based clarify (#46490).

Mirrors test_matrix_exec_approval.py (harness) for the ``send_clarify``
override and the clarify branch of ``_on_reaction``:

  * render — question + numbered choices, self-reactions 1️⃣.. + ✏️,
  * a number reaction resolves the clarify with the right choice string,
  * ✏️ arms the gateway text-capture (``mark_awaiting_text``),
  * an unauthorized reaction is ignored,
  * open-ended (no choices) falls back to the base plain-text path,
  * settlement cleanup (``cleanup_clarify``) retires the tracked prompt and
    redacts seed reactions for typed-text resolution and waiter timeout,
    and stays idempotent after a reaction already cleaned up.
"""

import types

import pytest
from unittest.mock import AsyncMock, patch

from gateway.config import PlatformConfig


def _make_adapter(monkeypatch):
    monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@liizfq:liizfq.top")
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="tok",
            extra={"homeserver": "https://matrix.example.org"},
        )
    )
    adapter._client = types.SimpleNamespace()
    # Resolve user_id so _is_self_sender doesn't defensively drop all traffic.
    adapter._user_id = "@bot:example.org"
    return adapter


def _number_emojis(count):
    from plugins.platforms.matrix.adapter import _MATRIX_MODEL_PICKER_REACTIONS

    return list(_MATRIX_MODEL_PICKER_REACTIONS[:count])


def _reaction_event(sender, reacts_to, key):
    return types.SimpleNamespace(
        sender=sender,
        event_id="$react1",
        room_id="!room:example.org",
        content={"m.relates_to": {"event_id": reacts_to, "key": key}},
    )


class TestMatrixClarifyReactions:
    @pytest.mark.asyncio
    async def test_send_clarify_renders_and_seeds_reactions(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.send = AsyncMock(
            return_value=types.SimpleNamespace(success=True, message_id="$clar1")
        )
        adapter._send_reaction = AsyncMock(return_value="$r")

        result = await adapter.send_clarify(
            chat_id="!room:example.org",
            question="Which fruit?",
            choices=["Apple", "Banana"],
            clarify_id="c1",
            session_key="sess-1",
        )

        assert result.success is True
        # Message body carries the question + numbered choices.
        body = adapter.send.await_args.args[1]
        assert "Which fruit?" in body
        assert "Apple" in body and "Banana" in body
        # Prompt is tracked by the message event_id for the reaction handler.
        prompt = adapter._clarify_prompts_by_event["$clar1"]
        assert prompt.clarify_id == "c1"
        assert prompt.session_key == "sess-1"
        # Self-reactions: one number per choice + the ✏️ "Other" control.
        emojis = [call.args[2] for call in adapter._send_reaction.await_args_list]
        assert emojis == _number_emojis(2) + ["✏️"]

    @pytest.mark.asyncio
    async def test_reaction_after_approval_timeout_still_resolves(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._approval_timeout_seconds = 300
        adapter.send = AsyncMock(
            return_value=types.SimpleNamespace(success=True, message_id="$clar1")
        )
        adapter._send_reaction = AsyncMock(return_value="$r")
        adapter._redact_bot_clarify_reactions = AsyncMock()

        from tools import clarify_gateway

        clarify_gateway.register(
            clarify_id="c1",
            session_key="sess-1",
            question="Which fruit?",
            choices=["Apple", "Banana"],
        )
        try:
            with (
                patch("tools.clarify_gateway.get_clarify_timeout", return_value=3600),
                patch("plugins.platforms.matrix.adapter.time.monotonic", return_value=1000.0),
            ):
                await adapter.send_clarify(
                    chat_id="!room:example.org",
                    question="Which fruit?",
                    choices=["Apple", "Banana"],
                    clarify_id="c1",
                    session_key="sess-1",
                )

            prompt = adapter._clarify_prompts_by_event["$clar1"]
            assert prompt.expires_at == 4600.0

            # The Matrix prompt must remain live after the approval timeout
            # (300s) while the gateway clarify waiter (3600s) is still active.
            event = _reaction_event("@liizfq:liizfq.top", "$clar1", _number_emojis(2)[1])
            with patch("plugins.platforms.matrix.adapter.time.monotonic", return_value=1301.0):
                await adapter._on_reaction(event)

            entry = clarify_gateway._entries["c1"]
            assert entry.response == "Banana"
            assert entry.event.is_set()
            assert "$clar1" not in adapter._clarify_prompts_by_event
        finally:
            clarify_gateway.clear_session("sess-1")

    @pytest.mark.asyncio
    async def test_reaction_resolves_choice(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        from plugins.platforms.matrix.adapter import _MatrixClarifyPrompt
        from tools import clarify_gateway

        number_emojis = _number_emojis(2)
        clarify_gateway.register(
            clarify_id="c1",
            session_key="sess-1",
            question="Which fruit?",
            choices=["Apple", "Banana"],
        )
        try:
            adapter._clarify_prompts_by_event["$clar1"] = _MatrixClarifyPrompt(
                chat_id="!room:example.org",
                message_id="$clar1",
                session_key="sess-1",
                clarify_id="c1",
                choices=["Apple", "Banana"],
                emoji_to_index={number_emojis[0]: 0, number_emojis[1]: 1},
                other_emoji="✏️",
            )

            event = _reaction_event("@liizfq:liizfq.top", "$clar1", number_emojis[1])
            await adapter._on_reaction(event)

            # The exact chosen string crosses into the gateway primitive.
            entry = clarify_gateway._entries["c1"]
            assert entry.response == "Banana"
            assert entry.event.is_set()
            assert "$clar1" not in adapter._clarify_prompts_by_event
        finally:
            clarify_gateway.clear_session("sess-1")

    @pytest.mark.asyncio
    async def test_other_reaction_arms_text_capture(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        from plugins.platforms.matrix.adapter import _MatrixClarifyPrompt

        adapter._clarify_prompts_by_event["$clar1"] = _MatrixClarifyPrompt(
            chat_id="!room:example.org",
            message_id="$clar1",
            session_key="sess-1",
            clarify_id="c1",
            choices=["Apple", "Banana"],
            emoji_to_index={e: i for i, e in enumerate(_number_emojis(2))},
            other_emoji="✏️",
        )
        adapter._send_invalid_reaction_feedback = AsyncMock()

        event = _reaction_event("@liizfq:liizfq.top", "$clar1", "✏️")
        with patch(
            "tools.clarify_gateway.mark_awaiting_text", return_value=True
        ) as mock_arm:
            await adapter._on_reaction(event)

        mock_arm.assert_called_once_with("c1")
        # Prompt stays pending — the typed reply resolves it via the gateway
        # text-intercept, not a second reaction.
        assert "$clar1" in adapter._clarify_prompts_by_event

    @pytest.mark.asyncio
    async def test_unauthorized_reaction_ignored(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        from plugins.platforms.matrix.adapter import _MatrixClarifyPrompt

        number_emojis = _number_emojis(2)
        adapter._clarify_prompts_by_event["$clar1"] = _MatrixClarifyPrompt(
            chat_id="!room:example.org",
            message_id="$clar1",
            session_key="sess-1",
            clarify_id="c1",
            choices=["Apple", "Banana"],
            emoji_to_index={number_emojis[0]: 0, number_emojis[1]: 1},
            other_emoji="✏️",
        )
        adapter._send_invalid_reaction_feedback = AsyncMock()

        event = _reaction_event("@stranger:example.org", "$clar1", number_emojis[1])
        with patch("tools.clarify_gateway.resolve_gateway_clarify") as mock_resolve:
            await adapter._on_reaction(event)

        mock_resolve.assert_not_called()
        # Prompt is left pending; only the auth-denied feedback fires.
        assert "$clar1" in adapter._clarify_prompts_by_event
        assert not adapter._clarify_prompts_by_event["$clar1"].resolved

    @pytest.mark.asyncio
    async def test_open_ended_falls_back_to_base(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.send = AsyncMock(
            return_value=types.SimpleNamespace(success=True, message_id="$open1")
        )
        adapter._send_reaction = AsyncMock(return_value="$r")

        result = await adapter.send_clarify(
            chat_id="!room:example.org",
            question="Anything else?",
            choices=None,
            clarify_id="c2",
            session_key="sess-2",
        )

        assert result.success is True
        # Base path renders a plain question and seeds no reactions / prompt.
        adapter._send_reaction.assert_not_awaited()
        assert "$open1" not in adapter._clarify_prompts_by_event
        # Base send_clarify calls self.send(content=...) by keyword.
        sent = adapter.send.await_args
        body = sent.kwargs.get("content") or sent.args[1]
        assert "Anything else?" in body

    @pytest.mark.asyncio
    async def test_typed_reply_settlement_retires_prompt(self, monkeypatch):
        """Regression (#46495 review): a typed reply resolves the clarify
        through the gateway text path, which bypasses the adapter's reaction
        handler — the settlement cleanup must retire the tracked prompt and
        redact the bot's seeded reactions."""
        adapter = _make_adapter(monkeypatch)
        adapter.send = AsyncMock(
            return_value=types.SimpleNamespace(success=True, message_id="$clar1")
        )
        adapter._send_reaction = AsyncMock(return_value="$r")
        adapter._redact_bot_clarify_reactions = AsyncMock()

        from tools import clarify_gateway

        clarify_gateway.register(
            clarify_id="c1",
            session_key="sess-1",
            question="Which fruit?",
            choices=["Apple", "Banana"],
        )
        try:
            await adapter.send_clarify(
                chat_id="!room:example.org",
                question="Which fruit?",
                choices=["Apple", "Banana"],
                clarify_id="c1",
                session_key="sess-1",
            )
            assert "$clar1" in adapter._clarify_prompts_by_event

            # The user types "2" — the gateway text-intercept
            # (gateway/run.py _handle_message) resolves via
            # resolve_text_response_for_session, never touching the adapter.
            assert clarify_gateway.resolve_text_response_for_session("sess-1", "2")
            entry = clarify_gateway._entries["c1"]
            assert entry.response == "Banana"
            assert entry.event.is_set()

            # The typed path leaves the Matrix prompt live — that's the bug
            # this hook fixes.
            assert "$clar1" in adapter._clarify_prompts_by_event

            # The gateway's clarify callback fires cleanup_clarify after
            # wait_for_response returns.
            await adapter.cleanup_clarify(
                chat_id="!room:example.org",
                clarify_id="c1",
                message_id="$clar1",
                session_key="sess-1",
            )

            assert "$clar1" not in adapter._clarify_prompts_by_event
            adapter._redact_bot_clarify_reactions.assert_awaited_once()
        finally:
            clarify_gateway.clear_session("sess-1")

    @pytest.mark.asyncio
    async def test_timeout_settlement_retires_prompt(self, monkeypatch):
        """Waiter timeout evicts only the gateway entry; the settlement
        cleanup must still retire the Matrix prompt.  Also exercises the
        clarify_id fallback scan (message_id=None)."""
        adapter = _make_adapter(monkeypatch)
        adapter.send = AsyncMock(
            return_value=types.SimpleNamespace(success=True, message_id="$clar1")
        )
        adapter._send_reaction = AsyncMock(return_value="$r")
        adapter._redact_bot_clarify_reactions = AsyncMock()

        from tools import clarify_gateway

        clarify_gateway.register(
            clarify_id="c1",
            session_key="sess-1",
            question="Which fruit?",
            choices=["Apple", "Banana"],
        )
        try:
            await adapter.send_clarify(
                chat_id="!room:example.org",
                question="Which fruit?",
                choices=["Apple", "Banana"],
                clarify_id="c1",
                session_key="sess-1",
            )
            assert "$clar1" in adapter._clarify_prompts_by_event

            # Timeout: wait_for_response evicts the gateway-side entry and
            # returns None — nothing tells the adapter.
            assert clarify_gateway.wait_for_response("c1", timeout=0) is None
            assert "c1" not in clarify_gateway._entries
            assert "$clar1" in adapter._clarify_prompts_by_event

            await adapter.cleanup_clarify(
                chat_id="!room:example.org",
                clarify_id="c1",
                message_id=None,
                session_key="sess-1",
            )

            assert "$clar1" not in adapter._clarify_prompts_by_event
            adapter._redact_bot_clarify_reactions.assert_awaited_once()
        finally:
            clarify_gateway.clear_session("sess-1")

    @pytest.mark.asyncio
    async def test_cleanup_after_reaction_resolution_is_noop(self, monkeypatch):
        """The reaction path pops the prompt itself; the settlement cleanup
        that follows must be idempotent (no double redaction)."""
        adapter = _make_adapter(monkeypatch)
        from plugins.platforms.matrix.adapter import _MatrixClarifyPrompt
        from tools import clarify_gateway

        number_emojis = _number_emojis(2)
        clarify_gateway.register(
            clarify_id="c1",
            session_key="sess-1",
            question="Which fruit?",
            choices=["Apple", "Banana"],
        )
        try:
            adapter._clarify_prompts_by_event["$clar1"] = _MatrixClarifyPrompt(
                chat_id="!room:example.org",
                message_id="$clar1",
                session_key="sess-1",
                clarify_id="c1",
                choices=["Apple", "Banana"],
                emoji_to_index={number_emojis[0]: 0, number_emojis[1]: 1},
                other_emoji="✏️",
            )
            adapter._redact_bot_clarify_reactions = AsyncMock()

            event = _reaction_event("@liizfq:liizfq.top", "$clar1", number_emojis[1])
            await adapter._on_reaction(event)
            assert "$clar1" not in adapter._clarify_prompts_by_event
            assert adapter._redact_bot_clarify_reactions.await_count == 1

            # Gateway settlement cleanup arrives after the waiter wakes.
            await adapter.cleanup_clarify(
                chat_id="!room:example.org",
                clarify_id="c1",
                message_id="$clar1",
                session_key="sess-1",
            )

            # Idempotent — no second redaction pass.
            assert adapter._redact_bot_clarify_reactions.await_count == 1
        finally:
            clarify_gateway.clear_session("sess-1")
