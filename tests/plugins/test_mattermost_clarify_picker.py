"""Mattermost plugin — reaction-driven clarify picker (v2 2026-09-27).

Mirror of matrix's ``send_choice_picker`` + ``_handle_choice_picker_reaction``:
when the agent asks the user a multiple-choice question (e.g. "Which of these
should I do? A, B, C"), Mattermost now posts the question with one reaction
per option (1️⃣ 2️⃣ 3️⃣ …). The user taps one and the agent proceeds without
typing. Open-ended questions still fall through to the gateway text fallback.

This is the SECOND platform adapter to ship reaction-driven clarify pickers
(Matrix was first; mirror matrix/adapter.py:1735 exactly). Tests assert the
matrix-parity contract: 12 emoji slots (1️⃣–9️⃣, 🔟, 🅰️ 🅱️), bot-self-check,
allowlist gate, expiry TTL, and fall-through on send failure.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.mattermost.adapter_clarify import (
    format_clarify_picker_body,
    numeric_labels_for,
)


# ─── helpers ────────────────────────────────────────────────────────────────


def _make_adapter():
    """Build a MattermostAdapter with a default allowlist of {user-1, user-99}.

    Mirrors the helper in test_mattermost_approval.py — keep behavior consistent
    so the same user_id conventions apply across the test suite.
    """
    os.environ.setdefault("MATTERMOST_URL", "https://mm.example.com")
    os.environ.setdefault("MATTERMOST_TOKEN", "test-token-xxx")
    from gateway.config import PlatformConfig
    from plugins.platforms.mattermost.adapter import MattermostAdapter
    cfg = PlatformConfig(enabled=True, token="test-token-xxx")
    adapter = MattermostAdapter(cfg)
    adapter._allowed_user_ids = {"user-1", "user-99"}
    return adapter


def _ok_send(post_id: str):
    from gateway.platforms.base import SendResult
    return SendResult(success=True, message_id=post_id)


def _ws_event_reaction_added(post_id: str, user_id: str, emoji_name: str):
    """Match the shape in test_mattermost_approval.py for handler tests."""
    reaction = {"user_id": user_id, "post_id": post_id, "emoji_name": emoji_name}
    return {"event": "reaction_added", "data": {"reaction": json.dumps(reaction)}}


# ─── outbound: send_clarify renders a native picker card ─────────────────────


class TestSendClarifySeedsReactions:
    """send_clarify with choices renders a picker card and seeds reactions."""

    @pytest.mark.asyncio
    async def test_send_clarify_with_3_choices_registers_picker_and_seeds_3_emojis(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter.send = AsyncMock(return_value=_ok_send("picker-post-1"))
        adapter._send_reaction = AsyncMock(return_value=True)

        result = await adapter.send_clarify(
            chat_id="dm-1",
            question="Which approach should I take?",
            choices=["Approach A", "Approach B", "Approach C"],
            clarify_id="clar-abc",
            session_key="sess-1",
            metadata={"requester_user_id": "user-1"},
        )

        assert result.success is True
        assert "picker-post-1" in adapter._clarify_picker_prompts_by_event
        picker = adapter._clarify_picker_prompts_by_event["picker-post-1"]
        assert picker.clarify_id == "clar-abc"
        assert picker.choices == ["one", "two", "three"]
        assert picker.responses == ["Approach A", "Approach B", "Approach C"]
        assert picker.chat_id == "dm-1"
        assert picker.resolved is False
        assert picker.expires_at > 0

        # Three reactions seeded, one per choice — matrix-parity emoji short names.
        seeded = [call.args[1] for call in adapter._send_reaction.await_args_list]
        assert seeded == ["one", "two", "three"]

    @pytest.mark.asyncio
    async def test_send_clarify_with_12_choices_caps_at_12_emojis(self):
        """Matrix caps choice pickers at 12. Mattermost v2 does the same."""
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter.send = AsyncMock(return_value=_ok_send("picker-post-2"))
        adapter._send_reaction = AsyncMock(return_value=True)

        choices = [f"Opt {i}" for i in range(15)]  # 15 choices
        result = await adapter.send_clarify(
            chat_id="dm-1", question="Pick one?", choices=choices,
            clarify_id="clar-15", session_key="sess-1",
        )
        assert result.success is True
        picker = adapter._clarify_picker_prompts_by_event["picker-post-2"]
        assert len(picker.choices) == 12
        assert len(picker.responses) == 12
        assert picker.choices[9] == "keycap_ten"
        assert picker.choices[10] == "a"
        assert picker.choices[11] == "b"
        assert picker.responses[:3] == ["Opt 0", "Opt 1", "Opt 2"]
        assert picker.responses[-1] == "Opt 11"  # 12th = Opt 11

    @pytest.mark.asyncio
    async def test_send_clarify_with_no_choices_falls_through_to_text(self):
        """Open-ended questions must not render a picker card; the gateway
        text fallback (just the question text) handles them. ``mark_awaiting_text``
        is only called when ``choices`` is truthy (see base.send_clarify:2891)."""
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter.send = AsyncMock(return_value=_ok_send("text-post-1"))

        with patch("tools.clarify_gateway.mark_awaiting_text") as mock_mark:
            result = await adapter.send_clarify(
                chat_id="dm-1", question="Tell me more?", choices=None,
                clarify_id="clar-open", session_key="sess-1",
            )

        assert result.success is True
        # No picker registered — open-ended went through super().send_clarify.
        assert "text-post-1" not in adapter._clarify_picker_prompts_by_event
        # No mark_awaiting_text for open-ended (base skips it when choices is empty).
        mock_mark.assert_not_called()
        # The text body contained the question.
        sent_text = adapter.send.await_args.kwargs.get("content") or adapter.send.await_args.args[1]
        assert "Tell me more?" in sent_text

    @pytest.mark.asyncio
    async def test_send_clarify_send_failure_falls_back_to_text(self):
        """If the picker card post fails (network, etc.), fall back to the
        gateway text path so the user can still answer."""
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"

        # First call (the picker card) fails; second call (text fallback) succeeds.
        fail_result = _ok_send("ignored")
        fail_result.success = False
        ok_result = _ok_send("text-post-fb")

        async def _maybe_send(*args, **kwargs):
            if not getattr(adapter, "_send_attempted", False):
                adapter._send_attempted = True
                return fail_result
            return ok_result

        adapter.send = AsyncMock(side_effect=_maybe_send)

        with patch("tools.clarify_gateway.mark_awaiting_text"):
            result = await adapter.send_clarify(
                chat_id="dm-1", question="Pick?", choices=["A", "B"],
                clarify_id="clar-fb", session_key="sess-1",
            )

        # send was called twice — first attempt (picker card) failed, second
        # was the super().send_clarify text fallback.
        assert adapter.send.await_count == 2
        # The picker registry was NOT populated (card send failed first).
        assert "ignored" not in adapter._clarify_picker_prompts_by_event
        # The fallback result is what the caller sees.
        assert result.success is True
        assert result.message_id == "text-post-fb"


# ─── inbound: reaction handler resolves the picker ──────────────────────────


class TestHandlePickerReaction:

    @pytest.mark.asyncio
    async def test_tap_on_one_resolves_clarify_with_first_choice(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        from plugins.platforms.mattermost.adapter import _MattermostClarifyPicker
        import time as _time
        adapter._clarify_picker_prompts_by_event["post-1"] = _MattermostClarifyPicker(
            chat_id="dm-1", post_id="post-1", clarify_id="clar-xyz",
            choices=["one", "two", "three"],
            responses=["A", "B", "C"],
            expires_at=_time.time() + 300,
        )

        with patch("tools.clarify_gateway.resolve_gateway_clarify",
                   return_value=True) as mock_resolve:
            ev = _ws_event_reaction_added(post_id="post-1", user_id="user-1", emoji_name="one")
            await adapter._handle_picker_reaction(ev["data"])

        mock_resolve.assert_called_once_with("clar-xyz", "A")

    @pytest.mark.asyncio
    async def test_tap_on_two_resolves_with_second_choice(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        from plugins.platforms.mattermost.adapter import _MattermostClarifyPicker
        import time as _time
        adapter._clarify_picker_prompts_by_event["post-2"] = _MattermostClarifyPicker(
            chat_id="dm-1", post_id="post-2", clarify_id="clar-xyz",
            choices=["one", "two", "three"],
            responses=["A", "B", "C"],
            expires_at=_time.time() + 300,
        )

        with patch("tools.clarify_gateway.resolve_gateway_clarify",
                   return_value=True) as mock_resolve:
            ev = _ws_event_reaction_added(post_id="post-2", user_id="user-1", emoji_name="two")
            await adapter._handle_picker_reaction(ev["data"])

        mock_resolve.assert_called_once_with("clar-xyz", "B")

    @pytest.mark.asyncio
    async def test_bot_self_reaction_does_not_resolve(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        from plugins.platforms.mattermost.adapter import _MattermostClarifyPicker
        import time as _time
        adapter._clarify_picker_prompts_by_event["post-3"] = _MattermostClarifyPicker(
            chat_id="dm-1", post_id="post-3", clarify_id="clar-self",
            choices=["one", "two"], responses=["A", "B"],
            expires_at=_time.time() + 300,
        )

        with patch("tools.clarify_gateway.resolve_gateway_clarify") as mock_resolve:
            ev = _ws_event_reaction_added(post_id="post-3", user_id="bot-id", emoji_name="one")
            await adapter._handle_picker_reaction(ev["data"])

        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_user_cannot_resolve(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter._allowed_user_ids = {"user-1"}  # user-99 is NOT allowed
        from plugins.platforms.mattermost.adapter import _MattermostClarifyPicker
        import time as _time
        adapter._clarify_picker_prompts_by_event["post-4"] = _MattermostClarifyPicker(
            chat_id="dm-1", post_id="post-4", clarify_id="clar-locked",
            choices=["one", "two"], responses=["A", "B"],
            expires_at=_time.time() + 300,
        )

        with patch("tools.clarify_gateway.resolve_gateway_clarify") as mock_resolve:
            ev = _ws_event_reaction_added(post_id="post-4", user_id="intruder-99", emoji_name="one")
            await adapter._handle_picker_reaction(ev["data"])

        mock_resolve.assert_not_called()
        # Card stays open for an allowlisted user.
        assert "post-4" in adapter._clarify_picker_prompts_by_event

    @pytest.mark.asyncio
    async def test_tap_after_resolution_is_noop(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        from plugins.platforms.mattermost.adapter import _MattermostClarifyPicker
        import time as _time
        picker = _MattermostClarifyPicker(
            chat_id="dm-1", post_id="post-5", clarify_id="clar-double",
            choices=["one", "two"], responses=["A", "B"],
            expires_at=_time.time() + 300,
        )
        adapter._clarify_picker_prompts_by_event["post-5"] = picker

        # First tap resolves and removes the entry from the registry.
        with patch("tools.clarify_gateway.resolve_gateway_clarify", return_value=True) as mock_resolve:
            ev = _ws_event_reaction_added(post_id="post-5", user_id="user-1", emoji_name="one")
            await adapter._handle_picker_reaction(ev["data"])
            # Re-register to simulate the race; picker.resolved still True.
            adapter._clarify_picker_prompts_by_event["post-5"] = picker
            ev2 = _ws_event_reaction_added(post_id="post-5", user_id="user-1", emoji_name="two")
            await adapter._handle_picker_reaction(ev2["data"])

        # Only one resolve call.
        assert mock_resolve.call_count == 1

    @pytest.mark.asyncio
    async def test_tap_on_unknown_emoji_is_noop(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        from plugins.platforms.mattermost.adapter import _MattermostClarifyPicker
        import time as _time
        adapter._clarify_picker_prompts_by_event["post-6"] = _MattermostClarifyPicker(
            chat_id="dm-1", post_id="post-6", clarify_id="clar-noop",
            choices=["one", "two"], responses=["A", "B"],
            expires_at=_time.time() + 300,
        )

        with patch("tools.clarify_gateway.resolve_gateway_clarify") as mock_resolve:
            ev = _ws_event_reaction_added(post_id="post-6", user_id="user-1", emoji_name="thumbsup")
            await adapter._handle_picker_reaction(ev["data"])

        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_tap_after_expiry_is_noop_and_drops_entry(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        from plugins.platforms.mattermost.adapter import _MattermostClarifyPicker
        # expires_at = 0 means we explicitly set it in the past.
        adapter._clarify_picker_prompts_by_event["post-7"] = _MattermostClarifyPicker(
            chat_id="dm-1", post_id="post-7", clarify_id="clar-expired",
            choices=["one", "two"], responses=["A", "B"],
            expires_at=0.0,  # already expired
        )

        with patch("tools.clarify_gateway.resolve_gateway_clarify") as mock_resolve:
            ev = _ws_event_reaction_added(post_id="post-7", user_id="user-1", emoji_name="one")
            await adapter._handle_picker_reaction(ev["data"])

        mock_resolve.assert_not_called()
        assert "post-7" not in adapter._clarify_picker_prompts_by_event

    @pytest.mark.asyncio
    async def test_resolve_tidies_reactions_and_edits_card(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        from plugins.platforms.mattermost.adapter import _MattermostClarifyPicker
        import time as _time
        adapter._clarify_picker_prompts_by_event["post-8"] = _MattermostClarifyPicker(
            chat_id="dm-1", post_id="post-8", clarify_id="clar-tidy",
            choices=["one", "two"], responses=["A", "B"],
            expires_at=_time.time() + 300,
        )

        deleted: list = []
        edited: list = []

        async def fake_delete(post_id, emoji_name):
            deleted.append((post_id, emoji_name))

        async def fake_edit(chat_id, post_id, text, **_):
            edited.append((chat_id, post_id, text))
            from gateway.platforms.base import SendResult
            return SendResult(success=True, message_id=post_id)

        adapter._delete_reaction = fake_delete
        adapter.edit_message = fake_edit

        with patch("tools.clarify_gateway.resolve_gateway_clarify", return_value=True):
            ev = _ws_event_reaction_added(post_id="post-8", user_id="user-1", emoji_name="one")
            await adapter._handle_picker_reaction(ev["data"])
            # Allow scheduled delete-tasks to run.
            await asyncio.sleep(0)

        # Registry tidied.
        assert "post-8" not in adapter._clarify_picker_prompts_by_event
        # Bot's reactions on the picker card retracted.
        assert ("post-8", "one") in deleted
        assert ("post-8", "two") in deleted
        # Card edited to show the chosen label.
        assert any("A" in text for _, _, text in edited)


# ─── dispatch: reaction_added routes to picker OR approval ──────────────────


class TestReactionDispatch:
    """``reaction_added`` events route to picker-first, then approval."""

    @pytest.mark.asyncio
    async def test_reaction_on_picker_post_routes_to_picker(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        # Same post_id is registered as a picker, but NOT as an approval.
        from plugins.platforms.mattermost.adapter import _MattermostClarifyPicker
        import time as _time
        adapter._clarify_picker_prompts_by_event["post-shared"] = _MattermostClarifyPicker(
            chat_id="dm-1", post_id="post-shared", clarify_id="clar-dispatch",
            choices=["one", "two"], responses=["A", "B"],
            expires_at=_time.time() + 300,
        )

        with patch("tools.clarify_gateway.resolve_gateway_clarify", return_value=True) as mock_clarify, \
             patch("tools.approval.resolve_gateway_approval") as mock_approve:
            ev = _ws_event_reaction_added(post_id="post-shared", user_id="user-1", emoji_name="one")
            await adapter._handle_ws_event(ev)

        # Picker handler ran (clarify was called), approval handler did NOT.
        mock_clarify.assert_called_once_with("clar-dispatch", "A")
        mock_approve.assert_not_called()

    @pytest.mark.asyncio
    async def test_reaction_on_approval_only_post_routes_to_approval(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        # Only an approval entry, no picker entry.
        adapter._approval_prompts_by_event["post-approval-only"] = {
            "session_key": "sess-X", "requester_user_id": "user-1",
            "choices": ["once", "session", "always", "deny"],
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_approve, \
             patch("tools.clarify_gateway.resolve_gateway_clarify") as mock_clarify:
            ev = _ws_event_reaction_added(post_id="post-approval-only", user_id="user-1",
                                          emoji_name="white_check_mark")
            await adapter._handle_ws_event(ev)

        # Approval handler ran, picker handler did NOT.
        mock_approve.assert_called_once()
        mock_clarify.assert_not_called()

# ─── pure-helper tests for format_clarify_picker_body (v3 display-fix) ─────


class TestClarifyPickerBodyHelper:
    """Layout helper tests (no adapter fixture needed)."""

    def test_single_choice_layout(self):
        out = format_clarify_picker_body("Pick one", ["one"], ["Yes"])
        # No typeable-hint line when only 1 choice (hint says "reply with 1"
        # alone is noise).
        assert "❓ Pick one" in out
        assert ":one: Yes" in out
        assert "Tap an emoji" not in out
        assert "reply with 1" not in out

    def test_three_choices_with_typeable_hint(self):
        out = format_clarify_picker_body(
            "Where?",
            ["one", "two", "three"],
            ["PL", "UK", "IS"],
        )
        # All three emoji short-names present, in one body line.
        assert ":one: PL" in out
        assert ":two: UK" in out
        assert ":three: IS" in out
        # Typeable-fallback hint shows "1 / 2 / 3" and the tap-instruction.
        assert "1 / 2 / 3" in out
        assert "Tap an emoji above to answer" in out

    def test_keycap_ten_maps_to_numeric_ten(self):
        labels = numeric_labels_for(
            ["one","two","three","four","five","six","seven","eight","nine","keycap_ten"]
        )
        # Index-based: 10th position → "10", NOT "ten".
        assert labels == ["1","2","3","4","5","6","7","8","9","10"]

    def test_twelve_choices_keeps_all_visible(self):
        # Real Mattermost picker caps at 12. The pure helper itself takes any
        # length; the cap is in the adapter (line ~1004 ``[:12]``). Here we
        # confirm the helper renders ALL 12 options without truncation when
        # the adapter passes them in.
        twelve = ["one","two","three","four","five","six",
                  "seven","eight","nine","keycap_ten","a","b"]
        out = format_clarify_picker_body(
            "Pick",
            twelve,
            [str(i) for i in range(1, 13)],
        )
        for emoji in twelve:
            assert f":{emoji}:" in out, f"missing emoji {emoji!r}"
        # Typeable hint lists all 12 numeric labels.
        assert "12" in out
        assert "Tap an emoji above to answer" in out
