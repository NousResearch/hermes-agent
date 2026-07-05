"""Regression tests for Discord oversized edit_message split-and-deliver.

Issue #27881 surfaced as silent truncation: ``edit_message`` clipped any
formatted payload over Discord's 2,000-char cap to ``[:1997] + "..."`` and
returned ``success=True``, so the stream consumer believed the full reply
landed and the user lost everything past the boundary.

The fix mirrors the proven Telegram contract (and its #48648 lesson):

* **Mid-stream** (``finalize=False``) — never split.  A mid-stream split moves
  the edit target to a continuation; the next accumulated-token tick re-edits
  the full text into it and re-splits, looping forever.  We truncate a
  one-message preview in place instead.
* **Final** (``finalize=True``) — split-and-deliver: edit chunk 1 in place,
  send chunks 2..N as reply-threaded continuations, return the LAST visible id
  in ``message_id`` plus every continuation in ``continuation_message_ids``.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


MAX = DiscordAdapter.MAX_MESSAGE_LENGTH  # 2000


def _make_adapter():
    return DiscordAdapter(PlatformConfig(enabled=True, token="***"))


def _wire_channel(adapter, *, original_msg, send_side_effect=None):
    """Wire a fake client whose channel returns ``original_msg`` on fetch and
    records every ``channel.send`` call."""
    sends = []

    async def fake_send(*, content, reference=None):
        sends.append({"content": content, "reference": reference})
        if send_side_effect is not None:
            res = send_side_effect(len(sends), content, reference)
            if res is not None:
                return res
        return SimpleNamespace(id=9000 + len(sends))

    channel = SimpleNamespace(
        id=555,
        fetch_message=AsyncMock(return_value=original_msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _cid: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )
    return channel, sends


# --------------------------------------------------------------------------- #
# Happy path — short edits unchanged
# --------------------------------------------------------------------------- #


class TestEditMessageHappyPath:
    @pytest.mark.asyncio
    async def test_short_edit_in_place(self):
        adapter = _make_adapter()
        edits = []
        msg = SimpleNamespace(
            id=42,
            edit=AsyncMock(side_effect=lambda *, content: edits.append(content)),
        )
        channel, sends = _wire_channel(adapter, original_msg=msg)

        result = await adapter.edit_message("555", "42", "short reply")

        assert result.success is True
        assert result.message_id == "42"
        assert result.continuation_message_ids == ()
        assert edits == ["short reply"]
        assert sends == []  # no continuations for a short edit

    @pytest.mark.asyncio
    async def test_no_client_returns_failure(self):
        adapter = _make_adapter()
        adapter._client = None
        result = await adapter.edit_message("555", "42", "x")
        assert result.success is False


# --------------------------------------------------------------------------- #
# Mid-stream overflow — TRUNCATE, never split (the #48648 lesson)
# --------------------------------------------------------------------------- #


class TestMidStreamOverflowTruncates:
    @pytest.mark.asyncio
    async def test_oversized_streaming_edit_truncates_in_place(self):
        adapter = _make_adapter()
        edits = []
        msg = SimpleNamespace(
            id=42,
            edit=AsyncMock(side_effect=lambda *, content: edits.append(content)),
        )
        channel, sends = _wire_channel(adapter, original_msg=msg)

        big = "p" * 6000
        result = await adapter.edit_message("555", "42", big, finalize=False)

        # No split: the original message stays the target, no continuations.
        assert result.success is True
        assert result.message_id == "42"
        assert result.continuation_message_ids == ()
        assert sends == [], "mid-stream overflow must NOT create continuations"
        # Exactly one in-place edit, clipped to a single chunk under the cap.
        assert len(edits) == 1
        assert len(edits[0]) <= MAX
        # No literal "..." truncation marker leaks in (fence-aware truncation).
        assert not edits[0].endswith("...")


# --------------------------------------------------------------------------- #
# Final overflow — SPLIT and deliver every chunk
# --------------------------------------------------------------------------- #


class TestFinalOverflowSplits:
    @pytest.mark.asyncio
    async def test_oversized_final_edit_splits_and_delivers(self):
        adapter = _make_adapter()
        edits = []
        msg = SimpleNamespace(
            id=42,
            to_reference=MagicMock(return_value=SimpleNamespace(kind="ref")),
            edit=AsyncMock(side_effect=lambda *, content: edits.append(content)),
        )
        channel, sends = _wire_channel(adapter, original_msg=msg)

        big = "q" * 6000  # ~3-4 chunks at 2000 cap
        result = await adapter.edit_message("555", "42", big, finalize=True)

        assert result.success is True
        # message_id points at the LAST visible continuation, not the original.
        assert result.continuation_message_ids, "expected continuations"
        assert result.message_id == result.continuation_message_ids[-1]
        # chunk 1 edited in place; chunks 2..N sent as new messages.
        assert len(edits) == 1
        assert len(sends) == len(result.continuation_message_ids)
        # Every delivered chunk is under the cap.
        for c in edits + [s["content"] for s in sends]:
            assert len(c) <= MAX
        # No "..." truncation marker anywhere.
        for c in edits + [s["content"] for s in sends]:
            assert not c.endswith("...")

    @pytest.mark.asyncio
    async def test_byte_coverage_preserved(self):
        adapter = _make_adapter()
        edits = []
        msg = SimpleNamespace(
            id=42,
            to_reference=MagicMock(return_value=object()),
            edit=AsyncMock(side_effect=lambda *, content: edits.append(content)),
        )
        channel, sends = _wire_channel(adapter, original_msg=msg)

        # Distinctive marker at the very end must survive end-to-end.
        body = "a" * 5000 + "END_MARKER_XYZ"
        result = await adapter.edit_message("555", "42", body, finalize=True)

        assert result.success is True
        delivered = "".join(edits + [s["content"] for s in sends])
        assert "END_MARKER_XYZ" in delivered

    @pytest.mark.asyncio
    async def test_continuations_threaded_as_replies(self):
        adapter = _make_adapter()
        msg = SimpleNamespace(
            id=42,
            to_reference=MagicMock(return_value=SimpleNamespace(tag="orig")),
            edit=AsyncMock(),
        )
        # Each sent continuation must also expose to_reference so the NEXT
        # chunk can thread under it.
        channel, sends = _wire_channel(
            adapter,
            original_msg=msg,
            send_side_effect=lambda n, content, ref: SimpleNamespace(
                id=9000 + n,
                to_reference=MagicMock(return_value=SimpleNamespace(tag=f"c{n}")),
            ),
        )

        result = await adapter.edit_message("555", "42", "z" * 6000, finalize=True)

        assert result.success is True
        # First continuation replies to the original message's reference.
        assert sends[0]["reference"] is not None
        # Later continuations reply to the previous continuation, not None.
        for s in sends[1:]:
            assert s["reference"] is not None

    @pytest.mark.asyncio
    async def test_first_chunk_edit_failure_propagates(self):
        adapter = _make_adapter()
        msg = SimpleNamespace(
            id=42,
            to_reference=MagicMock(return_value=object()),
            edit=AsyncMock(side_effect=RuntimeError("hard edit failure")),
        )
        channel, sends = _wire_channel(adapter, original_msg=msg)

        result = await adapter.edit_message("555", "42", "w" * 6000, finalize=True)

        assert result.success is False
        assert "hard edit failure" in (result.error or "")
        assert sends == []  # never reached the continuation loop

    @pytest.mark.asyncio
    async def test_mid_continuation_failure_reports_partial(self):
        adapter = _make_adapter()
        msg = SimpleNamespace(
            id=42,
            to_reference=MagicMock(return_value=object()),
            edit=AsyncMock(),
        )

        # First continuation succeeds; second fails both with and without ref.
        def side(n, content, ref):
            if n == 1:
                return SimpleNamespace(id=9001, to_reference=MagicMock(return_value=object()))
            raise RuntimeError("continuation send failed")

        channel, sends = _wire_channel(adapter, original_msg=msg, send_side_effect=side)

        result = await adapter.edit_message("555", "42", "k" * 6000, finalize=True)

        # Partial delivery still reports success (don't drop chunks the user
        # already saw) but flags partial_overflow so the consumer retries tail.
        assert result.success is True
        assert result.raw_response["partial_overflow"] is True
        assert result.raw_response["delivered_chunks"] < result.raw_response["total_chunks"]
        assert result.message_id == "9001"


# --------------------------------------------------------------------------- #
# Reactive overflow — Discord 50035 mid-edit triggers the same branch logic
# --------------------------------------------------------------------------- #


class TestReactiveOverflowDetection:
    @pytest.mark.asyncio
    async def test_50035_on_final_edit_triggers_split(self):
        adapter = _make_adapter()
        edit_calls = []

        # format_message leaves content under the cap, but the first edit
        # raises 50035 (server-side rejection); the split path then runs.
        def edit_effect(*, content):
            edit_calls.append(content)
            if len(edit_calls) == 1:
                raise RuntimeError(
                    "400 Bad Request (error code: 50035): Invalid Form Body\n"
                    "In content: Must be 2000 or fewer in length."
                )

        msg = SimpleNamespace(
            id=42,
            to_reference=MagicMock(return_value=object()),
            edit=AsyncMock(side_effect=edit_effect),
        )
        channel, sends = _wire_channel(adapter, original_msg=msg)

        # Content is UNDER the cap so pre-flight passes; the 50035 on edit
        # forces the reactive split.
        result = await adapter.edit_message("555", "42", "u" * 1500, finalize=True)

        assert result.success is True
        # Reactive split re-edited chunk 1 and may add continuations.
        assert len(edit_calls) >= 1

    @pytest.mark.asyncio
    async def test_unrelated_50035_is_not_treated_as_overflow(self):
        adapter = _make_adapter()
        msg = SimpleNamespace(
            id=42,
            edit=AsyncMock(side_effect=RuntimeError(
                "400 Bad Request (error code: 50035): In message_reference: "
                "Cannot reply to a system message"
            )),
        )
        channel, sends = _wire_channel(adapter, original_msg=msg)

        result = await adapter.edit_message("555", "42", "small", finalize=True)

        # Not a length error → propagates as a normal failure, no split.
        assert result.success is False
        assert sends == []


# --------------------------------------------------------------------------- #
# Overflow detector helper
# --------------------------------------------------------------------------- #


class TestLengthOverflowDetector:
    def test_matches_length_50035(self):
        err = RuntimeError(
            "error code: 50035 ... Must be 2000 or fewer in length."
        )
        assert DiscordAdapter._is_length_overflow_error(err) is True

    def test_ignores_non_length_50035(self):
        err = RuntimeError("error code: 50035: Cannot reply to a system message")
        assert DiscordAdapter._is_length_overflow_error(err) is False

    def test_ignores_other_errors(self):
        assert DiscordAdapter._is_length_overflow_error(RuntimeError("timeout")) is False
