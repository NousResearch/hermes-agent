"""Integration tests: SlackAdapter wiring of Block Kit into send paths.

Verifies the opt-in behaviour contract:
  * rich_blocks off (default)  => no ``blocks`` kwarg, plain ``text`` only
  * rich_blocks on             => ``blocks`` present AND ``text`` fallback set
  * edit_message: blocks only on finalize (streaming edits stay plain)
  * multi-chunk (>39k) messages fall back to plain text
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter, _apply_yaml_config


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="xoxb-fake", extra=extra or {})
    a = SlackAdapter(config)
    a._app = MagicMock()
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ts": "111.222"})
    client.chat_update = AsyncMock(return_value={"ts": "111.222"})
    a._get_client = MagicMock(return_value=client)
    a.stop_typing = AsyncMock()
    a._running = True
    return a, client


RICH_MD = "# Title\n\n- a\n  - nested\n\n---\n\nbody text"


class TestSendMessageBlocks:
    @pytest.mark.asyncio
    async def test_disabled_by_default_no_blocks(self):
        adapter, client = _make_adapter()
        await adapter.send("C1", RICH_MD)
        kwargs = client.chat_postMessage.await_args.kwargs
        assert "blocks" not in kwargs
        assert kwargs["text"]  # plain text still sent

    @pytest.mark.asyncio
    async def test_enabled_sends_blocks_with_text_fallback(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.send("C1", RICH_MD)
        kwargs = client.chat_postMessage.await_args.kwargs
        assert "blocks" in kwargs and kwargs["blocks"]
        # text fallback is ALWAYS present alongside blocks (notifications/a11y)
        assert kwargs["text"]
        types = [b["type"] for b in kwargs["blocks"]]
        assert "header" in types
        assert "divider" in types

    @pytest.mark.asyncio
    async def test_enabled_but_unrenderable_falls_back_to_text(self):
        # 60 dividers -> renderer returns None -> no blocks kwarg, text stands
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.send("C1", "\n\n".join(["---"] * 60))
        kwargs = client.chat_postMessage.await_args.kwargs
        assert "blocks" not in kwargs
        assert kwargs["text"]

    @pytest.mark.asyncio
    async def test_string_true_coerced(self):
        adapter, client = _make_adapter({"rich_blocks": "true"})
        await adapter.send("C1", RICH_MD)
        assert "blocks" in client.chat_postMessage.await_args.kwargs

    @pytest.mark.asyncio
    async def test_multichunk_message_no_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        huge = "word " * 20000  # well over MAX_MESSAGE_LENGTH -> chunked
        await adapter.send("C1", huge)
        # every posted chunk is plain text, none carry blocks
        for c in client.chat_postMessage.await_args_list:
            assert "blocks" not in c.kwargs
            assert c.kwargs["text"]


class TestEditMessageBlocks:
    @pytest.mark.asyncio
    async def test_intermediate_edit_no_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.edit_message("C1", "111.222", RICH_MD, finalize=False)
        kwargs = client.chat_update.await_args.kwargs
        assert "blocks" not in kwargs
        assert kwargs["text"]

    @pytest.mark.asyncio
    async def test_finalize_edit_gets_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.edit_message("C1", "111.222", RICH_MD, finalize=True)
        kwargs = client.chat_update.await_args.kwargs
        assert "blocks" in kwargs and kwargs["blocks"]
        assert kwargs["text"]

    @pytest.mark.asyncio
    async def test_finalize_edit_disabled_no_blocks(self):
        adapter, client = _make_adapter()  # rich_blocks off
        await adapter.edit_message("C1", "111.222", RICH_MD, finalize=True)
        assert "blocks" not in client.chat_update.await_args.kwargs


class TestApplyYamlConfigRichBlocksBridge:
    """``_apply_yaml_config`` must seed ``rich_blocks`` into PlatformConfig.extra.

    Regression guard for the config bridge: ``_rich_blocks_enabled()`` reads
    ``self.config.extra["rich_blocks"]``, but the YAML→config hook previously
    returned ``None`` and only ever set ``SLACK_*`` env vars, so the documented
    ``slack.rich_blocks`` opt-in never reached ``extra`` and was silently inert.
    The gateway merges this hook's returned dict into ``extra`` (see
    ``TestApplyYamlConfigFnDispatch`` for the dispatch contract), so returning
    ``{"rich_blocks": ...}`` here is what actually enables the feature.
    """

    def test_nested_extra_form_is_bridged(self):
        seeded = _apply_yaml_config({}, {"extra": {"rich_blocks": True}})
        assert seeded == {"rich_blocks": True}

    def test_flat_form_is_bridged(self):
        seeded = _apply_yaml_config({}, {"rich_blocks": True})
        assert seeded == {"rich_blocks": True}

    def test_string_value_is_bridged_verbatim(self):
        # Coercion is the reader's job (_rich_blocks_enabled); the bridge just
        # forwards the raw value so "true"/"1"/"on" all survive to the reader.
        seeded = _apply_yaml_config({}, {"rich_blocks": "true"})
        assert seeded == {"rich_blocks": "true"}

    def test_flat_form_wins_over_nested_when_both_present(self):
        # Matches the discord/telegram precedence idiom: the flat top-level key
        # takes precedence over the nested ``extra`` form.
        seeded = _apply_yaml_config(
            {}, {"rich_blocks": True, "extra": {"rich_blocks": False}}
        )
        assert seeded == {"rich_blocks": True}

    def test_false_value_is_bridged_and_stays_disabled(self):
        # ``rich_blocks: false`` is a non-empty dict, so it is NOT dropped by
        # ``seeded or None``; it is forwarded and the reader stays disabled.
        seeded = _apply_yaml_config({}, {"rich_blocks": False})
        assert seeded == {"rich_blocks": False}
        adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake", extra=seeded))
        assert adapter._rich_blocks_enabled() is False

    def test_absent_returns_none(self):
        # No rich_blocks anywhere -> nothing to merge; other keys still flow
        # through env, so the hook returns None (unchanged contract).
        assert _apply_yaml_config({}, {"require_mention": True}) is None

    def test_non_dict_extra_is_ignored(self):
        # A malformed ``extra:`` (not a mapping) must not raise; it degrades to
        # "no nested form present". ``None``/``int`` raise on the
        # ``"rich_blocks" in _extra_in`` membership test and a list raises on
        # the subsequent subscript, so these exercise the ``isinstance`` guard
        # directly (a plain string would pass even without it).
        for bad in (None, 123, ["rich_blocks"], "oops"):
            assert _apply_yaml_config({}, {"extra": bad}) is None

    def test_bridged_value_enables_reader(self):
        # End-to-end within the adapter: what the hook seeds must be what
        # _rich_blocks_enabled() reads back as truthy.
        seeded = _apply_yaml_config({}, {"extra": {"rich_blocks": True}})
        adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake", extra=seeded))
        assert adapter._rich_blocks_enabled() is True
