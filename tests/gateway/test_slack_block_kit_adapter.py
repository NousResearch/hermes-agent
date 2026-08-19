"""Integration tests: SlackAdapter wiring of Block Kit into send paths.

Verifies the opt-in behaviour contract:
  * rich_blocks off (default)  => no ``blocks`` kwarg, plain ``text`` only
  * rich_blocks on             => ``blocks`` present AND ``text`` fallback set
  * edit_message: blocks only on finalize (streaming edits stay plain)
  * multi-chunk (>39k) messages fall back to plain text
"""

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack import adapter as slack_module
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
RICH_TABLE_MD = (
    "| Item | Status | Note |\n"
    "|---|---:|---|\n"
    "| Hermes | ok | table |"
)


class SlackRejectedBlocks(Exception):
    def __init__(self, error="invalid_blocks"):
        super().__init__(f"Slack API rejected blocks: {error}")
        self.response = {"error": error}


def _slack_connection_key():
    from aiohttp.client_reqrep import ConnectionKey

    return ConnectionKey(
        host="slack.com",
        port=443,
        is_ssl=True,
        ssl=True,
        proxy=None,
        proxy_auth=None,
        proxy_headers_hash=None,
    )


class TestSendMessageBlocks:
    @pytest.mark.asyncio
    async def test_disabled_by_default_no_blocks(self):
        adapter, client = _make_adapter()
        await adapter.send("C1", RICH_MD)
        kwargs = client.chat_postMessage.await_args.kwargs
        assert "blocks" not in kwargs
        assert kwargs["text"]  # plain text still sent


    @pytest.mark.asyncio
    async def test_enabled_but_unrenderable_falls_back_to_text(self):
        # 60 dividers -> renderer returns None -> no blocks kwarg, text stands
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.send("C1", "\n\n".join(["---"] * 60))
        kwargs = client.chat_postMessage.await_args.kwargs
        assert "blocks" not in kwargs
        assert kwargs["text"]


    @pytest.mark.asyncio
    async def test_feedback_buttons_opt_in_appended_to_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True, "feedback_buttons": True})

        await adapter.send("C1", "final answer")

        blocks = client.chat_postMessage.await_args.kwargs["blocks"]
        feedback = blocks[-1]
        assert feedback["type"] == "context_actions"
        assert feedback["elements"][0]["type"] == "feedback_buttons"
        assert feedback["elements"][0]["action_id"] == "hermes_feedback"


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
    async def test_block_rejection_retries_edit_without_blocks_using_workspace_client(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        client.chat_update = AsyncMock(
            side_effect=[SlackRejectedBlocks("invalid_blocks"), {"ts": "111.222"}]
        )

        result = await adapter.edit_message(
            "C1",
            "111.222",
            RICH_TABLE_MD,
            finalize=True,
            metadata={"team_id": "T_SECONDARY"},
        )

        assert result.success is True
        assert adapter._get_client.call_args_list == [
            call("C1", team_id="T_SECONDARY"),
            call("C1", team_id="T_SECONDARY"),
        ]
        assert client.chat_update.await_count == 2
        first = client.chat_update.await_args_list[0].kwargs
        second = client.chat_update.await_args_list[1].kwargs
        assert "blocks" in first and first["blocks"]
        assert second["blocks"] == []
        assert second["text"]

    @pytest.mark.asyncio
    async def test_timeout_error_on_edit_is_retryable_transient(self):
        adapter, client = _make_adapter()
        client.chat_update = AsyncMock(side_effect=TimeoutError("timed out"))

        result = await adapter.edit_message("C1", "111.222", RICH_MD, finalize=True)

        assert result.success is False
        assert result.retryable is True
        assert result.error_kind == "transient"


# ---------------------------------------------------------------------------
# markdown_blocks mode — Slack's native ``markdown`` Block Kit block (#8552)
# ---------------------------------------------------------------------------


class TestMarkdownBlockMode:
    """Opt-in ``markdown_blocks`` renders raw standard markdown via Slack's
    native ``markdown`` block, keeping the mrkdwn ``text`` fallback."""

    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        adapter, client = _make_adapter()
        await adapter.send("C1", RICH_TABLE_MD)
        kwargs = client.chat_postMessage.await_args.kwargs
        assert "blocks" not in kwargs

    @pytest.mark.asyncio
    async def test_enabled_sends_markdown_block_with_raw_content(self):
        adapter, client = _make_adapter({"markdown_blocks": True})
        await adapter.send("C1", RICH_TABLE_MD)
        kwargs = client.chat_postMessage.await_args.kwargs
        blocks = kwargs["blocks"]
        assert blocks[0]["type"] == "markdown"
        # RAW standard markdown, not mrkdwn-converted — Slack translates it
        assert blocks[0]["text"] == RICH_TABLE_MD
        # mrkdwn fallback text is still present for notifications/search
        assert kwargs["text"]


    @pytest.mark.asyncio
    async def test_edit_finalize_uses_markdown_block(self):
        adapter, client = _make_adapter({"markdown_blocks": True})
        await adapter.edit_message("C1", "111.222", RICH_TABLE_MD, finalize=True)
        kwargs = client.chat_update.await_args.kwargs
        assert kwargs["blocks"][0]["type"] == "markdown"
        assert kwargs["blocks"][0]["text"] == RICH_TABLE_MD


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
