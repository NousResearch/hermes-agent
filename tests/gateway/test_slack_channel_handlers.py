"""Tests for Slack per-channel no_agent script handlers (``channel_handlers``).

These exercise the real ``SlackAdapter._handle_slack_message`` dispatch seam
plus the ``_run_channel_handler`` subprocess runner.  The handler is the Slack
analogue of the webhook ``deliver_only`` contract: on every plain user message
in a mapped channel, a script is run as a fire-and-forget subprocess with the
raw event JSON on stdin, and a handler failure/timeout must never break normal
message handling.  @mentions in a mapped channel must still reach the agent.

Conventions mirror ``test_slack_channel_session_scope.py``: ``handle_message``
is the agent path (an ``AsyncMock``); driving the real
``_handle_slack_message`` keeps the seam tight against production behaviour.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config
from gateway.platforms.slack import SlackAdapter


MAPPED_CHANNEL = "C_MAPPED"
UNMAPPED_CHANNEL = "C_OTHER"


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    # Map a channel to a handler. The script path is irrelevant for the
    # dispatch-decision tests because we patch the runner; the runner tests
    # below use a real tmp script via _run_channel_handler directly.
    a.config.extra["channel_handlers"] = {
        MAPPED_CHANNEL: {"script": "handler.py", "timeout": 30}
    }
    return a


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point document cache to tmp_path so tests don't touch ~/.hermes."""
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )


def _event(channel: str, text: str = "hello", **extra) -> dict:
    event = {
        "channel": channel,
        "channel_type": "channel",
        "user": "U_USER",
        "text": text,
        "ts": "1700000000.000001",
    }
    event.update(extra)
    return event


async def _drive(adapter, event):
    """Run _handle_slack_message with user-resolution stubbed."""
    with patch.object(
        adapter, "_resolve_user_name", new=AsyncMock(return_value="testuser")
    ):
        await adapter._handle_slack_message(event)


class TestDispatchDecision:
    """Which messages get dispatched to the channel handler."""

    @pytest.mark.asyncio
    async def test_mapped_channel_dispatches(self, adapter):
        with patch.object(adapter, "_dispatch_channel_handler") as disp:
            await _drive(adapter, _event(MAPPED_CHANNEL))
        assert disp.call_count == 1, "mapped channel must dispatch the handler"
        args = disp.call_args.args
        assert args[0] == MAPPED_CHANNEL
        assert args[1] == {"script": "handler.py", "timeout": 30}
        assert args[2]["channel"] == MAPPED_CHANNEL  # raw event passed through

    @pytest.mark.asyncio
    async def test_unmapped_channel_untouched(self, adapter):
        with patch.object(adapter, "_dispatch_channel_handler") as disp:
            await _drive(adapter, _event(UNMAPPED_CHANNEL))
        assert disp.call_count == 0, "unmapped channel must not dispatch"

    @pytest.mark.asyncio
    async def test_self_message_skipped(self, adapter):
        # A message authored by our own bot user must not feed the handler.
        ev = _event(MAPPED_CHANNEL, user="U_BOT")
        with patch.object(adapter, "_dispatch_channel_handler") as disp:
            await _drive(adapter, ev)
        assert disp.call_count == 0, "self-authored message must be skipped"

    @pytest.mark.asyncio
    async def test_bot_message_subtype_skipped(self, adapter):
        # bot_message subtype is dropped before dispatch (default allow_bots).
        ev = _event(MAPPED_CHANNEL, subtype="bot_message", bot_id="B123")
        with patch.object(adapter, "_dispatch_channel_handler") as disp:
            await _drive(adapter, ev)
        assert disp.call_count == 0, "bot_message subtype must be skipped"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("subtype", ["message_changed", "message_deleted"])
    async def test_edit_delete_subtypes_skipped(self, adapter, subtype):
        ev = _event(MAPPED_CHANNEL, subtype=subtype)
        with patch.object(adapter, "_dispatch_channel_handler") as disp:
            await _drive(adapter, ev)
        assert disp.call_count == 0, f"{subtype} must be skipped"

    @pytest.mark.asyncio
    async def test_self_thread_broadcast_skipped(self, adapter):
        ev = _event(MAPPED_CHANNEL, subtype="thread_broadcast", user="U_BOT")
        with patch.object(adapter, "_dispatch_channel_handler") as disp:
            await _drive(adapter, ev)
        assert disp.call_count == 0, "self thread-broadcast must be skipped"


class TestRunsInAdditionToAgent:
    """Handler runs in addition to normal processing — @mentions in a mapped
    channel must still reach the agent path."""

    @pytest.mark.asyncio
    async def test_mention_in_mapped_channel_still_reaches_agent(self, adapter):
        captured = []
        adapter.handle_message = AsyncMock(side_effect=lambda e: captured.append(e))
        with patch.object(adapter, "_dispatch_channel_handler") as disp:
            await _drive(adapter, _event(MAPPED_CHANNEL, text="<@U_BOT> ping"))
        assert disp.call_count == 1, "handler still fires on the mention"
        assert len(captured) == 1, (
            "an @mention in a mapped channel must still reach the agent path "
            "— channel_handlers must not short-circuit agent processing"
        )

    @pytest.mark.asyncio
    async def test_non_mention_in_mapped_channel_does_not_reach_agent(self, adapter):
        # Default gating: a non-mention channel message is dropped from the
        # agent path, but the handler still fires. This is the spam-triage
        # case: deterministic handling without an agent session.
        captured = []
        adapter.handle_message = AsyncMock(side_effect=lambda e: captured.append(e))
        with patch.object(adapter, "_dispatch_channel_handler") as disp:
            await _drive(adapter, _event(MAPPED_CHANNEL, text="buy cheap stuff"))
        assert disp.call_count == 1, "handler fires on the non-mention message"
        assert len(captured) == 0, (
            "a plain non-mention message must not reach the agent under "
            "default mention gating — only the handler runs"
        )


class TestConfigResolution:
    """``_slack_channel_handler_for`` normalisation."""

    def test_dict_entry(self, adapter):
        cfg = adapter._slack_channel_handler_for(MAPPED_CHANNEL)
        assert cfg == {"script": "handler.py", "timeout": 30}

    def test_bare_string_entry_uses_default_timeout(self, adapter):
        adapter.config.extra["channel_handlers"] = {MAPPED_CHANNEL: "s.py"}
        assert adapter._slack_channel_handler_for(MAPPED_CHANNEL) == {
            "script": "s.py",
            "timeout": 60,
        }

    def test_unmapped_returns_none(self, adapter):
        assert adapter._slack_channel_handler_for(UNMAPPED_CHANNEL) is None

    def test_no_channel_handlers_config_returns_none(self, adapter):
        adapter.config.extra.pop("channel_handlers", None)
        assert adapter._slack_channel_handler_for(MAPPED_CHANNEL) is None

    def test_empty_script_ignored(self, adapter):
        adapter.config.extra["channel_handlers"] = {MAPPED_CHANNEL: {"script": "  "}}
        assert adapter._slack_channel_handler_for(MAPPED_CHANNEL) is None

    def test_nonpositive_timeout_falls_back_to_default(self, adapter):
        adapter.config.extra["channel_handlers"] = {
            MAPPED_CHANNEL: {"script": "s.py", "timeout": 0}
        }
        assert adapter._slack_channel_handler_for(MAPPED_CHANNEL)["timeout"] == 60


class TestScriptPathResolution:
    """Path resolution mirrors cron's contract: relative under
    HERMES_HOME/scripts, traversal blocked."""

    def test_relative_resolves_under_scripts_dir(self, adapter, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home", lambda: tmp_path
        )
        path = adapter._resolve_handler_script_path("handler.py")
        assert path == (tmp_path / "scripts" / "handler.py").resolve()

    def test_traversal_is_blocked(self, adapter, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home", lambda: tmp_path
        )
        with pytest.raises(ValueError):
            adapter._resolve_handler_script_path("../../etc/passwd")


class TestRunChannelHandlerSubprocess:
    """The real subprocess runner: stdin payload, timeout, and failure
    isolation."""

    @pytest.mark.asyncio
    async def test_runs_script_and_passes_event_on_stdin(
        self, adapter, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        out_file = tmp_path / "received.json"
        script = scripts_dir / "handler.py"
        script.write_text(
            "import sys, json, pathlib\n"
            "data = sys.stdin.read()\n"
            f"pathlib.Path(r'{out_file}').write_text(data)\n"
        )

        event = _event(MAPPED_CHANNEL, text="spam?")
        await adapter._run_channel_handler(
            MAPPED_CHANNEL, {"script": "handler.py", "timeout": 30}, event
        )

        assert out_file.exists(), "handler did not run / did not read stdin"
        received = json.loads(out_file.read_text())
        assert received["channel"] == MAPPED_CHANNEL
        assert received["text"] == "spam?"

    @pytest.mark.asyncio
    async def test_timeout_does_not_raise(self, adapter, tmp_path, monkeypatch):
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script = scripts_dir / "slow.py"
        script.write_text("import time\ntime.sleep(5)\n")

        # timeout=1 < sleep 5 → must time out, kill the proc, and return
        # cleanly without raising.
        await adapter._run_channel_handler(
            MAPPED_CHANNEL, {"script": "slow.py", "timeout": 1}, _event(MAPPED_CHANNEL)
        )

    @pytest.mark.asyncio
    async def test_missing_script_does_not_raise(self, adapter, tmp_path, monkeypatch):
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
        (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
        await adapter._run_channel_handler(
            MAPPED_CHANNEL, {"script": "nope.py", "timeout": 5}, _event(MAPPED_CHANNEL)
        )

    @pytest.mark.asyncio
    async def test_nonzero_exit_does_not_raise(self, adapter, tmp_path, monkeypatch):
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script = scripts_dir / "boom.py"
        script.write_text("import sys\nsys.exit(3)\n")
        await adapter._run_channel_handler(
            MAPPED_CHANNEL, {"script": "boom.py", "timeout": 5}, _event(MAPPED_CHANNEL)
        )


class TestConfigBridging:
    """``slack.channel_handlers`` from config.yaml must reach the adapter's
    ``config.extra`` — otherwise the dispatch never sees the mapping."""

    def test_channel_handlers_bridged_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "slack:\n"
            "  channel_handlers:\n"
            "    C4R0C7FPZ:\n"
            "      script: ab_contact_spam_cleanup.py\n"
            "      timeout: 120\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()
        slack_cfg = config.platforms.get(Platform.SLACK)
        assert slack_cfg is not None
        assert slack_cfg.extra.get("channel_handlers") == {
            "C4R0C7FPZ": {"script": "ab_contact_spam_cleanup.py", "timeout": 120}
        }


class TestDispatchFailureIsolation:
    """A handler dispatch error must never break message handling."""

    @pytest.mark.asyncio
    async def test_dispatch_exception_does_not_break_handling(self, adapter):
        # Make dispatch raise; the message handler must still complete and the
        # agent path must still be reached for an @mention.
        captured = []
        adapter.handle_message = AsyncMock(side_effect=lambda e: captured.append(e))
        with patch.object(
            adapter, "_dispatch_channel_handler", side_effect=RuntimeError("boom")
        ):
            await _drive(adapter, _event(MAPPED_CHANNEL, text="<@U_BOT> hi"))
        assert len(captured) == 1, (
            "a channel_handler dispatch error must be swallowed and must not "
            "prevent the agent path from running"
        )
