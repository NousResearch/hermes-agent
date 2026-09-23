"""Tests for gateway.response_prefix — the opt-in model/provider prefix on
the FIRST gateway reply of a turn — and its streaming/command wiring."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.response_prefix import (
    _model_short,
    apply_prefix,
    build_prefix_line,
    interpolate_prefix_template,
    platform_has_prefix_override,
    resolve_prefix_config,
    strip_prefix,
)
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


# ---------------------------------------------------------------------------
# Template / config resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model,expected",
    [
        ("openai/gpt-5.4", "gpt-5.4"),
        ("anthropic/claude-sonnet-4.6", "claude-sonnet-4.6"),
        ("gpt-5.4", "gpt-5.4"),
        ("", ""),
        (None, ""),
    ],
)
def test_model_short_drops_vendor_prefix(model, expected):
    assert _model_short(model) == expected


def test_interpolate_all_variables_case_insensitive():
    out = interpolate_prefix_template(
        "[{PROVIDER}/{model}] {modelfull}",
        model="openai/gpt-5.4",
        provider="openai",
    )
    assert out == "[openai/gpt-5.4] openai/gpt-5.4"


def test_interpolate_unknown_placeholder_left_verbatim():
    assert interpolate_prefix_template("{nope} {model}", model="x/y") == "{nope} y"


def test_interpolate_missing_value_left_verbatim():
    assert interpolate_prefix_template("[{provider}]", model="gpt-5.4") == "[{provider}]"


def test_resolve_defaults_off():
    assert resolve_prefix_config({}) == {"enabled": False, "template": ""}
    assert resolve_prefix_config(None, "telegram")["enabled"] is False


def test_resolve_string_shorthand_enables():
    cfg = {"display": {"response_prefix": "[{model}]"}}
    assert resolve_prefix_config(cfg) == {"enabled": True, "template": "[{model}]"}
    assert resolve_prefix_config({"display": {"response_prefix": "   "}})["enabled"] is False


def test_resolve_platform_override_wins():
    cfg = {
        "display": {
            "response_prefix": {"enabled": True, "template": "[{model}]"},
            "platforms": {"telegram": {"response_prefix": {"enabled": False}}},
        }
    }
    assert resolve_prefix_config(cfg, "telegram") == {"enabled": False, "template": "[{model}]"}
    assert resolve_prefix_config(cfg, "discord")["enabled"] is True
    assert platform_has_prefix_override(cfg, "telegram") is True
    assert platform_has_prefix_override(cfg, "discord") is False


def test_build_prefix_line_disabled_returns_empty():
    assert build_prefix_line(user_config={}, platform_key="telegram", model="openai/gpt-5.4") == ""
    cfg = {"display": {"response_prefix": {"enabled": True, "template": "   "}}}
    assert build_prefix_line(user_config=cfg, platform_key=None, model="x/y") == ""


def test_build_prefix_line_derives_provider_from_model():
    cfg = {"display": {"response_prefix": {"enabled": True, "template": "[{provider}/{model}]"}}}
    assert build_prefix_line(user_config=cfg, platform_key=None, model="openai/gpt-5.4") == "[openai/gpt-5.4]"
    assert build_prefix_line(
        user_config=cfg, platform_key=None, model="openai/gpt-5.4", provider="custom",
    ) == "[custom/gpt-5.4]"


def test_build_prefix_line_collapses_to_single_line():
    cfg = {"display": {"response_prefix": {"enabled": True, "template": "  [{model}]\n\n"}}}
    assert build_prefix_line(user_config=cfg, platform_key=None, model="gpt-5.4") == "[gpt-5.4]"


def test_apply_and_strip_prefix_roundtrip():
    assert apply_prefix("[m]", "hi") == "[m] hi"
    assert apply_prefix("[m]", "[m] hi") == "[m] hi"  # idempotent
    assert apply_prefix("[m]", "") == ""
    assert apply_prefix("", "hi") == "hi"
    assert strip_prefix("[m]", "[m] hi") == "hi"
    assert strip_prefix("[m]", "hi") == "hi"


# ---------------------------------------------------------------------------
# Streaming delivery: prefix lands on the FIRST message exactly once
# ---------------------------------------------------------------------------

def _make_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    adapter.delete_message = AsyncMock(return_value=True)
    return adapter


def _make_consumer(adapter, prefix="[openai/gpt-5.4]"):
    return GatewayStreamConsumer(
        adapter=adapter,
        chat_id="chat",
        config=StreamConsumerConfig(edit_interval=0.0, fresh_final_after_seconds=0.0),
        prefix=prefix,
    )


def _sent_texts(adapter):
    return [c.kwargs.get("content", "") for c in adapter.send.call_args_list] + [
        c.kwargs.get("content", "") for c in adapter.edit_message.call_args_list
    ]


@pytest.mark.asyncio
async def test_streamed_first_message_and_edits_carry_prefix_once():
    adapter = _make_adapter()
    consumer = _make_consumer(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Hello")
    await asyncio.sleep(0.05)
    consumer.on_delta(" world")
    await asyncio.sleep(0.05)
    consumer.finish("Hello world")
    await task

    texts = [t for t in _sent_texts(adapter) if t]
    assert texts, "nothing was delivered"
    assert texts[0].startswith("[openai/gpt-5.4] Hello")
    for t in texts:
        assert t.count("[openai/gpt-5.4]") == 1, t
    # The seal adopted the authoritative final WITH the prefix.
    assert texts[-1] == "[openai/gpt-5.4] Hello world"
    # Delivered payload reconciles against the gateway's un-prefixed final.
    assert consumer.delivered_final_matches("Hello world") is True
    assert consumer.has_delivered_text("Hello world") is True


@pytest.mark.asyncio
async def test_streamed_prefix_not_repeated_on_later_segments():
    adapter = _make_adapter()
    consumer = _make_consumer(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Let me look that up.")
    await asyncio.sleep(0.05)
    consumer.on_delta(None)  # tool boundary → new message
    await asyncio.sleep(0.05)
    consumer.on_delta("The answer is 42.")
    await asyncio.sleep(0.05)
    consumer.finish("The answer is 42.")
    await task

    texts = [t for t in _sent_texts(adapter) if t]
    assert texts[0].startswith("[openai/gpt-5.4] Let me look")
    later = [t for t in texts if "42" in t]
    assert later and all("[openai/gpt-5.4]" not in t for t in later)
    assert consumer.delivered_final_matches("The answer is 42.") is True


@pytest.mark.asyncio
async def test_streamed_prefix_applied_after_think_block_filter():
    adapter = _make_adapter()
    consumer = _make_consumer(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("<think>secret</think>Visible")
    await asyncio.sleep(0.05)
    consumer.finish("Visible")
    await task

    texts = [t for t in _sent_texts(adapter) if t]
    assert texts and texts[0].startswith("[openai/gpt-5.4] Visible")
    assert all("secret" not in t for t in texts)


@pytest.mark.asyncio
async def test_no_prefix_configured_is_a_noop():
    adapter = _make_adapter()
    consumer = _make_consumer(adapter, prefix=None)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Hello")
    await asyncio.sleep(0.05)
    consumer.finish("Hello")
    await task
    texts = [t for t in _sent_texts(adapter) if t]
    assert texts and texts[0].startswith("Hello") and "[" not in texts[0]
    assert consumer.delivered_final_matches("Hello") is True


# ---------------------------------------------------------------------------
# /prefix gateway command — override-aware toggle
# ---------------------------------------------------------------------------

def _make_event(text: str):
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    source = SessionSource(
        platform=Platform.TELEGRAM, user_id="u1", chat_id="c1",
        user_name="tester", chat_type="dm",
    )
    return MessageEvent(text=text, source=source, message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner
    return object.__new__(GatewayRunner)


@pytest.fixture
def prefix_config(monkeypatch, tmp_path):
    """Patch config load/write so the handler edits an in-memory dict."""
    import gateway.run as run_mod
    import gateway.slash_commands as sc_mod

    state = {
        "cfg": {
            "display": {
                "response_prefix": {"enabled": True, "template": "[{model}]"},
                "platforms": {"telegram": {"response_prefix": {"enabled": True, "template": "[TG {model}]"}}},
            }
        },
        "written": None,
    }
    monkeypatch.setattr(run_mod, "_load_gateway_config", lambda: state["cfg"])
    monkeypatch.setattr(run_mod, "_resolve_gateway_model", lambda cfg: "openai/gpt-5.4")
    monkeypatch.setattr(run_mod, "_hermes_home", tmp_path)

    def _write(path, cfg):
        state["written"] = cfg

    monkeypatch.setattr(sc_mod, "atomic_config_write", _write)
    return state


@pytest.mark.asyncio
async def test_prefix_off_disables_platform_override_too(prefix_config):
    runner = _make_runner()
    reply = await runner._handle_prefix_command(_make_event("/prefix off"))
    written = prefix_config["written"]
    assert written is not None
    assert written["display"]["response_prefix"]["enabled"] is False
    assert written["display"]["platforms"]["telegram"]["response_prefix"]["enabled"] is False
    assert "OFF" in reply and "platform override" in reply
    # Effective state really is off for this platform now.
    assert resolve_prefix_config(written, "telegram")["enabled"] is False


@pytest.mark.asyncio
async def test_prefix_status_reports_effective_platform_state(prefix_config):
    prefix_config["cfg"]["display"]["platforms"]["telegram"]["response_prefix"]["enabled"] = False
    runner = _make_runner()
    reply = await runner._handle_prefix_command(_make_event("/prefix status"))
    assert "OFF" in reply and "[TG {model}]" in reply
    assert prefix_config["written"] is None  # status never writes


@pytest.mark.asyncio
async def test_prefix_on_promotes_string_shorthand_and_previews(prefix_config):
    prefix_config["cfg"] = {"display": {"response_prefix": "[{provider}/{model}]"}}
    prefix_config["cfg"]["display"]["response_prefix"] = "[{provider}/{model}]"
    runner = _make_runner()
    reply = await runner._handle_prefix_command(_make_event("/prefix on"))
    written = prefix_config["written"]
    assert written["display"]["response_prefix"] == {"enabled": True, "template": "[{provider}/{model}]"}
    assert "ON" in reply and "[openai/gpt-5.4] Hello!" in reply


@pytest.mark.asyncio
async def test_prefix_bad_arg_shows_usage(prefix_config):
    runner = _make_runner()
    reply = await runner._handle_prefix_command(_make_event("/prefix maybe"))
    assert "Usage" in reply and prefix_config["written"] is None


def test_prefix_registered_for_gateway_and_cli():
    from hermes_cli.commands import COMMAND_REGISTRY
    cmd = next(c for c in COMMAND_REGISTRY if c.name == "prefix")
    assert cmd.subcommands == ("on", "off", "status")
    assert not getattr(cmd, "cli_only", False)
