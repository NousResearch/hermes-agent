"""Regression test: /model strips Markdown link formatting from args (#64847).

Feishu auto-converts URL-like text (e.g. new-api.abrdns.com/DeepSeek-V4-Pro)
into rich-text "post" messages with <a> elements.  The adapter renders these
as Markdown links [text](url), which would get stored verbatim as the model
name, breaking API calls with 503 model_not_found.
"""

import asyncio

import yaml
import pytest

from hermes_cli.model_switch import ModelSwitchResult

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.feishu.adapter import parse_feishu_post_payload


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    return runner


def _make_event(text="/model"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


@pytest.mark.asyncio
async def test_markdown_link_stripped_from_model_arg(tmp_path, monkeypatch):
    """A Markdown link [label](url) in /model args should be reduced to plain text."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "gpt-4o", "provider": "openrouter"}}),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    # Capture the raw_input passed to switch_model; return failure to
    # short-circuit _finish_switch (which needs full runner state).
    switched_models = []

    def fake_switch(**kwargs):
        switched_models.append(kwargs.get("raw_input"))
        return ModelSwitchResult(success=False, error_message="test-stop")

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch)

    # Simulate what Feishu produces after _render_post_element with _escape_markdown_text
    feishu_mangled = (
        r"[new\-api\.abrdns\.com/DeepSeek\-V4\-Pro]"
        r"(http://new-api.abrdns.com/DeepSeek-V4-Pro)"
    )
    event = _make_event(f"/model {feishu_mangled}")

    # Patch asyncio.to_thread to run synchronously in test
    async def passthrough(fn, /, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", passthrough)

    await _make_runner()._handle_model_command(event)

    assert len(switched_models) == 1
    assert switched_models[0] == "new-api.abrdns.com/DeepSeek-V4-Pro"


@pytest.mark.asyncio
async def test_plain_model_arg_unchanged(tmp_path, monkeypatch):
    """A plain model name without Markdown links should pass through unchanged."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "gpt-4o", "provider": "openrouter"}}),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    switched_models = []

    def fake_switch(**kwargs):
        switched_models.append(kwargs.get("raw_input"))
        return ModelSwitchResult(success=False, error_message="test-stop")

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch)

    # Patch asyncio.to_thread to run synchronously in test
    async def passthrough(fn, /, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", passthrough)

    event = _make_event("/model deepseek/DeepSeek-V4-Pro")
    await _make_runner()._handle_model_command(event)

    assert len(switched_models) == 1
    assert switched_models[0] == "deepseek/DeepSeek-V4-Pro"


@pytest.mark.asyncio
async def test_feishu_post_a_tag_through_model_command(tmp_path, monkeypatch):
    """End-to-end: Feishu post <a> payload → parse → /model → correct model name."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "gpt-4o", "provider": "openrouter"}}),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    switched_models = []

    def fake_switch(**kwargs):
        switched_models.append(kwargs.get("raw_input"))
        return ModelSwitchResult(success=False, error_message="test-stop")

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch)

    async def passthrough(fn, /, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", passthrough)

    # Simulate a real Feishu post payload where the user typed:
    #   /model new-api.abrdns.com/DeepSeek-V4-Pro
    # Feishu auto-linked the URL-like text into an <a> element.
    feishu_post_payload = {
        "content": [
            [
                {"tag": "text", "text": "/model "},
                {
                    "tag": "a",
                    "text": "new-api.abrdns.com/DeepSeek-V4-Pro",
                    "href": "http://new-api.abrdns.com/DeepSeek-V4-Pro",
                },
            ]
        ]
    }

    # Parse through the real Feishu adapter post parser
    parsed = parse_feishu_post_payload(feishu_post_payload)
    # This produces something like:
    # "/model [new\-api\.abrdns\.com/DeepSeek\-V4\-Pro](http://...)"
    assert "[" in parsed.text_content, (
        "Feishu adapter should render <a> as Markdown link"
    )

    event = _make_event(parsed.text_content)
    await _make_runner()._handle_model_command(event)

    assert len(switched_models) == 1
    assert switched_models[0] == "new-api.abrdns.com/DeepSeek-V4-Pro"
