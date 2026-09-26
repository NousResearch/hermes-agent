"""Regression tests for selectable plugin middleware failure policies."""

from __future__ import annotations

import textwrap

import pytest
import yaml

from hermes_cli.middleware import (
    LLMStreamMiddlewareRefusal,
    run_llm_execution_middleware,
    run_llm_stream_text_middleware,
)
from agent.error_classifier import classify_api_error
from hermes_cli.plugin_validate import validate_plugin_dir
from hermes_cli.plugins import PluginManager


def _load_plugin(tmp_path, monkeypatch, name: str, register_body: str) -> PluginManager:
    home = tmp_path / "home"
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({
            "name": name,
            "version": "0.1.0",
            "description": "middleware failure-mode test",
        }),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n" + textwrap.indent(register_body.strip() + "\n", "    "),
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": [name]}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    return manager


def _use_manager(monkeypatch, manager: PluginManager) -> None:
    monkeypatch.setattr("hermes_cli.plugins._delivery_manager", lambda: manager)


def test_execution_failure_mode_is_selected_per_plugin_registration(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "failure-policy",
        """
def fail_open(**kwargs):
    raise RuntimeError("open failure")

def fail_closed(**kwargs):
    raise RuntimeError("closed failure")

ctx.register_middleware("llm_execution", fail_open, failure_mode="open")
ctx.register_middleware("tool_execution", fail_closed, failure_mode="closed")
""",
    )

    assert manager._middleware["llm_execution"][0]._hermes_failure_mode == "open"
    assert manager._middleware["tool_execution"][0]._hermes_failure_mode == "closed"


def test_fail_open_execution_keeps_legacy_fallthrough(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "fail-open",
        """
def protect(**kwargs):
    raise RuntimeError("plugin unavailable")

ctx.register_middleware("llm_execution", protect, failure_mode="open")
""",
    )
    _use_manager(monkeypatch, manager)

    calls = []

    def provider(request):
        calls.append(request)
        return {"ok": True}

    result = run_llm_execution_middleware({"messages": []}, provider)

    assert result == {"ok": True}
    assert calls == [{"messages": []}]


def test_fail_closed_execution_blocks_before_provider(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "fail-closed-pre",
        """
def protect(**kwargs):
    raise RuntimeError("privacy boundary unavailable")

ctx.register_middleware("llm_execution", protect, failure_mode="closed")
""",
    )
    _use_manager(monkeypatch, manager)

    calls = []

    def provider(request):
        calls.append(request)
        return {"ok": True}

    with pytest.raises(RuntimeError, match="privacy boundary unavailable"):
        run_llm_execution_middleware({"messages": []}, provider)

    assert calls == []


def test_fail_closed_execution_propagates_post_provider_failure(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "fail-closed-post",
        """
def protect(**kwargs):
    kwargs["next_call"](kwargs["request"])
    raise RuntimeError("restore failed")

ctx.register_middleware("llm_execution", protect, failure_mode="closed")
""",
    )
    _use_manager(monkeypatch, manager)

    calls = []

    def provider(request):
        calls.append(request)
        return {"unsafe": "provider response"}

    with pytest.raises(RuntimeError, match="restore failed"):
        run_llm_execution_middleware({"messages": []}, provider)

    assert calls == [{"messages": []}]


def test_fail_open_execution_preserves_post_provider_result(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "fail-open-post",
        """
def observe(**kwargs):
    kwargs["next_call"](kwargs["request"])
    raise RuntimeError("observer failed")

ctx.register_middleware("llm_execution", observe)
""",
    )
    _use_manager(monkeypatch, manager)

    result = run_llm_execution_middleware(
        {"messages": []},
        lambda request: {"result": request},
    )

    assert result == {"result": {"messages": []}}


def test_live_text_transform_receives_request_identity_and_rewrites(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "stream-transform",
        """
def transform(**kwargs):
    return {"text": f"{kwargs['kind']}:{kwargs['api_request_id']}:{kwargs['text']}"}

ctx.register_middleware("llm_stream_text", transform, failure_mode="closed")
""",
    )
    _use_manager(monkeypatch, manager)

    result = run_llm_stream_text_middleware(
        "tokenized",
        kind="text",
        provider="openrouter",
        model="model",
        session_id="session-1",
        turn_id="turn-1",
        api_request_id="request-1",
    )

    assert result == "text:request-1:tokenized"


def test_live_text_failure_policy_can_be_open_or_closed(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "stream-open",
        """
def transform(**kwargs):
    raise RuntimeError("stream transform failed")

ctx.register_middleware("llm_stream_text", transform, failure_mode="open")
""",
    )
    _use_manager(monkeypatch, manager)

    assert run_llm_stream_text_middleware("visible", kind="text") == "visible"

    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "stream-closed",
        """
def transform(**kwargs):
    raise RuntimeError("stream transform failed")

ctx.register_middleware("llm_stream_text", transform, failure_mode="closed")
""",
    )
    _use_manager(monkeypatch, manager)

    with pytest.raises(RuntimeError, match="stream transform failed"):
        run_llm_stream_text_middleware("must-not-deliver", kind="text")


def test_invalid_failure_mode_is_rejected(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "bad-mode",
        """
def callback(**kwargs):
    return None

ctx.register_middleware("llm_execution", callback, failure_mode="maybe")
""",
    )

    assert "failure_mode" in (manager._plugins["bad-mode"].error or "")

def test_direct_async_stream_transform_registration_is_rejected(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "async-stream",
        """
async def transform(**kwargs):
    return {"text": "unsafe"}

ctx.register_middleware("llm_stream_text", transform, failure_mode="closed")
""",
    )

    assert "must be synchronous" in (manager._plugins["async-stream"].error or "")


def test_wrapped_awaitable_closed_transform_refuses_delivery(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "wrapped-awaitable-closed",
        """
async def inner():
    return {"text": "unsafe"}

def transform(**kwargs):
    return inner()

ctx.register_middleware("llm_stream_text", transform, failure_mode="closed")
""",
    )
    _use_manager(monkeypatch, manager)

    with pytest.raises(LLMStreamMiddlewareRefusal, match="must be synchronous"):
        run_llm_stream_text_middleware("secret", kind="text")


def test_wrapped_awaitable_open_transform_keeps_legacy_passthrough(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "wrapped-awaitable-open",
        """
async def inner():
    return {"text": "unsafe"}

def transform(**kwargs):
    return inner()

ctx.register_middleware("llm_stream_text", transform, failure_mode="open")
""",
    )
    _use_manager(monkeypatch, manager)

    assert run_llm_stream_text_middleware("visible", kind="text") == "visible"


def test_direct_async_generator_stream_registration_is_rejected(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "async-generator-stream",
        """
async def transform(**kwargs):
    yield {"text": "unsafe"}

ctx.register_middleware("llm_stream_text", transform, failure_mode="closed")
""",
    )

    assert "must be synchronous" in (
        manager._plugins["async-generator-stream"].error or ""
    )


def test_validator_rejects_async_generator_stream_registration(tmp_path):
    plugin_dir = tmp_path / "validator-async-generator"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({
            "name": "validator-async-generator",
            "version": "0.1.0",
            "description": "validator async-generator regression",
            "provides_middleware": ["llm_stream_text"],
        }),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        """
def register(ctx):
    async def transform(**kwargs):
        yield {"text": "unsafe"}

    ctx.register_middleware(
        "llm_stream_text",
        transform,
        failure_mode="closed",
    )
""".lstrip(),
        encoding="utf-8",
    )

    report = validate_plugin_dir(plugin_dir)

    assert report.ok is False
    assert any("must be synchronous" in failure for failure in report.failures)


def test_wrapped_async_generator_closed_transform_refuses_delivery(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "wrapped-async-generator-closed",
        """
async def inner():
    yield {"text": "unsafe"}

def transform(**kwargs):
    return inner()

ctx.register_middleware("llm_stream_text", transform, failure_mode="closed")
""",
    )
    _use_manager(monkeypatch, manager)

    with pytest.raises(
        LLMStreamMiddlewareRefusal,
        match="asynchronous/deferred result",
    ):
        run_llm_stream_text_middleware("secret", kind="text")


def test_wrapped_async_generator_open_transform_keeps_passthrough(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "wrapped-async-generator-open",
        """
async def inner():
    yield {"text": "unsafe"}

def transform(**kwargs):
    return inner()

ctx.register_middleware("llm_stream_text", transform, failure_mode="open")
""",
    )
    _use_manager(monkeypatch, manager)

    assert run_llm_stream_text_middleware("visible", kind="text") == "visible"


def test_sync_stream_none_and_empty_string_controls(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "sync-controls",
        """
def no_change(**kwargs):
    return None

def suppress(**kwargs):
    return {"text": ""}

ctx.register_middleware("llm_stream_text", no_change, failure_mode="closed")
ctx.register_middleware("llm_stream_text", suppress, failure_mode="closed")
""",
    )
    _use_manager(monkeypatch, manager)

    assert run_llm_stream_text_middleware("visible", kind="text") == ""


def test_stream_refusal_classification_is_non_retryable_and_non_fallback():
    refusal = LLMStreamMiddlewareRefusal(ConnectionError("policy refused"))
    classified = classify_api_error(refusal, provider="openrouter", model="test/model")

    assert classified.retryable is False
    assert classified.should_rotate_credential is False
    assert classified.should_fallback is False
    assert classified.error_context["middleware"] == "llm_stream_text"


def test_malformed_sync_result_closed_refuses(tmp_path, monkeypatch):
    """A closed transform returning a malformed result (not dict/None) raises refusal."""
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "malformed-closed",
        """
def bad_transform(**kwargs):
    return {"txt": "SAFE"}

ctx.register_middleware("llm_stream_text", bad_transform, failure_mode="closed")
""",
    )
    _use_manager(monkeypatch, manager)

    with pytest.raises(LLMStreamMiddlewareRefusal):
        run_llm_stream_text_middleware("secret", kind="text")


def test_malformed_sync_result_open_preserves(tmp_path, monkeypatch):
    """An open transform returning a malformed result preserves current text."""
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "malformed-open",
        """
def bad_transform(**kwargs):
    return False

ctx.register_middleware("llm_stream_text", bad_transform, failure_mode="open")
""",
    )
    _use_manager(monkeypatch, manager)

    assert run_llm_stream_text_middleware("visible", kind="text") == "visible"


def test_malformed_sync_result_non_dict_closed_refuses(tmp_path, monkeypatch):
    """A closed transform returning a raw string raises refusal."""
    manager = _load_plugin(
        tmp_path,
        monkeypatch,
        "malformed-str-closed",
        """
def bad_transform(**kwargs):
    return "SAFE"

ctx.register_middleware("llm_stream_text", bad_transform, failure_mode="closed")
""",
    )
    _use_manager(monkeypatch, manager)

    with pytest.raises(LLMStreamMiddlewareRefusal):
        run_llm_stream_text_middleware("secret", kind="text")