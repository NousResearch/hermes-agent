"""Tests for the ``message:received`` decision hook.

The hook fires in the gateway just before ``agent:start``. Driving the full
gateway message pipeline from a unit test would be prohibitively heavy, so
these tests exercise the discovery and emit_collect dispatch semantics that
the wiring in ``gateway/run.py`` depends on:

    for _decision_result in _message_decisions:
        if not isinstance(_decision_result, dict):
            continue
        _decision = str(_decision_result.get("decision", "")).strip().lower()
        if _decision == "deny":
            ...  # swallow, optional platform-notice reply
        if _decision == "rewrite":
            ...  # mutate message_text / append context_prompt

Mirrors ``test_transform_llm_output_hook.py`` which tests the equivalent
contract for the LLM-output seam.
"""

import asyncio
from pathlib import Path

import yaml

import gateway.hooks as hooks_mod
from gateway.hooks import HookRegistry


def _make_hook(hooks_dir: Path, name: str, handler_body: str) -> None:
    hook_dir = hooks_dir / name
    hook_dir.mkdir(parents=True)
    (hook_dir / "HOOK.yaml").write_text(
        yaml.safe_dump({"name": name, "description": "t", "events": ["message:received"]}),
        encoding="utf-8",
    )
    (hook_dir / "handler.py").write_text(
        "async def handle(event_type, context):\n" + handler_body,
        encoding="utf-8",
    )


def _registry_for(tmp_path: Path, monkeypatch) -> HookRegistry:
    monkeypatch.setattr(hooks_mod, "HOOKS_DIR", tmp_path)
    registry = HookRegistry()
    registry.discover_and_load()
    return registry


def test_deny_decision_is_collected(tmp_path, monkeypatch):
    _make_hook(
        tmp_path,
        "denier",
        "    if context.get('message') == 'block me':\n"
        "        return {'decision': 'deny', 'reply': 'not allowed'}\n"
        "    return None\n",
    )
    registry = _registry_for(tmp_path, monkeypatch)

    results = asyncio.run(registry.emit_collect("message:received", {"message": "block me"}))
    assert results == [{"decision": "deny", "reply": "not allowed"}]

    passthrough = asyncio.run(registry.emit_collect("message:received", {"message": "hello"}))
    assert passthrough == []  # None-returning handlers keep telemetry semantics


def test_rewrite_decision_carries_message_and_context(tmp_path, monkeypatch):
    _make_hook(
        tmp_path,
        "rewriter",
        "    return {'decision': 'rewrite', 'message': 'rewritten',\n"
        "            'context_prompt_append': 'extra context'}\n",
    )
    registry = _registry_for(tmp_path, monkeypatch)
    results = asyncio.run(registry.emit_collect("message:received", {"message": "original"}))
    assert results == [
        {"decision": "rewrite", "message": "rewritten", "context_prompt_append": "extra context"}
    ]


def test_handler_errors_do_not_abort_other_handlers(tmp_path, monkeypatch):
    _make_hook(tmp_path, "a-broken", "    raise RuntimeError('boom')\n")
    _make_hook(tmp_path, "b-decider", "    return {'decision': 'deny'}\n")
    registry = _registry_for(tmp_path, monkeypatch)
    results = asyncio.run(registry.emit_collect("message:received", {"message": "x"}))
    assert results == [{"decision": "deny"}]


def test_full_message_text_documented_in_context(tmp_path, monkeypatch):
    # the wiring passes the full message text in the decision context while
    # agent:start keeps its 500-char preview; handlers can rely on it.
    captured = {}
    _make_hook(
        tmp_path,
        "capturer",
        "    return {'decision': '', 'seen': context.get('message')}\n",
    )
    registry = _registry_for(tmp_path, monkeypatch)
    long_text = "long " * 200
    results = asyncio.run(registry.emit_collect("message:received", {"message": long_text}))
    assert results[0]["seen"] == long_text
