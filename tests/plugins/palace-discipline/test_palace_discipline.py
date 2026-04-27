"""Tests for the palace-discipline plugin (Session 18).

Exercises:

* Session-type inference (keyword routing).
* Hook injection happy path (MCP responses present → rewrite returned).
* Idempotency (second fire on same conversation_id within window → no rewrite).
* Failure tolerance (every MCP call fails → no rewrite, never raises).
* Discord-specific path (additionally loads discord-delivery prompt).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Module loader — the plugin lives at plugins/palace-discipline/__init__.py.
# pytest can't normally import a path with a hyphen as a package, so we load
# it manually and register it under a clean module name.
# ---------------------------------------------------------------------------

_PLUGIN_PATH = (
    Path(__file__).resolve().parents[3]
    / "plugins" / "palace-discipline" / "__init__.py"
)


@pytest.fixture(scope="module")
def plugin():
    """Load the palace-discipline plugin module by file path."""
    spec = importlib.util.spec_from_file_location(
        "palace_discipline_under_test", _PLUGIN_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["palace_discipline_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _reset_state(plugin):
    """Reset per-test state on the loaded plugin."""
    plugin._reset_idempotency_for_tests()
    yield
    plugin._reset_idempotency_for_tests()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakePlatform:
    def __init__(self, value: str) -> None:
        self.value = value


class _FakeSource:
    def __init__(self, platform_value: str = "discord", chat_id: str = "c1") -> None:
        self.platform = _FakePlatform(platform_value)
        self.chat_id = chat_id


class _FakeEvent:
    def __init__(self, text: str, platform: str = "discord", chat_id: str = "c1") -> None:
        self.text = text
        self.source = _FakeSource(platform, chat_id)


def _ok_prompt(t: str) -> dict:
    return {"type": t, "body": f"## Body for {t}\nLine 2."}


def _ok_search(corpus: str, k: int) -> dict:
    return {
        "corpus": corpus,
        "query": "q",
        "k": k,
        "results": [
            {
                "id": f"{corpus}-{i}",
                "path": f"{corpus}/file{i}.md",
                "chunk": 0,
                "distance": 0.1 * i,
                "snippet": f"snippet {corpus} {i}",
            }
            for i in range(min(2, k))
        ],
    }


# ---------------------------------------------------------------------------
# Session-type inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("what's on the docket today?", "triage"),
        ("priorities for today?", "triage"),
        ("research best vector DB options", "research"),
        ("look into nomic embeddings", "research"),
        ("design the new auth flow", "design"),
        ("verify the deploy is live", "verification"),
        ("draft an email to Adam", "communication"),
        ("ingest this transcript", "ingestion"),
        ("build the new API endpoint", "implementation"),
        ("fix the bug in plugin loader", "implementation"),
        ("tighten the session close protocol", "system-correction"),
        ("update the rule about commits", "system-correction"),
        ("hi there", "triage"),  # default fallback
        ("", "triage"),
    ],
)
def test_session_type_inference(plugin, msg, expected):
    assert plugin._infer_session_type(msg) == expected


# ---------------------------------------------------------------------------
# Happy path — Discord conversation
# ---------------------------------------------------------------------------


def test_hook_happy_path_discord(plugin):
    """Every MCP call succeeds; hook returns rewrite with all blocks."""

    def fake_request(method, path, body=None, timeout=None):
        if path == "/sessions/init":
            return {"session": "2026-04-27", "ok": True}
        if path == "/prompts/triage":
            return _ok_prompt("triage")
        if path == "/prompts/discord-delivery":
            return _ok_prompt("discord-delivery")
        if path.startswith("/search/semantic") and "corpus=skills" in path:
            return _ok_search("skills", plugin.SKILLS_K)
        if path.startswith("/search/semantic") and "corpus=palace" in path:
            return _ok_search("palace", plugin.CANON_K)
        return None

    with mock.patch.object(plugin, "_http_request", side_effect=fake_request):
        evt = _FakeEvent("what's on the docket today?", platform="discord")
        result = plugin.on_pre_gateway_dispatch(event=evt)

    assert result is not None
    assert result["action"] == "rewrite"
    text = result["text"]
    # Required blocks
    assert "<palace_context>" in text
    assert "<inferred_session_type>triage</inferred_session_type>" in text
    assert "<memory_init>" in text
    assert "<session_type_prompt type='triage'>" in text
    assert "<discord_delivery_rules>" in text
    assert "<relevant_skills>" in text
    assert "<relevant_canon>" in text
    # Original message preserved at the end
    assert text.rstrip().endswith("what's on the docket today?")


def test_hook_non_discord_skips_discord_prompt(plugin):
    """Telegram conversation → no discord-delivery prompt loaded."""
    discord_prompt_calls = []

    def fake_request(method, path, body=None, timeout=None):
        if path == "/sessions/init":
            return {"ok": True}
        if path == "/prompts/research":
            return _ok_prompt("research")
        if path == "/prompts/discord-delivery":
            discord_prompt_calls.append(path)
            return _ok_prompt("discord-delivery")
        if path.startswith("/search/semantic"):
            return _ok_search(
                "skills" if "skills" in path else "palace", 3
            )
        return None

    with mock.patch.object(plugin, "_http_request", side_effect=fake_request):
        evt = _FakeEvent("research the best embedding model", platform="telegram")
        result = plugin.on_pre_gateway_dispatch(event=evt)

    assert result is not None
    text = result["text"]
    assert "<discord_delivery_rules>" not in text
    assert "<inferred_session_type>research</inferred_session_type>" in text
    assert discord_prompt_calls == []  # never requested


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_hook_is_idempotent_per_conversation(plugin):
    """Second fire for same (platform, chat_id) within window → no rewrite."""

    def fake_request(method, path, body=None, timeout=None):
        if path == "/sessions/init":
            return {"ok": True}
        if path.startswith("/prompts/"):
            return _ok_prompt(path.rsplit("/", 1)[-1])
        if path.startswith("/search/semantic"):
            return _ok_search(
                "skills" if "skills" in path else "palace", 3
            )
        return None

    with mock.patch.object(plugin, "_http_request", side_effect=fake_request):
        evt = _FakeEvent("priorities for today", platform="discord", chat_id="c-same")
        first = plugin.on_pre_gateway_dispatch(event=evt)
        second = plugin.on_pre_gateway_dispatch(event=evt)

    assert first is not None and first["action"] == "rewrite"
    assert second is None  # idempotent skip


def test_hook_different_chats_inject_independently(plugin):
    """Distinct chat_ids → both inject."""

    def fake_request(method, path, body=None, timeout=None):
        if path == "/sessions/init":
            return {"ok": True}
        if path.startswith("/prompts/"):
            return _ok_prompt(path.rsplit("/", 1)[-1])
        return _ok_search("skills" if "skills" in path else "palace", 3)

    with mock.patch.object(plugin, "_http_request", side_effect=fake_request):
        a = plugin.on_pre_gateway_dispatch(
            event=_FakeEvent("priorities", platform="discord", chat_id="A")
        )
        b = plugin.on_pre_gateway_dispatch(
            event=_FakeEvent("priorities", platform="discord", chat_id="B")
        )
    assert a is not None and b is not None


# ---------------------------------------------------------------------------
# Failure tolerance
# ---------------------------------------------------------------------------


def test_hook_returns_none_when_all_mcp_fails(plugin):
    """Every MCP call returns None → hook returns None (degraded silently)."""
    with mock.patch.object(plugin, "_http_request", return_value=None):
        evt = _FakeEvent("what's next", platform="discord")
        result = plugin.on_pre_gateway_dispatch(event=evt)
    assert result is None


def test_hook_does_not_raise_when_mcp_call_throws(plugin):
    """If the underlying request raises, the hook still returns gracefully."""
    def boom(*args, **kwargs):
        raise RuntimeError("simulated network blow-up")

    with mock.patch.object(plugin, "_http_request", side_effect=boom):
        evt = _FakeEvent("what's next", platform="discord")
        # Must not raise.
        result = plugin.on_pre_gateway_dispatch(event=evt)
    # Either degraded (None) or — if exception is wrapped per-future — still
    # a None outcome. We don't care which path; we care that it doesn't raise.
    assert result is None


def test_hook_partial_success_still_injects(plugin):
    """If only some MCP calls succeed, build the block from what we got."""

    def fake_request(method, path, body=None, timeout=None):
        if path == "/prompts/triage":
            return _ok_prompt("triage")
        # Everything else fails.
        return None

    with mock.patch.object(plugin, "_http_request", side_effect=fake_request):
        evt = _FakeEvent("priorities for today", platform="discord")
        result = plugin.on_pre_gateway_dispatch(event=evt)

    assert result is not None
    text = result["text"]
    assert "<session_type_prompt type='triage'>" in text
    assert "<memory_init>" not in text  # init failed
    assert "<relevant_skills>" not in text  # search failed
    assert "<relevant_canon>" not in text  # search failed


# ---------------------------------------------------------------------------
# Empty-message guard
# ---------------------------------------------------------------------------


def test_hook_skips_empty_message(plugin):
    evt = _FakeEvent("   ", platform="discord")
    assert plugin.on_pre_gateway_dispatch(event=evt) is None


# ---------------------------------------------------------------------------
# register() wiring
# ---------------------------------------------------------------------------


def test_register_wires_hook(plugin):
    """register(ctx) calls ctx.register_hook with the correct name."""
    calls = []

    class _Ctx:
        def register_hook(self, name, callback):
            calls.append((name, callback))

    plugin.register(_Ctx())
    assert len(calls) == 1
    assert calls[0][0] == "pre_gateway_dispatch"
    assert calls[0][1] is plugin.on_pre_gateway_dispatch
