"""Tests for LFDM wiki password gateway shortcut plugin."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_plugin():
    plugin_dir = _repo_root() / "plugins" / "lfdm_wiki_password"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.lfdm_wiki_password",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.lfdm_wiki_password"
    mod.__path__ = [str(plugin_dir)]
    sys.modules["hermes_plugins.lfdm_wiki_password"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SimpleNamespace(success=True)


class FakeGateway:
    def __init__(self, *, authorized=True):
        self.authorized = authorized
        self.adapter = FakeAdapter()
        self.adapters = {"discord": self.adapter}
        self._background_tasks = set()

    def _is_user_authorized(self, source):
        return self.authorized

    def _reply_anchor_for_event(self, event):
        return "reply-123"

    def _thread_metadata_for_source(self, source, reply_to_message_id=None):
        return {"thread_id": source.thread_id, "reply_anchor": reply_to_message_id}


def _event(text: str):
    source = SimpleNamespace(
        platform="discord",
        chat_id="channel-1",
        thread_id="thread-1",
        user_id="user-1",
    )
    return SimpleNamespace(text=text, source=source)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SASSY_WIKI_PASSWORD_FILE", raising=False)
    monkeypatch.delenv("LFDM_WIKI_PASSWORD_ALLOWED_CHATS", raising=False)


def test_request_matcher_accepts_direct_password_requests():
    mod = _load_plugin()

    assert mod._is_wiki_password_request("give me the password for wiki, no other output needed")
    assert mod._is_wiki_password_request("wiki password")
    assert mod._is_wiki_password_request("what is the current wiki pw?")


def test_request_matcher_rejects_debugging_discussion():
    mod = _load_plugin()

    assert not mod._is_wiki_password_request("let's figure out a workaround to fix this")
    assert not mod._is_wiki_password_request("fix the wiki password route")
    assert not mod._is_wiki_password_request("why did the wiki password request fail?")


@pytest.mark.asyncio
async def test_authorized_request_sends_password_and_skips_llm(tmp_path, monkeypatch):
    mod = _load_plugin()
    password_file = tmp_path / "password"
    password_file.write_text("test-wiki-password\n", encoding="utf-8")
    monkeypatch.setenv("SASSY_WIKI_PASSWORD_FILE", str(password_file))

    gateway = FakeGateway(authorized=True)
    result = mod._on_pre_gateway_dispatch(
        event=_event("give me the password for wiki, no other output needed"),
        gateway=gateway,
    )

    assert result == {"action": "skip", "reason": "lfdm_wiki_password_direct_response"}
    for _ in range(3):
        if gateway.adapter.sent:
            break
        await asyncio.sleep(0)

    assert gateway.adapter.sent == [
        {
            "chat_id": "channel-1",
            "content": "test-wiki-password",
            "reply_to": "reply-123",
            "metadata": {"thread_id": "thread-1", "reply_anchor": "reply-123"},
        }
    ]
    for _ in range(3):
        if not gateway._background_tasks:
            break
        await asyncio.sleep(0)
    assert not gateway._background_tasks


@pytest.mark.asyncio
async def test_unauthorized_request_falls_through_without_reading_or_sending(tmp_path, monkeypatch):
    mod = _load_plugin()
    missing = tmp_path / "missing-password"
    monkeypatch.setenv("SASSY_WIKI_PASSWORD_FILE", str(missing))

    gateway = FakeGateway(authorized=False)
    result = mod._on_pre_gateway_dispatch(
        event=_event("wiki password"),
        gateway=gateway,
    )

    assert result == {"action": "allow"}
    await asyncio.sleep(0)
    assert gateway.adapter.sent == []


@pytest.mark.asyncio
async def test_allowed_chat_env_can_narrow_exposure(tmp_path, monkeypatch):
    mod = _load_plugin()
    password_file = tmp_path / "password"
    password_file.write_text("test-wiki-password\n", encoding="utf-8")
    monkeypatch.setenv("SASSY_WIKI_PASSWORD_FILE", str(password_file))
    monkeypatch.setenv("LFDM_WIKI_PASSWORD_ALLOWED_CHATS", "other-channel")

    gateway = FakeGateway(authorized=True)
    result = mod._on_pre_gateway_dispatch(
        event=_event("wiki password"),
        gateway=gateway,
    )

    assert result == {"action": "allow"}
    await asyncio.sleep(0)
    assert gateway.adapter.sent == []
