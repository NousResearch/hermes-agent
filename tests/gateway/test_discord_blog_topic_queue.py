from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from plugins.platforms.discord.adapter import DiscordAdapter


def _fake_interaction(user_id: int = 123, name: str = "Tester"):
    return SimpleNamespace(user=SimpleNamespace(id=user_id, display_name=name))


def test_write_blog_topic_rejects_placeholder(monkeypatch, tmp_path):
    adapter = object.__new__(DiscordAdapter)
    fake_adapter_path = tmp_path / "repo" / "plugins" / "platforms" / "discord" / "adapter.py"
    fake_adapter_path.parent.mkdir(parents=True, exist_ok=True)
    fake_adapter_path.touch()

    monkeypatch.setattr(Path, "resolve", lambda self: fake_adapter_path)

    ok = adapter._write_blog_topic("ai", "New Concept", _fake_interaction())
    assert ok is False

    queue_path = tmp_path / "repo" / "content_engine" / "blog_topics" / "ai.jsonl"
    assert not queue_path.exists(), "placeholder input should not create queue entries"


def test_write_blog_topic_accepts_real_topic(monkeypatch, tmp_path):
    adapter = object.__new__(DiscordAdapter)
    fake_adapter_path = tmp_path / "repo" / "plugins" / "platforms" / "discord" / "adapter.py"
    fake_adapter_path.parent.mkdir(parents=True, exist_ok=True)
    fake_adapter_path.touch()

    monkeypatch.setattr(Path, "resolve", lambda self: fake_adapter_path)

    ok = adapter._write_blog_topic("ai", "A real topic worth generating", _fake_interaction())
    assert ok is True

    queue_path = tmp_path / "repo" / "content_engine" / "blog_topics" / "ai.jsonl"
    assert queue_path.exists()
    lines = queue_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["title_hint"] == "A real topic worth generating"
    assert entry["priority"] == 8
    assert entry["topic_id"].startswith("discord-123-")
