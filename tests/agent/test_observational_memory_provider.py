"""Tests for the Observational Memory memory provider."""

from __future__ import annotations

import importlib
import json
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path

from plugins.memory.observational_memory import ObservationalMemoryProvider


def _install_fake_om(monkeypatch, tmp_path):
    """Install lightweight fake observational_memory modules into sys.modules."""

    fake_pkg = types.ModuleType("observational_memory")
    fake_pkg.__path__ = []  # mark as package
    original_find_spec = importlib.util.find_spec

    reindex_calls = []
    observer_calls = []

    @dataclass
    class FakeConfig:
        memory_dir: Path | None = None
        env_file: Path | None = None
        llm_provider: str = "auto"
        llm_model: str | None = None
        search_backend: str = "bm25"
        min_messages: int = 5

        def __post_init__(self):
            self.memory_dir = (self.memory_dir or (tmp_path / "memory")).expanduser()
            self.env_file = (self.env_file or (tmp_path / "env")).expanduser()

        @property
        def observations_path(self) -> Path:
            return self.memory_dir / "observations.md"

        @property
        def reflections_path(self) -> Path:
            return self.memory_dir / "reflections.md"

        @property
        def profile_path(self) -> Path:
            return self.memory_dir / "profile.md"

        @property
        def active_path(self) -> Path:
            return self.memory_dir / "active.md"

        def ensure_memory_dir(self) -> None:
            self.memory_dir.mkdir(parents=True, exist_ok=True)

        def load_env_file(self) -> None:
            return None

        def validate_provider_config(self, provider=None):
            return "anthropic"

    class FakeResult:
        def __init__(self, rank, score, source, heading, content):
            self.rank = rank
            self.score = score
            self.document = types.SimpleNamespace(
                source=types.SimpleNamespace(value=source),
                heading=heading,
                content=content,
            )

    class FakeBackend:
        def is_ready(self):
            return True

        def search(self, query, limit=10):
            return [
                FakeResult(
                    1,
                    9.5,
                    "reflections",
                    "## Active Projects",
                    "## Active Projects\nHermes and Codex both use the same memory store.",
                )
            ][:limit]

    def get_backend(name, config):
        return FakeBackend()

    def reindex(config):
        reindex_calls.append(config.memory_dir)
        return 1

    def ensure_startup_memory(config):
        config.ensure_memory_dir()
        config.profile_path.write_text("# Startup Profile\n\n- prefers concise output\n")
        config.active_path.write_text("# Active Context\n\n- shipping a Hermes memory provider\n")

    def refresh_startup_memory(config):
        ensure_startup_memory(config)

    @dataclass
    class Message:
        role: str
        content: str
        timestamp: str
        source: str

    def run_observer(messages, config, dry_run=False):
        observer_calls.append(list(messages))
        config.ensure_memory_dir()
        config.observations_path.write_text("# Observations\n\n## 2026-04-03\n\n### Observations\n- 🔴 12:00 Observed Hermes sync\n")
        return "ok"

    config_mod = types.ModuleType("observational_memory.config")
    config_mod.Config = FakeConfig

    search_mod = types.ModuleType("observational_memory.search")
    search_mod.get_backend = get_backend
    search_mod.reindex = reindex

    startup_mod = types.ModuleType("observational_memory.startup_memory")
    startup_mod.ensure_startup_memory = ensure_startup_memory
    startup_mod.refresh_startup_memory = refresh_startup_memory

    transcripts_mod = types.ModuleType("observational_memory.transcripts")
    transcripts_mod.Message = Message

    observe_mod = types.ModuleType("observational_memory.observe")
    observe_mod.run_observer = run_observer

    monkeypatch.setitem(sys.modules, "observational_memory", fake_pkg)
    monkeypatch.setitem(sys.modules, "observational_memory.config", config_mod)
    monkeypatch.setitem(sys.modules, "observational_memory.search", search_mod)
    monkeypatch.setitem(sys.modules, "observational_memory.startup_memory", startup_mod)
    monkeypatch.setitem(sys.modules, "observational_memory.transcripts", transcripts_mod)
    monkeypatch.setitem(sys.modules, "observational_memory.observe", observe_mod)
    monkeypatch.setattr(
        "plugins.memory.observational_memory.importlib.util.find_spec",
        lambda name: object() if name == "observational_memory" else original_find_spec(name),
    )

    return reindex_calls, observer_calls


def test_save_config_writes_profile_scoped_json(tmp_path):
    provider = ObservationalMemoryProvider()
    provider.save_config({"writeback_mode": "session_end"}, str(tmp_path))

    path = tmp_path / "observational_memory.json"
    data = json.loads(path.read_text())
    assert data["writeback_mode"] == "session_end"
    assert data["memory_dir"].endswith("observational-memory")


def test_system_prompt_includes_startup_memory(monkeypatch, tmp_path):
    _install_fake_om(monkeypatch, tmp_path)
    provider = ObservationalMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path))

    prompt = provider.system_prompt_block()

    assert "Observational Memory" in prompt
    assert "Startup Profile" in prompt
    assert "Active Context" in prompt


def test_system_prompt_truncates_large_startup_sections(monkeypatch, tmp_path):
    _install_fake_om(monkeypatch, tmp_path)
    provider = ObservationalMemoryProvider()
    provider.initialize("session-1b", hermes_home=str(tmp_path))

    provider._config.profile_path.write_text("# Startup Profile\n\n" + ("p" * 5000))
    provider._config.active_path.write_text("# Active Context\n\n" + ("a" * 5000))

    prompt = provider.system_prompt_block()

    assert "Use om_context for the full text." in prompt
    assert prompt.count("truncated to 4000 chars for prompt safety") == 2
    assert len(prompt) < 9000


def test_om_remember_appends_local_observation(monkeypatch, tmp_path):
    reindex_calls, _ = _install_fake_om(monkeypatch, tmp_path)
    provider = ObservationalMemoryProvider()
    provider.initialize("session-2", hermes_home=str(tmp_path))

    result = json.loads(
        provider.handle_tool_call(
            "om_remember",
            {"content": "Bryan wants fixes grounded in artifacts", "importance": "high"},
        )
    )

    obs = provider._config.observations_path.read_text()
    assert result["stored"] is True
    assert "Bryan wants fixes grounded in artifacts" in obs
    assert "🔴" in obs
    assert reindex_calls


def test_incremental_sync_flushes_to_observer(monkeypatch, tmp_path):
    _, observer_calls = _install_fake_om(monkeypatch, tmp_path)
    provider = ObservationalMemoryProvider()
    provider.initialize("session-3", hermes_home=str(tmp_path))
    provider._config.min_messages = 5

    provider.sync_turn("first user", "first assistant")
    provider.sync_turn("second user", "second assistant")
    provider.sync_turn("third user", "third assistant")
    provider.shutdown()

    assert observer_calls
    assert len(observer_calls[0]) == 6
    assert {msg.source for msg in observer_calls[0]} == {"hermes"}


def test_session_end_defers_final_flush_until_active_sync_finishes(monkeypatch, tmp_path, caplog):
    _install_fake_om(monkeypatch, tmp_path)
    provider = ObservationalMemoryProvider()
    provider.initialize("session-4", hermes_home=str(tmp_path))

    flush_calls = []
    released = threading.Event()
    flushed = threading.Event()

    class FakeThread:
        def __init__(self):
            self._alive = True
            self.join_calls = []

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            self.join_calls.append(timeout)
            if timeout is None:
                released.wait(timeout=1.0)
                self._alive = False

    def _flush_pending(*, force: bool):
        flush_calls.append(force)
        flushed.set()

    active_thread = FakeThread()
    provider._sync_thread = active_thread
    monkeypatch.setattr(provider, "_flush_pending", _flush_pending)

    with caplog.at_level("WARNING"):
        provider.on_session_end([])

    followup = provider._sync_thread
    assert followup is not active_thread
    assert followup is not None
    assert "deferring final session flush" in caplog.text

    released.set()
    followup.join(timeout=1.0)

    assert flushed.wait(timeout=1.0)
    assert flush_calls == [True]
    assert active_thread.join_calls == [10.0, None]
