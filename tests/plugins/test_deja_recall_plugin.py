from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest


PLUGIN = Path(__file__).parents[2] / "plugins" / "deja-recall"


def load_module(name: str):
    package = "deja_recall"
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package, PLUGIN / "__init__.py", submodule_search_locations=[str(PLUGIN)]
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    if not name:
        return sys.modules[package]
    return __import__(f"{package}.{name}", fromlist=[name])


def source_db(path: Path, rows: list[tuple]) -> None:
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE messages (
          id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
          timestamp REAL, api_content TEXT, active INTEGER DEFAULT 1
        );
    """)
    con.executemany(
        "INSERT INTO messages(id,session_id,role,content,timestamp,api_content,active) VALUES(?,?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()


def test_index_is_incremental_and_uses_clean_content(tmp_path):
    index = load_module("index")
    source = tmp_path / "state.db"
    cache = tmp_path / "recall.db"
    source_db(source, [
        (1, "old", "user", "jwt refresh rotation failed", 100.0, "RECALLED SECRET", 1),
        (2, "old", "assistant", "fixed token cache", 101.0, None, 1),
        (3, "inactive", "user", "jwt hidden", 102.0, None, 0),
        (4, "bad-role", "tool", "jwt tool output", 103.0, None, 1),
    ])
    first = index.refresh_index([source], cache)
    second = index.refresh_index([source], cache)
    assert first.sessions_changed == 1
    assert second.sessions_changed == 0
    hits = index.recall(cache, "jwt refresh rotation", active_session_id="new", limit=3, now=200.0)
    assert [h.session_id for h in hits] == ["old"]
    assert "RECALLED SECRET" not in hits[0].text
    assert "hidden" not in hits[0].text


def test_ranking_is_deterministic_and_excludes_active_and_duplicates(tmp_path):
    index = load_module("index")
    source = tmp_path / "state.db"
    cache = tmp_path / "recall.db"
    source_db(source, [
        (1, "focused", "user", "jwt refresh rotation exact procedure", 190.0, None, 1),
        (2, "active", "user", "jwt refresh rotation exact procedure", 199.0, None, 1),
        (3, "partial", "user", "refresh generic prose", 198.0, None, 1),
        (4, "duplicate", "user", "jwt refresh rotation exact procedure", 180.0, None, 1),
    ])
    index.refresh_index([source], cache)
    one = index.recall(cache, "jwt refresh rotation", active_session_id="active", limit=5, now=200.0)
    two = index.recall(cache, "jwt refresh rotation", active_session_id="active", limit=5, now=200.0)
    assert [h.session_id for h in one] == [h.session_id for h in two]
    assert one[0].session_id == "focused"
    assert "active" not in [h.session_id for h in one]
    assert len([h for h in one if "exact procedure" in h.text]) == 1


def test_render_is_delimited_and_never_exceeds_budget(tmp_path):
    index = load_module("index")
    hits = [index.Hit("s1", "default", 10.0, 0.8, "User: " + "word " * 100, "/tmp/state.db")]
    text = index.render("word", hits, max_chars=240)
    assert text.startswith('<filesystem-recall query="word"')
    assert text.endswith("</filesystem-recall>")
    assert len(text) <= 240
    assert "untrusted historical context" in text


def test_malformed_and_missing_sources_fail_open(tmp_path):
    index = load_module("index")
    malformed = tmp_path / "bad.db"
    malformed.write_text("not sqlite", encoding="utf-8")
    report = index.refresh_index([tmp_path / "missing.db", malformed], tmp_path / "cache.db")
    assert report.sources_failed == 1
    assert index.recall(tmp_path / "missing-cache.db", "anything", active_session_id="x") == []


def test_atomic_update_rolls_back_on_interruption(tmp_path, monkeypatch):
    index = load_module("index")
    source = tmp_path / "state.db"
    cache = tmp_path / "recall.db"
    source_db(source, [(1, "s", "user", "stable jwt content", 1.0, None, 1)])
    index.refresh_index([source], cache)
    before = cache.read_bytes()
    monkeypatch.setattr(index, "_replace_source", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    report = index.refresh_index([source], cache, force=True)
    assert report.sources_failed == 1
    assert cache.read_bytes() == before


def test_hook_only_injects_on_first_turn(monkeypatch, tmp_path):
    plugin = load_module("")
    index = load_module("index")
    cache = tmp_path / "cache.db"
    cache.touch()
    monkeypatch.setattr(plugin, "_settings", plugin.Settings(enabled=True, source_paths=(), index_path=cache))
    monkeypatch.setattr(index, "recall", lambda *a, **k: [index.Hit("old", "default", 1.0, .9, "User: relevant", "/x")])
    assert plugin._pre_llm_call(user_message="relevant", session_id="new", is_first_turn=False) is None
    result = plugin._pre_llm_call(user_message="relevant", session_id="new", is_first_turn=True)
    assert "filesystem-recall" in result["context"]


def test_register_wires_supported_hooks(monkeypatch, tmp_path):
    plugin = load_module("")
    monkeypatch.setattr(plugin, "_load_settings", lambda: plugin.Settings(enabled=True, source_paths=(), index_path=tmp_path / "cache.db"))
    monkeypatch.setattr(plugin, "_enqueue_refresh", lambda **_: None)
    class Ctx:
        def __init__(self): self.hooks = []
        def register_hook(self, name, fn): self.hooks.append(name)
    ctx = Ctx()
    plugin.register(ctx)
    assert ctx.hooks == ["pre_llm_call", "on_session_finalize"]


def test_register_disabled_does_not_wire_hooks(monkeypatch, tmp_path):
    plugin = load_module("")
    monkeypatch.setattr(plugin, "_load_settings", lambda: plugin.Settings(enabled=False, source_paths=(), index_path=tmp_path / "cache.db"))
    monkeypatch.setattr(plugin, "_enqueue_refresh", lambda **_: None)
    class Ctx:
        def __init__(self): self.hooks = []
        def register_hook(self, name, fn): self.hooks.append(name)
    ctx = Ctx()
    plugin.register(ctx)
    assert ctx.hooks == []


def test_enqueue_refresh_skips_when_disabled(monkeypatch, tmp_path):
    plugin = load_module("")
    monkeypatch.setattr(plugin, "_settings", plugin.Settings(enabled=False, source_paths=(), index_path=tmp_path / "cache.db"))
    calls = []
    monkeypatch.setattr(plugin, "_ensure_worker", lambda: calls.append("ensure"))
    monkeypatch.setattr(plugin, "_jobs", plugin._settings.__dataclass_fields__)
    assert plugin._enqueue_refresh(on_session_finalize=lambda: None) is None


def test_missing_source_path_does_not_break_index(tmp_path):
    index = load_module("index")
    cache = tmp_path / "recall.db"
    report = index.refresh_index([tmp_path / "does-not-exist.db"], cache)
    assert report.sources_seen == 0
    assert report.sources_failed == 0
    assert report.sessions_changed == 0


def test_corrupted_source_is_isolated_and_other_sources_still_index(tmp_path):
    index = load_module("index")
    bad = tmp_path / "bad.db"
    bad.write_text("garbage", encoding="utf-8")
    good = tmp_path / "good.db"
    source_db(good, [
        (1, "s1", "user", "recovery rollback test", 1.0, None, 1),
    ])
    cache = tmp_path / "recall.db"
    report = index.refresh_index([bad, good], cache)
    assert report.sources_failed == 1
    assert report.sources_seen == 2
    assert report.sessions_changed == 1
    hits = index.recall(cache, "rollback test", active_session_id="x", now=2.0)
    assert [h.session_id for h in hits] == ["s1"]


def test_removed_index_path_recovers_from_missing_cache(tmp_path):
    index = load_module("index")
    source = tmp_path / "state.db"
    cache = tmp_path / "recall.db"
    source_db(source, [(1, "s1", "user", "cold cache recovery", 1.0, None, 1)])
    index.refresh_index([source], cache)
    cache.unlink()
    report = index.refresh_index([source], cache)
    assert report.sources_seen == 1
    assert report.sources_failed == 0
    assert report.sessions_changed == 1
    hits = index.recall(cache, "cold cache", active_session_id="x", now=2.0)
    assert [h.session_id for h in hits] == ["s1"]


def test_incremental_refresh_skips_unchanged_and_detects_changed(tmp_path):
    index = load_module("index")
    source = tmp_path / "state.db"
    cache = tmp_path / "recall.db"
    source_db(source, [(1, "s1", "user", "first draft", 1.0, None, 1)])
    first = index.refresh_index([source], cache)
    assert first.sessions_changed == 1
    second = index.refresh_index([source], cache)
    assert second.sessions_changed == 0
    con = sqlite3.connect(source)
    con.execute("INSERT INTO messages(id,session_id,role,content,timestamp,api_content,active) VALUES(?,?,?,?,?,?,?)",
                (2, "s1", "assistant", "revised draft", 2.0, None, 1))
    con.commit()
    con.close()
    third = index.refresh_index([source], cache)
    assert third.sessions_changed == 1
