"""Tests for the smart recall plugin (Phase D3)."""

import importlib
import os

from tools.memory_tool import MemoryStore

_PLUGIN_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "plugins", "hongxing-enhancements"
)
_spec = importlib.util.spec_from_file_location(
    "smart_recall",
    os.path.join(_PLUGIN_DIR, "smart_recall.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SmartRecall = _mod.SmartRecall


def _make_store(tmpdir, monkeypatch):
    hermes_home = tmpdir / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return MemoryStore()


def test_recall_returns_results(tmpdir, monkeypatch):
    store = _make_store(tmpdir, monkeypatch)
    store.add("memory", "Sprint deadline for auth rollout is Friday", memory_type="project")
    store.add("memory", "Database migration checklist for billing", memory_type="reference")

    recall = SmartRecall(store)
    results = recall.recall("auth rollout deadline", top_k=3)

    assert results
    assert results[0]["type"] == "project"
    assert "auth rollout" in results[0]["content"].lower()


def test_recall_type_filter(tmpdir, monkeypatch):
    store = _make_store(tmpdir, monkeypatch)
    store.add("memory", "Prefer concise writeups for release notes", memory_type="feedback")
    store.add("memory", "Release milestone is May 1", memory_type="project")

    recall = SmartRecall(store)
    results = recall.recall("release notes milestone", top_k=5, types=["feedback"])

    assert results
    assert {item["type"] for item in results} == {"feedback"}


def test_recall_top_k_limit(tmpdir, monkeypatch):
    store = _make_store(tmpdir, monkeypatch)
    for idx in range(5):
        store.add("memory", f"Auth token note {idx}", memory_type="reference")

    recall = SmartRecall(store)
    results = recall.recall("auth token", top_k=2)

    assert len(results) == 2


def test_recall_empty_memory(tmpdir, monkeypatch):
    store = _make_store(tmpdir, monkeypatch)

    recall = SmartRecall(store)
    results = recall.recall("anything", top_k=3)

    assert results == []


def test_recall_keyword_scoring(tmpdir, monkeypatch):
    store = _make_store(tmpdir, monkeypatch)
    store.add("memory", "alpha beta gamma release note")
    store.add("memory", "gamma delta epsilon note")

    recall = SmartRecall(store)
    results = recall.recall("alpha beta gamma", top_k=2)

    assert len(results) == 2
    assert "alpha beta gamma" in results[0]["content"].lower()


def test_recall_type_boost(tmpdir, monkeypatch):
    store = _make_store(tmpdir, monkeypatch)
    store.add("memory", "alpha beta guidance for engineers", memory_type="uncategorized")
    store.add("memory", "alpha beta guidance from feedback", memory_type="feedback")

    recall = SmartRecall(store)
    results = recall.recall("alpha beta guidance", top_k=2)

    assert len(results) == 2
    assert results[0]["type"] == "feedback"


def test_recall_degradation(tmpdir, monkeypatch):
    store = _make_store(tmpdir, monkeypatch)
    store.add("memory", "Apollo launch checklist", memory_type="project")
    store.add("memory", "Apollo retrospective notes", memory_type="reference")

    recall = SmartRecall(store)
    monkeypatch.setattr(recall, "_rank_candidates", lambda query, candidates: (_ for _ in ()).throw(RuntimeError("boom")))
    results = recall.recall("Apollo checklist", top_k=2)

    assert results
    assert any("checklist" in item["content"].lower() for item in results)


def test_different_types_participate(tmpdir, monkeypatch):
    store = _make_store(tmpdir, monkeypatch)
    store.add("user", "I work on Apollo operations", memory_type="user")
    store.add("memory", "Prefer Apollo updates in bullet points", memory_type="feedback")
    store.add("memory", "Apollo sprint milestone is locked", memory_type="project")
    store.add("memory", "Apollo docs live at https://example.com/apollo", memory_type="reference")

    recall = SmartRecall(store)
    results = recall.recall("Apollo", top_k=10)

    assert {"user", "feedback", "project", "reference"}.issubset(
        {item["type"] for item in results}
    )
