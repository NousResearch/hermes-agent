"""Regression tests: recent-model history for the Telegram /model picker."""
from __future__ import annotations

import json

from hermes_cli.telegram_recent_models import recent_models, record_recent


def test_record_and_read_most_recent_first(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record_recent("openrouter", "a/one", path=tmp_path / "r.json")
    record_recent("opencode-zen", "two", path=tmp_path / "r.json")
    got = recent_models(path=tmp_path / "r.json")
    assert [(r["provider"], r["model"]) for r in got] == [
        ("opencode-zen", "two"),
        ("openrouter", "a/one"),
    ]


def test_same_model_is_deduped_and_moves_to_front(tmp_path):
    p = tmp_path / "r.json"
    record_recent("openrouter", "a/one", path=p)
    record_recent("opencode-go", "b/two", path=p)
    record_recent("openrouter", "a/one", path=p)
    got = recent_models(path=p)
    assert [(r["provider"], r["model"]) for r in got] == [
        ("openrouter", "a/one"),
        ("opencode-go", "b/two"),
    ]


def test_keep_cap_is_enforced(tmp_path):
    p = tmp_path / "r.json"
    for i in range(25):
        record_recent("openrouter", f"m/{i}", path=p)
    got = recent_models(path=p, limit=100)
    assert len(got) == 20
    assert got[0]["model"] == "m/24"
    assert got[-1]["model"] == "m/5"


def test_limit_slices_result(tmp_path):
    p = tmp_path / "r.json"
    for i in range(5):
        record_recent("openrouter", f"m/{i}", path=p)
    assert [r["model"] for r in recent_models(path=p, limit=2)] == ["m/4", "m/3"]


def test_corrupt_file_is_ignored_and_healed(tmp_path):
    p = tmp_path / "r.json"
    p.write_text("{not json at all")
    assert recent_models(path=p) == []
    record_recent("openrouter", "x/one", path=p)
    assert [r["model"] for r in recent_models(path=p)] == ["x/one"]
    json.loads(p.read_text())


def test_blank_inputs_are_ignored(tmp_path):
    p = tmp_path / "r.json"
    record_recent("", "model", path=p)
    record_recent("openrouter", "", path=p)
    assert recent_models(path=p) == []