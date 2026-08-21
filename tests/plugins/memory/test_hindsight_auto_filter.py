"""Focused regressions for automatic Hindsight noise filtering."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from plugins.memory.hindsight import HindsightMemoryProvider


def _provider():
    p = HindsightMemoryProvider()
    p._config = {}
    p._load_auto_retain_filter_settings()
    p._memory_mode = "hybrid"
    p._auto_recall = True
    p._shutting_down.clear()
    p._prefetch_waits_for_retain = False
    p._recall_max_input_chars = 800
    p._prefetch_method = "recall"
    p._bank_id = "bank"
    p._budget = "mid"
    p._recall_max_tokens = 4096
    p._recall_tags = []
    p._recall_types = ["observation"]
    return p


def test_recall_filters_each_result_without_dropping_useful_siblings():
    p = _provider()
    p._recall_skip_patterns = [r"(?i)temporary debug"]
    results = [
        SimpleNamespace(text="Useful fact before."),
        SimpleNamespace(text="temporary debug lifecycle artifact"),
        SimpleNamespace(text="Useful preference after."),
    ]

    assert p._filtered_recall_result_texts(results) == [
        "Useful fact before.",
        "Useful preference after.",
    ]


def test_standalone_artifact_line_is_removed_but_neighbors_survive():
    p = _provider()
    text = (
        "Useful fact before.\n"
        "[Note: model was just switched from old to new]\n"
        "Useful fact after."
    )

    assert p._filtered_recall_result_texts([SimpleNamespace(text=text)]) == [
        "Useful fact before.\nUseful fact after."
    ]


def test_queue_prefetch_filters_noisy_result_without_dropping_useful_sibling():
    p = _provider()
    p._recall_skip_patterns = [r"(?i)temporary debug"]
    response = SimpleNamespace(
        results=[
            SimpleNamespace(text="temporary debug lifecycle artifact"),
            SimpleNamespace(text="Useful preference after."),
        ]
    )
    p._run_hindsight_operation = MagicMock(return_value=response)

    p.queue_prefetch("normal query")
    p._prefetch_thread.join(timeout=2)

    assert p._prefetch_result == "- Useful preference after."
    assert p._prefetch_count == 1


def test_explicit_disable_does_not_read_yaml_or_rewrite(tmp_path, monkeypatch):
    p = HindsightMemoryProvider()
    p._config = {
        "auto_retain_filter_enabled": False,
        "auto_retain_filter_path": str(tmp_path / "must-not-be-read.yaml"),
    }
    monkeypatch.setattr(
        "plugins.memory.hindsight._load_mapping_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("YAML was read")),
    )

    p._load_auto_retain_filter_settings()
    original_user = "[Note: model was just switched from old to new]"
    original_assistant = "Useful answer"

    assert p._sanitize_auto_retain_turn(original_user, original_assistant) == (
        original_user,
        original_assistant,
        "",
        "",
    )
    assert p._filtered_recall_result_texts(
        [SimpleNamespace(text=original_user)]
    ) == [original_user]
    assert p._recall_text_is_noise("anything") is False
    assert p._auto_retain_audit_log_path == ""


def test_audit_log_contains_counts_but_not_transcript_text(tmp_path):
    p = _provider()
    p._session_id = "session"
    p._thread_id = "thread"
    path = tmp_path / "audit.jsonl"
    p._auto_retain_audit_log_path = str(path)

    p._audit_auto_retain_filter(
        "sanitize_turn",
        "strip_patterns",
        "private user text",
        "private assistant text",
        "user",
        "assistant",
    )

    entry = json.loads(path.read_text(encoding="utf-8"))
    assert entry["action"] == "sanitize_turn"
    assert entry["raw_total_chars"] > entry["sanitized_total_chars"]
    assert "private user text" not in path.read_text(encoding="utf-8")
    assert "private assistant text" not in path.read_text(encoding="utf-8")
