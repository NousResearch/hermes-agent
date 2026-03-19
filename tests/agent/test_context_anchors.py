"""Tests for agent.context_anchors module."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from agent.context_anchors import (
    parse_anchor_config,
    get_max_total_chars,
    is_auto_save_enabled,
    load_anchor_file,
    load_all_anchors,
    get_anchor_paths_for_summary,
    detect_active_anchors,
    build_anchor_save_prompt,
    _truncate_content,
    ANCHOR_INJECTION_PREFIX,
    DEFAULT_MAX_CHARS_PER_ANCHOR,
    DEFAULT_MAX_TOTAL_CHARS,
)


# ---------------------------------------------------------------------------
# parse_anchor_config
# ---------------------------------------------------------------------------

class TestParseAnchorConfig:
    def test_empty_config(self):
        assert parse_anchor_config({}) == []

    def test_no_anchors_key(self):
        assert parse_anchor_config({"model": "gpt-4"}) == []

    def test_anchors_not_list(self):
        assert parse_anchor_config({"context_anchors": "not a list"}) == []

    def test_simple_path_string(self):
        """Simple form: just a path string in the list."""
        anchors = parse_anchor_config({
            "context_anchors": ["/tmp/test.md"]
        })
        assert len(anchors) == 1
        assert anchors[0]["path"] == "/tmp/test.md"
        assert "test" in anchors[0]["keywords"]  # auto-derived from stem
        assert anchors[0]["max_chars"] == DEFAULT_MAX_CHARS_PER_ANCHOR

    def test_full_config(self):
        anchors = parse_anchor_config({
            "context_anchors": [
                {
                    "path": "/home/user/.hermes/context/myproject.md",
                    "keywords": ["myproject", "/var/www/myproject"],
                    "max_chars": 3000,
                }
            ]
        })
        assert len(anchors) == 1
        a = anchors[0]
        assert "myproject" in a["keywords"]
        assert "/var/www/myproject" in a["keywords"]
        assert a["max_chars"] == 3000

    def test_tilde_expansion(self):
        anchors = parse_anchor_config({
            "context_anchors": [{"path": "~/test.md"}]
        })
        assert anchors[0]["path"].startswith("/")
        assert "~" not in anchors[0]["path"]

    def test_keywords_lowercased(self):
        anchors = parse_anchor_config({
            "context_anchors": [{"path": "/tmp/x.md", "keywords": ["MyProject", "FOO"]}]
        })
        assert anchors[0]["keywords"] == ["myproject", "foo"]

    def test_keywords_string_coerced_to_list(self):
        anchors = parse_anchor_config({
            "context_anchors": [{"path": "/tmp/x.md", "keywords": "single_keyword"}]
        })
        assert anchors[0]["keywords"] == ["single_keyword"]

    def test_skip_entry_without_path(self):
        anchors = parse_anchor_config({
            "context_anchors": [{"keywords": ["orphan"]}]
        })
        assert len(anchors) == 0

    def test_multiple_anchors(self):
        anchors = parse_anchor_config({
            "context_anchors": [
                {"path": "/tmp/a.md", "keywords": ["alpha"]},
                {"path": "/tmp/b.md", "keywords": ["beta"]},
            ]
        })
        assert len(anchors) == 2


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

class TestConfigHelpers:
    def test_get_max_total_chars_default(self):
        assert get_max_total_chars({}) == DEFAULT_MAX_TOTAL_CHARS

    def test_get_max_total_chars_custom(self):
        assert get_max_total_chars({"context_anchors_max_total_chars": 10000}) == 10000

    def test_auto_save_default_true(self):
        assert is_auto_save_enabled({}) is True

    def test_auto_save_disabled(self):
        assert is_auto_save_enabled({"context_anchors_auto_save": False}) is False


# ---------------------------------------------------------------------------
# _truncate_content
# ---------------------------------------------------------------------------

class TestTruncateContent:
    def test_short_content_unchanged(self):
        assert _truncate_content("hello", 1000) == "hello"

    def test_long_content_truncated(self):
        content = "x" * 10000
        result = _truncate_content(content, 1000)
        assert len(result) < 10000
        assert "truncated" in result
        assert "10000" in result  # original size mentioned

    def test_head_tail_preserved(self):
        content = "HEAD" + "x" * 10000 + "TAIL"
        result = _truncate_content(content, 1000)
        assert result.startswith("HEAD")
        assert result.endswith("TAIL")


# ---------------------------------------------------------------------------
# load_anchor_file
# ---------------------------------------------------------------------------

class TestLoadAnchorFile:
    def test_load_existing_file(self, tmp_path):
        f = tmp_path / "project.md"
        f.write_text("# My Project\nStatus: running")
        anchor = {"path": str(f), "keywords": ["project"], "max_chars": 5000}
        result = load_anchor_file(anchor)
        assert result is not None
        assert "My Project" in result

    def test_load_missing_file(self):
        anchor = {"path": "/nonexistent/file.md", "keywords": [], "max_chars": 5000}
        assert load_anchor_file(anchor) is None

    def test_load_empty_file(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("")
        anchor = {"path": str(f), "keywords": [], "max_chars": 5000}
        assert load_anchor_file(anchor) is None

    def test_truncation_on_large_file(self, tmp_path):
        f = tmp_path / "large.md"
        f.write_text("x" * 10000)
        anchor = {"path": str(f), "keywords": [], "max_chars": 500}
        result = load_anchor_file(anchor)
        assert result is not None
        assert len(result) < 10000


# ---------------------------------------------------------------------------
# load_all_anchors
# ---------------------------------------------------------------------------

class TestLoadAllAnchors:
    def test_empty_anchors(self):
        assert load_all_anchors([], 20000) is None

    def test_loads_and_formats(self, tmp_path):
        f1 = tmp_path / "a.md"
        f1.write_text("Project A state")
        f2 = tmp_path / "b.md"
        f2.write_text("Project B state")

        anchors = [
            {"path": str(f1), "keywords": ["a"], "max_chars": 5000},
            {"path": str(f2), "keywords": ["b"], "max_chars": 5000},
        ]
        result = load_all_anchors(anchors, 20000)
        assert result is not None
        assert ANCHOR_INJECTION_PREFIX in result
        assert "Project A state" in result
        assert "Project B state" in result

    def test_respects_total_limit(self, tmp_path):
        f1 = tmp_path / "big.md"
        f1.write_text("x" * 5000)
        f2 = tmp_path / "small.md"
        f2.write_text("y" * 100)

        anchors = [
            {"path": str(f1), "keywords": ["big"], "max_chars": 5000},
            {"path": str(f2), "keywords": ["small"], "max_chars": 5000},
        ]
        # Total limit smaller than both files combined
        result = load_all_anchors(anchors, 3000)
        assert result is not None
        # Should have truncated or skipped the second file

    def test_skips_missing_files(self, tmp_path):
        f1 = tmp_path / "exists.md"
        f1.write_text("I exist")

        anchors = [
            {"path": "/nonexistent.md", "keywords": ["nope"], "max_chars": 5000},
            {"path": str(f1), "keywords": ["exists"], "max_chars": 5000},
        ]
        result = load_all_anchors(anchors, 20000)
        assert result is not None
        assert "I exist" in result


# ---------------------------------------------------------------------------
# get_anchor_paths_for_summary
# ---------------------------------------------------------------------------

class TestGetAnchorPathsForSummary:
    def test_returns_existing_paths(self, tmp_path):
        f = tmp_path / "exists.md"
        f.write_text("content")
        anchors = [
            {"path": str(f), "keywords": [], "max_chars": 5000},
            {"path": "/nonexistent.md", "keywords": [], "max_chars": 5000},
        ]
        paths = get_anchor_paths_for_summary(anchors)
        assert str(f) in paths
        assert "/nonexistent.md" not in paths


# ---------------------------------------------------------------------------
# detect_active_anchors
# ---------------------------------------------------------------------------

class TestDetectActiveAnchors:
    def test_no_anchors(self):
        assert detect_active_anchors([], [{"role": "user", "content": "hello"}]) == []

    def test_keyword_in_user_message(self):
        anchors = [{"path": "/tmp/x.md", "keywords": ["eclatauto"], "max_chars": 5000}]
        messages = [{"role": "user", "content": "fix the eclatauto reservation bug"}]
        result = detect_active_anchors(anchors, messages)
        assert len(result) == 1

    def test_keyword_in_tool_call_args(self):
        anchors = [{"path": "/tmp/x.md", "keywords": ["/var/www/eclatauto"], "max_chars": 5000}]
        messages = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "/var/www/eclatauto/api.py"}'
                }
            }],
        }]
        result = detect_active_anchors(anchors, messages)
        assert len(result) == 1

    def test_no_match(self):
        anchors = [{"path": "/tmp/x.md", "keywords": ["eclatauto"], "max_chars": 5000}]
        messages = [{"role": "user", "content": "tell me about sirius trading bot"}]
        result = detect_active_anchors(anchors, messages)
        assert len(result) == 0

    def test_multiple_anchors_match(self):
        anchors = [
            {"path": "/tmp/a.md", "keywords": ["eclatauto"], "max_chars": 5000},
            {"path": "/tmp/b.md", "keywords": ["sirius"], "max_chars": 5000},
        ]
        messages = [
            {"role": "user", "content": "check eclatauto and sirius status"},
        ]
        result = detect_active_anchors(anchors, messages)
        assert len(result) == 2

    def test_lookback_limit(self):
        anchors = [{"path": "/tmp/x.md", "keywords": ["eclatauto"], "max_chars": 5000}]
        # keyword only in old message beyond lookback
        messages = [
            {"role": "user", "content": "fix eclatauto"},
        ] + [
            {"role": "user", "content": "unrelated message"}
            for _ in range(25)
        ]
        result = detect_active_anchors(anchors, messages, lookback=5)
        assert len(result) == 0  # old message is beyond lookback


# ---------------------------------------------------------------------------
# build_anchor_save_prompt
# ---------------------------------------------------------------------------

class TestBuildAnchorSavePrompt:
    def test_prompt_contains_path(self):
        anchor = {"path": "/root/.hermes/context/eclatauto.md", "keywords": ["eclatauto"], "max_chars": 5000}
        prompt = build_anchor_save_prompt(anchor)
        assert "/root/.hermes/context/eclatauto.md" in prompt
        assert "eclatauto" in prompt
        assert "read_file" in prompt or "read" in prompt.lower()
        assert "patch" in prompt


# ---------------------------------------------------------------------------
# build_batch_anchor_save_prompt
# ---------------------------------------------------------------------------

class TestBuildBatchAnchorSavePrompt:
    def test_batch_contains_all_paths(self):
        from agent.context_anchors import build_batch_anchor_save_prompt
        anchors = [
            {"path": "/a.md", "keywords": ["a"], "max_chars": 5000},
            {"path": "/b.md", "keywords": ["b"], "max_chars": 5000},
        ]
        prompt = build_batch_anchor_save_prompt(anchors)
        assert "/a.md" in prompt
        assert "/b.md" in prompt
        assert "read_file" in prompt
        assert "patch" in prompt

    def test_batch_single_anchor_same_as_legacy(self):
        from agent.context_anchors import build_batch_anchor_save_prompt
        anchor = {"path": "/test.md", "keywords": ["test"], "max_chars": 5000}
        legacy = build_anchor_save_prompt(anchor)
        batch = build_batch_anchor_save_prompt([anchor])
        assert legacy == batch


# ---------------------------------------------------------------------------
# should_pre_flush
# ---------------------------------------------------------------------------

class TestShouldPreFlush:
    def test_below_threshold(self):
        from agent.context_anchors import should_pre_flush
        assert should_pre_flush(threshold_tokens=100000, current_tokens=50000) is False

    def test_at_threshold(self):
        from agent.context_anchors import should_pre_flush
        assert should_pre_flush(threshold_tokens=100000, current_tokens=70000) is True

    def test_above_threshold(self):
        from agent.context_anchors import should_pre_flush
        assert should_pre_flush(threshold_tokens=100000, current_tokens=90000) is True

    def test_zero_tokens(self):
        from agent.context_anchors import should_pre_flush
        assert should_pre_flush(threshold_tokens=100000, current_tokens=0) is False
        assert should_pre_flush(threshold_tokens=0, current_tokens=50000) is False


# ---------------------------------------------------------------------------
# snapshot_anchor_hashes / anchors_changed_since
# ---------------------------------------------------------------------------

class TestAnchorHashes:
    def test_snapshot_and_unchanged(self, tmp_path):
        from agent.context_anchors import snapshot_anchor_hashes, anchors_changed_since
        f = tmp_path / "test.md"
        f.write_text("hello world")
        anchors = [{"path": str(f), "keywords": ["test"], "max_chars": 5000}]
        hashes = snapshot_anchor_hashes(anchors)
        assert str(f) in hashes
        assert anchors_changed_since(anchors, hashes) is False

    def test_changed_after_snapshot(self, tmp_path):
        from agent.context_anchors import snapshot_anchor_hashes, anchors_changed_since
        f = tmp_path / "test.md"
        f.write_text("hello world")
        anchors = [{"path": str(f), "keywords": ["test"], "max_chars": 5000}]
        hashes = snapshot_anchor_hashes(anchors)
        f.write_text("updated content")
        assert anchors_changed_since(anchors, hashes) is True

    def test_missing_file(self):
        from agent.context_anchors import snapshot_anchor_hashes, anchors_changed_since
        anchors = [{"path": "/nonexistent.md", "keywords": ["x"], "max_chars": 5000}]
        hashes = snapshot_anchor_hashes(anchors)
        assert len(hashes) == 0
        assert anchors_changed_since(anchors, hashes) is False
