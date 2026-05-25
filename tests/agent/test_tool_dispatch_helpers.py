"""Tests for agent/tool_dispatch_helpers.py.

Covers parallelisation gating, multimodal envelopes, mutation tracking,
error preview, trajectory normalisation, and tool-result message building.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from agent.tool_dispatch_helpers import (
    _append_subdir_hint_to_multimodal,
    _extract_error_preview,
    _extract_file_mutation_targets,
    _extract_parallel_scope_path,
    _is_destructive_command,
    _is_multimodal_tool_result,
    _multimodal_text_summary,
    _paths_overlap,
    _should_parallelize_tool_batch,
    _trajectory_normalize_msg,
    make_tool_result_message,
)


# ── _is_destructive_command ────────────────────────────────────────────

def test_destructive_rm():
    assert _is_destructive_command("rm -rf /tmp/x") is True


def test_destructive_cp():
    assert _is_destructive_command("cp a b") is True


def test_destructive_mv():
    assert _is_destructive_command("mv old new") is True


def test_destructive_sed_inplace():
    assert _is_destructive_command("sed -i 's/x/y/' file") is True


def test_destructive_redirect_overwrite():
    assert _is_destructive_command("echo hi > file.txt") is True


def test_safe_readonly_command():
    assert _is_destructive_command("cat file.txt") is False


def test_safe_echo():
    assert _is_destructive_command("echo hello") is False


def test_destructive_empty_command():
    assert _is_destructive_command("") is False


def test_destructive_git_reset():
    assert _is_destructive_command("git reset --hard HEAD") is True


def test_destructive_git_checkout():
    assert _is_destructive_command("git checkout -- file") is True


# ── _paths_overlap ────────────────────────────────────────────────────

def test_paths_overlap_same_path():
    assert _paths_overlap(Path("/a/b/c"), Path("/a/b/c")) is True


def test_paths_overlap_subpath():
    assert _paths_overlap(Path("/a/b"), Path("/a/b/c")) is True


def test_paths_overlap_different_roots():
    assert _paths_overlap(Path("/a/b"), Path("/x/y")) is False


def test_paths_overlap_relative_and_absolute():
    # Both absolute but different
    assert _paths_overlap(Path("/home/user"), Path("/etc/conf")) is False


# ── _is_multimodal_tool_result ─────────────────────────────────────────

def test_is_multimodal_valid_envelope():
    envelope = {"_multimodal": True, "content": [{"type": "text", "text": "hi"}]}
    assert _is_multimodal_tool_result(envelope) is True


def test_is_multimodal_missing_multimodal_key():
    assert _is_multimodal_tool_result({"content": []}) is False


def test_is_multimodal_non_dict():
    assert _is_multimodal_tool_result("not a dict") is False


def test_is_multimodal_content_not_list():
    assert _is_multimodal_tool_result({"_multimodal": True, "content": "str"}) is False


# ── _multimodal_text_summary ───────────────────────────────────────────

def test_text_summary_from_envelope():
    envelope = {
        "_multimodal": True,
        "content": [{"type": "text", "text": "hello"}, {"type": "image", "source": "..."}],
    }
    assert _multimodal_text_summary(envelope) == "hello"


def test_text_summary_falls_back_to_text_summary_field():
    envelope = {
        "_multimodal": True,
        "content": [],
        "text_summary": "fallback summary",
    }
    assert _multimodal_text_summary(envelope) == "fallback summary"


def test_text_summary_no_text_parts():
    envelope = {"_multimodal": True, "content": [{"type": "image"}]}
    assert _multimodal_text_summary(envelope) == "[multimodal tool result]"


def test_text_summary_plain_string():
    assert _multimodal_text_summary("plain string") == "plain string"


def test_text_summary_dict():
    assert _multimodal_text_summary({"key": "value"}) == '{"key": "value"}'


# ── _append_subdir_hint_to_multimodal ─────────────────────────────────

def test_append_hint_updates_text_part():
    envelope = {
        "_multimodal": True,
        "content": [{"type": "text", "text": "details"}],
    }
    _append_subdir_hint_to_multimodal(envelope, "\n(hint)")
    assert envelope["content"][0]["text"] == "details\n(hint)"


def test_append_hint_inserts_when_no_text_part():
    envelope = {
        "_multimodal": True,
        "content": [{"type": "image"}],
    }
    _append_subdir_hint_to_multimodal(envelope, "hint")
    assert envelope["content"][0]["type"] == "text"
    assert envelope["content"][0]["text"] == "hint"


def test_append_hint_updates_text_summary():
    envelope = {
        "_multimodal": True,
        "content": [{"type": "text", "text": "body"}],
        "text_summary": "summary",
    }
    _append_subdir_hint_to_multimodal(envelope, " (hint)")
    assert envelope["text_summary"] == "summary (hint)"


def test_append_hint_non_multimodal_noop():
    value = {"not": "multimodal"}
    _append_subdir_hint_to_multimodal(value, "hint")
    assert value == {"not": "multimodal"}


# ── _extract_file_mutation_targets ────────────────────────────────────

def test_extract_mutation_write_file():
    targets = _extract_file_mutation_targets("write_file", {"path": "/tmp/x.py"})
    assert targets == ["/tmp/x.py"]


def test_extract_mutation_patch_replace_mode():
    targets = _extract_file_mutation_targets("patch", {"mode": "replace", "path": "/tmp/y.py"})
    assert targets == ["/tmp/y.py"]


def test_extract_mutation_patch_v4a_mode():
    body = "*** Update File: foo.py\n@@ ... @@\n*** Add File: bar.py\n@@ ... @@\n"
    targets = _extract_file_mutation_targets("patch", {"mode": "patch", "patch": body})
    assert targets == ["foo.py", "bar.py"]


def test_extract_mutation_unknown_tool_returns_empty():
    targets = _extract_file_mutation_targets("read_file", {"path": "/tmp/z.py"})
    assert targets == []


def test_extract_mutation_missing_path():
    targets = _extract_file_mutation_targets("write_file", {})
    assert targets == []


def test_extract_mutation_patch_delete_file():
    body = "*** Delete File: stale.py\n"
    targets = _extract_file_mutation_targets("patch", {"mode": "patch", "patch": body})
    assert targets == ["stale.py"]


# ── _extract_error_preview ─────────────────────────────────────────────

def test_error_preview_from_error_field():
    result = '{"success": false, "error": "something went wrong"}'
    assert _extract_error_preview(result) == "something went wrong"


def test_error_preview_string_result():
    assert _extract_error_preview("plain error message here") == "plain error message here"


def test_error_preview_truncation():
    long_msg = "x" * 200
    preview = _extract_error_preview(long_msg, max_len=50)
    assert len(preview) <= 50
    assert preview.endswith("…")


def test_error_preview_none():
    assert _extract_error_preview(None) == ""


def test_error_preview_non_json_dict_like():
    assert _extract_error_preview("{not valid json") == "{not valid json"


# ── _trajectory_normalize_msg ──────────────────────────────────────────

def test_trajectory_strips_multimodal_content():
    msg = {
        "role": "tool",
        "content": {
            "_multimodal": True,
            "content": [{"type": "image"}],
            "text_summary": "summary text",
        },
    }
    result = _trajectory_normalize_msg(msg)
    assert result["content"] == "summary text"


def test_trajectory_replaces_image_parts():
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ],
    }
    result = _trajectory_normalize_msg(msg)
    assert result["content"] == [
        {"type": "text", "text": "describe"},
        {"type": "text", "text": "[screenshot]"},
    ]


def test_trajectory_input_image_replaced():
    msg = {
        "role": "user",
        "content": [{"type": "input_image"}],
    }
    result = _trajectory_normalize_msg(msg)
    assert result["content"] == [{"type": "text", "text": "[screenshot]"}]


def test_trajectory_text_only_unchanged():
    msg = {"role": "user", "content": "hello"}
    result = _trajectory_normalize_msg(msg)
    assert result == msg


def test_trajectory_non_dict_passthrough():
    assert _trajectory_normalize_msg("not a dict") == "not a dict"


# ── _extract_parallel_scope_path ───────────────────────────────────────

def test_extract_scope_absolute_path():
    result = _extract_parallel_scope_path("write_file", {"path": "/tmp/file.py"})
    assert result is not None
    assert str(result) == "/tmp/file.py"


def test_extract_scope_relative_path():
    result = _extract_parallel_scope_path("write_file", {"path": "subdir/file.py"})
    assert result is not None
    assert result.name == "file.py"


def test_extract_scope_missing_path():
    result = _extract_parallel_scope_path("write_file", {})
    assert result is None


def test_extract_scope_empty_path():
    result = _extract_parallel_scope_path("write_file", {"path": ""})
    assert result is None


def test_extract_scope_non_path_tool():
    result = _extract_parallel_scope_path("read_file", {"path": "/tmp/x"})
    assert result is not None  # read_file IS path-scoped


def test_extract_scope_non_scoped_tool():
    result = _extract_parallel_scope_path("terminal", {"path": "/tmp/x"})
    assert result is None


# ── _should_parallelize_tool_batch ────────────────────────────────────

def _tc(name: str, args: dict | str) -> MagicMock:
    """Build a mock tool_call with given name and JSON arguments."""
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args) if isinstance(args, dict) else args
    return tc


def test_should_parallelize_single_call():
    assert _should_parallelize_tool_batch([_tc("read_file", {"path": "/x"})]) is False


def test_should_parallelize_two_safe_reads():
    tcs = [_tc("read_file", {"path": "/a"}), _tc("read_file", {"path": "/b"})]
    assert _should_parallelize_tool_batch(tcs) is True


def test_should_parallelize_clarify_blocks():
    tcs = [_tc("read_file", {"path": "/a"}), _tc("clarify", {"question": "?"})]
    assert _should_parallelize_tool_batch(tcs) is False


def test_should_parallelize_same_file_blocks():
    tcs = [_tc("write_file", {"path": "/a/x.py"}), _tc("write_file", {"path": "/a/x.py"})]
    assert _should_parallelize_tool_batch(tcs) is False  # same file


def test_should_parallelize_file_and_parent_dir_blocks():
    tcs = [_tc("write_file", {"path": "/a"}), _tc("write_file", {"path": "/a/x.py"})]
    assert _should_parallelize_tool_batch(tcs) is False  # dir + file inside


def test_should_parallelize_non_overlapping_paths():
    tcs = [_tc("write_file", {"path": "/a/x.py"}), _tc("write_file", {"path": "/b/y.py"})]
    assert _should_parallelize_tool_batch(tcs) is True


def test_should_parallelize_unparseable_args():
    tcs = [_tc("read_file", "not json"), _tc("read_file", {"path": "/b"})]
    assert _should_parallelize_tool_batch(tcs) is False


def test_should_parallelize_non_dict_args():
    tc = MagicMock()
    tc.function.name = "read_file"
    tc.function.arguments = json.dumps("just a string")  # not a dict
    tcs = [tc, _tc("read_file", {"path": "/b"})]
    assert _should_parallelize_tool_batch(tcs) is False


# ── make_tool_result_message ───────────────────────────────────────────

def test_make_tool_result_message():
    msg = make_tool_result_message("read_file", "file contents", "call_123")
    assert msg["role"] == "tool"
    assert msg["name"] == "read_file"
    assert msg["tool_name"] == "read_file"
    assert msg["content"] == "file contents"
    assert msg["tool_call_id"] == "call_123"
