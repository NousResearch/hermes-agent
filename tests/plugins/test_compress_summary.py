"""Tests for the compress-summary plugin.

Covers: register(), _on_pre_compress(), _build_summary(), all extractors,
helper functions, and edge cases.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure repo root is on sys.path so the plugin can be imported
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Directory uses a hyphen ("compress-summary"), so we need importlib
import importlib
_mod = importlib.import_module("plugins.compress-summary")

register = _mod.register
_on_pre_compress = _mod._on_pre_compress
_build_summary = _mod._build_summary
_find_original_goal = _mod._find_original_goal
_extract_actions = _mod._extract_actions
_extract_progress_notes = _mod._extract_progress_notes
_extract_recent_user = _mod._extract_recent_user
_extract_key_files = _mod._extract_key_files
_trunc = _mod._trunc
_jfield = _mod._jfield
_MARKER = _mod._MARKER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _user(content):
    return {"role": "user", "content": content}


def _assistant(content, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool_call(name, arguments="{}"):
    return {"function": {"name": name, "arguments": arguments}}


def _make_conversation(n=15):
    """Build a realistic conversation with tool calls."""
    msgs = [
        _user("帮我在 /tmp/app 下创建一个 Flask 项目"),
        _assistant("好的，我来帮你创建 Flask 项目。", tool_calls=[
            _tool_call("terminal", '{"command": "mkdir -p /tmp/app"}'),
        ]),
        _assistant(None, tool_calls=[
            _tool_call("write_file", '{"path": "/tmp/app/app.py", "content": "from flask import Flask"}'),
        ]),
        _assistant("Flask 项目基础结构已创建。"),
        _user("加上 requirements.txt"),
        _assistant(None, tool_calls=[
            _tool_call("write_file", '{"path": "/tmp/app/requirements.txt", "content": "flask==3.0"}'),
        ]),
        _assistant("requirements.txt 已添加。"),
        _user("帮我加个 Dockerfile"),
        _assistant(None, tool_calls=[
            _tool_call("write_file", '{"path": "/tmp/app/Dockerfile", "content": "FROM python:3.12"}'),
        ]),
        _assistant("Dockerfile 已创建。"),
        _user("运行测试看看"),
        _assistant(None, tool_calls=[
            _tool_call("terminal", '{"command": "cd /tmp/app && python -m pytest"}'),
        ]),
        _assistant("测试通过了。"),
        _user("好的，现在部署到服务器"),
        _assistant("开始部署流程。"),
    ]
    return msgs[:n] if n < len(msgs) else msgs


# ===========================================================================
# register()
# ===========================================================================

class TestRegister:
    def test_registers_pre_compress_hook(self):
        ctx = MagicMock()
        register(ctx)
        ctx.register_hook.assert_called_once_with("pre_compress", _on_pre_compress)


# ===========================================================================
# _on_pre_compress()
# ===========================================================================

class TestOnPreCompress:
    def test_skips_when_messages_none(self):
        _on_pre_compress(messages=None)  # should not raise

    def test_skips_when_too_few_messages(self):
        msgs = [_user("hi")] * 5
        original_len = len(msgs)
        _on_pre_compress(messages=msgs)
        assert len(msgs) == original_len  # nothing appended

    def test_injects_summary_for_long_conversation(self):
        msgs = _make_conversation(15)
        original_len = len(msgs)
        _on_pre_compress(messages=msgs)
        assert len(msgs) == original_len + 1
        assert _MARKER in msgs[-1]["content"]
        assert msgs[-1]["role"] == "user"

    def test_no_double_injection(self):
        msgs = _make_conversation(15)
        _on_pre_compress(messages=msgs)
        count_after_first = len(msgs)
        _on_pre_compress(messages=msgs)
        assert len(msgs) == count_after_first  # no second injection

    def test_passes_session_id(self):
        msgs = _make_conversation(15)
        _on_pre_compress(session_id="test-123", messages=msgs)
        assert len(msgs) > 15  # summary was injected


# ===========================================================================
# _build_summary()
# ===========================================================================

class TestBuildSummary:
    def test_returns_none_for_empty(self):
        assert _build_summary([]) is None

    def test_returns_none_when_no_goal_no_actions(self):
        msgs = [{"role": "system", "content": "you are helpful"}] * 5
        assert _build_summary(msgs) is None

    def test_includes_marker(self):
        msgs = _make_conversation()
        summary = _build_summary(msgs)
        assert summary.startswith(_MARKER)

    def test_includes_original_request(self):
        msgs = _make_conversation()
        summary = _build_summary(msgs)
        assert "Original request:" in summary
        assert "Flask" in summary

    def test_includes_actions(self):
        msgs = _make_conversation()
        summary = _build_summary(msgs)
        assert "Actions taken:" in summary
        assert "[write_file]" in summary

    def test_includes_key_files(self):
        msgs = _make_conversation()
        summary = _build_summary(msgs)
        assert "Key files:" in summary
        assert "/tmp/app/app.py" in summary

    def test_includes_progress(self):
        msgs = _make_conversation()
        summary = _build_summary(msgs)
        assert "Progress:" in summary


# ===========================================================================
# _find_original_goal()
# ===========================================================================

class TestFindOriginalGoal:
    def test_finds_first_user_message(self):
        msgs = [
            {"role": "system", "content": "system prompt"},
            _user("帮我创建一个 Python 项目"),
        ]
        assert "Python" in _find_original_goal(msgs)

    def test_skips_short_messages(self):
        msgs = [_user("ok"), _user("帮我创建一个完整的 Flask 应用")]
        assert "Flask" in _find_original_goal(msgs)

    def test_skips_injected_content(self):
        msgs = [
            _user("[MEMORY] some injected stuff"),
            _user("帮我写一个脚本来处理数据"),
        ]
        assert "脚本" in _find_original_goal(msgs)

    def test_returns_none_when_no_goal(self):
        msgs = [_user("hi"), _user("ok")]
        assert _find_original_goal(msgs) is None

    def test_truncates_long_goal(self):
        msgs = [_user("x" * 1000)]
        result = _find_original_goal(msgs)
        assert len(result) <= 503  # 500 + "..."


# ===========================================================================
# _extract_actions()
# ===========================================================================

class TestExtractActions:
    def test_extracts_tool_calls(self):
        msgs = [
            _assistant(None, tool_calls=[
                _tool_call("write_file", '{"path": "/tmp/test.py"}'),
            ]),
        ]
        actions = _extract_actions(msgs)
        assert len(actions) == 1
        assert "[write_file]" in actions[0]

    def test_deduplicates(self):
        tc = _tool_call("write_file", '{"path": "/tmp/test.py"}')
        msgs = [
            _assistant(None, tool_calls=[tc]),
            _assistant(None, tool_calls=[tc]),
        ]
        actions = _extract_actions(msgs)
        assert len(actions) == 1

    def test_caps_at_20(self):
        msgs = []
        for i in range(30):
            msgs.append(_assistant(None, tool_calls=[
                _tool_call("write_file", f'{{"path": "/tmp/file_{i}.py"}}'),
            ]))
        actions = _extract_actions(msgs)
        assert len(actions) == 20  # 5 + marker + 14

    def test_terminal_shows_command(self):
        msgs = [_assistant(None, tool_calls=[
            _tool_call("terminal", '{"command": "ls -la /tmp"}'),
        ])]
        actions = _extract_actions(msgs)
        assert "ls -la" in actions[0]

    def test_search_shows_pattern(self):
        msgs = [_assistant(None, tool_calls=[
            _tool_call("search_files", '{"pattern": "TODO"}'),
        ])]
        actions = _extract_actions(msgs)
        assert "TODO" in actions[0]

    def test_delegate_shows_goal(self):
        msgs = [_assistant(None, tool_calls=[
            _tool_call("delegate_task", '{"goal": "check upstream repo"}'),
        ])]
        actions = _extract_actions(msgs)
        assert "check upstream" in actions[0]

    def test_unknown_tool_still_logged(self):
        msgs = [_assistant(None, tool_calls=[
            _tool_call("some_new_tool", "{}"),
        ])]
        actions = _extract_actions(msgs)
        assert "[some_new_tool]" in actions[0]


# ===========================================================================
# _extract_progress_notes()
# ===========================================================================

class TestExtractProgressNotes:
    def test_extracts_first_sentence(self):
        msgs = [_assistant("项目已创建。接下来配置数据库。")]
        notes = _extract_progress_notes(msgs)
        assert len(notes) == 1
        assert notes[0] == "项目已创建。"

    def test_skips_short_content(self):
        msgs = [_assistant("ok")]
        assert _extract_progress_notes(msgs) == []

    def test_keeps_last_10(self):
        msgs = [_assistant(f"步骤 {i} 已完成。") for i in range(20)]
        notes = _extract_progress_notes(msgs)
        assert len(notes) <= 10

    def test_deduplicates(self):
        msgs = [_assistant("这个任务已经完成了。"), _assistant("这个任务已经完成了。")]
        notes = _extract_progress_notes(msgs)
        assert len(notes) == 1


# ===========================================================================
# _extract_recent_user()
# ===========================================================================

class TestExtractRecentUser:
    def test_extracts_recent_messages(self):
        msgs = [_user(f"指令 {i}，请执行这个操作") for i in range(25)]
        recent = _extract_recent_user(msgs)
        assert len(recent) == 5  # last 5

    def test_skips_injected(self):
        msgs = [_user("[SYSTEM] injected"), _user("真正的用户消息来了")]
        recent = _extract_recent_user(msgs)
        assert len(recent) == 1

    def test_skips_marker(self):
        msgs = [_user(f"{_MARKER}\nsome summary"), _user("继续工作吧")]
        recent = _extract_recent_user(msgs)
        assert len(recent) == 1
        assert "继续" in recent[0]


# ===========================================================================
# _extract_key_files()
# ===========================================================================

class TestExtractKeyFiles:
    def test_collects_paths(self):
        msgs = [
            _assistant(None, tool_calls=[
                _tool_call("write_file", '{"path": "/a.py"}'),
                _tool_call("read_file", '{"path": "/b.py"}'),
                _tool_call("patch", '{"path": "/c.py"}'),
            ]),
        ]
        files = _extract_key_files(msgs)
        assert files == {"/a.py", "/b.py", "/c.py"}

    def test_ignores_non_file_tools(self):
        msgs = [_assistant(None, tool_calls=[
            _tool_call("terminal", '{"command": "ls"}'),
        ])]
        assert _extract_key_files(msgs) == set()

    def test_deduplicates(self):
        msgs = [
            _assistant(None, tool_calls=[_tool_call("write_file", '{"path": "/a.py"}')]),
            _assistant(None, tool_calls=[_tool_call("read_file", '{"path": "/a.py"}')]),
        ]
        assert len(_extract_key_files(msgs)) == 1


# ===========================================================================
# Helpers
# ===========================================================================

class TestTrunc:
    def test_short_unchanged(self):
        assert _trunc("hello", 10) == "hello"

    def test_long_truncated(self):
        assert _trunc("a" * 20, 10) == "a" * 10 + "..."

    def test_newlines_collapsed(self):
        assert _trunc("a\nb\nc", 100) == "a b c"


class TestJfield:
    def test_extracts_string_field(self):
        assert _jfield('{"path": "/tmp/test.py"}', "path") == "/tmp/test.py"

    def test_returns_none_for_missing(self):
        assert _jfield('{"path": "/tmp"}', "command") is None

    def test_returns_none_for_empty(self):
        assert _jfield("", "path") is None
        assert _jfield(None, "path") is None

    def test_handles_escaped_quotes(self):
        assert _jfield('{"cmd": "echo \\"hi\\""}', "cmd") == 'echo "hi"'

    def test_handles_escaped_newlines(self):
        assert _jfield('{"text": "line1\\nline2"}', "text") == "line1\nline2"
