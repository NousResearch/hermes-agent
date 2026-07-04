"""Tests for _summarize_tool_result dict coercion (#58317)."""
import pytest
from agent.context_compressor import _summarize_tool_result


class TestSummarizeToolResultDictArgs:
    def test_write_file_with_dict_content(self):
        """write_file with content as a dict should coerce to str."""
        args = '{"path": "/tmp/out.txt", "content": {"key": "value"}}'
        result = _summarize_tool_result("write_file", args, "ok")
        assert "write_file" in result
        assert "lines" in result

    def test_write_file_with_list_content(self):
        """write_file with content as a list should coerce to str."""
        args = '{"path": "/tmp/out.txt", "content": [1, 2, 3]}'
        result = _summarize_tool_result("write_file", args, "ok")
        assert "write_file" in result
        assert "lines" in result

    def test_write_file_with_string_content(self):
        """write_file with normal string content should work unchanged."""
        args = '{"path": "/tmp/out.txt", "content": "hello\nworld"}'
        result = _summarize_tool_result("write_file", args, "ok")
        assert "write_file" in result

    def test_execute_code_with_dict_code(self):
        """execute_code with code as a dict should coerce to str."""
        args = '{"code": {"language": "python", "content": "print(1)"}}'
        result = _summarize_tool_result("execute_code", args, "ok")
        assert "execute_code" in result

    def test_execute_code_with_string_code(self):
        """execute_code with normal string code should work unchanged."""
        args = '{"code": "print(1)"}'
        result = _summarize_tool_result("execute_code", args, "ok")
        assert "execute_code" in result

    def test_vision_analyze_with_dict_question(self):
        """vision_analyze with question as a dict should coerce to str."""
        args = '{"question": {"text": "what is this?"}}'
        result = _summarize_tool_result("vision_analyze", args, "ok")
        assert "vision_analyze" in result

    def test_vision_analyze_with_none_question(self):
        """vision_analyze with missing question should use default."""
        args = '{}'
        result = _summarize_tool_result("vision_analyze", args, "ok")
        assert "vision_analyze" in result
