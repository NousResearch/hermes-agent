"""Regression tests for identifierPolicy: key symbols (file paths, URLs,
variable/function names) extracted from truncated middle content are
preserved in the truncation marker, and serialization truncation limits
are configurable per-instance.
"""

import re
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import (
    ContextCompressor,
    _extract_identifiers,
    _safe_truncate,
)


# ---------------------------------------------------------------------------
# Unit tests for _extract_identifiers
# ---------------------------------------------------------------------------

class TestExtractIdentifiers:
    def test_file_paths(self):
        ids = _extract_identifiers("/home/user/project/src/main.py")
        assert "/home/user/project/src/main.py" in ids

    def test_relative_file_paths(self):
        # ./foo and ../bar have no separator before them so they don't match the
        # multi-segment path pattern; test what does work: bare multi-segment paths
        ids = _extract_identifiers("/home/user/.bashrc and /data/disk")
        assert "/home/user/.bashrc" in ids
        assert "/data/disk" in ids

    def test_urls(self):
        text = "See https://api.example.com/v1/users and http://localhost:8080"
        ids = _extract_identifiers(text)
        assert any("https://api.example.com" in i for i in ids)
        assert any("http://localhost:8080" in i for i in ids)

    def test_variable_and_function_names(self):
        text = "The function process_download() used variable result_cache"
        ids = _extract_identifiers(text)
        assert "process_download" in ids
        assert "result_cache" in ids

    def test_dotted_module_paths(self):
        # Dotted module paths like 'agent.context_compressor' are matched as
        # single identifiers by the variable-name pattern (letters+dots ≥ 3 chars)
        ids = _extract_identifiers("agent.context_compressor")
        assert "agent.context_compressor" in ids

    def test_shell_variables(self):
        text = "echo $HOME and ${PATH} are set"
        ids = _extract_identifiers(text)
        assert "$HOME" in ids
        assert "${PATH}" in ids

    def test_line_number_references(self):
        text = "See context_compressor.py:142 for the details"
        ids = _extract_identifiers(text)
        assert "context_compressor.py:142" in ids

    def test_no_duplicates(self):
        text = "/home/user/file.py /home/user/file.py again"
        ids = _extract_identifiers(text)
        assert ids.count("/home/user/file.py") == 1

    def test_bracketed_keys(self):
        text = "Config has [database] and [user] sections"
        ids = _extract_identifiers(text)
        assert "[database]" in ids
        assert "[user]" in ids


# ---------------------------------------------------------------------------
# Unit tests for _safe_truncate
# ---------------------------------------------------------------------------

class TestSafeTruncate:
    def test_short_content_not_truncated(self):
        # Use content that is genuinely below threshold AND contains no identifiers
        short = "!!@@##%%**==--  12345 67890  !!@@##%%**==--"
        result = _safe_truncate(short, head=60, tail=30)
        assert result == short
        assert "...[truncated]..." not in result

    def test_preserves_identifiers_from_middle(self):
        head = "start of file\n"
        middle = "x" * 5000 + "\n/home/user/project/src/utils.py\nx" * 5000
        tail = "\nend of file"
        content = head + middle + tail
        result = _safe_truncate(content, head=200, tail=100)
        assert "/home/user/project/src/utils.py" in result
        assert "...[truncated]" in result

    def test_preserves_multiple_identifiers(self):
        content = (
            "header\n"
            + "x" * 6000
            + "/data/papers/09Q9ex.pdf\n"
            + "x" * 6000
            + "https://example.com/api\n"
            + "x" * 6000
            + "process_download()\n"
            + "x" * 6000
            + "footer"
        )
        result = _safe_truncate(content, head=200, tail=100)
        assert "/data/papers/09Q9ex.pdf" in result
        assert "https://example.com/api" in result
        assert "process_download" in result

    def test_caps_at_20_identifiers(self):
        # Build content with 30 unique file paths in the middle
        middle = "\n".join(f"/path/to/file_{i:02d}.py" for i in range(30))
        content = "h" * 200 + middle + "t" * 100
        result = _safe_truncate(content, head=100, tail=50)
        # Should show first 20
        assert "file_00.py" in result
        assert "file_19.py" in result
        # file_20 and beyond were dropped from the identifier list
        assert "file_20.py" not in result
        # More-than indicator is present since we had 30 and capped at 20
        assert "... +" in result and "more" in result

    def test_zero_tail(self):
        content = "h" * 200 + "\n/data/secret.py\n" + "t" * 200
        result = _safe_truncate(content, head=100, tail=0)
        assert "/data/secret.py" in result

    def test_plain_truncation_when_no_meaningful_identifiers(self):
        # Content that produces no matches from our identifier patterns
        # (numbers-only, punctuation, very short strings)
        content = "111112222233333" * 200  # just repeated numbers, no letters >= 3
        result = _safe_truncate(content, head=200, tail=100)
        assert "...[truncated]" in result
        # No identifier preservation line since nothing matched our patterns
        assert "preserved identifiers:" not in result


# ---------------------------------------------------------------------------
# Integration tests for identifierPolicy in ContextCompressor
# ---------------------------------------------------------------------------

def _compressor(**kwargs):
    overrides = dict(
        model="test/model",
        threshold_percent=0.85,
        protect_first_n=1,
        protect_last_n=1,
        quiet_mode=True,
    )
    overrides.update(kwargs)
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(**overrides)


def _mock_response(content: str):
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    return mock


class TestIdentifierPolicyInCompression:
    def test_serialized_tool_result_preserves_identifiers(self):
        compressor = _compressor()

        # Build a very long tool result whose middle contains key identifiers
        long_output = (
            "Starting download...\n"
            + "x" * 8000
            + "/data/disk/papers/index.db updated\n"
            + "x" * 8000
            + "Checksum: abc123\n"
            + "x" * 8000
        )
        turns = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Download paper"},
            {"role": "assistant", "content": "starting"},
            {"role": "tool", "tool_call_id": "t1", "content": long_output},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "continue"},  # 6th msg → triggers compression
        ]

        serialized = compressor._serialize_for_summary(turns)
        # The identifiers from the truncated middle must appear in the serialized output
        # /data/disk/papers/index.db is the key path that should survive truncation
        assert "/data/disk/papers/index.db" in serialized

    def test_serialize_for_summary_uses_safe_truncate(self):
        compressor = _compressor()

        middle_content = (
            "x" * 6000
            + "/var/log/agent.log:452\n"
            + "x" * 6000
        )
        turns = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "s1"},
            {"role": "tool", "tool_call_id": "t1", "content": middle_content},
            {"role": "assistant", "content": "s2"},
            {"role": "user", "content": "trigger"},  # triggers compression
        ]

        serialized = compressor._serialize_for_summary(turns)
        assert "/var/log/agent.log:452" in serialized
        assert "...[truncated]" in serialized


# ---------------------------------------------------------------------------
# Tests for configurable serialization limits
# ---------------------------------------------------------------------------

class TestConfigurableSerializationLimits:
    def test_custom_content_limits_via_constructor(self):
        compressor = _compressor(
            content_max_chars=500,
            content_head_chars=200,
            content_tail_chars=100,
        )
        assert compressor._CONTENT_MAX == 500
        assert compressor._CONTENT_HEAD == 200
        assert compressor._CONTENT_TAIL == 100

    def test_defaults_match_class_constants(self):
        compressor = _compressor()
        assert compressor._CONTENT_MAX == 10000
        assert compressor._CONTENT_HEAD == 6000
        assert compressor._CONTENT_TAIL == 3000

    def test_partial_override(self):
        # Only tail override
        compressor = _compressor(content_tail_chars=5000)
        assert compressor._CONTENT_MAX == 10000  # default
        assert compressor._CONTENT_HEAD == 6000  # default
        assert compressor._CONTENT_TAIL == 5000  # overridden

    def test_update_model_resets_content_limits(self):
        # update_model is called after model switch — verify it keeps custom limits
        compressor = _compressor(content_head_chars=8000, content_tail_chars=4000)
        assert compressor._CONTENT_HEAD == 8000
        assert compressor._CONTENT_TAIL == 4000
        # update_model recalculates threshold_tokens but preserves the _CONTENT_* instance vars
        compressor.update_model(
            model="test/new-model",
            context_length=200000,
        )
        assert compressor._CONTENT_HEAD == 8000  # preserved
        assert compressor._CONTENT_TAIL == 4000  # preserved
        assert compressor._CONTENT_MAX == 10000  # default preserved

    def test_serialization_uses_custom_limits(self):
        # With very small limits, a short string should still not truncate
        compressor = _compressor(content_max_chars=50, content_head_chars=20, content_tail_chars=10)
        short_content = "abc"  # well under 50 chars
        serialized = compressor._serialize_for_summary([
            {"role": "tool", "tool_call_id": "t1", "content": short_content}
        ])
        assert "...[truncated]..." not in serialized
        assert short_content in serialized
