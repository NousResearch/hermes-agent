"""Tests for _ensure_args_is_list() — the single normalization point that
guarantees runtime['args'] is always list[str].

This covers the root cause of the TypeError: 'types.SimpleNamespace' object
is not iterable crash that killed all delegation paths.
"""
import types

import pytest

from hermes_cli.runtime_provider import _ensure_args_is_list


class TestEnsureArgsIsList:
    """Unit tests for _ensure_args_is_list."""

    def test_none_becomes_empty_list(self):
        runtime = {"provider": "nous"}
        result = _ensure_args_is_list(runtime)
        assert result["args"] == []
        assert result is runtime  # mutates in place and returns

    def test_list_passthrough(self):
        runtime = {"provider": "copilot-acp", "args": ["--acp", "--stdio"]}
        result = _ensure_args_is_list(runtime)
        assert result["args"] == ["--acp", "--stdio"]
        assert isinstance(result["args"], list)

    def test_tuple_converted(self):
        runtime = {"provider": "custom", "args": ("--flag", "value")}
        result = _ensure_args_is_list(runtime)
        assert result["args"] == ["--flag", "value"]
        assert isinstance(result["args"], list)

    def test_set_converted(self):
        runtime = {"provider": "custom", "args": {"--flag"}}
        result = _ensure_args_is_list(runtime)
        assert isinstance(result["args"], list)
        assert "--flag" in result["args"]

    def test_frozenset_converted(self):
        runtime = {"provider": "custom", "args": frozenset(["--flag"])}
        result = _ensure_args_is_list(runtime)
        assert isinstance(result["args"], list)
        assert "--flag" in result["args"]

    def test_string_wrapped_with_warning(self, caplog):
        runtime = {"provider": "custom", "args": "--stdio"}
        with caplog.at_level("WARNING"):
            result = _ensure_args_is_list(runtime)
        assert result["args"] == ["--stdio"]
        assert "string" in caplog.text.lower()

    def test_empty_list_stays_empty(self):
        runtime = {"provider": "nous", "args": []}
        result = _ensure_args_is_list(runtime)
        assert result["args"] == []

    def test_simplenamespace_raises_typeerror(self):
        """The original crash: types.SimpleNamespace is not iterable."""
        ns = types.SimpleNamespace(acp=True, stdio=True)
        runtime = {"provider": "copilot-acp", "args": ns}
        with pytest.raises(TypeError, match="must be a list of strings"):
            _ensure_args_is_list(runtime)

    def test_dict_raises_typeerror(self):
        """Dict would silently lose keys if coerced — must fail loudly."""
        runtime = {"provider": "custom", "args": {"--acp": True}}
        with pytest.raises(TypeError, match="must be a list of strings"):
            _ensure_args_is_list(runtime)

    def test_int_raises_typeerror(self):
        runtime = {"provider": "custom", "args": 42}
        with pytest.raises(TypeError, match="must be a list of strings"):
            _ensure_args_is_list(runtime)

    def test_runtime_without_args_key_gets_empty_list(self):
        """Most provider paths don't include 'args' at all."""
        runtime = {"provider": "nous", "api_key": "test"}
        result = _ensure_args_is_list(runtime)
        assert "args" in result
        assert result["args"] == []

    def test_list_is_shallow_copy(self):
        """Returned list should be a new object, not the same reference."""
        original = ["--acp", "--stdio"]
        runtime = {"provider": "copilot-acp", "args": original}
        result = _ensure_args_is_list(runtime)
        assert result["args"] == original
        assert result["args"] is not original  # different object

    def test_nested_dicts_in_list_preserved(self):
        """Lists with non-string items are preserved as-is (shallow copy)."""
        runtime = {"provider": "custom", "args": ["--flag", 42]}
        result = _ensure_args_is_list(runtime)
        assert result["args"] == ["--flag", 42]

    def test_return_value_is_same_dict(self):
        """Function mutates in-place and returns the same dict for convenience."""
        runtime = {"provider": "nous"}
        result = _ensure_args_is_list(runtime)
        assert result is runtime
