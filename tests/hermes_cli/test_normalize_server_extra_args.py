"""Tests for normalize_server_extra_args (#103533)."""

from __future__ import annotations

from hermes_cli.local_runtime.supervisor import normalize_server_extra_args


def test_passes_valid_flags():
    """Valid flags must pass through unchanged."""
    result = normalize_server_extra_args(["--main-gpu", "0", "--no-mmap"])
    assert result == ["--main-gpu", "0", "--no-mmap"]


def test_refuses_managed_flags():
    """Managed flags (--host, --port, etc.) must be dropped."""
    result = normalize_server_extra_args(["--host", "0.0.0.0", "--port", "8080", "--main-gpu", "0"])
    assert result == ["--main-gpu", "0"]


def test_refuses_managed_flags_with_equals():
    """Managed flags in --flag=value form must be dropped."""
    result = normalize_server_extra_args(["--host=0.0.0.0", "--port=8080", "--main-gpu=0"])
    assert result == ["--main-gpu=0"]


def test_refuses_managed_flag_aliases():
    """Alias forms (--api-key, --models-dir) must also be dropped."""
    result = normalize_server_extra_args(["--api-key", "abc", "--models-dir", "/models"])
    assert result == []


def test_non_list_input_returns_empty():
    """Non-list inputs (None, string, int) must return empty list."""
    assert normalize_server_extra_args(None) == []
    assert normalize_server_extra_args("") == []
    assert normalize_server_extra_args("--main-gpu") == []


def test_blank_and_non_string_items_are_skipped():
    """Blank strings and non-string items must be filtered out."""
    result = normalize_server_extra_args(["", "  ", 42, "--main-gpu", "0"])
    assert result == ["--main-gpu", "0"]


def test_overlong_args_are_truncated():
    """Args exceeding _MAX_SERVER_EXTRA_ARG_LEN must be dropped."""
    long_arg = "--" + "x" * 2000
    result = normalize_server_extra_args([long_arg, "--main-gpu", "0"])
    assert result == ["--main-gpu", "0"]


def test_too_many_args_are_capped():
    """More than _MAX_SERVER_EXTRA_ARGS items must be truncated."""
    many = [f"--flag-{i}" for i in range(200)]
    result = normalize_server_extra_args(many)
    assert len(result) <= 128
