"""Tests for gateway.platforms.qqbot.utils — User-Agent, headers, config coercion."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from gateway.platforms.qqbot.utils import (
    _get_hermes_version,
    build_user_agent,
    coerce_list,
    get_api_headers,
)


# ============================================================================
# _get_hermes_version
# ============================================================================
class TestGetHermesVersion:
    def test_returns_version_when_available(self):
        with patch("importlib.metadata.version", return_value="1.2.3"):
            assert _get_hermes_version() == "1.2.3"

    def test_returns_dev_on_error(self):
        with patch("importlib.metadata.version", side_effect=ModuleNotFoundError):
            assert _get_hermes_version() == "dev"

    def test_returns_dev_on_any_exception(self):
        with patch("importlib.metadata.version", side_effect=RuntimeError("bad")):
            assert _get_hermes_version() == "dev"


# ============================================================================
# build_user_agent
# ============================================================================
class TestBuildUserAgent:
    def test_format_structure(self):
        ua = build_user_agent()
        assert ua.startswith("QQBotAdapter/")
        assert "(Python/" in ua
        assert "; Hermes/" in ua

    def test_contains_python_version(self):
        ua = build_user_agent()
        assert f"{sys.version_info.major}.{sys.version_info.minor}" in ua

    @patch("platform.system", return_value="Linux")
    def test_contains_os_name(self, _mock_system):
        ua = build_user_agent()
        assert "; linux;" in ua


# ============================================================================
# get_api_headers
# ============================================================================
class TestGetApiHeaders:
    def test_returns_expected_keys(self):
        headers = get_api_headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"
        assert "User-Agent" in headers

    def test_user_agent_is_string(self):
        headers = get_api_headers()
        assert isinstance(headers["User-Agent"], str)
        assert len(headers["User-Agent"]) > 0


# ============================================================================
# coerce_list
# ============================================================================
class TestCoerceList:
    def test_none_returns_empty(self):
        assert coerce_list(None) == []

    def test_empty_string(self):
        assert coerce_list("") == []

    def test_whitespace_string(self):
        assert coerce_list("   ") == []

    def test_comma_separated_string(self):
        result = coerce_list("a,b,c")
        assert result == ["a", "b", "c"]

    def test_comma_separated_with_whitespace(self):
        result = coerce_list(" apple , banana , cherry ")
        assert result == ["apple", "banana", "cherry"]

    def test_comma_separated_with_empty_slots(self):
        result = coerce_list("a,,b, ,c")
        assert result == ["a", "b", "c"]

    def test_list_of_strings(self):
        assert coerce_list(["a", "b", "c"]) == ["a", "b", "c"]

    def test_list_with_whitespace(self):
        assert coerce_list([" a ", " b "]) == ["a", "b"]

    def test_list_with_empty_strings_filtered(self):
        result = coerce_list(["a", "", "  ", "b"])
        assert result == ["a", "b"]

    def test_tuple(self):
        assert coerce_list(("a", "b")) == ["a", "b"]

    def test_set(self):
        result = coerce_list({"a", "b"})
        assert sorted(result) == ["a", "b"]  # set order not guaranteed

    def test_single_scalar_value(self):
        assert coerce_list(42) == ["42"]

    def test_single_scalar_whitespace_only(self):
        """str(0) = '0' which is truthy, but str of empty-ish..."""
        # bool → str(True) = "True" → truthy → ["True"]
        assert coerce_list(True) == ["True"]

    def test_empty_list(self):
        assert coerce_list([]) == []

    def test_list_with_non_strings_coerced(self):
        result = coerce_list([1, 2.5, True])
        assert result == ["1", "2.5", "True"]

    def test_empty_tuple(self):
        assert coerce_list(()) == []

    def test_single_item_string(self):
        assert coerce_list("hello") == ["hello"]
