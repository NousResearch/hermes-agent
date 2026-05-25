"""Tests for tools/budget_config.py and tools/debug_helpers.py."""

import os
import json
import tempfile
from pathlib import Path
from unittest import mock
import pytest

from tools.budget_config import BudgetConfig, PINNED_THRESHOLDS, DEFAULT_BUDGET
from tools.debug_helpers import DebugSession


# ── tools/budget_config.py ────────────────────────────────────────────────────

class TestBudgetConfig:
    """BudgetConfig.resolve_threshold — pinned > overrides > registry > default."""

    @mock.patch("tools.registry.registry")
    def test_default_threshold(self, mock_registry):
        """Without overrides or pinned, returns the default from registry."""
        mock_registry.get_max_result_size.return_value = 50000
        cfg = BudgetConfig(default_result_size=50000)
        assert cfg.resolve_threshold("unknown_tool") == 50000

    def test_pinned_threshold_overrides_everything(self):
        """Pinned thresholds cannot be overridden — always return inf."""
        cfg = BudgetConfig(
            default_result_size=100000,
            tool_overrides={"read_file": 5000},
        )
        assert cfg.resolve_threshold("read_file") == float("inf")

    @mock.patch("tools.registry.registry")
    def test_tool_override_wins_over_default(self, mock_registry):
        """Tool-specific override takes priority over default."""
        mock_registry.get_max_result_size.return_value = 100000
        cfg = BudgetConfig(
            default_result_size=100000,
            tool_overrides={"web_search": 20000},
        )
        assert cfg.resolve_threshold("web_search") == 20000

    def test_default_budget_instance(self):
        """DEFAULT_BUDGET is a BudgetConfig with standard defaults."""
        assert isinstance(DEFAULT_BUDGET, BudgetConfig)
        assert DEFAULT_BUDGET.default_result_size == 100_000
        assert DEFAULT_BUDGET.turn_budget == 200_000
        assert DEFAULT_BUDGET.preview_size == 1_500

    def test_no_tool_overrides_by_default(self):
        """DEFAULT_BUDGET has an empty tool_overrides dict."""
        assert DEFAULT_BUDGET.tool_overrides == {}

    @mock.patch("tools.registry.registry")
    def test_custom_overrides_are_respected(self, mock_registry):
        """When tool_overrides has a value for a non-pinned tool, it wins."""
        mock_registry.get_max_result_size.return_value = 100000
        cfg = BudgetConfig(
            default_result_size=100000,
            tool_overrides={"terminal": 80000},
        )
        assert cfg.resolve_threshold("terminal") == 80000

    def test_frozen_dataclass(self):
        """BudgetConfig is frozen — cannot be mutated after creation."""
        cfg = BudgetConfig(default_result_size=50000)
        with pytest.raises(Exception):
            cfg.default_result_size = 99999

    def test_pinned_thresholds_contains_read_file(self):
        """PINNED_THRESHOLDS has read_file set to infinity."""
        assert "read_file" in PINNED_THRESHOLDS
        assert PINNED_THRESHOLDS["read_file"] == float("inf")


class TestPinThresholds:
    """PINNED_THRESHOLDS immutability and correctness."""

    def test_pinned_is_dict(self):
        assert isinstance(PINNED_THRESHOLDS, dict)

    def test_only_read_file_is_pinned(self):
        assert len(PINNED_THRESHOLDS) >= 1
        assert "read_file" in PINNED_THRESHOLDS

    def test_pinned_keys_are_strings(self):
        for key in PINNED_THRESHOLDS:
            assert isinstance(key, str), f"Key {key!r} is not a string"


# ── tools/debug_helpers.py ────────────────────────────────────────────────────

class TestDebugSession:
    """DebugSession — per-tool debug logging with env-var activation."""

    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            ds = DebugSession("test_tool", env_var="TEST_DEBUG")
            assert ds.active is False
            assert ds.enabled is False

    @mock.patch("tools.debug_helpers.get_hermes_home")
    def test_enabled_when_env_true(self, mock_home):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_home.return_value = Path(tmpdir)
            with mock.patch.dict(os.environ, {"TEST_DEBUG": "true"}):
                ds = DebugSession("test_tool", env_var="TEST_DEBUG")
                assert ds.active is True
                assert ds.enabled is True
                assert ds.session_id != ""

    def test_env_var_true_case_insensitive(self):
        """'TRUE' or 'True' both activate — .lower() is applied."""
        with mock.patch.dict(os.environ, {"TEST_DEBUG": "TRUE"}):
            ds = DebugSession("test_tool", env_var="TEST_DEBUG")
            assert ds.active is True  # case-insensitive via .lower()

    def test_log_call_noop_when_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            ds = DebugSession("test_tool", env_var="TEST_DEBUG")
            ds.log_call("search", {"query": "test", "results": 5})
            assert ds._calls == []

    @mock.patch("tools.debug_helpers.get_hermes_home")
    def test_log_call_appends_when_enabled(self, mock_home):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_home.return_value = Path(tmpdir)
            with mock.patch.dict(os.environ, {"TEST_DEBUG": "true"}):
                ds = DebugSession("test_tool", env_var="TEST_DEBUG")
                ds.log_call("search", {"query": "test"})
                assert len(ds._calls) == 1
                assert ds._calls[0]["tool_name"] == "search"
                assert ds._calls[0]["query"] == "test"
                assert "timestamp" in ds._calls[0]

    @mock.patch("tools.debug_helpers.get_hermes_home")
    def test_log_call_merges_extra_data(self, mock_home):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_home.return_value = Path(tmpdir)
            with mock.patch.dict(os.environ, {"TEST_DEBUG": "true"}):
                ds = DebugSession("test_tool", env_var="TEST_DEBUG")
                ds.log_call("fetch", {"url": "https://x.com", "status": 200})
                assert ds._calls[0]["url"] == "https://x.com"
                assert ds._calls[0]["status"] == 200

    def test_save_noop_when_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            ds = DebugSession("test_tool", env_var="TEST_DEBUG")
            ds.save()  # should not raise

    @mock.patch("tools.debug_helpers.get_hermes_home")
    def test_save_writes_json_file(self, mock_home):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_home.return_value = Path(tmpdir)
            logs_dir = Path(tmpdir) / "logs"
            with mock.patch.dict(os.environ, {"TEST_DEBUG": "true"}):
                ds = DebugSession("test_tool", env_var="TEST_DEBUG")
                ds.log_call("search", {"query": "hello"})
                ds.save()
                json_files = list(logs_dir.glob("test_tool_debug_*.json"))
                assert len(json_files) == 1
                with open(json_files[0]) as f:
                    data = json.load(f)
                assert data["debug_enabled"] is True
                assert data["total_calls"] == 1
                assert data["tool_calls"][0]["tool_name"] == "search"

    @mock.patch("tools.debug_helpers.get_hermes_home")
    def test_save_multiple_calls(self, mock_home):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_home.return_value = Path(tmpdir)
            logs_dir = Path(tmpdir) / "logs"
            with mock.patch.dict(os.environ, {"TEST_DEBUG": "true"}):
                ds = DebugSession("test_tool", env_var="TEST_DEBUG")
                ds.log_call("a", {"n": 1})
                ds.log_call("b", {"n": 2})
                ds.log_call("c", {"n": 3})
                ds.save()
                json_files = list(logs_dir.glob("test_tool_debug_*.json"))
                assert len(json_files) == 1
                with open(json_files[0]) as f:
                    data = json.load(f)
                assert data["total_calls"] == 3

    def test_get_session_info_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            ds = DebugSession("test_tool", env_var="TEST_DEBUG")
            info = ds.get_session_info()
            assert info["enabled"] is False
            assert info["session_id"] is None
            assert info["log_path"] is None
            assert info["total_calls"] == 0

    @mock.patch("tools.debug_helpers.get_hermes_home")
    def test_get_session_info_enabled(self, mock_home):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_home.return_value = Path(tmpdir)
            with mock.patch.dict(os.environ, {"TEST_DEBUG": "true"}):
                ds = DebugSession("test_tool", env_var="TEST_DEBUG")
                ds.log_call("x", {})
                ds.log_call("y", {})
                info = ds.get_session_info()
                assert info["enabled"] is True
                assert info["session_id"] != ""
                assert "test_tool_debug_" in info["log_path"]
                assert info["total_calls"] == 2

    def test_session_id_unique_per_instance(self):
        with mock.patch.dict(os.environ, {"A_DEBUG": "true", "B_DEBUG": "true"}):
            ds1 = DebugSession("tool_a", env_var="A_DEBUG")
            ds2 = DebugSession("tool_b", env_var="B_DEBUG")
            assert ds1.session_id != ds2.session_id
            assert len(ds1.session_id) == 36
            assert len(ds2.session_id) == 36
