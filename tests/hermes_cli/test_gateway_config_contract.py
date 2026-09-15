"""Tests for gateway run.py/config.py cross-file attribute contract validation.

The import-time probe in ``_restore_stashed_changes`` stays green when the
stash restore brings ``gateway/run.py`` and ``gateway/config.py`` from
different commits: the module imports cleanly, and the AttributeError only
fires inside ``GatewayRunner.__init__`` at gateway start. These tests cover the
AST contract check that catches this class of mismatch (issue #105806).
"""

import ast
from pathlib import Path

import pytest

from hermes_cli import gateway_config_contract as gcc


# ---------------------------------------------------------------------------
# Unit tests for the AST extractors
# ---------------------------------------------------------------------------

def _parse_str(src: str) -> ast.Module:
    return ast.parse(src)


class TestExtractConfigAttrReads:
    def test_self_config_attr_reads(self):
        src = """
class GatewayRunner:
    def __init__(self):
        self.config.sessions_dir
        _ = self.config.reset_triggers
    def _init_runtime_settings(self):
        return self.config.channel_overrides
"""
        attrs = gcc.extract_config_attr_reads(_parse_str(src))
        assert attrs == {"sessions_dir", "reset_triggers", "channel_overrides"}

    def test_mixin_methods_covered(self):
        """self.config reads in mixin-style classes must be captured too."""
        src = """
class GatewayRunner(GatewayAuthorizationMixin, GatewayShutdownMixin):
    def __init__(self):
        self.config = None

class GatewayAuthorizationMixin:
    def authorize(self):
        return self.config.max_concurrent_sessions
"""
        # Only the GatewayRunner class body is walked; the mixin class defined
        # elsewhere in the same file is NOT part of GatewayRunner.
        attrs = gcc.extract_config_attr_reads(_parse_str(src))
        assert attrs == set()  # no self.config reads inside GatewayRunner itself

    def test_plain_config_attr_reads(self):
        src = """
class GatewayRunner:
    def start(self, config):
        return config.systemd_watchdog_seconds
"""
        attrs = gcc.extract_config_attr_reads(_parse_str(src))
        assert attrs == {"systemd_watchdog_seconds"}

    def test_method_calls_not_attrs(self):
        """config.get_connected_platforms() is a method call — attr is
        get_connected_platforms which must resolve to a GatewayConfig method."""
        src = """
class GatewayRunner:
    def check(self):
        return self.config.get_connected_platforms()
"""
        attrs = gcc.extract_config_attr_reads(_parse_str(src))
        assert attrs == {"get_connected_platforms"}


class TestExtractGatewayConfigMembers:
    def test_dataclass_fields_and_methods(self):
        src = """
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class GatewayConfig:
    platforms: Dict[str, Any] = field(default_factory=dict)
    sessions_dir: Path = field(default_factory=Path)
    write_sessions_json: bool = True
    def get_connected_platforms(self):
        return []
"""
        members = gcc.extract_gateway_config_members(_parse_str(src))
        assert members == {"platforms", "sessions_dir", "write_sessions_json", "get_connected_platforms"}

    def test_tuple_assignment_targets(self):
        """ast.Assign uses .targets (plural) — tuple unpacking must be handled."""
        src = """
@dataclass
class GatewayConfig:
    a: int = 0
"""
        members = gcc.extract_gateway_config_members(_parse_str(src))
        assert "a" in members


# ---------------------------------------------------------------------------
# Integration test: the contract check against real gateway files
# ---------------------------------------------------------------------------

def test_real_gateway_contract_consistent(hermes_agent_root: Path):
    """The actual gateway/run.py + gateway/config.py in the repo must be
    consistent (no missing attrs)."""
    missing = gcc.check_gateway_config_contract(hermes_agent_root)
    assert missing is None, f"expected consistent contract, got missing: {missing}"


def test_real_gateway_contract_detects_mismatch(hermes_agent_root: Path, tmp_path: Path, monkeypatch):
    """A stale run.py referencing a dropped config attr must be flagged."""
    run_py = hermes_agent_root / "gateway" / "run.py"
    src = run_py.read_text(encoding="utf-8")
    injected = src.replace(
        "def __init__(self, config: Optional[GatewayConfig] = None):",
        "def __init__(self, config: Optional[GatewayConfig] = None):\n"
        "        _ = self.config.default_reset_policy  # test injection",
        1,
    )
    assert injected != src, "could not find __init__ to inject into"

    # Stage the injected run.py in a temp tree alongside the real config.py
    staged = tmp_path / "gateway"
    staged.mkdir()
    (staged / "run.py").write_text(injected, encoding="utf-8")
    (staged / "config.py").write_text(
        (hermes_agent_root / "gateway" / "config.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    missing = gcc.check_gateway_config_contract(tmp_path)
    assert missing == ["default_reset_policy"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hermes_agent_root() -> Path:
    """Repo root (tests run from repo root or tests/hermes_cli)."""
    return Path(__file__).resolve().parents[2]
