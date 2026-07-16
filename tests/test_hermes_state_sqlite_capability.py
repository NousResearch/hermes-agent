"""Tests for hermes_state.py SQLite capability warning.

Covers the one-shot startup warning that surfaces when the runtime SQLite
lacks the FTS5 trigram tokenizer (issue #41030). Without this warning,
operators on RHEL 8 / Rocky 8 / Alma 8 / Amazon Linux 2 silently lose
substring + CJK session search.
"""

import logging
import sqlite3

import pytest

import hermes_state


@pytest.fixture(autouse=True)
def _reset_capability_warning():
    """Re-arm the one-shot capability check before and after every test."""
    hermes_state._reset_sqlite_capability_warning_for_tests()
    yield
    hermes_state._reset_sqlite_capability_warning_for_tests()


class TestSqliteCapabilityWarning:
    """The capability check must emit exactly one actionable warning when
    trigram FTS5 is unavailable, and stay silent otherwise."""

    def test_warns_when_sqlite_below_trigram_minimum(self, caplog, monkeypatch):
        """SQLite 3.26 (RHEL 8) must produce one warning with remediation."""
        monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version", "3.26.0")
        monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version_info", (3, 26, 0))

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            hermes_state._check_sqlite_capabilities()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f"expected 1 warning, got {len(warnings)}"
        msg = warnings[0].getMessage()
        assert "3.26.0" in msg
        assert "trigram" in msg
        assert "modern-sqlite" in msg, "warning must point at the supported extra"
        assert "pip install" in msg, "warning must include the install command"

    def test_silent_on_modern_sqlite_with_trigram(self, caplog, monkeypatch):
        """A new-enough SQLite with working trigram must produce zero warnings."""
        monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version", "3.45.0")
        monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version_info", (3, 45, 0))
        # Force the probe to succeed without depending on the host's real
        # SQLite build (CI images vary).
        monkeypatch.setattr(hermes_state, "_probe_trigram_tokenizer", lambda: None)

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            hermes_state._check_sqlite_capabilities()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == [], f"unexpected warnings: {[r.getMessage() for r in warnings]}"

    def test_warns_when_modern_sqlite_omits_trigram(self, caplog, monkeypatch):
        """Custom SQLite build that omits trigram (rare) must produce one warning."""
        monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version", "3.45.0")
        monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version_info", (3, 45, 0))
        monkeypatch.setattr(
            hermes_state,
            "_probe_trigram_tokenizer",
            lambda: "no such tokenizer: trigram",
        )

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            hermes_state._check_sqlite_capabilities()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "tokenizer" in msg
        assert "no such tokenizer" in msg

    def test_warning_fires_only_once_per_process(self, caplog, monkeypatch):
        """Multiple calls must collapse to a single log line. This protects
        against startup-warning spam from import cycles or worker fan-out."""
        monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version", "3.26.0")
        monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version_info", (3, 26, 0))

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            hermes_state._check_sqlite_capabilities()
            hermes_state._check_sqlite_capabilities()
            hermes_state._check_sqlite_capabilities()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, "capability warning must be idempotent"

    def test_reset_helper_re_arms_the_warning(self, caplog, monkeypatch):
        """The test-only reset hook must let a second warning emit."""
        monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version", "3.26.0")
        monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version_info", (3, 26, 0))

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            hermes_state._check_sqlite_capabilities()
            hermes_state._reset_sqlite_capability_warning_for_tests()
            hermes_state._check_sqlite_capabilities()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2

    def test_check_never_raises_even_if_probe_explodes(self, caplog, monkeypatch):
        """Capability reporting must never break SessionDB init. If the probe
        itself raises an unexpected error, the check must still return cleanly."""
        monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version", "3.45.0")
        monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version_info", (3, 45, 0))

        def _explode():
            raise RuntimeError("synthetic failure")

        monkeypatch.setattr(hermes_state, "_probe_trigram_tokenizer", _explode)

        # Currently the check propagates non-OperationalError out of the probe.
        # We tolerate either behaviour as long as a real-world OperationalError
        # path stays clean. Confirm RuntimeError does NOT crash the warning
        # path by wrapping the call.
        try:
            hermes_state._check_sqlite_capabilities()
        except RuntimeError:
            pytest.fail("capability check must isolate probe failures")


class TestPysqlite3Shim:
    """The auto-shim must be opt-in (only when pysqlite3 is installed) and
    must respect HERMES_DISABLE_PYSQLITE3_SHIM."""

    def test_stdlib_sqlite3_used_when_pysqlite3_not_installed(self):
        """Default install path: sqlite3 module is the stdlib one."""
        # Best-effort sanity: when pysqlite3 isn't importable, the in-process
        # sqlite3 module must be the stdlib version. We don't try to uninstall
        # it inside the test; instead we assert the swap only happened if the
        # extra is actually present.
        try:
            import pysqlite3  # noqa: F401
            shim_active = hermes_state.sqlite3 is pysqlite3
            assert shim_active, (
                "when pysqlite3 is installed, hermes_state.sqlite3 must be the shim"
            )
        except ImportError:
            assert hermes_state.sqlite3 is sqlite3, (
                "when pysqlite3 is not installed, hermes_state.sqlite3 must be stdlib"
            )

    def test_probe_returns_string_or_none(self):
        """The probe contract: None on success, str on failure. Whichever
        path the host takes, the return type must match."""
        result = hermes_state._probe_trigram_tokenizer()
        assert result is None or isinstance(result, str)
