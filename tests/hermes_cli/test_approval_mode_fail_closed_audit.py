"""Approvals.mode fail-closed default + audit logging (issue #84547).

Two independent defects were fixed together:

(a) The shipped default for ``approvals.mode`` was the PERMISSIVE value
    (``smart``). Because ``load_config()`` deep-merges DEFAULT_CONFIG, an
    absent key silently resolved to LLM-adjudicated approvals — an operator
    who deliberately set ``manual`` could be returned to ``smart`` with
    nothing to notice it by. The default is now ``manual`` (fail closed):
    absence resolves to the restrictive value.

(b) Changes to ``approvals.mode`` were never audit-logged. Every change is
    now recorded in ``$HERMES_HOME/logs/approvals.log`` (who, when, from
    which mode to which) — both explicit writes (CLI ``hermes config set``,
    ``/approvals``, TUI ``config.set``) and silent transitions detected at
    read time (hand edits, re-serializations that drop the key).
"""

import json
import os
from unittest.mock import patch

import pytest

from hermes_cli.config import DEFAULT_CONFIG, set_config_value


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir and drop config caches."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.config import _LOAD_CONFIG_CACHE, _RAW_CONFIG_CACHE

    _LOAD_CONFIG_CACHE.clear()
    _RAW_CONFIG_CACHE.clear()
    return tmp_path


def _read_audit_log(home):
    path = home / "logs" / "approvals.log"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _approval_mode():
    from tools.approval import _get_approval_mode

    return _get_approval_mode()


def _write_config(home, text):
    (home / "config.yaml").write_text(text, encoding="utf-8")
    from hermes_cli.config import _LOAD_CONFIG_CACHE, _RAW_CONFIG_CACHE

    _LOAD_CONFIG_CACHE.clear()
    _RAW_CONFIG_CACHE.clear()


# ---------------------------------------------------------------------------
# (a) fail-closed default
# ---------------------------------------------------------------------------


class TestFailClosedDefault:
    def test_shipped_default_is_manual(self):
        # The schema default must be the restrictive value, not smart.
        assert DEFAULT_CONFIG["approvals"]["mode"] == "manual"

    def test_absent_key_resolves_to_manual(self, _isolated_home):
        # No config.yaml at all -> merged default -> manual (fail closed).
        assert _approval_mode() == "manual"

    def test_key_missing_from_approvals_block_resolves_to_manual(self, _isolated_home):
        # The exact repro from the issue: the approvals block exists but the
        # mode key was dropped (e.g. by a strip_defaults re-serialization).
        _write_config(_isolated_home, "approvals:\n  timeout: 300\n")
        assert _approval_mode() == "manual"

    def test_explicit_manual_is_respected(self, _isolated_home):
        _write_config(_isolated_home, "approvals:\n  mode: manual\n")
        assert _approval_mode() == "manual"

    def test_explicit_smart_is_still_allowed(self, _isolated_home):
        # Fail-closed default must not break the documented opt-in.
        _write_config(_isolated_home, "approvals:\n  mode: smart\n")
        assert _approval_mode() == "smart"

    def test_explicit_off_is_still_allowed(self, _isolated_home):
        _write_config(_isolated_home, "approvals:\n  mode: off\n")
        assert _approval_mode() == "off"


# ---------------------------------------------------------------------------
# (b) audit logging — explicit write paths
# ---------------------------------------------------------------------------


class TestWritePathAudit:
    def test_set_config_value_audits_mode_change(self, _isolated_home):
        _write_config(_isolated_home, "approvals:\n  mode: manual\n")
        set_config_value("approvals.mode", "off")

        entries = _read_audit_log(_isolated_home)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["event"] == "approvals.mode_changed"
        assert entry["old_mode"] == "manual"
        assert entry["new_mode"] == "off"
        assert entry["source"] == "cli-config-set"
        assert entry["actor"]  # who — never empty

    def test_set_config_value_same_mode_is_not_audited(self, _isolated_home):
        _write_config(_isolated_home, "approvals:\n  mode: manual\n")
        set_config_value("approvals.mode", "manual")
        assert _read_audit_log(_isolated_home) == []

    def test_approvals_command_audits_change_without_duplicate_observer_line(
        self, _isolated_home
    ):
        # The /approvals command writes through set_config_value; the
        # runtime observer must NOT re-log the same change (it was already
        # audited on the write path).
        from hermes_cli.approval_mode import run_approval_mode_command

        _write_config(_isolated_home, "approvals:\n  mode: manual\n")
        result = run_approval_mode_command("smart")

        assert result.ok is True
        entries = _read_audit_log(_isolated_home)
        assert len(entries) == 1
        assert entries[0]["old_mode"] == "manual"
        assert entries[0]["new_mode"] == "smart"
        assert entries[0]["source"] == "cli-config-set"

    def test_tui_write_path_audits_mode_change(self, _isolated_home, monkeypatch):
        import tui_gateway.server as server

        monkeypatch.setattr(server, "_hermes_home", _isolated_home)
        _write_config(_isolated_home, "approvals:\n  mode: manual\n")
        server._write_config_key("approvals.mode", "off")

        entries = _read_audit_log(_isolated_home)
        assert len(entries) == 1
        assert entries[0]["old_mode"] == "manual"
        assert entries[0]["new_mode"] == "off"
        assert entries[0]["source"] == "tui-config-set"


# ---------------------------------------------------------------------------
# (b) audit logging — silent transitions detected at read time
# ---------------------------------------------------------------------------


class TestRuntimeTransitionObserver:
    def test_silent_transition_is_audited_and_warned(self, _isolated_home, caplog, monkeypatch):
        # Simulate the issue's scenario: operator set manual; config is then
        # re-serialized with the mode silently flipped. The write path never
        # runs — only the read-time observer can catch it.
        import tools.approval as approval

        monkeypatch.setattr(approval, "_get_approval_config", lambda: {"mode": "manual"})
        assert approval._get_approval_mode() == "manual"

        monkeypatch.setattr(approval, "_get_approval_config", lambda: {"mode": "smart"})
        with caplog.at_level("WARNING", logger="tools.approval"):
            assert approval._get_approval_mode() == "smart"

        assert any("approvals.mode changed" in record.message for record in caplog.records)
        entries = _read_audit_log(_isolated_home)
        assert len(entries) == 1
        assert entries[0]["source"] == "effective-mode-observer"
        assert entries[0]["old_mode"] == "manual"
        assert entries[0]["new_mode"] == "smart"

    def test_first_observation_is_a_baseline_not_a_transition(self, _isolated_home):
        import tools.approval as approval

        with patch.object(approval, "_get_approval_config", lambda: {"mode": "smart"}):
            assert approval._get_approval_mode() == "smart"
        assert _read_audit_log(_isolated_home) == []

    def test_explicit_write_suppresses_observer_duplicate(self, _isolated_home):
        # A change made through the write path is audited once there; the
        # observer sees the new value on the next read and must stay quiet.
        import tools.approval as approval

        _write_config(_isolated_home, "approvals:\n  mode: manual\n")
        assert approval._get_approval_mode() == "manual"
        set_config_value("approvals.mode", "smart")
        assert approval._get_approval_mode() == "smart"

        entries = _read_audit_log(_isolated_home)
        assert len(entries) == 1
        assert entries[0]["source"] == "cli-config-set"
