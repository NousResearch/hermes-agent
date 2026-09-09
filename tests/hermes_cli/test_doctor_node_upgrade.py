"""hermes doctor --upgrade-node[=MAJOR]. See issue #106456."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from hermes_cli import doctor_node_upgrade


def test_bare_flag_targets_the_default_upgrade_major(monkeypatch):
    monkeypatch.setattr(doctor_node_upgrade, "engines_node_default_upgrade_major", lambda: 26)
    monkeypatch.setattr(doctor_node_upgrade, "engines_node_allows_major", lambda major: major == 26)
    monkeypatch.setattr(doctor_node_upgrade, "_current_managed_major", lambda: 22)
    with patch.object(doctor_node_upgrade, "_install_target_major", return_value=True) as install, \
         patch.object(doctor_node_upgrade, "_update_node_dependencies", return_value=[]) as rebuild:
        doctor_node_upgrade.upgrade_node("<newest>")
    install.assert_called_once_with(26)
    rebuild.assert_called_once_with(force=True)


def test_explicit_major_outside_engines_node_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(doctor_node_upgrade, "engines_node_allows_major", lambda major: major in (22, 24))
    monkeypatch.setattr(doctor_node_upgrade, "_current_managed_major", lambda: 22)
    with pytest.raises(SystemExit) as exc_info:
        doctor_node_upgrade.upgrade_node("23")
    assert exc_info.value.code == 2
    assert "HERMES_NODE_TARGET_MAJOR" in capsys.readouterr().out

def test_already_on_target_major_is_a_no_op(monkeypatch, capsys):
    monkeypatch.setattr(doctor_node_upgrade, "engines_node_default_upgrade_major", lambda: 22)
    monkeypatch.setattr(doctor_node_upgrade, "engines_node_allows_major", lambda major: major == 22)
    monkeypatch.setattr(doctor_node_upgrade, "_current_managed_major", lambda: 22)
    with patch.object(doctor_node_upgrade, "_install_target_major") as install:
        doctor_node_upgrade.upgrade_node("<newest>")
    install.assert_not_called()
    assert "already on Node 22" in capsys.readouterr().out


def test_install_failure_rebuilds_nothing_and_exits_nonzero(monkeypatch):
    monkeypatch.setattr(doctor_node_upgrade, "engines_node_default_upgrade_major", lambda: 24)
    monkeypatch.setattr(doctor_node_upgrade, "engines_node_allows_major", lambda major: major == 24)
    monkeypatch.setattr(doctor_node_upgrade, "_current_managed_major", lambda: 22)
    with patch.object(doctor_node_upgrade, "_install_target_major", return_value=False), \
         patch.object(doctor_node_upgrade, "_update_node_dependencies") as rebuild:
        with pytest.raises(SystemExit) as exc_info:
            doctor_node_upgrade.upgrade_node("<newest>")
    assert exc_info.value.code == 1
    rebuild.assert_not_called()


def test_non_numeric_explicit_major_is_refused_without_crashing(monkeypatch, capsys):
    """A typo like `--upgrade-node=banana` or `--upgrade-node=24.5` must hit the
    "not a Node major version number" branch (int() raising ValueError), not an uncaught
    exception, and must not attempt an install."""
    with patch.object(doctor_node_upgrade, "_install_target_major") as install:
        for bad_arg in ("banana", "24.5", "", "0x18"):
            with pytest.raises(SystemExit) as exc_info:
                doctor_node_upgrade.upgrade_node(bad_arg)
            assert exc_info.value.code == 2, bad_arg
    install.assert_not_called()
    assert "not a Node major version number" in capsys.readouterr().out


def test_explicit_major_matching_current_is_also_a_no_op(monkeypatch, capsys):
    """The already-on-target no-op isn't only reachable via the bare-flag/default-major path —
    an explicit `--upgrade-node=22` when already on 22 must short-circuit the same way, without
    the tests above (which only exercise the "<newest>" sentinel) ever having proven that."""
    monkeypatch.setattr(doctor_node_upgrade, "engines_node_allows_major", lambda major: major == 22)
    monkeypatch.setattr(doctor_node_upgrade, "_current_managed_major", lambda: 22)
    with patch.object(doctor_node_upgrade, "_install_target_major") as install:
        doctor_node_upgrade.upgrade_node("22")
    install.assert_not_called()
    assert "already on Node 22" in capsys.readouterr().out


def test_validation_runs_before_the_already_on_target_check(monkeypatch, capsys):
    """An explicit MAJOR that engines.node doesn't allow must be refused even when it happens to
    equal the currently-installed major (e.g. a stray major installed earlier via the unvalidated
    HERMES_NODE_TARGET_MAJOR override) — validation must not be skipped just because there'd be
    nothing to do otherwise."""
    monkeypatch.setattr(doctor_node_upgrade, "engines_node_allows_major", lambda major: major in (22, 24))
    monkeypatch.setattr(doctor_node_upgrade, "_current_managed_major", lambda: 23)
    with patch.object(doctor_node_upgrade, "_install_target_major") as install:
        with pytest.raises(SystemExit) as exc_info:
            doctor_node_upgrade.upgrade_node("23")
    assert exc_info.value.code == 2
    install.assert_not_called()
    assert "does not allow that major" in capsys.readouterr().out


class TestInstallTargetMajorDispatch:
    """`_install_target_major` itself has no direct coverage above — every other test
    patches it away entirely. Its POSIX/Windows branch selection is the actual wiring that
    determines which real installer runs, so it needs its own tests."""

    def test_posix_path_calls_run_node_bootstrap_with_target_major(self, monkeypatch):
        monkeypatch.setattr(doctor_node_upgrade.sys, "platform", "linux")
        with patch("hermes_constants._run_node_bootstrap", return_value=True) as bootstrap:
            result = doctor_node_upgrade._install_target_major(24)
        assert result is True
        bootstrap.assert_called_once_with(
            "_nb_install_bundled_node", timeout=600, HERMES_NODE_TARGET_MAJOR="24")

    def test_windows_path_calls_heal_managed_node_windows_with_target_major(self, monkeypatch):
        monkeypatch.setattr(doctor_node_upgrade.sys, "platform", "win32")
        with patch("hermes_constants._heal_managed_node_windows", return_value=True) as heal:
            result = doctor_node_upgrade._install_target_major(24)
        assert result is True
        heal.assert_called_once_with(target_major=24)

    def test_windows_deferred_in_use_result_is_treated_as_failure(self, monkeypatch):
        """`_heal_managed_node_windows` can return `None` (deferred: tree in use), which is not a
        success — `_install_target_major` coerces it with `bool(...)`, so a caller checking
        `if not _install_target_major(...)` must see this as a failure to retry later, not crash
        on a truthy/falsy `None` mixup or silently report success."""
        monkeypatch.setattr(doctor_node_upgrade.sys, "platform", "win32")
        with patch("hermes_constants._heal_managed_node_windows", return_value=None):
            assert doctor_node_upgrade._install_target_major(24) is False
