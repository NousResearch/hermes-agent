"""Tests for interrupted-install self-heal (the ``.update-incomplete`` marker).

Covers the breadcrumb lifecycle so a ``hermes update`` killed mid-install
(Ctrl-C, terminal close, WSL OOM) leaves a marker the next launch can act on,
and the boot recovery that finishes it (#81594: bounded, never a reinstall per boot).
"""

from __future__ import annotations

import pytest

import hermes_cli.main as m
from hermes_cli import _early_recovery as er
from hermes_cli import _install_repair as ir
from hermes_cli import main_install_repair as mir


def test_marker_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    marker = m._update_marker_path()
    assert marker == tmp_path / ".update-incomplete"
    assert not marker.exists()

    m._write_update_incomplete_marker()
    assert marker.exists()
    body = marker.read_text()
    assert "started=" in body
    assert "pid=" in body

    m._clear_update_incomplete_marker()
    assert not marker.exists()


@pytest.fixture
def pending_core_install(tmp_path, monkeypatch):
    """A source checkout whose last ``hermes update`` (owner now dead) left the core install
    pending; ``installs`` records every reinstall the boot recovery starts."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
    marker = tmp_path / ".update-incomplete"
    marker.write_text("started=1\npid=999999\n", encoding="utf-8")
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(er, "_marker_owner_is_live", lambda _marker: False)
    monkeypatch.setattr(mir, "_windows_running_hermes_launcher_locked", lambda: False)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda *a, **k: None)
    installs: list[str] = []
    outcome = {"fail": True}

    def install(root):
        installs.append(str(root))
        if outcome["fail"]:
            raise RuntimeError("os error 5: _cffi_backend.pyd is in use")

    monkeypatch.setattr(ir, "run_core_install", install)
    return tmp_path, marker, installs, outcome


def _launch(root):
    """One ``hermes`` boot: the pre-import early pass, then main()'s post-import recovery."""
    er.recover_if_needed(project_root=root, argv=[])
    m._recover_from_interrupted_install()


def test_persistently_failing_install_stops_rerunning_every_boot(pending_core_install, capsys):
    """#81594: once the early pass stood down, the post-import path reinstalled on every
    launch forever. Both paths share one attempt budget: after it, boots stop reinstalling
    and print the exact recovery command instead."""
    root, marker, installs, _ = pending_core_install
    for _ in range(4):
        _launch(root)
    spent = len(installs)
    capsys.readouterr()

    for _ in range(4):
        _launch(root)

    assert len(installs) == spent, "boots past the attempt budget must not reinstall"
    assert marker.exists()
    output = capsys.readouterr()
    shown = output.out + output.err
    assert "pip install -e" in shown
    assert str(marker) in shown, "the recovery command must say how to clear the marker"


def test_post_import_recovery_counts_failures_and_clears_marker_on_success(pending_core_install):
    root, marker, installs, outcome = pending_core_install

    m._recover_from_interrupted_install()
    assert marker.exists()
    assert er._read_marker_attempts(marker) == 1, "a late-path failure spends one attempt"

    outcome["fail"] = False
    m._recover_from_interrupted_install()
    assert len(installs) == 2
    assert not marker.exists(), "a successful recovery clears the marker"
