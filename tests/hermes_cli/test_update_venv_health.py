"""Tests for the Windows half-updated-venv hardening (July 2026 incident).

Covers the commit_count == 0 repair branch: an "Already up to date" checkout
must still re-sync a venv whose installed distribution lags the checkout.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


from hermes_cli import main as cli_main
from hermes_cli import update_cmd


# ---------------------------------------------------------------------------
# _venv_core_imports_healthy
# ---------------------------------------------------------------------------




def _fake_venv_python(tmp_path, *, windows: bool = False):
    bin_dir = tmp_path / "venv" / ("Scripts" if windows else "bin")
    bin_dir.mkdir(parents=True)
    py = bin_dir / ("python.exe" if windows else "python")
    py.write_bytes(b"")
    return py




# ---------------------------------------------------------------------------
# "Already up to date" must not hide a venv the last pull never re-synced (#97208)
# ---------------------------------------------------------------------------


def test_venv_dependency_set_stale_compares_installed_distribution_with_checkout(tmp_path):
    """Reporter's state: git current at 0.21.2, venv metadata still 0.20.6 (the hand-off child
    refused the sync); core imports pass, so only the distribution version reveals the drift."""
    from hermes_cli import update_cmd_deps

    (tmp_path / "pyproject.toml").write_text('[project]\nname = "hermes-agent"\nversion = "0.21.2"\n', encoding="utf-8")
    venv_python = _fake_venv_python(tmp_path)
    probes = []

    def fake_run(cmd, **kwargs):
        probes.append(cmd)
        return SimpleNamespace(returncode=0, stdout=f"{fake_run.installed}\n", stderr="")

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(cli_main, "_is_windows", return_value=False), \
            patch.object(update_cmd_deps.subprocess, "run", fake_run):
        fake_run.installed = "0.20.6"
        assert update_cmd._venv_dependency_set_stale() == (True, "installed hermes-agent 0.20.6, checkout is 0.21.2")
        fake_run.installed = "0.21.2"
        assert update_cmd._venv_dependency_set_stale() == (False, "")
    # Asked in the venv's own interpreter (the updater may run under another Python).
    assert {cmd[0] for cmd in probes} == {str(venv_python)}


def test_current_checkout_with_stale_dependency_set_runs_the_sync(monkeypatch, capsys):
    """Drift on the commit_count == 0 path triggers the same repair as an unhealthy venv instead
    of ``✓ Already up to date!``; a synced venv keeps the cheap node-only path."""
    from hermes_cli import main as hm

    calls = []
    monkeypatch.setattr(update_cmd, "_venv_core_imports_healthy", lambda: (True, ""))
    monkeypatch.setattr(hm, "_is_windows", lambda: False)
    monkeypatch.delenv(hm._UPDATE_REEXEC_ENV, raising=False)
    monkeypatch.setattr("hermes_cli.managed_uv.update_managed_uv", lambda **kwargs: None)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda **kwargs: "uv")
    monkeypatch.setattr(update_cmd, "_repair_venv_on_current_checkout",
                        lambda **kwargs: calls.append("sync") or True)
    monkeypatch.setattr(update_cmd, "_repair_node_deps_on_current_checkout",
                        lambda *a, **kwargs: calls.append(kwargs["completion_message"]) or True)

    def run(stale):
        monkeypatch.setattr(update_cmd, "_venv_dependency_set_stale", lambda: stale)
        return update_cmd._repair_current_checkout(
            assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None,
            had_desktop_app_before_update=False, active_lazy_features=[], active_tool_dependencies=[],
            upstream_checked=True, _windows_gateway_resume=None)

    assert run((True, "installed hermes-agent 0.20.6, checkout is 0.21.2")) is True
    assert calls == ["sync"]
    assert run((False, "")) is True
    assert calls == ["sync", "✓ Already up to date!"]
