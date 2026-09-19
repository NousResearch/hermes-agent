"""The already-current update path must honor a pending ``.update-incomplete`` marker (#103532).

A prior ``hermes update`` whose dependency install failed (e.g. PyPI unreachable) leaves the
marker behind. The next ``hermes update`` reaches the already-current branch, where the
4-module import probe still passes — an editable install resolves the probed modules against
the new checkout while the venv stays on stale pins. Without consulting the marker, the
update printed "✓ Already up to date!" with exit 0 and a success receipt, masking the skew.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_cli import update_cmd


def _run_repair_current_checkout(tmp_path: Path, *, marker: bool):
    """Drive ``_repair_current_checkout`` with an inert environment; returns what ran."""
    marker_path = tmp_path / ".update-incomplete"
    if marker:
        marker_path.write_text("started=probe pid=0\n")
    calls = {"venv_repair": False, "node_repair": False}
    with patch.object(update_cmd, "_venv_core_imports_healthy", return_value=(True, "")), \
            patch.object(update_cmd, "_repair_venv_on_current_checkout") as venv_repair, \
            patch.object(update_cmd, "_repair_node_deps_on_current_checkout") as node_repair, \
            patch("hermes_cli.managed_uv.ensure_uv"), \
            patch("hermes_cli.managed_uv.update_managed_uv"), \
            patch.object(update_cmd, "_m") as m:
        m.return_value._update_marker_path = lambda: marker_path
        m.return_value._UPDATE_REEXEC_ENV = "HERMES_UPDATE_REEXEC"
        venv_repair.return_value = True
        node_repair.return_value = True
        result = update_cmd._repair_current_checkout(
            assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None,
            desktop_dir=None, had_desktop_app_before_update=False,
            active_lazy_features=[], active_tool_dependencies=[],
            upstream_checked=True, _windows_gateway_resume=None)
        calls["venv_repair"] = venv_repair.called
        calls["node_repair"] = node_repair.called
    assert result is True
    return calls


def test_pending_marker_routes_to_venv_repair_despite_healthy_probe(tmp_path):
    """Marker present + probe green ⇒ repair the venv, never report 'Already up to date!'."""
    calls = _run_repair_current_checkout(tmp_path, marker=True)
    assert calls["venv_repair"] is True
    assert calls["node_repair"] is False


def test_clean_checkout_keeps_already_up_to_date_path(tmp_path):
    """No marker + probe green ⇒ the node-deps catch-up path still reports up to date."""
    calls = _run_repair_current_checkout(tmp_path, marker=False)
    assert calls["venv_repair"] is False
    assert calls["node_repair"] is True


def test_pending_marker_prints_hint(capsys, tmp_path):
    """The marker branch surfaces why the repair is running."""
    _run_repair_current_checkout(tmp_path, marker=True)
    out = capsys.readouterr().out
    assert "previous dependency install did not finish" in out
    assert ".update-incomplete" in out
