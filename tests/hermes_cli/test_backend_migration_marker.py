"""A backend migration's ``backend_migration.json`` keeps a retired old backend down across ``hermes update``.

``hermes update``'s managed-dashboard step started an inactive ``hermes-dashboard.service`` whose unit
file reads ``static`` (no ``[Install]`` section, so ``systemctl disable`` leaves it ``static``): after a
migration stopped it, the next update brought it back beside the gateway. With the migration's
``backend_migration.json`` in a retiring state the step leaves a stopped unit stopped. That test answers
for ``systemctl`` the way ``test_update_stale_dashboard.py`` does (a real unit would be the host's); the
guest run in the track-2 report drives the real one.
"""
import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize("state, retired", [("done", True), ("running", True), ("failed", True),
                                            ("rolled_back", False), ("restored", False)])
def test_the_migration_manifest_state_decides_whether_the_old_backend_stays_down(tmp_path, state, retired):
    from hermes_cli.backend_migration_marker import backend_retired_by_migration
    (tmp_path / "backend_migration.json").write_text(json.dumps({"state": state, "step": "x"}))
    assert backend_retired_by_migration(tmp_path) is retired


def test_no_manifest_changes_nothing(tmp_path):
    from hermes_cli.backend_migration_marker import backend_retired_by_migration
    assert backend_retired_by_migration(tmp_path) is False


@pytest.mark.parametrize("content", ["{not json", "[]", '{"step": "write_config"}'])
def test_an_unreadable_manifest_keeps_the_old_backend_down(tmp_path, content):
    """A migration may have stopped half-way: nothing starts the old backend on a guess."""
    from hermes_cli.backend_migration_marker import backend_retired_by_migration
    (tmp_path / "backend_migration.json").write_text(content)
    assert backend_retired_by_migration(tmp_path) is True


def _systemctl(active: str, enabled: str, calls: list):
    def run(args, *a, **kw):
        calls.append(list(args))
        if args[:3] == ["systemctl", "--user", "list-unit-files"]:
            return MagicMock(returncode=0, stdout=f"hermes-dashboard.service {enabled}\n", stderr="")
        if args[:3] == ["systemctl", "--user", "is-active"]:
            return MagicMock(returncode=3, stdout=f"{active}\n", stderr="")
        if args[:3] == ["systemctl", "--user", "is-enabled"]:
            return MagicMock(returncode=0, stdout=f"{enabled}\n", stderr="")
        if args[:3] == ["systemctl", "--user", "restart"]:
            return MagicMock(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected subprocess.run call: {args}")
    return run


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("manifest, restarted", [(None, True), ("done", False), ("rolled_back", True)])
def test_update_starts_a_stopped_static_dashboard_unit_only_without_a_retiring_migration(
        manifest, restarted, capsys):
    from hermes_cli.dashboard_procs import _kill_stale_dashboard_processes
    from hermes_constants import get_hermes_home
    if manifest is not None:
        (get_hermes_home() / "backend_migration.json").write_text(json.dumps({"state": manifest}))
    calls: list = []
    with patch("subprocess.run", side_effect=_systemctl("inactive", "static", calls)), \
         patch("hermes_cli.main_dashboard._find_stale_dashboard_pids", return_value=[]):
        _kill_stale_dashboard_processes(restart_managed=True)
    assert (["systemctl", "--user", "restart", "hermes-dashboard.service"] in calls) is restarted
    out = capsys.readouterr().out
    assert ("stays stopped" in out) is (not restarted)
