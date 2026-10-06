"""Executed second-hop update dependencies must retain their real-update routes.

These are scoped call edges, not a claim that a recursive static import walk
identifies the executed dependency closure of every lazy branch.
"""

import sys
from pathlib import Path

import pytest

from tests.ci.test_update_ci_routing import _REPO, _ci_run, _consumers_reached, _real_classifier


# bounded_probe_run -> spawn_server, kill_process_tree -> deadline; the
# remaining edges are migrate_all_homes' provider/profile/install decisions.
# backup is the update transaction's pre-build step, not a build dependency.
@pytest.mark.parametrize("path", [
    "hermes_cli/local_runtime/processes.py",
    "agent/deadline.py",
    "agent/memory_provider.py",
    "pm/plugins_state.py",
    "pm/install.py",
    "hermes_cli/backup.py",
])
def test_second_hop_update_change_dispatches_real_update_consumers(path):
    lanes = _real_classifier([path])
    for lane in ("e2e_upgrade", "e2e_desktop_update"):
        assert lanes[lane], f"{path}: classifier leaves {lane} off"
    run = _ci_run(lanes)
    for lane in ("e2e_upgrade", "e2e_desktop_update"):
        assert all(_consumers_reached(run, lane).values()), f"{path}: {lane} never reaches its consumers"


# Off the update lanes (review K132346-ci-cost: 4 of 7 replayed main commits that newly started
# both update journeys touched only these): the journeys cannot observe code their update never runs.
_PLUGIN_INSTALL = ("hermes_cli/plugins_cmd.py", "hermes_cli/plugins_cmd_install.py")


def test_update_migration_on_a_journey_home_runs_no_plugin_install_code():
    """The journeys' homes configure no memory provider that left core. If the migration ever runs
    plugin-install code there, the journeys exercise it: route these modules again."""
    from hermes_cli.memory_provider_migration import migrate_all_homes

    ran: set[tuple[str, str]] = set()

    def profile(frame, event, _arg):
        if event != "call" or frame.f_code.co_name == "<module>":
            return
        # A class statement or module-level comprehension runs at import, called by its own
        # file's <module> frame: importing a module is not running its install code.
        caller = frame.f_back
        if caller and caller.f_code.co_name == "<module>" and caller.f_code.co_filename == frame.f_code.co_filename:
            return
        ran.add((Path(frame.f_code.co_filename).resolve().as_posix(), frame.f_code.co_name))

    sys.setprofile(profile)
    try:
        assert migrate_all_homes() == []
    finally:
        sys.setprofile(None)
    files = {f for f, _name in ran}
    assert (_REPO / "hermes_cli/memory_provider_migration.py").resolve().as_posix() in files
    hit = [rel for rel in _PLUGIN_INSTALL if (_REPO / rel).resolve().as_posix() in files]
    assert hit == [], f"the update runs {hit} on a journey home"
