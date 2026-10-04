"""Executed second-hop update dependencies must retain their real-update routes.

These are scoped call edges, not a claim that a recursive static import walk
identifies the executed dependency closure of every lazy branch.
"""

import pytest

from tests.ci.test_update_ci_routing import _ci_run, _consumers_reached, _real_classifier


# bounded_probe_run -> spawn_server, kill_process_tree -> deadline; the
# remaining edges are migrate_all_homes' provider/profile/install decisions.
# backup is the update transaction's pre-build step, not a build dependency.
@pytest.mark.parametrize("path", [
    "hermes_cli/local_runtime/processes.py",
    "agent/deadline.py",
    "agent/memory_provider.py",
    "hermes_cli/plugins_cmd_install.py",
    "hermes_cli/plugins_cmd.py",
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
