"""Fork upstream sync must ride the bounded network path (#93759 bug class).

``_sync_with_upstream_if_needed`` runs ``git fetch upstream`` and
``git pull --ff-only upstream`` — network operations that dead-stall on a
black-holed proxy. Every other network git step in the update path goes
through ``_git_run(..., network=True)`` which bounds the wait
(``NETWORK_GIT_TIMEOUT_SECONDS``); the sync's own fetch/pull previously ran a
bare ``subprocess.run`` and hung ``hermes update`` forever on a stalled
transport. These tests pin the invariant: the fetch and the pull are bounded
the same way the fork push already is.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from hermes_cli import update_cmd


def _fake_git_run(calls):
    """Stand-in for update_cmd._git_run: records kwargs, answers rev-list counts.

    First ``rev-list`` (origin ahead of upstream) reports 0, the second (upstream
    ahead of origin) reports 5 — the strictly-behind fork state that triggers
    pull + push.
    """
    counts = iter(["0", "5"])

    def run(git_cmd, args, cwd=None, *, check=False, network=False):
        calls.append({"args": list(args), "network": network})
        stdout = next(counts) if args[0] == "rev-list" else ""
        return subprocess.CompletedProcess(git_cmd + args, 0, stdout=stdout, stderr="")

    return run


def _sync_behind_fork(tmp_path):
    """Drive the sync with upstream present and origin strictly behind upstream."""
    calls: list[dict] = []
    with (
        patch.object(update_cmd, "_git_run", _fake_git_run(calls)),
        patch.object(update_cmd, "_has_upstream_remote", return_value=True),
    ):
        checked = update_cmd._sync_with_upstream_if_needed(["git"], tmp_path)
    return checked, calls


class TestUpstreamSyncNetworkBounded:
    def test_behind_fork_sync_is_verified(self, tmp_path):
        checked, _calls = _sync_behind_fork(tmp_path)
        assert checked is True

    def test_fetch_and_pull_are_network_bounded_like_the_push(self, tmp_path):
        _checked, calls = _sync_behind_fork(tmp_path)
        network_steps = [c for c in calls if c["args"][0] in {"fetch", "pull", "push"}]
        assert [c["args"][0] for c in network_steps] == ["fetch", "pull", "push"]
        assert all(c["network"] is True for c in network_steps), (
            "every upstream network step must opt into the bounded network path"
        )
