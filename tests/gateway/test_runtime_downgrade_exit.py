"""A gateway stops once its checkout is moved back to a release that predates the gateway runtime.

Downgrading is unsupported (#106742, jquesnelle item 6). After a manual ``git checkout <older
release>`` under a running head gateway, the older client cannot attach to it, and older
``gateway run`` refuses because this process still holds the profile. Nothing on the older side
can stop it, so it stops itself: only on a proven, confirmed move to pre-runtime code, never
mid-update, and without asking for a restart — the stop is a plain stop, but it exits with the
service-restart code so the supervisor relaunches and the settled tree on disk serves.
"""
import asyncio
import subprocess

from gateway import runtime_downgrade_exit as rde
from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE

_SKEW = ("aaaaaaaaaa", "bbbbbbbbbb")


class _Runner:
    def __init__(self):
        self._running, self.stopped, self.restart_kwargs = True, 0, None
        self._exit_code: int | None = None

    exit_code = property(lambda self: self._exit_code)

    async def stop(self, **kwargs):
        self.stopped += 1
        self.restart_kwargs = kwargs
        self._running = False


def _watch(skews, *, update=False, predates=lambda: True):
    runner, seq = _Runner(), iter(skews)

    async def run():
        await rde.runtime_downgrade_watcher(
            runner, poll_s=0, skew_fn=lambda: next(seq, None), update_probe=lambda: update,
            predates=predates, max_polls=len(skews) + 1)
        await asyncio.sleep(0)  # let the detached stop task run
    asyncio.run(run())
    return runner


def test_a_confirmed_move_to_a_pre_runtime_tree_stops_without_a_restart(tmp_path):
    runner = _watch([_SKEW, _SKEW])
    assert runner.stopped == 1 and runner.restart_kwargs == {}  # a plain stop: the relaunch runs the tree on disk, not this process
    assert runner.exit_code == GATEWAY_SERVICE_RESTART_EXIT_CODE  # the supervisor relaunches the settled tree; the launchd plist parks exit 0 (SuccessfulExit=false)
    assert "predates the gateway runtime" in runner._exit_reason
    # A single blip, a live update, or a move to code that still has the runtime keep it up.
    assert _watch([_SKEW, None, _SKEW, None]).stopped == 0
    assert _watch([_SKEW, _SKEW, _SKEW], update=True).stopped == 0
    assert _watch([_SKEW, _SKEW], predates=lambda: False).stopped == 0
    # Live: the stop lazily imported gateway.session_acp_lifecycle, absent at v0.21.6, and raised;
    # the shutdown event was never set and the process idled on holding its profile lock.
    exits = []

    class _Broken(_Runner):
        async def stop(self, **kwargs):
            raise ModuleNotFoundError("No module named 'gateway.session_acp_lifecycle'")

    asyncio.run(rde._stop_or_exit(_Broken(), hard_exit=exits.append))
    asyncio.run(rde._stop_or_exit(_Runner(), hard_exit=exits.append))
    assert exits == [1]


def test_the_tree_probe_reads_the_checkout_on_disk(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-b", "main")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "T")
    (tmp_path / "hermes_state_common.py").write_text("SCHEMA = 'CREATE TABLE IF NOT EXISTS sessions'\n")
    git("add", ".")
    git("commit", "-m", "old release")
    git("tag", "old")
    (tmp_path / "hermes_state_common.py").write_text("SCHEMA = 'CREATE TABLE IF NOT EXISTS session_admissions'\n")
    git("commit", "-am", "runtime")
    assert rde.tree_predates_runtime(tmp_path) is False
    git("checkout", "-q", "--detach", "old")
    assert rde.tree_predates_runtime(tmp_path) is True
    assert rde.tree_predates_runtime(tmp_path / "missing") is False  # nothing readable proves nothing


def test_the_downgrade_exit_code_reaches_the_process_exit():
    """The bootstrap's exit verdict raises the runner's code, so launchd (SuccessfulExit=false) sees 75."""
    import pytest
    from gateway.run_shutdown import _resolve_gateway_exit_verdict
    runner = _watch([_SKEW, _SKEW])
    runner.should_exit_with_failure, runner._restart_requested, runner._restart_via_service = False, False, False
    with pytest.raises(SystemExit) as exc:
        _resolve_gateway_exit_verdict(runner, signal_initiated_shutdown=False)
    assert exc.value.code == GATEWAY_SERVICE_RESTART_EXIT_CODE
