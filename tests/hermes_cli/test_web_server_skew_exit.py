"""A Desktop-over-SSH ``serve --isolated`` backend retires itself once the install under it moved to
new code: nothing on its host may restart it (only the remote client holds its token), so without
this it runs the pre-update code against the updated tree until that client happens to reconnect.
It exits only between turns (the retirement fence proves idle and closes admission first), never
while an update is still in flight, and its client respawns it from the new code."""

from __future__ import annotations

import subprocess

from hermes_cli.web_server_skew_exit import should_retire_for_skew, start_code_skew_watchdog

_SKEW = ("aaaaaaaaaa", "bbbbbbbbbb")


class _Server:
    should_exit = False


class _Fence:
    """Retirement fence stand-in: ``idle`` answers prepare; commit records the exit permit."""

    def __init__(self, idle: bool):
        self.idle, self.committed = idle, False

    def prepare(self):
        return {"ok": True, "idle": True, "token": "t"} if self.idle else {"ok": False, "idle": False}

    def commit(self, token):
        self.committed = token == "t"
        return {"ok": self.committed}


def test_retires_only_on_a_proven_code_change_outside_an_update():
    assert should_retire_for_skew(skew=_SKEW, update_in_progress=False) is True
    # No drift, or an unreadable revision (detect_code_skew returns None for both): stay up.
    assert should_retire_for_skew(skew=None, update_in_progress=False) is False
    # The tree is mid-swap: respawning now would import a half-updated install.
    assert should_retire_for_skew(skew=_SKEW, update_in_progress=True) is False


def _run(fence, skews, *, update=False):
    server = _Server()
    seq = iter(skews)
    thread = start_code_skew_watchdog(
        server, skew_fn=lambda: next(seq, skews[-1]),
        update_probe=lambda: update, fence=fence, poll_s=0.01, max_polls=len(skews) + 2)
    thread.join(timeout=5)
    return server


def test_skew_retires_through_the_fence_after_two_consecutive_observations():
    fence = _Fence(idle=True)
    assert _run(fence, [_SKEW, _SKEW]).should_exit is True
    assert fence.committed is True  # admission closed before exit: no turn can start mid-teardown


def test_a_single_blip_does_not_retire():
    fence = _Fence(idle=True)
    assert _run(fence, [_SKEW, None, _SKEW, None]).should_exit is False
    assert fence.committed is False


def test_a_busy_backend_waits_instead_of_cutting_off_its_client():
    fence = _Fence(idle=False)
    assert _run(fence, [_SKEW, _SKEW, _SKEW]).should_exit is False


def test_never_retires_while_an_update_is_in_flight():
    fence = _Fence(idle=True)
    assert _run(fence, [_SKEW, _SKEW, _SKEW], update=True).should_exit is False


def test_default_probe_retires_once_the_checkout_moves_past_the_boot_revision(tmp_path, monkeypatch):
    """The real wiring: the watchdog reads the boot revision the serve lifespan records
    (``gateway.code_skew``) against a checkout whose HEAD then moves."""
    import gateway.code_skew as code_skew

    repo = tmp_path / "repo"
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "one"], check=True)
    monkeypatch.setattr(code_skew, "_PROJECT_ROOT", repo)
    monkeypatch.setattr(code_skew, "_boot_fingerprint", None)
    code_skew.record_boot_fingerprint()

    fence = _Fence(idle=True)
    unchanged = _Server()
    start_code_skew_watchdog(unchanged, update_probe=lambda: False, fence=fence,
                             poll_s=0.01, max_polls=4).join(timeout=5)
    assert unchanged.should_exit is False

    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "two"], check=True)
    moved = _Server()
    start_code_skew_watchdog(moved, update_probe=lambda: False, fence=fence,
                             poll_s=0.01, max_polls=4).join(timeout=5)
    assert moved.should_exit is True
