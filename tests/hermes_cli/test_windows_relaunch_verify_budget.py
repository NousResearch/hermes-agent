"""Post-update gateway relaunch verification: watcher identity and liveness budget (#107002).

Two invariants, both host-independent (the platform enters as DATA, never by faking ``sys.platform``):

1. The detached restart watcher ``hermes_cli.gateway._spawn_gateway_restart_watcher`` actually
   spawns must not be classified as a running gateway. Its argv carries the gateway command it
   will spawn LATER, so an argv-substring identity read vouches for the watcher itself.
2. ``hermes update``'s post-relaunch liveness poll must not expire before the watchers it is
   verifying are even scheduled to respawn.
"""

from __future__ import annotations

import pytest

from gateway.status import (
    _gateway_command_subcommand, looks_like_gateway_command_line, looks_like_gateway_runtime_command_line,
)
from hermes_cli import gateway as gateway_mod
from hermes_cli.gateway import GATEWAY_RESTART_WATCHER_TIMEOUT_S
from hermes_cli.update_cmd_windows import (
    _hermes_holder_subcommand, _pending_relaunch_pids, _relaunch_verify_timeout_s,
)


def _spawned_watcher_cmdline(monkeypatch) -> str:
    """The real watcher command line, captured from a stubbed ``Popen`` (nothing is started)."""
    captured: list[list[str]] = []

    def fake_popen(argv, **_kwargs):
        captured.append(list(argv))
        return object()

    monkeypatch.setattr(gateway_mod.subprocess, "Popen", fake_popen)
    assert gateway_mod.launch_detached_gateway_restart_by_cmdline(
        14980, ["python", "-m", "hermes_cli.main", "gateway", "run"]
    )
    return " ".join(captured[0])


def test_spawned_restart_watcher_is_not_identified_as_a_gateway(monkeypatch):
    cmdline = _spawned_watcher_cmdline(monkeypatch)
    # The watcher really does carry the gateway argv — that is what made this bite.
    assert "hermes_cli.main gateway run" in cmdline
    # No Hermes subcommand at all may be read off it — not even a wrong one (the watcher source
    # text tokenizes into whatever words it happens to contain).
    assert _gateway_command_subcommand(cmdline) is None
    assert looks_like_gateway_command_line(cmdline) is False
    assert looks_like_gateway_runtime_command_line(cmdline) is False
    assert _hermes_holder_subcommand(cmdline) is None


@pytest.mark.parametrize(
    "profiles, unmapped, alive, expected",
    [
        ({}, [], set(), []),
        ({"default": 14980}, [], set(), []),
        ({"default": 14980}, [], {14980}, [14980]),
        # An unmapped entry without captured argv was never handed to a watcher.
        ({}, [{"pid": 4242, "argv": None}], {4242}, []),
        ({"default": 14980}, [{"pid": 4242, "argv": ["python"]}], {14980, 4242}, [4242, 14980]),
    ],
)
def test_pending_relaunch_pids(profiles, unmapped, alive, expected):
    assert _pending_relaunch_pids(profiles, unmapped, alive.__contains__) == expected


def test_verify_budget_covers_the_watchers_own_wait_when_the_old_pid_is_still_alive():
    profiles = {"default": 14980}
    settled = _relaunch_verify_timeout_s(profiles, [], lambda _pid: False)
    pending = _relaunch_verify_timeout_s(profiles, [], lambda _pid: True)
    # A watcher only respawns once its PID is gone: verifying for less than its own deadline
    # reports "no stable gateway process appeared" before the relaunch could have happened.
    assert pending > GATEWAY_RESTART_WATCHER_TIMEOUT_S
    assert settled < pending


# ---------------------------------------------------------------------------
# The runpy launcher form IS this process's own subcommand (#124588)
# ---------------------------------------------------------------------------

def _runpy_launcher_cmdline(root) -> str:
    """The real managed-runtime launcher command line (hermes_cli/_launchers.runtime_command)."""
    from hermes_cli._launchers import runtime_command

    return " ".join(runtime_command(root, ["gateway", "run", "--external-supervisor"]))


def test_runpy_launcher_form_is_recognized_as_gateway_run(tmp_path, monkeypatch):
    """#124588: the launchd-supervised gateway runs as the inline-source launcher itself —
    ``python -I -c "<runpy source>" gateway run --external-supervisor``. The #107002 guard
    read it as "not a gateway" and the Desktop app reported "Messaging gateway stopped"
    while the gateway was running and connected. The runpy bootstrap runs hermes_cli.main
    IN THIS PROCESS, so the trailing argv IS this process's own subcommand."""
    cmdline = _runpy_launcher_cmdline(tmp_path)
    assert "runpy.run_module" in cmdline
    assert _gateway_command_subcommand(cmdline) == "run"
    assert looks_like_gateway_command_line(cmdline) is True


def test_runpy_launcher_form_strips_profile_flags(tmp_path):
    cmdline = _runpy_launcher_cmdline(tmp_path).replace(
        "--external-supervisor", "-p work --external-supervisor")
    assert _gateway_command_subcommand(cmdline) == "run"


def test_watcher_form_still_refused(tmp_path):
    """The #107002 premise holds for the watcher (its ``-c`` code spawns later, no runpy)."""
    from hermes_cli._launchers import runtime_command

    cmdline = " ".join(runtime_command(
        tmp_path, ["1234", "30.0", "python", "-m", "hermes_cli.main", "gateway", "restart"],
        code="watcher"))
    assert "runpy.run_module" not in cmdline
    assert _gateway_command_subcommand(cmdline) is None
