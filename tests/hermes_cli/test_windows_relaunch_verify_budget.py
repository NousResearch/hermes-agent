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


def test_windows_relaunch_argv_is_visible_to_the_liveness_poll(monkeypatch):
    """The argv the Windows relaunch spawns must be one the poll can actually find.

    The poll discovers gateways by command line, and an inline-source (``-c``) command line is
    never a gateway (#107002) — a store-Python bootstrap relaunch is invisible to it, so the
    update reports "no stable gateway process appeared" and exits 1 with the gateway up.
    """
    monkeypatch.setattr(gateway_mod, "is_windows", lambda: True)
    monkeypatch.setattr(gateway_mod, "get_python_path", lambda: "/usr/bin/python3")
    default = gateway_mod._gateway_run_args_for_profile("default")
    named = gateway_mod._gateway_run_args_for_profile("work")
    assert looks_like_gateway_command_line(" ".join(default)) is True
    assert looks_like_gateway_command_line(" ".join(named)) is True
    assert "-c" not in default
    assert " ".join(default).endswith("gateway run --replace")
    assert "--profile work" in " ".join(named)
