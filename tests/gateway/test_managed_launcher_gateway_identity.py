"""A store launcher IS the gateway: ``python -I -c <bootstrap> gateway run`` must be identified.

``hermes_cli._launchers._launcher_script`` mints every store launcher as
``exec <store python> -I -c '<bootstrap>' "$@"`` (``.hermes/bin/hermes``), so the live gateway's own
command line is an *inline source* one. The #107002 rule — ``_gateway_command_subcommand`` refuses
``python -c <src> …`` wholesale, because the detached restart watcher's trailing argv is a command it
will spawn LATER — therefore classified the real gateway as "not a gateway" on every store
deployment: ``hermes gateway list``, the dashboard and ``hermes -p X status`` all read "not running"
while the process was up and serving platforms.

The refusal must stay for every other inline source; only a launcher bootstrap is peeled, and only
its trailing argv is read as identity.
"""

from __future__ import annotations

import json

import pytest

import gateway.status as gw_status
from gateway.status import (
    _MANAGED_LAUNCHER_SENTINEL,
    _gateway_command_subcommand,
    looks_like_gateway_command_line,
    looks_like_gateway_runtime_command_line,
)
from hermes_cli._launchers import MANAGED_LAUNCHER_SENTINEL, _launcher_script

STORE_PYTHON = "/root/.hermes/tools/python-3.14.7-linux-x64/bin/python3"
START_TIME = 904324


def _minted_launcher_cmdline(repo_root, *, sentinel: bool = True) -> str:
    """The command line a minted ``.hermes/bin/hermes`` shim actually runs.

    ``exec <python> -I -c '<script>' "$@"`` — the script is ONE argv and keeps its newlines, exactly
    as ``/proc/<pid>/cmdline`` reports it (the reader joins NUL-separated argv with spaces).
    """
    script = _launcher_script("hermes", repo_root, None)
    if not sentinel:
        script = script.replace(MANAGED_LAUNCHER_SENTINEL + "\n", "", 1)
    return f"{STORE_PYTHON} -I -c {script} gateway run"


def test_the_producer_and_the_matcher_agree_on_one_sentinel(tmp_path):
    assert MANAGED_LAUNCHER_SENTINEL == _MANAGED_LAUNCHER_SENTINEL
    assert _launcher_script("hermes", tmp_path, None).startswith(MANAGED_LAUNCHER_SENTINEL)


def test_minted_launcher_is_identified_as_a_running_gateway(tmp_path):
    cmdline = _minted_launcher_cmdline(tmp_path)
    assert _gateway_command_subcommand(cmdline) == "run"
    assert looks_like_gateway_command_line(cmdline) is True
    assert looks_like_gateway_runtime_command_line(cmdline) is True


def test_launcher_minted_before_the_sentinel_is_still_identified(tmp_path):
    """Store launchers already on disk predate the marker: the generated prologue must keep them
    identifiable, or existing installs stay invisible until they are re-minted."""
    cmdline = _minted_launcher_cmdline(tmp_path, sentinel=False)
    assert MANAGED_LAUNCHER_SENTINEL not in cmdline
    assert _gateway_command_subcommand(cmdline) == "run"


@pytest.mark.parametrize(
    "cmdline",
    [
        # Not a launcher: a plain inline program with a gateway-looking trailing argv.
        'python3 -c "print(1)" gateway run',
        # The #107002 shape: the watcher hides a LATER gateway spawn behind its inline source.
        'python3 -c "import sys, time" 14980 /venv/bin/python -m hermes_cli.main gateway run',
    ],
)
def test_every_other_inline_source_is_still_refused(cmdline):
    assert _gateway_command_subcommand(cmdline) is None
    assert looks_like_gateway_command_line(cmdline) is False
    assert looks_like_gateway_runtime_command_line(cmdline) is False


def test_launcher_identity_carries_through_the_runtime_record_rung(tmp_path, monkeypatch):
    """The rung every status surface hangs off: with no ``gateway.pid`` (a service-managed store
    gateway has none), ``live_gateway_pid_for_home`` reads the PID off ``gateway_state.json`` only
    when the LIVE command line proves it is a gateway."""
    cmdline = _minted_launcher_cmdline(tmp_path)
    record = {
        "pid": 4242, "kind": "hermes-gateway", "argv": ["-c", "gateway", "run"],
        "start_time": START_TIME, "gateway_state": "running", "platforms": {},
    }
    (tmp_path / "gateway_state.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(gw_status, "_read_process_cmdline", lambda pid: cmdline)
    monkeypatch.setattr(gw_status, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(gw_status, "_get_process_start_time", lambda pid: START_TIME)
    assert gw_status.live_gateway_pid_for_home(tmp_path) == 4242
