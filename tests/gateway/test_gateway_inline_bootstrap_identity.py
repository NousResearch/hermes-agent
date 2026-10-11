"""A gateway started through one of Hermes' own inline bootstraps is a gateway on every OS (#124318).

The store launcher (``_launchers.runtime_command``, also the Windows updater's relaunch), the
published launcher script (POSIX shell launcher and the Windows ``.cmd`` base64 wrapper) and the
``venv_sync`` re-entry all run ``python -I -c <source> …`` with the entry point IN that process.
The #107002 guard that keeps an inline program's trailing argv as data made all of them invisible.
The readers hand the matchers different strings: ``/proc``, psutil and ``ps`` space-join argv (the
source splits across tokens); Windows CIM reports the CreateProcess line (``list2cmdline``).
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

import pytest

from gateway.status import looks_like_gateway_command_line
from hermes_cli import _launchers, venv_sync
from hermes_cli.update_cmd_windows import _hermes_holder_subcommand

ROOT = Path("/opt/Hermes Agent/hermes-agent")
PY = "/opt/venv/bin/python3"
_SCRIPT = _launchers._launcher_script("hermes", ROOT, None)
_JOINS = {"space-joined": " ".join, "windows": subprocess.list2cmdline}


def _forms(argv: list[str]) -> dict[str, list[str]]:
    forms = {
        "store-launcher": _launchers.runtime_command(ROOT, argv, python=Path(PY)),
        "launcher-script": [PY, "-I", "-c", _SCRIPT, *argv],
        "cmd-launcher": [PY, "-I", "-c", f"import base64; exec(base64.b64decode('{base64.b64encode(_SCRIPT.encode()).decode()}'))", *argv],
        "venv-reentry": venv_sync.relaunch_command(
            Path(PY), ROOT, [str(ROOT / "hermes_cli" / "main.py"), *argv], ["/old/python", "-m", "hermes_cli.main", *argv],
            "hermes_cli.main"),
    }
    # A launch through a published launcher that synced dependencies first re-enters it via exec().
    for name in ("launcher-script", "cmd-launcher"):
        forms[f"venv-reentry-{name}"] = venv_sync.relaunch_command(
            Path(PY), ROOT, ["-c", *argv], ["/old/python", *forms[name][1:]], None)
    return forms


@pytest.mark.parametrize("join", _JOINS)
@pytest.mark.parametrize("form", _forms([]))
def test_bootstrap_launched_gateway_is_a_gateway(form: str, join: str) -> None:
    command_line = _JOINS[join]([str(t) for t in _forms(["gateway", "run", "--replace"])[form]])
    assert looks_like_gateway_command_line(command_line)
    assert _hermes_holder_subcommand(command_line) == "gateway"


def test_spaced_venv_reentry_requires_argv_boundary_preservation() -> None:
    """The actual venv-sync ``-I -c`` re-entry must survive a spaced interpreter path."""
    spaced_py = Path("/opt/Hermes Agent/hermes/tools/python-3.14.7-x86_64/python.exe")
    argv = [
        str(token)
        for token in venv_sync.relaunch_command(
            spaced_py,
            ROOT,
            [str(ROOT / "hermes_cli" / "main.py"), "gateway", "run"],
            ["/old/python", "-m", "hermes_cli.main", "gateway", "run"],
            "hermes_cli.main",
        )
    ]
    old_reader = " ".join(argv)
    new_reader = subprocess.list2cmdline(argv)

    assert not looks_like_gateway_command_line(old_reader)
    assert _hermes_holder_subcommand(old_reader) != "gateway"
    assert looks_like_gateway_command_line(new_reader)
    assert _hermes_holder_subcommand(new_reader) == "gateway"

    # A space-free interpreter remains valid with either representation.
    space_free_argv = [
        str(token)
        for token in venv_sync.relaunch_command(
            Path(PY),
            ROOT,
            [str(ROOT / "hermes_cli" / "main.py"), "gateway", "run"],
            ["/old/python", "-m", "hermes_cli.main", "gateway", "run"],
            "hermes_cli.main",
        )
    ]
    for reader in (" ".join(space_free_argv), subprocess.list2cmdline(space_free_argv)):
        assert looks_like_gateway_command_line(reader)
        assert _hermes_holder_subcommand(reader) == "gateway"


@pytest.mark.parametrize("join", _JOINS)
def test_bootstrap_argv_is_identity_only_for_the_process_running_it(join: str) -> None:
    store = [str(t) for t in _launchers.runtime_command(ROOT, ["gateway", "run"], python=Path(PY))]
    chat = _JOINS[join]([str(t) for t in _launchers.runtime_command(ROOT, ["chat"], python=Path(PY))])
    # The restart watcher CARRIES a store-launcher gateway command it spawns later (#107002).
    watcher = _JOINS[join]([PY, "-c", "import os, sys, time\npid = int(sys.argv[1]); cmd = sys.argv[2:]\n", "1234", *store])
    assert not looks_like_gateway_command_line(chat) and _hermes_holder_subcommand(chat) == "chat"
    assert not looks_like_gateway_command_line(watcher) and _hermes_holder_subcommand(watcher) is None
