"""A gateway started through one of Hermes' own inline bootstraps is a gateway on every OS (#124318).

The store launcher (``_launchers.runtime_command``, also the Windows updater's relaunch), the
published launcher script (POSIX shell launcher and the Windows ``.cmd`` base64 wrapper) and the
``venv_sync`` re-entry and the native payload's ``bin/hermes`` (``scripts/build/launchers``) all run
``python -c <source> …`` with the entry point IN that process.
The #107002 guard that keeps an inline program's trailing argv as data made all of them invisible.
The readers hand the matchers different strings: ``/proc``, psutil and ``ps`` space-join argv (the
source splits across tokens); Windows CIM reports the CreateProcess line (``list2cmdline``).
"""

from __future__ import annotations

import base64
import shlex
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


def _payload_launcher(argv: list[str]) -> list[str]:
    """The native payload's ``bin/hermes`` exec line, rendered by the real payload builder."""
    from scripts.build.launchers import posix_launcher

    script = posix_launcher("hermes", "hermes_cli.main:main", python="tools/python/bin/python3",
                            repo="hermes-agent", site="venv/lib/python3.14/site-packages", target="linux-x64")
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))
    _, _python, *launch, _args = shlex.split(exec_line)
    return [PY, *launch, *argv]


def _forms(argv: list[str]) -> dict[str, list[str]]:
    return {
        "store-launcher": _launchers.runtime_command(ROOT, argv, python=Path(PY)),
        "launcher-script": [PY, "-I", "-c", _SCRIPT, *argv],
        "cmd-launcher": [PY, "-I", "-c", f"import base64; exec(base64.b64decode('{base64.b64encode(_SCRIPT.encode()).decode()}'))", *argv],
        "venv-reentry": venv_sync.relaunch_command(
            Path(PY), ROOT, [str(ROOT / "hermes_cli" / "main.py"), *argv], ["/old/python", "-m", "hermes_cli.main", *argv],
            "hermes_cli.main"),
        "payload-launcher": _payload_launcher(argv),
    }


@pytest.mark.parametrize("join", _JOINS)
@pytest.mark.parametrize("form", _forms([]))
def test_bootstrap_launched_gateway_is_a_gateway(form: str, join: str) -> None:
    command_line = _JOINS[join]([str(t) for t in _forms(["gateway", "run", "--replace"])[form]])
    assert looks_like_gateway_command_line(command_line)
    assert _hermes_holder_subcommand(command_line) == "gateway"


@pytest.mark.parametrize("join", _JOINS)
def test_bootstrap_argv_is_identity_only_for_the_process_running_it(join: str) -> None:
    store = [str(t) for t in _launchers.runtime_command(ROOT, ["gateway", "run"], python=Path(PY))]
    chat = _JOINS[join]([str(t) for t in _launchers.runtime_command(ROOT, ["chat"], python=Path(PY))])
    # The restart watcher CARRIES a store-launcher gateway command it spawns later (#107002).
    watcher = _JOINS[join]([PY, "-c", "import os, sys, time\npid = int(sys.argv[1]); cmd = sys.argv[2:]\n", "1234", *store])
    assert not looks_like_gateway_command_line(chat) and _hermes_holder_subcommand(chat) == "chat"
    assert not looks_like_gateway_command_line(watcher) and _hermes_holder_subcommand(watcher) is None


def test_native_payload_gateway_under_systemd_is_a_gateway() -> None:
    # /proc of a gateway a service manager started through a native payload's bin/hermes.
    command_line = (
        "/opt/hermes/releases/04ccd391ddfdd1f89eb6c2fd10a75518c64b2fd3/tools/python-3.14.7+20260901-linux-x64"
        "/bin/python3.14 -P -c import os, site, sys; sys.argv[0]='hermes'; site.addsitedir(os.environ['HERMES_SITE']);"
        " from hermes_cli.main import main; sys.exit(main()) gateway run"
    )
    assert looks_like_gateway_command_line(command_line)
    assert not looks_like_gateway_command_line(command_line.replace(" gateway run", " gateway status"))


@pytest.mark.parametrize("join", _JOINS)
def test_payload_launcher_argv_is_identity_only_for_the_process_running_it(join: str) -> None:
    chat = _JOINS[join](_payload_launcher(["chat"]))
    watcher = _JOINS[join]([PY, "-c", "import os, sys, time\npid = int(sys.argv[1]); cmd = sys.argv[2:]\n", "1234",
                            *_payload_launcher(["gateway", "run"])])
    assert not looks_like_gateway_command_line(chat) and _hermes_holder_subcommand(chat) == "chat"
    assert not looks_like_gateway_command_line(watcher) and _hermes_holder_subcommand(watcher) is None
