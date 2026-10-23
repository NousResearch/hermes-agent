"""A gateway started through one of Hermes' own inline bootstraps is a gateway on every OS (#124318).

The store launcher (``_launchers.runtime_command``, also the Windows updater's relaunch), the
published launcher script (POSIX shell launcher and the Windows ``.cmd`` base64 wrapper) and the
``venv_sync`` re-entry and the bundled/Docker payload's ``bin/hermes`` (``scripts/build/launchers``,
#127865) all run ``python -c <source> …`` with the entry point IN that process.
The #107002 guard that keeps an inline program's trailing argv as data made all of them invisible.
The readers hand the matchers different strings: ``/proc``, psutil and ``ps`` space-join argv (the
source splits across tokens); Windows CIM reports the CreateProcess line (``list2cmdline``).
"""

from __future__ import annotations

import base64
import json
import shlex
import subprocess
from pathlib import Path

import pytest

from gateway import status
from gateway.status import looks_like_gateway_command_line, looks_like_gateway_runtime_command_line
from gateway.status_inline_source import inline_bootstrap_argv
from hermes_cli import _launchers, venv_sync
from hermes_cli.update_cmd_windows import _hermes_holder_subcommand

ROOT = Path("/opt/Hermes Agent/hermes-agent")
PY = "/opt/venv/bin/python3"
_SCRIPT = _launchers._launcher_script("hermes", ROOT, None)
_JOINS = {"space-joined": " ".join, "windows": subprocess.list2cmdline}


def _payload_launcher(argv: list[str]) -> list[str]:
    """The payload ``bin/hermes`` exec line, rendered by the real payload builder."""
    from scripts.build.launchers import posix_launcher

    script = posix_launcher("hermes", "hermes_cli.main:main", python="tools/python/bin/python3",
                            repo="hermes-agent", site="venv/lib/python3.14/site-packages", target="linux-x64")
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))
    _, _python, *launch, _args = shlex.split(exec_line)
    return [PY, *launch, *argv]


def _forms(argv: list[str]) -> dict[str, list[str]]:
    forms = {
        "store-launcher": _launchers.runtime_command(ROOT, argv, python=Path(PY)),
        "launcher-script": [PY, "-I", "-c", _SCRIPT, *argv],
        "cmd-launcher": [PY, "-I", "-c", f"import base64; exec(base64.b64decode('{base64.b64encode(_SCRIPT.encode()).decode()}'))", *argv],
        "venv-reentry": venv_sync.relaunch_command(
            Path(PY), ROOT, [str(ROOT / "hermes_cli" / "main.py"), *argv], ["/old/python", "-m", "hermes_cli.main", *argv],
            "hermes_cli.main"),
        "payload-launcher": _payload_launcher(argv),
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


def test_bundled_posix_launcher_bootstrap_is_a_gateway() -> None:
    source = (
        "import os, site, sys; sys.argv[0]='hermes'; "
        "site.addsitedir(os.environ['HERMES_SITE']); "
        "from hermes_cli.main import main; sys.exit(main())"
    )
    command_line = " ".join([PY, "-P", "-c", source, "gateway", "run", "--replace"])
    assert looks_like_gateway_command_line(command_line)



_GATEWAY = (["gateway", "run"], ["gateway", "run", "--replace"], ["--profile", "work", "gateway", "run"],
            ["-p", "work", "gateway", "run"], ["gateway", "run", "-p", "work"])
_NOT_GATEWAY = (["gateway", "status"], ["gateway", "stop"], ["status"], ["chat"])


@pytest.mark.parametrize("join", _JOINS)
@pytest.mark.parametrize("argv", [*_GATEWAY, ["gateway", "restart"], *_NOT_GATEWAY], ids=" ".join)
def test_payload_launcher_reads_as_the_hermes_command_it_runs(argv: list[str], join: str) -> None:
    command_line = _JOINS[join](_payload_launcher(argv))
    # ``gateway restart`` hosts the runtime in-process without a service manager, but is no ``run``.
    assert looks_like_gateway_command_line(command_line) is (argv in _GATEWAY)
    assert looks_like_gateway_runtime_command_line(command_line) is (argv in (*_GATEWAY, ["gateway", "restart"]))


@pytest.mark.parametrize("split", [list, lambda argv: " ".join(argv).split(" ")], ids=["argv", "space-split"])
def test_payload_launcher_maps_to_the_module_spelling(split) -> None:
    tokens = split(_payload_launcher(["gateway", "run", "--replace"]))
    assert inline_bootstrap_argv(tokens) == [PY, "-m", "hermes_cli.main", "gateway", "run", "--replace"]


@pytest.mark.parametrize("join", _JOINS)
@pytest.mark.parametrize("module", ["not_hermes_cli.main", "foreign.hermes_cli.main", "hermes_cli.main_x"])
def test_payload_launcher_shape_with_another_entry_module_is_not_hermes(module: str, join: str) -> None:
    launch = _payload_launcher(["gateway", "run"])
    launch[launch.index("-c") + 1] = launch[launch.index("-c") + 1].replace(
        "from hermes_cli.main import", f"from {module} import")
    command_line = _JOINS[join](launch)
    assert inline_bootstrap_argv(launch) is None
    assert not looks_like_gateway_command_line(command_line)
    assert _hermes_holder_subcommand(command_line) is None


@pytest.mark.parametrize("join", _JOINS)
@pytest.mark.parametrize("carrier", [
    "import os, sys, time\npid = int(sys.argv[1]); cmd = sys.argv[2:]\n",  # the restart watcher (#107002)
    "import os, site, sys; cmd = sys.argv[2:]",  # opens like the payload bootstrap, runs no entry point
], ids=["restart-watcher", "payload-lookalike"])
def test_a_program_carrying_the_payload_launcher_is_not_a_gateway(carrier: str, join: str) -> None:
    carried = [PY, "-c", carrier, "1234", *_payload_launcher(["gateway", "run"])]
    command_line = _JOINS[join](carried)
    assert inline_bootstrap_argv(carried) is None
    assert not looks_like_gateway_command_line(command_line)
    assert _hermes_holder_subcommand(command_line) is None


@pytest.mark.parametrize("argv,expected", [
    ([], None), (["gateway", "run"], None), (["-p", "work", "gateway", "run"], "work"),
    (["gateway", "run", "--profile=Work"], "work"),
])
def test_payload_launcher_profile_is_read_from_the_hermes_argv(argv: list[str], expected: str | None) -> None:
    # ``/proc`` space-joins argv; the interpreter's ``-P`` before ``-c`` is no ``-p`` (#127865).
    assert status.profile_flag_value(" ".join(_payload_launcher(argv))) == expected


@pytest.mark.parametrize("argv,profile,found", [
    (["gateway", "run"], None, True),
    (["-p", "work", "gateway", "run"], "work", True),
    (["-p", "work", "gateway", "run"], None, False),  # another profile's gateway is not the default's
    (["chat"], None, False),
], ids=["default", "named", "other-profile", "chat"])
def test_live_payload_launched_gateway_is_found_for_its_home(argv, profile, found, tmp_path, monkeypatch) -> None:
    """The dashboard / ``gateway stop`` / ``restart`` lookup: the live ``/proc`` cmdline of a gateway
    started through the payload ``bin/hermes`` proves its home's runtime record (#127865)."""
    import hermes_constants

    root = tmp_path / ".hermes"
    home = root / "profiles" / profile if profile else root
    home.mkdir(parents=True)
    monkeypatch.setattr(hermes_constants.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None, raising=False)
    record = {
        "pid": 4242, "kind": "hermes-gateway", "gateway_state": "running", "start_time": 1000,
        "argv": ["/opt/hermes/hermes-agent/hermes_cli/main.py", *argv], "hermes_home": str(home),
    }
    (home / "gateway_state.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(status, "_pid_exists", lambda pid: pid == 4242)
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 1000)
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: " ".join(_payload_launcher(argv)))

    assert status.live_gateway_pid_for_home(home) == (4242 if found else None)
