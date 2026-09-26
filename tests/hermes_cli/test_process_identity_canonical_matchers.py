"""Process-identity contract: the two kill/relaunch predicates and the profile liveness probe
defer to the canonical matchers instead of argv substrings (root AGENTS.md process-identity rule).
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli.dashboard_procs import _is_desktop_local_serve_cmdline
from hermes_cli.update_cmd_windows import _hermes_holder_subcommand, _is_backend_argv
from hermes_state_holders import (
    _PYTHON_LONG_OPTIONS_WITH_OPERANDS,
    _PYTHON_SHORT_OPTIONS_WITH_OPERANDS,
    _looks_like_hermes,
)

LOOPBACK = "--host 127.0.0.1 --port 0"

# (cmdline, holder subcommand, desktop-local reap?, Windows updater "Desktop backend"?). Substring
# scanners get every "trap" row wrong: "serve" appears inside --preserve-cache / observer.py / a flag
# value. The updater's kill set additionally requires the Desktop's `-m hermes_cli.main` spawn shape —
# a user-launched `hermes serve` / `hermes dashboard` is refused on, never tree-killed.
CMDLINES = [
    ("python -m hermes_cli.main serve " + LOOPBACK, "serve", True, True),
    ("python -m hermes_cli.main dashboard", "dashboard", False, True),
    ("/venv/bin/hermes serve --isolated --host=127.0.0.1 --port=0 --ssh-owner-nonce abc", "serve", True, False),
    (r"C:\hermes\.venv\Scripts\hermes.exe serve --host 100.106.105.2 --port 9119", "serve", False, False),
    ("hermes.exe dashboard", "dashboard", False, False),
    ("hermes --profile ops serve " + LOOPBACK, "serve", True, False),
    ("hermes -m serve kanban --preserve-cache " + LOOPBACK, "kanban", False, False),
    ("python -m hermes_cli.main kanban --preserve-cache " + LOOPBACK, "kanban", False, False),
    ("hermes --reasoning high dashboard " + LOOPBACK, "dashboard", False, False),
    ("hermes gateway run --replace", "gateway", False, False),
    ("hermes chat --model serve", "chat", False, False),
    ("python observer.py serve " + LOOPBACK, None, False, False),
    # Entry tokens in another program's arguments are data, not execution targets.
    ("python -c 'import time; time.sleep(60)' hermes webapp", None, False, False),
    ("python -uc 'pass' -m hermes_cli.main serve " + LOOPBACK, None, False, False),
    ("python -X utf8 -c 'pass' 14980 python -m hermes_cli.main gateway run", None, False, False),
    ("python -uWignore -Xdev -c 'pass' -m hermes_cli.main webapp", None, False, False),
    ("python observer.py -m hermes_cli.main serve " + LOOPBACK, None, False, False),
    ("python -m other hermes dashboard", None, False, False),
    ("python - hermes webapp", None, False, False),
    ("vim hermes serve " + LOOPBACK, None, False, False),
    ("bash -c 'hermes serve " + LOOPBACK + "'", None, False, False),
    # Python flags belong to the interpreter; Hermes flags belong to its target.
    ("python3.14 -uI -W ignore -X dev -m hermes_cli.main --model serve webapp", "webapp", False, True),
    ("python --check-hash-based-pycs always -m hermes_cli.main -p webapp serve " + LOOPBACK,
     "serve", True, True),
    ("python -Wignore -Xdev -- /venv/bin/hermes --reasoning high dashboard", "dashboard", False, False),
    (r'"C:\Program Files\Python\pythonw.exe" -u "C:\Hermes App\hermes_cli\main.py" webapp',
     "webapp", False, False),
    ("python /venv/bin/hermes --model='serve' webapp", "webapp", False, False),
    ("python -mhermes_cli.main webapp", "webapp", False, True),
    ("python -X hermes serve", None, False, False),
    ("hermes serve --model hermes_cli.main " + LOOPBACK, "serve", True, False),
]


@pytest.mark.parametrize("cmdline,subcommand,reapable,desktop_backend", CMDLINES)
def test_kill_and_relaunch_predicates_agree_with_the_canonical_holder_matcher(
        cmdline, subcommand, reapable, desktop_backend):
    assert _hermes_holder_subcommand(cmdline) == subcommand
    # Desktop-local reap (a KILL path): serve + loopback + ephemeral port, decided by tokens.
    assert _is_desktop_local_serve_cmdline(cmdline) is reapable
    # Windows updater backend classifier (taskkill /T on orphans): canonical subcommand AND Desktop spawn shape.
    assert _is_backend_argv(cmdline) is desktop_backend


@pytest.mark.parametrize("flag", sorted(
    {f"-{letter}" for letter in _PYTHON_SHORT_OPTIONS_WITH_OPERANDS} | _PYTHON_LONG_OPTIONS_WITH_OPERANDS))
def test_holder_matcher_and_state_holder_scan_share_the_interpreter_operand_table(flag):
    """A flag the canonical table says takes an operand never ends the option block, in either
    matcher: the operand ``x`` is not a script, the ``-m`` target behind it is."""
    argv = ["python", flag, "x", "-m", "hermes_cli.main", "serve"]
    assert _looks_like_hermes(argv)
    assert _hermes_holder_subcommand(subprocess.list2cmdline(argv)) == "serve"


def test_desktop_local_serve_spares_fixed_port_and_remote_hosts():
    assert not _is_desktop_local_serve_cmdline("hermes serve --host 100.106.105.2 --port 9119 --skip-build")
    assert not _is_desktop_local_serve_cmdline("hermes serve --host 127.0.0.1 --port 9119")
    assert _is_desktop_local_serve_cmdline("hermes serve --host localhost --port 0")


@pytest.mark.parametrize("argv,expected,desktop", [
    ([r"C:\Program Files\Python\python.exe", "-W", "ignore", "-m", "hermes_cli.main", "webapp"], "webapp", True),
    ([r"C:\Python Home\python.exe", r"C:\Hermes App\hermes_cli\main.py", "dashboard"], "dashboard", False),
    (["/Python Home/bin/python", "/Hermes App/hermes", "serve"], "serve", False),
    (["/Python Home/bin/python", "-c", "import time; time.sleep(60)", "hermes", "webapp"], None, False),
])
def test_live_argv_preserves_executable_and_script_boundaries(argv, expected, desktop):
    from types import SimpleNamespace
    from hermes_cli.update_cmd_windows import _cmdline_or_empty, _live_argv

    process = SimpleNamespace(cmdline=lambda: argv)
    psutil = SimpleNamespace(Process=lambda pid: process)
    for command in (_cmdline_or_empty(process), _live_argv(psutil, 1, "")):
        assert command is not None
        assert _hermes_holder_subcommand(command) == expected
        assert _is_backend_argv(command) is desktop


def test_profile_liveness_is_the_shared_ladder(tmp_path, monkeypatch):
    """``_check_gateway_running`` is ``resolve_gateway_liveness`` scoped to the profile dir, with the
    PID rung reading (never cleaning) THAT profile's ``gateway.pid``."""
    import gateway.status as gw_status
    from hermes_cli.profiles import _check_gateway_running

    seen: dict = {}

    def fake_resolve(**kwargs):
        seen.update(kwargs)
        return gw_status.GatewayLiveness(running=True, pid=1, source="pid")

    monkeypatch.setattr(gw_status, "resolve_gateway_liveness", fake_resolve)
    calls: list = []
    monkeypatch.setattr(gw_status, "get_running_pid",
                        lambda path, cleanup_stale=True: calls.append((path, cleanup_stale)))
    assert _check_gateway_running(tmp_path) is True
    assert seen["profile_dir"] == tmp_path
    seen["pid_probe"](tmp_path / "gateway.pid")
    assert calls == [(tmp_path / "gateway.pid", False)]
