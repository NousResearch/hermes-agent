"""Process-identity contract: the two kill/relaunch predicates and the profile liveness probe
defer to the canonical matchers instead of argv substrings (root AGENTS.md process-identity rule).
"""

from __future__ import annotations

import pytest

from hermes_cli.dashboard_procs import _is_desktop_local_serve_cmdline
from hermes_cli.update_cmd_windows import _hermes_holder_subcommand, _is_backend_argv

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
]


@pytest.mark.parametrize("cmdline,subcommand,reapable,desktop_backend", CMDLINES)
def test_kill_and_relaunch_predicates_agree_with_the_canonical_holder_matcher(
        cmdline, subcommand, reapable, desktop_backend):
    assert _hermes_holder_subcommand(cmdline) == subcommand
    # Desktop-local reap (a KILL path): serve + loopback + ephemeral port, decided by tokens.
    assert _is_desktop_local_serve_cmdline(cmdline) is reapable
    # Windows updater backend classifier (taskkill /T on orphans): canonical subcommand AND Desktop spawn shape.
    assert _is_backend_argv(cmdline.lower()) is desktop_backend


def test_desktop_local_serve_spares_fixed_port_and_remote_hosts():
    assert not _is_desktop_local_serve_cmdline("hermes serve --host 100.106.105.2 --port 9119 --skip-build")
    assert not _is_desktop_local_serve_cmdline("hermes serve --host 127.0.0.1 --port 9119")
    assert _is_desktop_local_serve_cmdline("hermes serve --host localhost --port 0")


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


# ``hermes serve`` is a substring of ``hermes server``: a terminal multiplexer started as
# ``herdr --session hermes server`` was SIGTERMed by ``hermes update`` and its unit restarted (#121156).
DECOY = "tool --name hermes server 30"
BACKENDS = [
    "/opt/hermes/venv/bin/python -m hermes_cli.main serve --port 0",
    "/usr/bin/python3 /opt/hermes/hermes_cli/main.py dashboard --no-open",
]


def test_dashboard_scan_selects_entrypoint_plus_subcommand_tokens_never_substrings(monkeypatch):
    from hermes_cli import dashboard_procs
    from hermes_cli import process_identity

    monkeypatch.setattr(dashboard_procs, "_iter_process_table",
                        lambda: [(4242, DECOY), *((5000 + i, cmd) for i, cmd in enumerate(BACKENDS))])
    monkeypatch.setattr(process_identity, "ledger_entries", list)
    assert dashboard_procs._scan_dashboard_processes() == [(5000, BACKENDS[0]), (5001, BACKENDS[1])]


def test_dashboard_runtime_parse_refuses_the_decoy_and_reads_a_real_backend():
    from hermes_cli.main_dashboard import _parse_dashboard_runtime

    assert _parse_dashboard_runtime(DECOY) is None
    assert _parse_dashboard_runtime(BACKENDS[0]) == ("serve", "127.0.0.1", 0)
    # launchd ProgramArguments arrive ``shlex.join``ed: a quoted path with spaces is still the entry.
    assert _parse_dashboard_runtime("'/Users/a b/venv/bin/hermes' dashboard --port 9200") == ("dashboard", "127.0.0.1", 9200)
