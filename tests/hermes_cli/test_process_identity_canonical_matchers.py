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
    # Inline-bootstrap launcher (-c "sys.argv = [...]") correctly extracts host and port.
    inline = ("/usr/bin/python3 -I -c \"import sys, runpy; sys.path.insert(0, '/root'); "
              "sys.argv = ['main.py', 'serve', '--host', '0.0.0.0', '--port', '9119']; "
              "sys._hermes_relaunched = True; runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)\"")
    assert _parse_dashboard_runtime(inline) == ("serve", "0.0.0.0", 9119)


@pytest.mark.parametrize(
    "shape,inner_argv,expected",
    [
        # macOS launchd relaunch (runpy.run_module): wildcard 0.0.0.0 must not narrow to 127.0.0.1.
        (
            "run_module",
            ["/Users/alice/.hermes/hermes-agent/hermes_cli/main.py", "dashboard", "--host", "0.0.0.0", "--port", "9120", "--no-open"],
            ("0.0.0.0", 9120),
        ),
        (
            "run_module",
            ["/Users/alice/.hermes/hermes-agent/hermes_cli/main.py", "serve", "--host=0.0.0.0", "--port=9121"],
            ("0.0.0.0", 9121),
        ),
        # Linux systemd relaunch (runpy.run_path): wildcard, specific IP, IPv6, and = syntax.
        (
            "run_path",
            ["/home/alice/.hermes/hermes-agent/venv/bin/hermes", "dashboard", "--host", "0.0.0.0", "--port", "9122", "--no-open"],
            ("0.0.0.0", 9122),
        ),
        (
            "run_path",
            ["/home/alice/.local/bin/hermes", "serve", "--host", "10.0.1.5", "--port", "8080"],
            ("10.0.1.5", 8080),
        ),
        (
            "run_path",
            ["/home/alice/.local/bin/hermes", "dashboard", "--host", "::", "--port", "9119"],
            ("::", 9119),
        ),
        (
            "run_path",
            ["/home/alice/.local/bin/hermes", "serve", "--host=0.0.0.0", "--port=9125"],
            ("0.0.0.0", 9125),
        ),
    ],
    ids=[
        "launchd-run-module-wildcard",
        "launchd-run-module-equals",
        "systemd-run-path-wildcard",
        "systemd-run-path-specific-ip",
        "systemd-run-path-ipv6-wildcard",
        "systemd-run-path-equals",
    ],
)
def test_backend_address_reads_relaunched_inline_bootstrap_bind(shape, inner_argv, expected):
    from pathlib import Path
    from hermes_cli import venv_sync
    from hermes_cli.gateway_migrate_backends_plan import backend_address
    from hermes_cli.main_dashboard import _parse_dashboard_runtime

    py = Path("/opt/hermes/venv/bin/python")
    root = Path("/opt/hermes/hermes-agent")
    if shape == "run_module":
        relaunched = venv_sync.relaunch_command(
            py, root, inner_argv, ["/old/python", "-m", "hermes_cli.main", *inner_argv[1:]], "hermes_cli.main"
        )
    else:
        relaunched = venv_sync.relaunch_command(py, root, inner_argv, ["/old/python", *inner_argv], None)

    assert backend_address(relaunched) == expected
    # Space-joined /proc / ps output (used by dashboard --status) parses the same bind.
    assert _parse_dashboard_runtime(" ".join(relaunched))[1:] == expected


def test_backend_address_preserves_plain_non_relaunched_commands():
    from hermes_cli.gateway_migrate_backends_plan import backend_address

    assert backend_address(["/home/alice/.local/bin/hermes", "dashboard", "--host", "0.0.0.0", "--port", "9130"]) == ("0.0.0.0", 9130)
    assert backend_address(["/usr/bin/python3", "-m", "hermes_cli.main", "serve"]) == ("127.0.0.1", 9119)
    assert backend_address(["/usr/bin/python3", "-m", "hermes_cli.main", "serve", "--host", "127.0.0.1", "--port", "0"]) is None


def test_backend_address_ignores_path_decoys_and_invalid_ports():
    from pathlib import Path
    from hermes_cli import venv_sync
    from hermes_cli.gateway_migrate_backends_plan import backend_address
    from hermes_cli.main_dashboard import _parse_dashboard_runtime

    # A root path containing "--host ..." must not leak when the unwrapped command omits --host.
    decoy_root = Path("/tmp/dir --host 1.2.3.4")
    relaunched = venv_sync.relaunch_command(
        decoy_root / "venv/bin/python",
        decoy_root,
        [str(decoy_root / "venv/bin/hermes"), "dashboard", "--port", "9142"],
        ["/old/python", str(decoy_root / "venv/bin/hermes"), "dashboard", "--port", "9142"],
        None,
    )
    assert backend_address(relaunched) == ("127.0.0.1", 9142)

    # Exact argv lists keep path arguments containing quotes and flag fragments intact.
    assert backend_address(
        [
            "/usr/bin/hermes",
            "serve",
            "--ssh-session-token-file",
            "/home/o'brien/keys --host 10.0.0.9 --port 8111 notes/token",
            "--host",
            "0.0.0.0",
            "--port",
            "9120",
        ]
    ) == ("0.0.0.0", 9120)

    # Non-numeric or negative ports are rejected rather than defaulting silently.
    assert _parse_dashboard_runtime("/usr/bin/hermes dashboard --port notaport") is None
    assert _parse_dashboard_runtime("/usr/bin/hermes dashboard --port=-1") is None


@pytest.mark.parametrize(
    "cmd",
    [
        # Malformed --port fragment in flattened path text.
        "hermes serve --host 0.0.0.0 --port 9120 --ssh-session-token-file /home/user/keys --port 8111/token",
        # Numeric --port fragment in flattened path text must not overwrite the real port.
        "hermes serve --host 0.0.0.0 --port 9120 --ssh-session-token-file /home/user/keys --port 8111 notes/token",
        # --host fragment in flattened path text must not overwrite the real host.
        "hermes serve --host 0.0.0.0 --port 9120 --ssh-session-token-file /home/user/keys --host 10.0.0.9 notes/token",
    ],
    ids=["malformed-port-fragment", "numeric-port-fragment", "host-fragment"],
)
def test_parse_dashboard_runtime_keeps_valid_port_before_flattened_path_fragment(cmd):
    from hermes_cli.main_dashboard import _parse_dashboard_runtime

    # Flattened ps output strips quotes from paths containing flag fragments; keep the first
    # valid explicit --host/--port instead of dropping or overwriting the real bind.
    assert _parse_dashboard_runtime(cmd) == ("serve", "0.0.0.0", 9120)


@pytest.mark.parametrize(
    "cmd,expected",
    [
        (
            "hermes serve --port 9120 --ssh-session-token-file /home/user/keys --port 8111 notes/token",
            ("serve", "127.0.0.1", 9120),
        ),
        (
            "hermes serve --host 0.0.0.0 --ssh-session-token-file /home/user/keys --host 10.0.0.9 notes/token",
            ("serve", "0.0.0.0", 9119),
        ),
    ],
    ids=["port-only", "host-only"],
)
def test_parse_dashboard_runtime_keeps_first_flag_when_only_one_is_named(cmd, expected):
    from hermes_cli.main_dashboard import _parse_dashboard_runtime

    # With only one bind flag named, each first-value guard independently keeps the real bind.
    assert _parse_dashboard_runtime(cmd) == expected


@pytest.mark.parametrize(
    "exact_argv,expected_bind",
    [
        (
            [
                "/usr/bin/hermes",
                "serve",
                "--ssh-session-token-file",
                "/home/user/keys --host 10.0.0.9 --port 8111 notes/token",
            ],
            ("127.0.0.1", 9119),
        ),
        (
            [
                "/usr/bin/hermes",
                "serve",
                "--ssh-session-token-file",
                "/home/user/keys --host 10.0.0.9 --port 8111 notes/token",
                "--host",
                "0.0.0.0",
                "--port",
                "9120",
            ],
            ("0.0.0.0", 9120),
        ),
        (
            [
                "/usr/bin/python3",
                "-I",
                "-c",
                "import os, re, sys\nif sys.argv[1:2] == ['--run-module']:\n    sys.exit(0)\nfrom hermes_cli.main import main\nsys.exit(main())",
                "serve",
                "--host",
                "0.0.0.0",
                "--port",
                "9120",
                "--ssh-session-token-file",
                "/home/user/keys --port 8111 notes/token",
            ],
            ("0.0.0.0", 9120),
        ),
    ],
    ids=[
        "default-bind-with-path-fragments",
        "explicit-bind-after-path-fragments",
        "posix-launcher-source-newlines",
    ],
)
def test_report_dashboard_status_preserves_argv_boundaries_from_psutil(monkeypatch, exact_argv, expected_bind):
    import psutil
    from hermes_cli import dashboard_procs, main_dashboard

    pid = 424242
    # ps prints argv joined by spaces, the published launcher's multi-line -c source included.
    flattened = " ".join(exact_argv).replace("\n", " ")

    class _FakeProcess:
        def __init__(self, target_pid):
            assert target_pid == pid

        def cmdline(self):
            return list(exact_argv)

    probed: list[tuple[str, int]] = []
    monkeypatch.setattr(dashboard_procs, "_scan_dashboard_processes", lambda **_: [(pid, flattened)])
    monkeypatch.setattr(dashboard_procs, "_ledger_serve_binds", dict)
    monkeypatch.setattr(dashboard_procs, "_caller_ancestor_pids", set)
    monkeypatch.setattr("gateway.status._pid_exists", lambda p: p == pid)
    monkeypatch.setattr(psutil, "Process", _FakeProcess)
    monkeypatch.setattr(
        main_dashboard,
        "_dashboard_listening",
        lambda host, port: probed.append((host, port)) or ("127.0.0.1" if (host, port) == expected_bind else None),
    )

    assert main_dashboard._report_dashboard_status() == 1
    assert probed == [expected_bind]


def test_backend_address_and_runtime_use_last_bind_flag_for_exact_and_unwrapped_argv():
    from pathlib import Path
    from hermes_cli import venv_sync
    from hermes_cli.gateway_migrate_backends_plan import backend_address
    from hermes_cli.main_dashboard import _parse_dashboard_runtime

    # Exact argv sequences follow argparse/CLI last-value semantics for repeated flags.
    repeated = [
        "hermes", "serve",
        "--host", "127.0.0.1", "--port", "9120",
        "--host", "0.0.0.0", "--port", "9121",
    ]
    assert backend_address(repeated) == ("0.0.0.0", 9121)
    assert _parse_dashboard_runtime(repeated) == ("serve", "0.0.0.0", 9121)
    assert backend_address(
        ["hermes", "serve", "--host=127.0.0.1", "--port=9120", "--host=0.0.0.0", "--port=9121"]
    ) == ("0.0.0.0", 9121)
    # Invalid, negative, or missing trailing --port on exact argv is rejected even after a prior valid port.
    assert _parse_dashboard_runtime(["hermes", "serve", "--port", "9120", "--port", "notaport"]) is None
    assert _parse_dashboard_runtime(["hermes", "serve", "--port", "9120", "--port=-1"]) is None
    assert _parse_dashboard_runtime(["hermes", "serve", "--port", "9120", "--port"]) is None

    # Unwrapped relaunched inline-bootstrap commands also preserve last-value semantics
    # whether passed as exact argv or space-joined ps text.
    py = Path("/opt/hermes/venv/bin/python")
    root = Path("/opt/hermes/hermes-agent")
    inner_argv = [
        "/home/alice/.local/bin/hermes", "serve",
        "--host", "127.0.0.1", "--port", "9120",
        "--host", "0.0.0.0", "--port", "9121",
    ]
    for shape in ("run_module", "run_path"):
        if shape == "run_module":
            relaunched = venv_sync.relaunch_command(
                py, root, inner_argv, ["/old/python", "-m", "hermes_cli.main", *inner_argv[1:]], "hermes_cli.main"
            )
        else:
            relaunched = venv_sync.relaunch_command(py, root, inner_argv, ["/old/python", *inner_argv], None)
        assert backend_address(relaunched) == ("0.0.0.0", 9121)
        assert _parse_dashboard_runtime(" ".join(relaunched)) == ("serve", "0.0.0.0", 9121)
