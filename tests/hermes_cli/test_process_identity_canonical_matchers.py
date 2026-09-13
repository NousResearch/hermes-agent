"""Process-identity contract: the two kill/relaunch predicates and the profile liveness probe
defer to the canonical matchers instead of argv substrings (root AGENTS.md process-identity rule).
"""

from __future__ import annotations

import pytest

from hermes_cli.dashboard_procs import _is_desktop_local_serve_cmdline
from hermes_cli.update_cmd_windows import _hermes_holder_subcommand, _is_backend_argv

LOOPBACK = "--host 127.0.0.1 --port 0"

# (cmdline, holder subcommand, desktop-local reap?). Substring scanners get every "trap" row wrong:
# "serve" appears inside --preserve-cache / observer.py / a flag value.
CMDLINES = [
    ("python -m hermes_cli.main serve " + LOOPBACK, "serve", True),
    ("/venv/bin/hermes serve --isolated --host=127.0.0.1 --port=0 --ssh-owner-nonce abc", "serve", True),
    ("hermes --profile ops serve " + LOOPBACK, "serve", True),
    ("hermes -m serve kanban --preserve-cache " + LOOPBACK, "kanban", False),
    ("python -m hermes_cli.main kanban --preserve-cache " + LOOPBACK, "kanban", False),
    ("hermes --reasoning high dashboard " + LOOPBACK, "dashboard", False),
    ("hermes gateway run --replace", "gateway", False),
    ("hermes chat --model serve", "chat", False),
    ("python observer.py serve " + LOOPBACK, None, False),
]


@pytest.mark.parametrize("cmdline,subcommand,reapable", CMDLINES)
def test_kill_and_relaunch_predicates_agree_with_the_canonical_holder_matcher(cmdline, subcommand, reapable):
    assert _hermes_holder_subcommand(cmdline) == subcommand
    # Desktop-local reap (a KILL path): serve + loopback + ephemeral port, decided by tokens.
    assert _is_desktop_local_serve_cmdline(cmdline) is reapable
    # Windows updater backend classifier (stop + relaunch path).
    assert _is_backend_argv(cmdline.lower()) is (subcommand in ("serve", "dashboard"))


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
