"""A gateway-mounted dashboard that refuses to bind for lack of auth must exit EX_CONFIG (78).

`_configure_auth_gate` reports the refusal as `SystemExit(<message>)`. A string code reached
`run_gateway`'s SystemExit handler, which maps non-int codes to 1 — and the generated units park
the service on `RestartPreventExitStatus=78` only, so the refusal became an unbounded restart loop
with the actionable text confined to logs/gateway-exit-diag.log (#142281).
"""
import sys
import types

import pytest

import hermes_cli.gateway as gateway_cli
import gateway.run as gateway_run
from gateway.run_api import GatewayDashboardMount

MSG = "Refusing to bind dashboard to 127.0.0.1 — no auth providers are registered."


def _mount(host: str = "127.0.0.1") -> GatewayDashboardMount:
    runner = types.SimpleNamespace(session_authority=object())
    return GatewayDashboardMount(runner, types.SimpleNamespace(bound_host=host, bound_port=1))


def _gate_engaged(monkeypatch, providers):
    """Declare an operator-set public URL so the real gate logic runs for this bind."""
    from hermes_cli import web_server

    monkeypatch.setattr(web_server, "_dashboard_public_hosts", lambda: {"dashboard.example.invalid"})
    monkeypatch.setattr(web_server, "_desktop_loopback_auth_exempt", lambda *args: False)
    monkeypatch.setitem(
        sys.modules, "hermes_cli.dashboard_auth",
        types.SimpleNamespace(list_providers=lambda: providers),
    )
    return web_server


class _ProcessExit(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _stub_run_gateway_shell(monkeypatch):
    """Neutralise everything `run_gateway` does before and after `start_gateway`."""
    for name in (
        "_guard_official_docker_root_gateway",
        "_attach_to_host_gateway_or_guard",
        "_guard_supervised_gateway_conflict",
        "_guard_existing_gateway_process_conflict",
        "_apply_startup_watchdog_config",
        "_respawn_storm_backoff",
    ):
        monkeypatch.setattr(gateway_cli, name, lambda *a, **k: None)
    monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda *a, **k: False)
    monkeypatch.setattr(gateway_cli, "_stdin_is_tty", lambda: False)
    monkeypatch.setattr(gateway_cli, "_windows_console_window_attached", lambda: False)
    monkeypatch.setattr(gateway_cli, "_windows_gateway_breakaway_state", lambda: None)
    monkeypatch.setattr(gateway_cli, "_windows_gateway_should_absorb_console_controls", lambda: False)
    monkeypatch.setattr(gateway_cli, "_gateway_detached_env", lambda: False)
    monkeypatch.setattr(gateway_cli, "is_linux", lambda: False)
    monkeypatch.setattr(gateway_cli, "_make_exit_diag", lambda: (lambda *a, **k: None))
    monkeypatch.setattr(
        gateway_run, "_exit_after_graceful_shutdown",
        lambda code: (_ for _ in ()).throw(_ProcessExit(code)),
    )


def test_refusal_exits_config_code_and_prints_the_reason(monkeypatch, capsys):
    web = _gate_engaged(monkeypatch, [])

    with pytest.raises(SystemExit) as refusal:
        _mount()._bind_state(web)

    assert refusal.value.code == 78, "a permanent config refusal must exit EX_CONFIG, not 1"
    err = capsys.readouterr().err
    assert "no auth providers are registered" in err
    assert "dashboard.public_url" in err


def test_process_exit_code_is_78_through_run_gateway(monkeypatch, capsys):
    """The property the issue is about: the PROCESS exit code, not just the in-process exception."""
    _stub_run_gateway_shell(monkeypatch)
    _gate_engaged(monkeypatch, [])

    async def fake_start_gateway(**kwargs):
        runner = types.SimpleNamespace(session_authority=object())
        state = types.SimpleNamespace(bound_host="127.0.0.1", bound_port=1)
        await GatewayDashboardMount(runner, state).ensure()
        return True

    monkeypatch.setattr(gateway_run, "start_gateway", fake_start_gateway)

    with pytest.raises(_ProcessExit) as exit_code:
        gateway_cli.run_gateway()

    assert exit_code.value.code == 78
    assert "no auth providers are registered" in capsys.readouterr().err


def test_deliberate_int_code_is_not_rewritten(monkeypatch):
    web = _gate_engaged(monkeypatch, [])
    monkeypatch.setattr(
        web, "_configure_auth_gate",
        lambda *a: (_ for _ in ()).throw(SystemExit(75)),  # EX_TEMPFAIL: "restart me"
    )

    with pytest.raises(SystemExit) as refusal:
        _mount()._bind_state(web)

    assert refusal.value.code == 75


def test_provider_present_still_mounts(monkeypatch):
    web = _gate_engaged(monkeypatch, [types.SimpleNamespace(name="basic")])
    mount = _mount()

    mount._bind_state(web)

    assert web.app.state.auth_required is True, "the gated branch must be the one exercised"
    assert web.app.state.gateway_runner is mount.runner
    assert web.app.state.bound_host == "127.0.0.1"


def test_gate_not_engaged_still_mounts(monkeypatch):
    """A loopback bind with no public URL leaves the gate off (real predicate, no stub)."""
    from hermes_cli import web_server

    monkeypatch.setattr(web_server, "_dashboard_public_hosts", lambda: set())
    monkeypatch.setattr(web_server, "_desktop_loopback_auth_exempt", lambda *args: False)
    mount = _mount()

    mount._bind_state(web_server)

    assert web_server.app.state.auth_required is False
    assert web_server.app.state.gateway_runner is mount.runner


@pytest.mark.parametrize(
    "code, expected",
    [
        (78, 78),      # EX_CONFIG survives to RestartPreventExitStatus
        (75, 75),      # EX_TEMPFAIL survives to RestartForceExitStatus
        (0, 0),
        (None, 0),
        ("boom", 1),   # sys.exit("message") keeps the historical code
    ],
)
def test_system_exit_code_mapping(code, expected):
    from hermes_cli.gateway import _system_exit_code

    assert _system_exit_code(SystemExit(code)) == expected
