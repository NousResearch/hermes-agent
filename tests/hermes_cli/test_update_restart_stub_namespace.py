"""Namespace regression guard for the update flow's gateway-restart stubs.

``hermes_cli.main`` re-exports the update helpers through a lazy module
``__getattr__``, and most of ``update_cmd``'s own call sites resolve through
``_m()`` so the historical ``hermes_cli.main.<helper>`` monkeypatch surface
keeps working.  Two restart-phase helpers are the exception:
``_restart_macos_launchd_gateways`` and ``_finish_dashboard_update_cleanup``
are invoked as bare module globals inside ``hermes_cli.update_cmd``, so a test
stub only intercepts them when patched on the ``hermes_cli.update_cmd``
namespace.  A stub patched on ``hermes_cli.main`` looks like isolation but
never intercepts (review finding on #96435).

These tests pin that binding so a future refactor that moves the call sites
(or the fixtures' patch targets) to the wrong namespace fails loudly instead
of silently re-arming the live-gateway escape on developer machines.

This environment cannot run launchd (Linux), so the runtime test drives
``_run_pending_fleet_restart`` — whose launchd call is the same bare-global
binding as the ``_cmd_update_impl`` restart phase — with ``is_macos`` forced
True, and asserts that the sentinel installed on ``hermes_cli.update_cmd`` is
the object the call site actually invokes.
"""

import dis
import types

import hermes_cli.gateway as hermes_gateway
import hermes_cli.main as hermes_main
import hermes_cli.update_cmd as update_cmd


def _loaded_global_names(code):
    """Names a code object (or any nested one) loads via LOAD_GLOBAL.

    ``co_names`` alone cannot distinguish a bare-global call from an
    attribute access (``_m()._helper`` also lands there), so inspect the
    actual instructions.
    """
    names = {
        ins.argval
        for ins in dis.get_instructions(code)
        if ins.opname == "LOAD_GLOBAL"
    }
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _loaded_global_names(const)
    return names


def test_launchd_stub_on_update_cmd_namespace_reaches_call_site(monkeypatch):
    """The recording sentinel patched on hermes_cli.update_cmd IS what the
    bare-global launchd call site invokes."""
    launchd_calls = []
    kill_calls = []
    probe_state = {"calls": 0}

    def _fake_find_gateway_pids(*a, **kw):
        # Non-empty on the first probe so the phase does not early-return;
        # empty afterwards so the leftover sweep has nothing to stop.
        probe_state["calls"] += 1
        return [424242] if probe_state["calls"] == 1 else []

    monkeypatch.setattr(hermes_main, "_purge_stale_hermes_modules", lambda: None)
    monkeypatch.setattr(hermes_gateway, "find_gateway_pids", _fake_find_gateway_pids)
    monkeypatch.setattr(hermes_gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(hermes_gateway, "is_macos", lambda: True)
    monkeypatch.setattr(hermes_gateway, "is_windows", lambda: False)
    monkeypatch.setattr(
        hermes_gateway,
        "kill_gateway_processes",
        lambda *a, **kw: kill_calls.append((a, kw)),
    )
    monkeypatch.setattr(
        hermes_gateway, "_wait_for_gateway_exit", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "hermes_cli.update_cmd._restart_macos_launchd_gateways",
        lambda *a, **kw: launchd_calls.append((a, kw)),
    )

    assert update_cmd._run_pending_fleet_restart() is True

    assert len(launchd_calls) == 1, (
        "the update_cmd-namespace stub was not what the launchd call site "
        "invoked — the bare-global binding has moved; move the no-live-gateway "
        "fixtures' patch target with it"
    )
    args, _kw = launchd_calls[0]
    assert isinstance(args[0], list) and isinstance(args[1], list)
    assert kill_calls == []  # no real process was ever a stop candidate


def test_restart_helpers_bind_as_update_cmd_globals(monkeypatch):
    """The restart-phase call sites resolve these helpers from
    hermes_cli.update_cmd's module globals, and a hermes_cli.main patch does
    not reach them."""
    cases = {
        "_restart_macos_launchd_gateways": (
            update_cmd._cmd_update_impl,
            update_cmd._run_pending_fleet_restart,
        ),
        "_finish_dashboard_update_cleanup": (
            update_cmd._cmd_update_impl,
            update_cmd._update_via_zip,
        ),
    }
    for name, call_site_holders in cases.items():
        update_cmd_sentinel = lambda *a, **kw: None  # noqa: E731
        main_decoy = lambda *a, **kw: None  # noqa: E731
        monkeypatch.setattr(f"hermes_cli.update_cmd.{name}", update_cmd_sentinel)
        monkeypatch.setattr(hermes_main, name, main_decoy, raising=False)
        for func in call_site_holders:
            # The call site's global namespace is update_cmd's own dict...
            assert func.__globals__ is vars(update_cmd), (
                f"{func.__name__} no longer lives in hermes_cli.update_cmd; "
                f"re-verify where the {name} stubs must be patched"
            )
            # ...the name is referenced there as a bare global (not via
            # _m()), so the update_cmd patch is the one that intercepts...
            assert name in _loaded_global_names(func.__code__), (
                f"{func.__name__} no longer references {name} as a bare "
                f"global — if it now routes through _m(), move the stubs to "
                f"the hermes_cli.main surface"
            )
            # ...and a hermes_cli.main patch is a dead stub for it.
            assert func.__globals__[name] is update_cmd_sentinel
            assert func.__globals__[name] is not main_decoy
