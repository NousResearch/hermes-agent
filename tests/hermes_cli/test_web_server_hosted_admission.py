"""Standalone ``hermes dashboard`` / ``hermes serve`` run the hosted boot admission before they serve.

The gateway's boot runs :func:`hermes_cli.hosted_mode.assert_hosted_boot` before anything binds;
the standalone web backend (``main:cmd_dashboard`` -> ``web_server:start_server``, also the
update resume's relaunch) serves HTTP without that boot, so it runs the same admission at its
own entry. These tests drive the real ``start_server`` and, when it gets that far, the real
uvicorn startup on port 0 with an actual HTTP request: a hosted process whose storage refuses
exits with the fatal-config code and never binds; a local process serves exactly as before.
"""

from __future__ import annotations

import asyncio
import sys
import urllib.request
from contextlib import asynccontextmanager

import pytest


@pytest.fixture
def served(monkeypatch):
    """Seams for the unrelated background services and auth of a real ``start_server`` run — NOT
    for hosted admission. ``_run_serve`` drives the real uvicorn startup on the real app and
    issues one HTTP request; the returned list records each status it served."""
    from hermes_cli import hosted_code_fence, web_server

    monkeypatch.setattr(hosted_code_fence, "_armed_by_start_flag", False)
    monkeypatch.setattr("hermes_cli.update_proxy_env.restore_user_proxy_env", lambda: None)
    monkeypatch.setattr("hermes_cli.nous_auth_keepalive.start_nous_auth_keepalive", lambda: None)
    monkeypatch.setattr(web_server, "_configure_auth_gate", lambda *a: None)
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr("hermes_cli.isolated_scope.record_launch_scope", lambda **k: None)
    monkeypatch.setattr("tui_gateway.server.install_exit_flush_signal_handlers", lambda: None)
    monkeypatch.setattr("tui_gateway.launch_profile_policy.activate_multi_profile_hosting_eagerly",
                        lambda: None)
    monkeypatch.setattr(web_server, "_on_server_started", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.observability.shared_metrics_startup.record_process_ready",
                        lambda *a, **k: None)

    @asynccontextmanager
    async def empty_lifespan(app):
        yield

    monkeypatch.setattr(web_server.app.router, "lifespan_context", empty_lifespan)
    statuses: list[int] = []

    async def drive(config):
        import uvicorn

        config.load()
        server = uvicorn.Server(config)
        server.lifespan = config.lifespan_class(config)
        await server.startup()
        try:
            port = server.servers[0].sockets[0].getsockname()[1]

            def get():
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/openapi.json") as response:
                    return response.status

            statuses.append(await asyncio.to_thread(get))
        finally:
            await server.shutdown()

    monkeypatch.setattr(web_server, "_run_serve",
                        lambda serve, config, host, port: asyncio.run(drive(config)))
    return statuses


def _hosted_deployment_reaching_storage(monkeypatch, attempts):
    """A hosted deployment whose every other admission condition passes, so the verdict is the
    storage check's: the import-free preflight and loaded-code checks have their own tests; the
    admission entry, ``_require_hosted_backends`` and its storage call are real, and the storage
    check refuses like the built-in SQLite would."""
    from hermes_cli import hosted_mode
    import persistence_assembly
    from persistence import IsolationUnsupported

    config = {"hosted": {"enabled": True, "identity_verifier": "stub-verifier"}}
    monkeypatch.setattr(hosted_mode, "_VERDICT", {"hosted": True, "config": config,
                                                  "plugins_permitted": True})
    monkeypatch.setattr(hosted_mode, "hosted_boot_preflight", lambda **kwargs: None)
    monkeypatch.setattr(hosted_mode, "_refuse_direct_platform_adapters", lambda gateway_config: None)
    monkeypatch.setattr(hosted_mode, "_refuse_non_sandbox_terminal", lambda config: None)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda *a, **k: None)

    def refuse():
        attempts.append("storage-check")
        raise IsolationUnsupported("sqlite must not serve hosted traffic")

    monkeypatch.setattr(persistence_assembly, "require_hosted_persistence", refuse)


@pytest.mark.parametrize("headless", [False, True], ids=["dashboard", "serve"])
def test_a_hosted_standalone_server_refuses_before_serving_when_storage_refuses(
        monkeypatch, capsys, served, headless):
    from hermes_cli import web_server
    from hermes_cli.hosted_code_fence import HOSTED_BOOT_REFUSED_EXIT

    monkeypatch.setattr(sys, "argv", ["hermes", "serve" if headless else "dashboard"])
    attempts: list[str] = []
    _hosted_deployment_reaching_storage(monkeypatch, attempts)

    with pytest.raises(SystemExit) as exit_info:
        web_server.start_server(host="127.0.0.1", port=0, open_browser=False, headless=headless)
    assert exit_info.value.code == HOSTED_BOOT_REFUSED_EXIT == 78
    assert attempts == ["storage-check"], "hosted server reached HTTP without storage admission"
    assert served == [], "the listener served before hosted admission"
    out = capsys.readouterr().out
    assert "Hosted boot refused:" in out and "isolate tenants in storage" in out


def test_the_start_flag_alone_refuses_a_non_hosted_config(monkeypatch, capsys, served):
    """Either hosted signal runs the admission: a ``--hosted`` start on a deployment whose
    config is not hosted is the mismatch the gateway preflight refuses, and nothing serves."""
    from hermes_cli import hosted_mode, web_server

    monkeypatch.setattr(sys, "argv", ["hermes", "serve", "--hosted"])
    monkeypatch.setattr(hosted_mode, "_VERDICT", {"hosted": False, "config": {},
                                                  "plugins_permitted": False})
    with pytest.raises(SystemExit) as exit_info:
        web_server.start_server(host="127.0.0.1", port=0, open_browser=False, headless=True)
    assert exit_info.value.code == 78
    assert served == []
    assert "Hosted boot refused:" in capsys.readouterr().out


@pytest.mark.parametrize("headless", [False, True], ids=["dashboard", "serve"])
def test_a_local_standalone_server_serves_without_hosted_checks(monkeypatch, served, headless):
    from hermes_cli import hosted_mode, web_server

    monkeypatch.setattr(sys, "argv", ["hermes", "serve" if headless else "dashboard"])
    monkeypatch.setattr(hosted_mode, "_VERDICT", {"hosted": False, "config": {},
                                                  "plugins_permitted": False})

    def must_not_run(*args, **kwargs):
        raise AssertionError("a hosted admission check ran on a local install")

    for check in ("_refuse_credential_environment", "_refuse_profile_secret_files",
                  "_refuse_non_sandbox_terminal", "_refuse_direct_platform_adapters",
                  "_refuse_uncurated_inprocess_code", "_require_hosted_backends"):
        monkeypatch.setattr(hosted_mode, check, must_not_run)

    web_server.start_server(host="127.0.0.1", port=0, open_browser=False, headless=headless)
    assert served == [200]
