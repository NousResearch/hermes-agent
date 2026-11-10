"""The gateway runs the hosted boot checks before it serves, and freezes the registries at the end
of hosted boot. A local install consults neither (the switch is off), which the fake-runner boot
below drives through the real ``start_gateway``.
"""

from __future__ import annotations

import threading
from unittest.mock import AsyncMock

import pytest

from gateway import run_bootstrap


@pytest.fixture
def _startup(monkeypatch, tmp_path):
    """Drive the real ``start_gateway`` with a fake runner; record startup milestones. Modelled on
    the deferred-integrity-scan boot fixture (a full wiring pass with nothing real started)."""
    from agent import memory_provider
    from gateway import lifecycle_ledger as ledger
    from gateway import run_api, run_runtime
    from gateway import run as run_mod

    events, release = [], threading.Event()

    class _LiveRunner:
        def __init__(self, config):
            from gateway.session import SessionStore
            from hermes_constants import get_hermes_home
            from hermes_state import SessionDB

            self.session_store = SessionStore(get_hermes_home() / "sessions", config)
            self._session_db = SessionDB(get_hermes_home() / "state.db")
            self.session_store._db = self._session_db
            self._draining = False
            self._running = False
            self._closed = False
            self.config = config
            self.should_exit_cleanly = False
            self.exit_reason = None
            self.exit_code = None
            self.adapters = {}

        async def start(self):
            events.append("runner_started")
            self._running = True
            return True

        def _start_systemd_watchdog(self):
            pass

        async def stop(self):
            self._running = False
            if not self._closed:
                self._closed = True
                self.session_store.close_all_db_handles()
                self._session_db.close()

    monkeypatch.setattr(run_mod, "_host_attach_or_none", AsyncMock(return_value=None))
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_mod, "GatewayRunner", _LiveRunner)
    monkeypatch.setattr(ledger, "record_startup", lambda: None)
    monkeypatch.setattr(run_bootstrap, "_start_gateway_start_cron_and_housekeeping",
                        lambda runner: (threading.Event(), None, None, None))
    monkeypatch.setattr(run_bootstrap, "_start_gateway_shutdown_tail", AsyncMock(return_value=True))
    monkeypatch.setattr(run_runtime, "publish_gateway_runtime_ready", lambda runner: events.append("ready"))
    monkeypatch.setattr(run_runtime, "wait_gateway_runtime", AsyncMock())
    monkeypatch.setattr(run_api, "warm_gateway_dashboard", lambda runner: None)
    yield events
    release.set()
    from gateway import host_rendezvous as hr

    hr.release_host_lock(hr.ROLE_GATEWAY)
    hr.clear_record(hr.ROLE_GATEWAY)


@pytest.mark.asyncio
async def test_a_local_boot_never_consults_the_hosted_checks(_startup, monkeypatch):
    """hosted false (the default): the boot check runs and consults nothing — no refusal check
    fires, no freeze fires — and the local install's gateway starts exactly as before."""
    from hermes_cli import hosted_mode

    def _must_not_run(*args, **kwargs):
        raise AssertionError("a hosted boot check ran on a local install")

    for check in ("_refuse_credential_environment", "_refuse_profile_secret_files",
                  "_refuse_non_sandbox_terminal", "_refuse_direct_platform_adapters",
                  "_refuse_uncurated_inprocess_code", "_require_hosted_backends"):
        monkeypatch.setattr(hosted_mode, check, _must_not_run)
    from gateway.config import GatewayConfig

    assert await run_bootstrap.start_gateway(config=GatewayConfig(), replace=False,
                                             verbosity=None) is True
    assert _startup == ["runner_started", "ready"]
    import registry_freeze

    assert registry_freeze.registries_frozen() is False  # the freeze never fired


@pytest.mark.asyncio
async def test_a_hosted_refusal_stops_the_boot_before_anything_serves(_startup, monkeypatch):
    """The checks run before the host attach, the PID claim or the runner: a refused condition
    exits with the fatal-config code and nothing started."""
    from hermes_cli import hosted_mode
    from hermes_cli.hosted_mode import HostedBootRefused

    monkeypatch.setattr(hosted_mode, "is_hosted", lambda: True)

    def refuse(*args, **kwargs):
        raise HostedBootRefused("a condition the process may not serve with. the fix")

    monkeypatch.setattr(hosted_mode, "assert_hosted_boot", refuse)
    # The real start command armed the code fence at process entry, so the preflight's
    # arm-confirmation sees it; this in-process boot simulation fakes exactly that.
    monkeypatch.setattr("hermes_cli.hosted_code_fence.armed_by_start_flag", lambda: True)
    from gateway.config import GatewayConfig
    from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE

    with pytest.raises(SystemExit) as exit_info:
        await run_bootstrap.start_gateway(config=GatewayConfig(), replace=False, verbosity=None)
    assert exit_info.value.code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    assert _startup == []  # no runner, no adapters, nothing served


# A user plugin whose module-level code writes a marker file: the boot-level probe for whether
# plugin code can run before the hosted refusal (config resolution discovers and imports plugins,
# so the refusal must come first) — and for a local boot importing it exactly as before.
_MARKER_PLUGIN = '''
from pathlib import Path

Path(__file__).with_name("imported").write_text("ran")


def register(ctx):
    pass
'''


def _marker_plugin(name):
    """Install an enabled, non-curated user plugin under the test home; its marker is a file the
    plugin's own directory never gets unless its module really imported."""
    import hermes_yaml as yaml

    from hermes_constants import get_hermes_home

    plugin = get_hermes_home() / "plugins" / name
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "__init__.py").write_text(_MARKER_PLUGIN, encoding="utf-8")
    (plugin / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "1.0"}, sort_keys=False), encoding="utf-8")
    (get_hermes_home() / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": [name]}}, sort_keys=False), encoding="utf-8")
    return plugin / "imported"


@pytest.mark.asyncio
async def test_uncurated_user_plugin_code_never_runs_before_the_hosted_refusal(_startup, monkeypatch):
    """An enabled, non-curated user plugin whose module-level code writes a marker: the hosted
    boot refuses BEFORE the config resolution that would import it — the marker never appears,
    nothing starts, and the exit is the fatal-config one."""
    import hermes_yaml as yaml

    from hermes_constants import get_hermes_home
    from hermes_cli import hosted_mode

    marker = _marker_plugin("boot-hosted-probe")
    config_path = get_hermes_home() / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config["hosted"] = {"enabled": True}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(hosted_mode, "_VERDICT",
                        {"hosted": None, "config": None, "plugins_permitted": False})
    # The real start command armed the code fence at process entry (--hosted) and the
    # preflight's install is an idempotent no-op there; this in-process boot simulation fakes
    # both — an in-process fence is irremovable and would fence the local-boot tests below.
    monkeypatch.setattr("hermes_cli.hosted_code_fence.armed_by_start_flag", lambda: True)
    monkeypatch.setattr("hermes_cli.hosted_code_fence.install_hosted_code_fence", lambda: False)
    # The runner's own environment carries credential-shaped tool settings that are this test's
    # noise, not the probe's subject (the environment check has its own tests): skip it here so
    # the refusal under test is the plugin's.
    monkeypatch.setattr(hosted_mode, "_refuse_credential_environment", lambda env=None: None)
    from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE

    with pytest.raises(SystemExit) as exit_info:
        await run_bootstrap.start_gateway(replace=False, verbosity=None)
    assert exit_info.value.code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    assert _startup == []  # no runner, no adapters, nothing served
    assert not marker.exists()  # its module-level code never ran


@pytest.mark.asyncio
async def test_a_local_boot_still_runs_user_plugin_code(_startup, monkeypatch):
    """The same enabled user plugin on a local install: the boot resolves config (which imports
    it) exactly as before — the marker is written and the gateway serves."""
    from hermes_cli import hosted_mode

    marker = _marker_plugin("boot-local-probe")
    monkeypatch.setattr(hosted_mode, "_VERDICT",
                        {"hosted": None, "config": None, "plugins_permitted": False})

    assert await run_bootstrap.start_gateway(replace=False, verbosity=None) is True
    assert _startup == ["runner_started", "ready"]
    assert marker.exists()  # a local boot imported it, exactly as before


_LOCAL_LOGIN_PLUGIN = '''
from agent.vault_backends.base import LoginBackend


class Backend(LoginBackend):
    name, prefix, display_name = "probevault", "probevault:", "Probe Vault"

    def list_items(self):
        return []

    def get_meta(self, handle):
        return None

    def resolve_password(self, handle):
        raise AssertionError("registry tests never resolve passwords")


def register(ctx):
    ctx.register_login_backend(Backend())
'''


@pytest.mark.asyncio
async def test_a_local_boot_loads_a_login_backend_from_a_legacy_catalog_record(_startup, monkeypatch):
    """A catalog install whose installer record predates any newer record fields still
    registers its login backend on a local boot: the record's own tier and pin decide
    (the installer wrote it, not the plugin), and a local home is its user's own machine.
    Driven through the real ``start_gateway`` — config resolution discovers and imports
    the plugin, which registers through the real PluginManager."""
    import json

    import hermes_yaml as yaml

    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    plugin = home / "plugins" / "probe-vault"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text('name: probe-vault\nversion: "1.0"\n', encoding="utf-8")
    (plugin / "__init__.py").write_text(_LOCAL_LOGIN_PLUGIN, encoding="utf-8")
    sha = "a" * 40
    record = {"source": "https://github.com/example/probe-vault", "revision": sha, "pinned": True,
              "catalog": {"name": "probe-vault", "repo": "https://github.com/example/probe-vault",
                          "tier": "verified", "pin": sha, "sha": sha}}
    (home / "plugins" / ".install-metadata.json").write_text(json.dumps({"probe-vault": record}),
                                                             encoding="utf-8")
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["probe-vault"]}, "vault": {"probevault": {"enabled": True}}},
                       sort_keys=False), encoding="utf-8")

    assert await run_bootstrap.start_gateway(replace=False, verbosity=None) is True
    assert _startup == ["runner_started", "ready"]
    from agent import vault_backend_registry

    assert vault_backend_registry.get_provider("probevault", scope=str(home)) is not None


@pytest.mark.asyncio
async def test_the_registries_freeze_at_the_end_of_hosted_boot(_startup, monkeypatch):
    """Hosted boot finishes once everything the process serves is loaded: the freeze runs after
    the runner started (adapters, MCP) and before the runtime is published ready."""
    from hermes_cli import hosted_mode

    events = _startup
    monkeypatch.setattr(hosted_mode, "is_hosted", lambda: True)
    monkeypatch.setattr(hosted_mode, "hosted_boot_preflight", lambda **kwargs: None)
    monkeypatch.setattr(hosted_mode, "assert_hosted_boot", lambda *a, **k: None)
    monkeypatch.setattr(hosted_mode, "finish_hosted_boot",
                        lambda: events.append("registries_frozen"))
    from gateway.config import GatewayConfig

    assert await run_bootstrap.start_gateway(config=GatewayConfig(), replace=False,
                                             verbosity=None) is True
    assert events == ["runner_started", "registries_frozen", "ready"]
