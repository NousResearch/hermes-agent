"""Secondary profile preparation preserves loop responsiveness and profile scope."""

import asyncio
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest
import hermes_yaml as yaml


@pytest.mark.asyncio
async def test_secondary_profile_load_keeps_loop_available_and_scopes_plugins(
    tmp_path, monkeypatch
):
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner, _profile_runtime_scope
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    from hermes_constants import get_hermes_home
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    home = tmp_path / ".hermes"
    profiles = [home / "profiles" / name for name in ("first", "second")]
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PROFILE_PREPARATION_TOKEN", "launch")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    entered, loop_progress, release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    loaded, responsive, rewired = [], [], []

    def registered(profile_home, token):
        loaded.append((profile_home, token))
        if profile_home == profiles[1]:
            entered.set()
            assert release.wait(10), "profile preparation was never released"

    monkeypatch.setitem(
        sys.modules, "profile_preparation_probe", SimpleNamespace(registered=registered)
    )
    for index, profile in enumerate(profiles):
        plugin = profile / "plugins" / f"preparation_{index}"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text(
            f"name: preparation_{index}\nversion: '0.1'\ndescription: t\n"
        )
        (plugin / "__init__.py").write_text("""
def register(ctx):
    from agent.secret_scope import get_secret
    from hermes_constants import get_hermes_home
    from profile_preparation_probe import registered
    registered(get_hermes_home(), get_secret("PROFILE_PREPARATION_TOKEN"))
    ctx.register_command("prepared", lambda raw: "ready", description="prepared")
""")
        (profile / ".env").write_text(f"PROFILE_PREPARATION_TOKEN=profile-{index}\n")
        (profile / "config.yaml").write_text(
            yaml.safe_dump({
                "plugins": {"enabled": [f"preparation_{index}"]},
                "platforms": {
                    "telegram": {
                        "enabled": False,
                        "token": "${PROFILE_PREPARATION_TOKEN}",
                    }
                },
                "hooks": {
                    "outbound": [
                        {
                            "url": f"https://example.invalid/{index}",
                            "events": ["on_session_start"],
                        }
                    ]
                },
            })
        )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {}
    runner._profile_adapters = {}
    loop_thread = threading.get_ident()

    class PreparationAdapter(RestartTestAdapter):
        def rewire_plugin_handlers(self) -> None:
            rewired.append((get_hermes_home(), threading.get_ident()))

    runner._profile_adapters["first"] = {Platform.TELEGRAM: PreparationAdapter()}
    context = set_multiplex_context(True)
    before = dict(os.environ)

    def watchdog():
        if entered.wait(10):
            responsive.append(loop_progress.wait(5))
        release.set()

    watcher = threading.Thread(target=watchdog)
    try:
        first = await runner._load_secondary_profile_config("first", profiles[0])
        watcher.start()
        task = asyncio.create_task(
            runner._load_secondary_profile_config("second", profiles[1])
        )
        assert await asyncio.to_thread(entered.wait, 10), (
            "plugin registration was never entered"
        )
        loop_progress.set()
        second = await task
        again = await runner._load_secondary_profile_config("first", profiles[0])
        with _profile_runtime_scope(profiles[0], hydrate_secrets=False):
            manager = get_plugin_manager()
            initial_hooks = len(manager._hooks.get("on_session_start", []))
            late = profiles[0] / "plugins" / "late_prepared"
            late.mkdir()
            (late / "plugin.yaml").write_text(
                "name: late_prepared\nversion: '0.1'\ndescription: t\n"
            )
            (late / "__init__.py").write_text(
                'def register(ctx):\n    ctx.register_command("late_prepared", lambda raw: "ready", description="late")\n'
            )
            cfg = yaml.safe_load((profiles[0] / "config.yaml").read_text())
            cfg["plugins"]["enabled"].append("late_prepared")
            (profiles[0] / "config.yaml").write_text(yaml.safe_dump(cfg))
            await asyncio.to_thread(discover_plugins, force=True)
        await asyncio.sleep(0)
        subscriptions = runner._plugin_rewire_unsubscribe
        assert subscriptions is not None
        assert {
            "responsive": responsive,
            "loaded": loaded,
            "tokens": [
                cfg.platforms[Platform.TELEGRAM].token for cfg in (first, second, again)
            ],
            "subscriptions": set(subscriptions),
            "configured_hooks": initial_hooks > 0,
            "rewired": rewired,
            "ambient_home": get_hermes_home(),
            "environment": dict(os.environ),
        } == {
            "responsive": [True],
            "loaded": [
                (profiles[0], "profile-0"),
                (profiles[1], "profile-1"),
                (profiles[0], "profile-0"),
            ],
            "tokens": ["profile-0", "profile-1", "profile-0"],
            "subscriptions": {str(profile.resolve()) for profile in profiles},
            "configured_hooks": True,
            "rewired": [(profiles[0], loop_thread)],
            "ambient_home": home,
            "environment": before,
        }
    finally:
        loop_progress.set()
        release.set()
        if watcher.ident is not None:
            watcher.join(10)
        for unsubscribe in (runner._plugin_rewire_unsubscribe or {}).values():
            unsubscribe()
        reset_multiplex_context(context)


@pytest.mark.asyncio
async def test_secondary_profile_recovers_catalog_plugins_before_discovery(
    tmp_path, monkeypatch
):
    from agent.secret_scope import (
        get_secret,
        reset_multiplex_context,
        set_multiplex_context,
    )
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from hermes_cli import left_core_migration, memory_provider_migration, plugins_cmd
    from hermes_constants import get_hermes_home
    from tools.terminal_scope import get_terminal_scope, terminal_env

    home = tmp_path / ".hermes"
    home.mkdir()
    profiles = [home / "profiles" / name for name in ("first", "second")]
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HASS_TOKEN", "launch")
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setattr(left_core_migration, "_attempted", set())
    monkeypatch.setattr(left_core_migration, "_undelivered", {})
    monkeypatch.setattr(memory_provider_migration, "catalog_source", lambda name: name)
    loop_thread = threading.get_ident()
    installed, loaded = [], {}

    def observation():
        return (
            get_hermes_home(),
            get_secret("HASS_TOKEN"),
            terminal_env("TERMINAL_CWD"),
            threading.get_ident() != loop_thread,
        )

    def registered():
        loaded[get_hermes_home()] = observation()

    monkeypatch.setitem(
        sys.modules,
        "profile_recovery_probe",
        SimpleNamespace(registered=registered),
    )
    for index, profile in enumerate(profiles):
        profile.mkdir(parents=True)
        (profile / ".env").write_text(f"HASS_TOKEN=profile-{index}\n")
        (profile / "config.yaml").write_text(
            yaml.safe_dump({
                "security": {"allow_lazy_installs": True},
                "terminal": {"cwd": str(profile / "before")},
            })
        )

    def install(source, *, force, enable, catalog_name, assume_deps_consent):
        profile = get_hermes_home()
        installed.append(observation())
        plugin = profile / "plugins" / catalog_name
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text(
            f"name: {catalog_name}\nversion: '0.1'\ndescription: t\n"
        )
        (plugin / "__init__.py").write_text(
            "def register(ctx):\n"
            "    from profile_recovery_probe import registered\n"
            "    registered()\n"
            '    ctx.register_command("recovered", lambda raw: "ready", description="recovered")\n'
        )
        cfg = yaml.safe_load((profile / "config.yaml").read_text())
        cfg["plugins"] = {"enabled": [catalog_name]}
        cfg["terminal"]["cwd"] = str(profile / "after")
        (profile / "config.yaml").write_text(yaml.safe_dump(cfg))
        return {"ok": True}

    monkeypatch.setattr(plugins_cmd, "dashboard_install_plugin", install)
    runner = GatewayRunner(
        GatewayConfig(
            sessions_dir=tmp_path / "sessions",
            multiplex_profiles=True,
        )
    )
    context = set_multiplex_context(True)
    before = dict(os.environ)
    try:
        for profile in (profiles[0], profiles[1], profiles[0]):
            await runner._load_secondary_profile_config(profile.name, profile)
        assert {
            "installed": installed,
            "loaded": loaded,
            "ambient_home": get_hermes_home(),
            "terminal_scope": get_terminal_scope(),
            "environment": dict(os.environ),
        } == {
            "installed": [
                (profile, f"profile-{index}", str(profile / "before"), True)
                for index, profile in enumerate(profiles)
            ],
            "loaded": {
                profile: (profile, f"profile-{index}", str(profile / "after"), True)
                for index, profile in enumerate(profiles)
            },
            "ambient_home": home,
            "terminal_scope": None,
            "environment": before,
        }
    finally:
        for unsubscribe in (runner._plugin_rewire_unsubscribe or {}).values():
            unsubscribe()
        reset_multiplex_context(context)
        runner.session_store.close_all_db_handles()
