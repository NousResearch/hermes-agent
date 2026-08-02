from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest
import yaml

from hermes_cli import plugins


def _write_plugin(home: Path, marker: str, *, body: str | None = None) -> None:
    plugin_dir = home / "plugins" / "same-name"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "same-name", "version": "1"}),
        encoding="utf-8",
    )
    register_body = body or (
        "ctx.register_hook('pre_llm_call', lambda **kw: %r)\n"
        "    ctx.register_command('who', lambda raw: %r)\n"
        "    ctx.register_auxiliary_task('side_%s', display_name='Side', description='d')"
        % (marker, marker, marker)
    )
    (plugin_dir / "__init__.py").write_text(
        f"MARKER = {marker!r}\ndef register(ctx):\n    {register_body}\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["same-name"]}}),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _reset_profile_managers(monkeypatch):
    monkeypatch.setattr(plugins, "_plugin_manager", None)
    monkeypatch.setattr(plugins, "_plugin_managers", {})
    yield


def test_profile_keyed_managers_isolate_modules_and_dispatch(tmp_path):
    home_a = tmp_path / "profiles" / "a"
    home_b = tmp_path / "profiles" / "b"
    _write_plugin(home_a, "a")
    _write_plugin(home_b, "b")

    manager_a = plugins.get_plugin_manager(profile_home=home_a)
    manager_b = plugins.get_plugin_manager(profile_home=home_b)
    manager_a.discover_and_load()
    manager_b.discover_and_load()

    assert manager_a is plugins.get_plugin_manager(profile_home=home_a)
    assert manager_b is plugins.get_plugin_manager(profile_home=home_b)
    assert manager_a is not manager_b
    assert manager_a.profile_home == home_a.resolve()
    assert manager_b.profile_home == home_b.resolve()
    assert manager_a.module_namespace != manager_b.module_namespace
    assert manager_a._plugins["same-name"].module.MARKER == "a"
    assert manager_b._plugins["same-name"].module.MARKER == "b"
    assert manager_a._plugins["same-name"].module.__name__.startswith(
        manager_a.module_namespace + "."
    )
    assert manager_b._plugins["same-name"].module.__name__.startswith(
        manager_b.module_namespace + "."
    )

    token = plugins.bind_plugin_manager(manager_a)
    try:
        assert plugins.invoke_hook("pre_llm_call") == ["a"]
        assert plugins.get_plugin_command_handler("who") ("") == "a"
        assert [entry["key"] for entry in plugins.get_plugin_auxiliary_tasks()] == [
            "side_a"
        ]
    finally:
        plugins.reset_plugin_manager(token)

    token = plugins.bind_plugin_manager(manager_b)
    try:
        assert plugins.invoke_hook("pre_llm_call") == ["b"]
        assert plugins.get_plugin_command_handler("who") ("") == "b"
        assert [entry["key"] for entry in plugins.get_plugin_auxiliary_tasks()] == [
            "side_b"
        ]
    finally:
        plugins.reset_plugin_manager(token)


@pytest.mark.parametrize("first_marker,second_marker", [("a", "b"), ("b", "a")])
def test_bound_manager_dispatches_its_own_same_name_tool_after_peer_discovery(
    tmp_path, first_marker, second_marker
):
    from tools.registry import registry

    first_home = tmp_path / "profiles" / first_marker
    second_home = tmp_path / "profiles" / second_marker
    tool_body = (
        "ctx.register_tool('profile_probe', 'profile-probe', "
        "{'name': 'profile_probe', 'description': %r, "
        "'parameters': {'type': 'object', 'properties': {}}}, "
        "lambda args, **kwargs: %r)"
    )
    _write_plugin(first_home, first_marker, body=tool_body % (first_marker, first_marker))
    _write_plugin(second_home, second_marker, body=tool_body % (second_marker, second_marker))

    first_manager = plugins.discover_plugins(profile_home=first_home)
    plugins.discover_plugins(profile_home=second_home)

    token = plugins.bind_plugin_manager(first_manager)
    try:
        entry = registry.get_entry("profile_probe")
        assert entry is not None
        assert entry.schema["description"] == first_marker
        assert entry.handler({}) == first_marker
        assert registry.dispatch("profile_probe", {}) == first_marker
    finally:
        plugins.reset_plugin_manager(token)
        registry.clear_profile(first_home)
        registry.clear_profile(second_home)


def test_concurrent_first_discovery_runs_once(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "race"
    home.mkdir(parents=True)
    manager = plugins.get_plugin_manager(profile_home=home)
    calls = 0
    calls_lock = threading.Lock()
    gate = threading.Barrier(8)

    def discover_inner():
        nonlocal calls
        with calls_lock:
            calls += 1

    monkeypatch.setattr(manager, "_discover_and_load_inner", discover_inner)

    def discover():
        gate.wait()
        manager.discover_and_load()

    threads = [threading.Thread(target=discover) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert calls == 1
    assert manager._discovered is True


def test_failed_real_plugin_registration_rolls_back_provider_publication(tmp_path):
    from agent import image_gen_registry

    home = tmp_path / "profiles" / "failed-provider"
    _write_plugin(home, "failed")
    plugin_init = home / "plugins" / "same-name" / "__init__.py"
    plugin_init.write_text(
        """\
from agent.image_gen_provider import ImageGenProvider

class PartialProvider(ImageGenProvider):
    @property
    def name(self):
        return "partial-provider"

    def is_available(self):
        return True

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        return {"success": True}


def register(ctx):
    ctx.register_image_gen_provider(PartialProvider())
    raise RuntimeError("deliberate registration failure")
""",
        encoding="utf-8",
    )

    manager = plugins.discover_plugins(profile_home=home)
    loaded = manager._plugins["same-name"]

    try:
        assert loaded.enabled is False
        assert loaded.error == "deliberate registration failure"
        assert image_gen_registry.get_provider(
            "partial-provider", profile_key=home
        ) is None
    finally:
        image_gen_registry._reset_for_tests()


def test_failed_real_plugin_registration_rolls_back_manager_and_tool_publications(tmp_path):
    from tools.registry import registry

    home = tmp_path / "profiles" / "failed-publications"
    plugin_dir = home / "plugins" / "same-name"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "same-name", "version": "1"}),
        encoding="utf-8",
    )
    skill_path = plugin_dir / "SKILL.md"
    skill_path.write_text("# Failed skill\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        """\
from pathlib import Path


def register(ctx):
    ctx.register_tool(
        "partial_tool", "partial-tool",
        {"name": "partial_tool", "description": "partial",
         "parameters": {"type": "object", "properties": {}}},
        lambda args, **kwargs: "partial",
    )
    ctx.register_hook("pre_llm_call", lambda **kwargs: "partial")
    ctx.register_middleware("pre_tool", lambda **kwargs: "partial")
    ctx.register_command("partial", lambda raw: "partial")
    ctx.register_auxiliary_task(
        "partial_task", display_name="Partial", description="partial"
    )
    ctx.register_skill("partial", Path(__file__).with_name("SKILL.md"))
    raise RuntimeError("deliberate publication failure")
""",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["same-name"]}}),
        encoding="utf-8",
    )

    manager = plugins.discover_plugins(profile_home=home)
    loaded = manager._plugins["same-name"]

    try:
        assert loaded.enabled is False
        assert loaded.error == "deliberate publication failure"
        token = plugins.bind_plugin_manager(manager)
        try:
            assert registry.get_entry("partial_tool") is None
        finally:
            plugins.reset_plugin_manager(token)
        assert "partial_tool" not in manager._plugin_tool_names
        assert "partial" not in manager.invoke_hook("pre_llm_call")
        assert "partial" not in manager.invoke_middleware("pre_tool")
        assert "partial" not in manager._plugin_commands
        assert "partial_task" not in manager._aux_tasks
        assert "same-name:partial" not in manager._plugin_skills
    finally:
        registry.clear_profile(home)


def test_failed_real_plugin_registration_rolls_back_launch_global_publications(tmp_path):
    from agent.secret_sources import registry as secret_registry
    from gateway.platform_registry import platform_registry
    from hermes_cli.dashboard_auth import registry as dashboard_registry

    home = tmp_path / "profiles" / "failed-global"
    _write_plugin(home, "failed")
    plugin_init = home / "plugins" / "same-name" / "__init__.py"
    plugin_init.write_text(
        """\
from agent.context_engine import ContextEngine
from agent.secret_sources.base import FetchResult, SecretSource
from hermes_cli.dashboard_auth import DashboardAuthProvider


class PartialEngine(ContextEngine):
    @property
    def name(self):
        return "partial-engine"
    def update_from_response(self, usage):
        pass
    def should_compress(self, prompt_tokens=None):
        return False
    def compress(self, messages, current_tokens=None, focus_topic=None,
                 force=False, memory_context=""):
        return messages


class PartialDashboardProvider(DashboardAuthProvider):
    name = "partial-dashboard"
    display_name = "Partial dashboard"
    def start_login(self, *, redirect_uri):
        return None
    def complete_login(self, **kwargs):
        return None
    def verify_session(self, *, access_token):
        return None
    def refresh_session(self, *, refresh_token):
        return None
    def revoke_session(self, *, refresh_token):
        pass


class PartialSecretSource(SecretSource):
    name = "partial_secret"
    label = "Partial secret"
    def fetch(self, cfg, home_path):
        return FetchResult()


def register(ctx):
    ctx.register_context_engine(PartialEngine())
    ctx.register_cli_command("partial-cli", "partial", lambda parser: None)
    ctx.register_slack_action_handler("partial-action", lambda *args: None)
    ctx.register_dashboard_auth_provider(PartialDashboardProvider())
    ctx.register_secret_source(PartialSecretSource())
    ctx.register_platform(
        "partial-platform", "Partial platform", lambda cfg: object(), lambda: True
    )
    raise RuntimeError("deliberate global publication failure")
""",
        encoding="utf-8",
    )

    manager = plugins.PluginManager(
        profile_home=home,
        allow_global_publication=True,
    )
    manager.discover_and_load()
    loaded = manager._plugins["same-name"]

    assert loaded.enabled is False
    assert loaded.error == "deliberate global publication failure"
    assert manager._context_engine is None
    assert "partial-cli" not in manager._cli_commands
    assert all(entry[0] != "partial-action" for entry in manager._slack_action_handlers)
    assert "partial-platform" not in manager._plugin_platform_names
    assert dashboard_registry.get_provider("partial-dashboard") is None
    assert secret_registry._SOURCES.get("partial_secret") is None
    assert platform_registry._entries.get("partial-platform") is None


def test_force_discovery_replaces_cached_manager_without_mutating_snapshot(tmp_path):
    home = tmp_path / "profiles" / "snapshot"
    _write_plugin(home, "old")
    old = plugins.discover_plugins(profile_home=home)
    assert old._plugins["same-name"].module.MARKER == "old"

    plugin_init = home / "plugins" / "same-name" / "__init__.py"
    plugin_init.write_text(
        "MARKER = 'new'\ndef register(ctx):\n    ctx.register_hook('pre_llm_call', lambda **kw: 'new')\n",
        encoding="utf-8",
    )
    new = plugins.discover_plugins(profile_home=home, force=True)

    assert new is not old
    assert plugins.get_plugin_manager(profile_home=home) is new
    assert old._plugins["same-name"].module.MARKER == "old"
    assert old.invoke_hook("pre_llm_call") == ["old"]
    assert new._plugins["same-name"].module.MARKER == "new"
    assert new.invoke_hook("pre_llm_call") == ["new"]
    assert old._plugins["same-name"].module.__name__ in sys.modules
    assert new._plugins["same-name"].module.__name__ in sys.modules


def test_force_discovery_preserves_live_manager_tool_snapshot(tmp_path):
    from tools.registry import registry

    home = tmp_path / "profiles" / "tool-snapshot"
    tool_body = (
        "ctx.register_tool('snapshot_probe', 'snapshot-probe', "
        "{'name': 'snapshot_probe', 'description': %r, "
        "'parameters': {'type': 'object', 'properties': {}}}, "
        "lambda args, **kwargs: %r, check_fn=lambda: True)"
    )
    _write_plugin(home, "old", body=tool_body % ("old", "old"))
    old = plugins.discover_plugins(profile_home=home)

    plugin_init = home / "plugins" / "same-name" / "__init__.py"
    plugin_init.write_text(
        "MARKER = 'new'\ndef register(ctx):\n    "
        + tool_body % ("new", "new")
        + "\n",
        encoding="utf-8",
    )
    new = plugins.discover_plugins(profile_home=home, force=True)

    try:
        old_token = plugins.bind_plugin_manager(old)
        try:
            old_definition = registry.get_definitions({"snapshot_probe"})[0]
            assert old_definition["function"]["description"] == "old"
            assert registry.dispatch("snapshot_probe", {}) == "old"
        finally:
            plugins.reset_plugin_manager(old_token)

        new_token = plugins.bind_plugin_manager(new)
        try:
            new_definition = registry.get_definitions({"snapshot_probe"})[0]
            assert new_definition["function"]["description"] == "new"
            assert registry.dispatch("snapshot_probe", {}) == "new"
        finally:
            plugins.reset_plugin_manager(new_token)
    finally:
        registry.clear_profile(home)


def test_force_discovery_keeps_delayed_old_tool_callbacks_and_resets_binding(tmp_path):
    from tools.registry import registry

    home = tmp_path / "profiles" / "delayed-snapshot"
    old_body = """\
import threading
MARKER = 'old'
CHECK_STARTED = threading.Event()
CHECK_RELEASE = threading.Event()
HANDLER_STARTED = threading.Event()
HANDLER_RELEASE = threading.Event()
def check():
    CHECK_STARTED.set()
    assert CHECK_RELEASE.wait(5)
    return True
def handle(args, **kwargs):
    HANDLER_STARTED.set()
    assert HANDLER_RELEASE.wait(5)
    raise RuntimeError(MARKER)
def register(ctx):
    ctx.register_tool(
        'delayed_probe', 'delayed-probe',
        {'name': 'delayed_probe', 'description': MARKER,
         'parameters': {'type': 'object', 'properties': {}}},
        handle, check_fn=check,
    )
"""
    _write_plugin(home, "old")
    plugin_init = home / "plugins" / "same-name" / "__init__.py"
    plugin_init.write_text(old_body, encoding="utf-8")
    old = plugins.discover_plugins(profile_home=home)
    old_module = old._plugins["same-name"].module
    assert old_module is not None

    observations = {}

    def old_session():
        token = plugins.bind_plugin_manager(old)
        try:
            definitions = registry.get_definitions({"delayed_probe"})
            observations["schema"] = definitions[0]["function"]["description"]
            observations["dispatch"] = registry.dispatch("delayed_probe", {})
        finally:
            plugins.reset_plugin_manager(token)
            observations["reset"] = plugins.get_bound_plugin_manager() is None

    thread = threading.Thread(target=old_session)
    thread.start()
    assert old_module.CHECK_STARTED.wait(5)

    new_body = (
        "ctx.register_tool('delayed_probe', 'delayed-probe', "
        "{'name': 'delayed_probe', 'description': 'new', "
        "'parameters': {'type': 'object', 'properties': {}}}, "
        "lambda args, **kwargs: 'new', check_fn=lambda: True)"
    )
    plugin_init.write_text(
        f"MARKER = 'new'\ndef register(ctx):\n    {new_body}\n",
        encoding="utf-8",
    )
    new = plugins.discover_plugins(profile_home=home, force=True)

    old_module.CHECK_RELEASE.set()
    assert old_module.HANDLER_STARTED.wait(5)
    old_module.HANDLER_RELEASE.set()
    thread.join(timeout=5)

    try:
        assert not thread.is_alive()
        assert observations["schema"] == "old"
        assert "RuntimeError: old" in observations["dispatch"]
        assert observations["reset"] is True

        token = plugins.bind_plugin_manager(new)
        try:
            assert registry.get_definitions({"delayed_probe"})[0]["function"][
                "description"
            ] == "new"
            assert registry.dispatch("delayed_probe", {}) == "new"
        finally:
            plugins.reset_plugin_manager(token)
    finally:
        registry.clear_profile(home)


def test_force_discovery_reloads_same_size_source_with_unchanged_timestamp(tmp_path):
    home = tmp_path / "profiles" / "same-size"
    _write_plugin(home, "old")
    old = plugins.discover_plugins(profile_home=home)
    assert old._plugins["same-name"].module.MARKER == "old"

    plugin_init = home / "plugins" / "same-name" / "__init__.py"
    original_stat = plugin_init.stat()
    source = plugin_init.read_text(encoding="utf-8").replace("old", "new")
    plugin_init.write_text(source, encoding="utf-8")
    os.utime(
        plugin_init,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert plugin_init.stat().st_size == original_stat.st_size

    new = plugins.discover_plugins(profile_home=home, force=True)

    assert new._plugins["same-name"].module.MARKER == "new"
    assert new.invoke_hook("pre_llm_call") == ["new"]


def test_force_discovery_replaces_only_target_profile_cache(tmp_path):
    home_a = tmp_path / "profiles" / "a"
    home_b = tmp_path / "profiles" / "b"
    _write_plugin(home_a, "a-old")
    _write_plugin(home_b, "b-old")
    old_a = plugins.discover_plugins(profile_home=home_a)
    old_b = plugins.discover_plugins(profile_home=home_b)

    replacement_a = plugins.discover_plugins(profile_home=home_a, force=True)

    assert replacement_a is not old_a
    assert plugins.get_plugin_manager(profile_home=home_a) is replacement_a
    assert plugins.get_plugin_manager(profile_home=home_b) is old_b
    assert old_b._plugins["same-name"].module.MARKER == "b-old"


@pytest.mark.parametrize(
    "registration",
    [
        lambda ctx: ctx.register_cli_command("x", "x", lambda parser: None),
        lambda ctx: ctx.register_slack_action_handler("x", lambda *args: None),
        lambda ctx: ctx.register_dashboard_auth_provider(object()),
        lambda ctx: ctx.register_secret_source(object()),
        lambda ctx: ctx.register_platform("x", "X", lambda cfg: None, lambda: True),
    ],
)
def test_non_launch_manager_fails_closed_for_process_global_publication(
    tmp_path, registration
):
    manager = plugins.PluginManager(
        profile_home=tmp_path / "profiles" / "remote",
        allow_global_publication=False,
    )
    ctx = plugins.PluginContext(plugins.PluginManifest(name="scoped"), manager)

    with pytest.raises(plugins.PluginScopeError, match="launch profile"):
        registration(ctx)


def test_non_launch_manager_rejects_deferred_platform_publication(tmp_path):
    manager = plugins.PluginManager(
        profile_home=tmp_path / "profiles" / "remote",
        allow_global_publication=False,
    )
    manifest = plugins.PluginManifest(
        name="remote-platform",
        kind="platform",
        source="bundled",
    )

    with pytest.raises(plugins.PluginScopeError, match="launch profile"):
        manager._register_deferred_platform(manifest)
