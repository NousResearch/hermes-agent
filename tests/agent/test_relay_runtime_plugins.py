"""Tests for native NeMo Relay plugin configuration ownership."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from agent import relay_runtime


class _FakeRelay:
    def __init__(
        self,
        *,
        initialize_error: Exception | None = None,
        dynamic_initialize_error: Exception | None = None,
        activation_close_error: Exception | None = None,
    ) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.initialize_error = initialize_error
        self.dynamic_initialize_error = dynamic_initialize_error
        self.activation_close_error = activation_close_error
        self.dynamic_plugin_specs: list[dict[str, Any]] = []
        self.ScopeType = SimpleNamespace(Agent="agent")
        self.plugin = SimpleNamespace(
            initialize=self._initialize_plugins,
            initialize_with_dynamic_plugins=self._initialize_dynamic_plugins,
            load_dynamic_plugin_activation_specs=self._load_dynamic_plugin_specs,
            clear_async=self._clear_plugins_async,
        )
        self.scope = SimpleNamespace(
            push=self._scope_push,
            pop=self._scope_pop,
        )
        self.subscribers = SimpleNamespace(flush_async=self._flush_async)

    def get_scope_stack(self) -> None:
        return None

    async def _initialize_plugins(self, config: dict[str, Any]) -> dict[str, Any]:
        self.events.append(("plugin.initialize", config))
        if self.initialize_error is not None:
            raise self.initialize_error
        return {"diagnostics": []}

    async def _initialize_dynamic_plugins(
        self,
        config: dict[str, Any],
        dynamic_plugins: list[dict[str, Any]],
    ) -> Any:
        self.events.append(("plugin.initialize_dynamic", config, dynamic_plugins))
        if self.dynamic_initialize_error is not None:
            raise self.dynamic_initialize_error

        relay = self

        class _Activation:
            async def close(self) -> None:
                relay.events.append(("plugin.activation.close",))
                if relay.activation_close_error is not None:
                    raise relay.activation_close_error

        return _Activation()

    def _load_dynamic_plugin_specs(self, config_path: Any) -> list[dict[str, Any]]:
        self.events.append(("plugin.load_dynamic_specs", str(config_path)))
        return self.dynamic_plugin_specs

    async def _clear_plugins_async(self) -> None:
        self.events.append(("plugin.clear_async",))

    def _scope_push(self, name: str, scope_type: Any, **kwargs: Any) -> Any:
        handle = ("scope", name, len(self.events))
        self.events.append(("scope.push", name, scope_type, kwargs))
        return handle

    def _scope_pop(self, handle: Any, **kwargs: Any) -> None:
        self.events.append(("scope.pop", handle, kwargs))

    async def _flush_async(self) -> None:
        self.events.append(("subscribers.flush_async",))


class _ConcurrentPublicationRelay(_FakeRelay):
    def __init__(self) -> None:
        super().__init__()
        self.publication_finished = threading.Event()

    async def _flush_async(self) -> None:
        self.events.append(("subscribers.flush_async",))
        assert await asyncio.to_thread(self.publication_finished.wait, 5)


@pytest.fixture(autouse=True)
def _reset_runtime():
    relay_runtime._reset_for_tests()
    yield
    relay_runtime._reset_for_tests()


@pytest.fixture
def explicit_static_config(tmp_path, monkeypatch):
    config = tmp_path / "plugins.toml"
    config.write_text("", encoding="utf-8")
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(config))
    return config


def test_unset_config_disables_plugin_initialization(monkeypatch):
    monkeypatch.delenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, raising=False)
    relay = _FakeRelay()
    host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")

    try:
        assert not host.managed_execution_enabled()
        host.ensure_session({"session_id": "session"})
        assert relay.events[0][0:2] == ("scope.push", relay_runtime.SESSION_SCOPE)
        assert not any(event[0].startswith("plugin.") for event in relay.events)
    finally:
        host.shutdown()

    assert not any(event[0] == "subscribers.flush_async" for event in relay.events)


def test_relay_initializes_explicit_plugins_before_first_session_scope(
    explicit_static_config,
):
    relay = _FakeRelay()
    host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")

    try:
        assert host.managed_execution_enabled()
        host.ensure_session({"session_id": "session"})
        assert relay.events[0] == ("plugin.initialize", {})
        assert relay.events[1][0:2] == ("scope.push", relay_runtime.SESSION_SCOPE)
    finally:
        host.shutdown()


def test_initialization_failure_is_fail_open(explicit_static_config, caplog):
    relay = _FakeRelay(initialize_error=RuntimeError("rejected config"))

    with caplog.at_level("WARNING"):
        host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")

    try:
        assert not host.managed_execution_enabled()
        assert "Hermes Relay plugin initialization failed" in caplog.text
    finally:
        host.shutdown()


def test_later_host_retries_after_initialization_failure(explicit_static_config):
    relay = _FakeRelay(initialize_error=RuntimeError("transient failure"))
    failed_host = relay_runtime.RelayRuntime(relay=relay, profile_key="failed")
    assert not failed_host.managed_execution_enabled()

    relay.initialize_error = None
    recovered_host = relay_runtime.RelayRuntime(relay=relay, profile_key="recovered")
    try:
        assert recovered_host.managed_execution_enabled()
        assert relay.events == [
            ("plugin.initialize", {}),
            ("plugin.initialize", {}),
        ]
    finally:
        failed_host.shutdown()
        recovered_host.shutdown()


def test_missing_explicit_config_does_not_fall_back_to_discovery(
    tmp_path,
    monkeypatch,
    caplog,
):
    missing_config = tmp_path / "missing" / "plugins.toml"
    monkeypatch.setenv(
        relay_runtime.RELAY_PLUGINS_CONFIG_ENV,
        str(missing_config),
    )
    relay = _FakeRelay()

    with caplog.at_level("WARNING"):
        host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")
    try:
        assert not host.managed_execution_enabled()
        assert relay.events == []
        assert "continuing without Relay plugins" in caplog.text
    finally:
        host.shutdown()


def test_malformed_explicit_config_does_not_fall_back_to_discovery(
    tmp_path,
    monkeypatch,
    caplog,
):
    config = tmp_path / "plugins.toml"
    config.write_text("[[components]\nkind =", encoding="utf-8")
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(config))
    relay = _FakeRelay()

    with caplog.at_level("WARNING"):
        host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")
    try:
        assert not host.managed_execution_enabled()
        assert relay.events == []
        assert "continuing without Relay plugins" in caplog.text
    finally:
        host.shutdown()


def test_present_plugins_section_is_validated_even_when_falsey(
    tmp_path,
    monkeypatch,
    caplog,
):
    config = tmp_path / "plugins.toml"
    config.write_text("plugins = []", encoding="utf-8")
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(config))
    relay = _FakeRelay()

    def reject_invalid_plugins(_config_path):
        raise ValueError("'plugins' must be a table")

    relay.plugin.load_dynamic_plugin_activation_specs = reject_invalid_plugins

    with caplog.at_level("WARNING"):
        host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")
    try:
        assert not host.managed_execution_enabled()
        assert relay.events == []
        assert "'plugins' must be a table" in caplog.text
        assert "continuing without Relay plugins" in caplog.text
    finally:
        host.shutdown()


def test_two_profile_hosts_initialize_once_and_clear_after_final_shutdown(
    explicit_static_config,
):
    relay = _FakeRelay()
    host_a = relay_runtime.RelayRuntime(relay=relay, profile_key="profile-a")
    host_b = relay_runtime.RelayRuntime(relay=relay, profile_key="profile-b")

    assert relay.events == [("plugin.initialize", {})]
    assert host_a.managed_execution_enabled()
    assert host_b.managed_execution_enabled()
    host_b.ensure_session({"session_id": "profile-b-session"})

    host_a.shutdown()
    assert ("plugin.clear_async",) not in relay.events

    host_b.shutdown()
    assert relay.events[-2:] == [
        ("subscribers.flush_async",),
        ("plugin.clear_async",),
    ]
    assert relay.events.count(("plugin.initialize", {})) == 1
    assert relay.events.count(("plugin.clear_async",)) == 1
    pop_index = next(
        index for index, event in enumerate(relay.events) if event[0] == "scope.pop"
    )
    assert pop_index < relay.events.index(("plugin.clear_async",))


def test_plugin_initialization_inside_running_event_loop(explicit_static_config):
    relay = _FakeRelay()

    async def construct_host() -> relay_runtime.RelayRuntime:
        return relay_runtime.RelayRuntime(relay=relay, profile_key="profile")

    host = asyncio.run(construct_host())
    try:
        assert relay.events == [("plugin.initialize", {})]
        assert host.managed_execution_enabled()
    finally:
        host.shutdown()


def test_static_plugin_cleanup_uses_async_apis_inside_running_event_loop(
    explicit_static_config,
):
    relay = _FakeRelay()

    async def run_lifecycle() -> None:
        host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")
        host.shutdown()

    asyncio.run(run_lifecycle())

    assert relay.events == [
        ("plugin.initialize", {}),
        ("subscribers.flush_async",),
        ("plugin.clear_async",),
    ]


def test_dynamic_plugins_share_owned_activation_until_final_host_shutdown(
    tmp_path,
    monkeypatch,
):
    config = tmp_path / ".nemo-relay" / "plugins.toml"
    config.parent.mkdir()
    config.write_text(
        """
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[[plugins.dynamic]]
manifest = "plugins/native/relay-plugin.toml"

[plugins.dynamic.config]
mode = "strict"

[[plugins.dynamic]]
manifest = "plugins/worker/relay-plugin.toml"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(config))
    relay = _FakeRelay()
    relay.dynamic_plugin_specs = [
        {
            "plugin_id": "native.policy",
            "kind": "rust_dynamic",
            "manifest_ref": str(
                config.parent / "plugins/native/relay-plugin.toml"
            ),
            "config": {"mode": "strict"},
        },
        {
            "plugin_id": "worker.policy",
            "kind": "worker",
            "manifest_ref": str(
                config.parent / "plugins/worker/relay-plugin.toml"
            ),
            "environment_ref": str(config.parent / "environments/worker"),
            "config": {},
        },
    ]

    host_a = relay_runtime.RelayRuntime(relay=relay, profile_key="profile-a")
    host_b = relay_runtime.RelayRuntime(relay=relay, profile_key="profile-b")

    assert host_a.managed_execution_enabled()
    assert host_b.managed_execution_enabled()
    assert relay.events == [
        ("plugin.load_dynamic_specs", str(config)),
        (
            "plugin.initialize_dynamic",
            {
                "version": 1,
                "components": [
                    {
                        "kind": "observability",
                        "enabled": True,
                        "config": {"version": 1},
                    }
                ],
            },
            relay.dynamic_plugin_specs,
        )
    ]

    host_b.ensure_session({"session_id": "profile-b-session"})
    host_a.shutdown()
    assert ("plugin.activation.close",) not in relay.events

    host_b.shutdown()
    assert relay.events[-2:] == [
        ("subscribers.flush_async",),
        ("plugin.activation.close",),
    ]
    assert ("plugin.clear_async",) not in relay.events
    assert relay.events.count(("plugin.activation.close",)) == 1
    pop_index = next(
        index for index, event in enumerate(relay.events) if event[0] == "scope.pop"
    )
    assert pop_index < relay.events.index(("plugin.activation.close",))


def test_dynamic_activation_failure_disables_plugins(
    tmp_path,
    monkeypatch,
    caplog,
):
    config = tmp_path / "plugins.toml"
    config.write_text(
        """
[[plugins.dynamic]]
manifest = "relay-plugin.toml"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(config))
    relay = _FakeRelay(
        dynamic_initialize_error=RuntimeError("worker rejected config")
    )
    relay.dynamic_plugin_specs = [{"plugin_id": "worker.policy"}]

    with caplog.at_level("WARNING"):
        host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")
    try:
        assert not host.managed_execution_enabled()
        assert [event[0] for event in relay.events] == [
            "plugin.load_dynamic_specs",
            "plugin.initialize_dynamic",
        ]
        assert "dynamic plugin activation failed" in caplog.text
    finally:
        host.shutdown()

    assert ("subscribers.flush_async",) not in relay.events
    assert ("plugin.clear_async",) not in relay.events


def test_dynamic_activation_lifecycle_inside_running_event_loop(
    tmp_path,
    monkeypatch,
):
    config = tmp_path / "plugins.toml"
    config.write_text(
        """
[[plugins.dynamic]]
manifest = "relay-plugin.toml"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(config))
    relay = _FakeRelay()
    relay.dynamic_plugin_specs = [{"plugin_id": "native.policy"}]

    async def run_lifecycle() -> None:
        host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")
        assert host.managed_execution_enabled()
        host.shutdown()

    asyncio.run(run_lifecycle())

    assert [event[0] for event in relay.events] == [
        "plugin.load_dynamic_specs",
        "plugin.initialize_dynamic",
        "subscribers.flush_async",
        "plugin.activation.close",
    ]


def test_shutdown_defers_dynamic_unload_until_async_operation_finishes(
    tmp_path,
    monkeypatch,
):
    config = tmp_path / "plugins.toml"
    config.write_text(
        """
[[plugins.dynamic]]
manifest = "relay-plugin.toml"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(config))
    relay = _FakeRelay()
    relay.dynamic_plugin_specs = [{"plugin_id": "worker.policy"}]

    async def run_lifecycle() -> None:
        host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")
        session = host.ensure_session({"session_id": "session"})
        assert session is not None
        started = asyncio.Event()
        finish = asyncio.Event()

        async def in_flight_call() -> None:
            relay.events.append(("operation.start",))
            started.set()
            await finish.wait()
            relay.events.append(("operation.end",))

        operation = asyncio.create_task(
            host.run_in_session_async(session, in_flight_call)
        )
        await started.wait()
        host.shutdown()
        assert host.ensure_session({"session_id": "late-session"}) is None
        assert ("plugin.activation.close",) not in relay.events

        finish.set()
        await operation
        assert await asyncio.to_thread(host._shutdown_complete.wait, 5)

    asyncio.run(run_lifecycle())

    assert relay.events.index(("operation.end",)) < relay.events.index(
        ("plugin.activation.close",)
    )


def test_session_close_does_not_flush_during_concurrent_managed_publication(
    explicit_static_config,
):
    relay = _ConcurrentPublicationRelay()
    completed = threading.Event()
    errors: list[BaseException] = []

    async def run_lifecycle() -> None:
        host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")
        closing_session = host.ensure_session({"session_id": "closing"})
        active_session = host.ensure_session({"session_id": "active"})
        assert closing_session is not None
        assert active_session is not None
        publication_started = asyncio.Event()
        finish_publication = asyncio.Event()

        async def managed_publication() -> None:
            relay.events.append(("publication.start",))
            publication_started.set()
            await finish_publication.wait()
            relay.events.append(("publication.end",))
            relay.publication_finished.set()

        publication = asyncio.create_task(
            host.run_in_session_async(active_session, managed_publication)
        )
        await publication_started.wait()

        host.close_session({"session_id": "closing"})
        relay.events.append(("session.close.returned",))
        finish_publication.set()
        await publication
        host.shutdown()
        assert host._shutdown_complete.is_set()

    def run_on_event_loop_thread() -> None:
        try:
            asyncio.run(run_lifecycle())
        except BaseException as exc:
            errors.append(exc)
        finally:
            completed.set()

    event_loop_thread = threading.Thread(
        target=run_on_event_loop_thread,
        name="hermes-relay-session-close-regression",
        daemon=True,
    )
    event_loop_thread.start()

    if not completed.wait(3):
        # Release a broken implementation so the test process can clean up
        # after reporting the same deadlock guarded in production.
        relay.publication_finished.set()
        assert completed.wait(5)
        pytest.fail("session close blocked the active asyncio event loop")

    event_loop_thread.join()
    assert errors == []
    assert relay.events.count(("subscribers.flush_async",)) == 1
    assert relay.events.index(("session.close.returned",)) < relay.events.index(
        ("publication.end",)
    )
    assert relay.events.index(("publication.end",)) < relay.events.index(
        ("subscribers.flush_async",)
    )
    assert relay.events[-1] == ("plugin.clear_async",)


def test_failed_dynamic_teardown_retains_activation_and_blocks_replacement(
    tmp_path,
    monkeypatch,
    caplog,
):
    config = tmp_path / "plugins.toml"
    config.write_text(
        """
[[plugins.dynamic]]
manifest = "relay-plugin.toml"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(config))
    relay = _FakeRelay(activation_close_error=RuntimeError("worker still busy"))
    relay.dynamic_plugin_specs = [
        {
            "plugin_id": "worker.policy",
            "kind": "worker",
            "manifest_ref": str(tmp_path / "relay-plugin.toml"),
            "environment_ref": str(tmp_path / "environment"),
            "config": {},
        }
    ]
    host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")

    with caplog.at_level("WARNING"):
        host.shutdown()

    assert "plugin configuration cleanup failed" in caplog.text
    activation = relay_runtime._PLUGIN_CONFIGURATION._activation
    assert activation is not None

    with caplog.at_level("WARNING"):
        replacement = relay_runtime.RelayRuntime(
            relay=relay,
            profile_key="replacement",
        )
    try:
        assert not replacement.managed_execution_enabled()
        assert relay_runtime._PLUGIN_CONFIGURATION._activation is activation
        assert relay.events.count(
            ("plugin.initialize_dynamic", {}, relay.dynamic_plugin_specs)
        ) == 1
        assert relay.events.count(("plugin.activation.close",)) == 2
        assert "refusing to replace" in caplog.text
    finally:
        replacement.shutdown()
        # Relay treats a close failure as terminal; only reset the permissive
        # fake so this process-global fixture cannot leak into later tests.
        relay.activation_close_error = None
        relay_runtime._PLUGIN_CONFIGURATION.reset_for_tests()


def test_standard_dynamic_records_use_relay_toml_loader(
    tmp_path,
    monkeypatch,
):
    config = tmp_path / "plugins.toml"
    config.write_text(
        """
[[plugins.dynamic]]
manifest = "relay-plugin.toml"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(config))
    relay = _FakeRelay()
    relay.dynamic_plugin_specs = [
        {
            "plugin_id": "native.policy",
            "kind": "rust_dynamic",
            "manifest_ref": str(tmp_path / "relay-plugin.toml"),
            "config": {},
        }
    ]

    host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")
    try:
        assert host.managed_execution_enabled()
        assert relay.events == [
            ("plugin.load_dynamic_specs", str(config)),
            ("plugin.initialize_dynamic", {}, relay.dynamic_plugin_specs),
        ]
    finally:
        host.shutdown()


def test_legacy_dynamic_records_are_rejected(
    tmp_path,
    monkeypatch,
    caplog,
):
    config = tmp_path / "plugins.toml"
    config.write_text(
        """
version = 1

[[dynamic_plugins]]
plugin_id = "native.policy"
kind = "rust_dynamic"
manifest_ref = "relay-plugin.toml"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(config))
    relay = _FakeRelay()

    with caplog.at_level("WARNING"):
        host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")
    try:
        assert not host.managed_execution_enabled()
        assert relay.events == []
        assert "Hermes [[dynamic_plugins]] records are unsupported" in caplog.text
        assert "use Relay [[plugins.dynamic]] records" in caplog.text
        assert "continuing without Relay plugins" in caplog.text
    finally:
        host.shutdown()


def test_real_binding_loads_standard_dynamic_specs_from_explicit_toml(
    tmp_path,
    monkeypatch,
):
    relay = pytest.importorskip("nemo_relay")
    manifest = tmp_path / "plugins" / "relay-plugin.toml"
    manifest.parent.mkdir()
    manifest.write_text(
        """
manifest_version = 1

[plugin]
id = "fixture.native"
kind = "rust_dynamic"
""".strip(),
        encoding="utf-8",
    )
    config = tmp_path / "plugins.toml"
    config.write_text(
        """
version = 1

[[plugins.dynamic]]
manifest = "plugins/relay-plugin.toml"

[plugins.dynamic.config]
mode = "strict"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(config))

    plugin_config, specs = relay_runtime._configured_plugin_inputs(relay)

    assert plugin_config == {"version": 1}
    assert [spec.to_dict() for spec in specs] == [
        {
            "plugin_id": "fixture.native",
            "kind": "rust_dynamic",
            "manifest_ref": str(manifest.resolve()),
            "config": {"mode": "strict"},
        }
    ]


def test_real_binding_ignores_project_config_without_explicit_opt_in(
    tmp_path,
    monkeypatch,
):
    relay = pytest.importorskip("nemo_relay")
    if getattr(relay, "_native", None) is None:
        pytest.skip("NeMo Relay native binding is unavailable on this platform")

    project_root = tmp_path / "project"
    working_directory = project_root / "workspace"
    config_directory = project_root / ".nemo-relay"
    atof_dir = tmp_path / "atof"
    working_directory.mkdir(parents=True)
    config_directory.mkdir()
    (config_directory / "plugins.toml").write_text(
        f"""
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 3

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = "{atof_dir}"
filename = "events.jsonl"
mode = "overwrite"
""".strip(),
        encoding="utf-8",
    )
    xdg_config_home = tmp_path / "xdg"
    xdg_config_home.mkdir()
    monkeypatch.chdir(working_directory)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_home))
    monkeypatch.delenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, raising=False)
    relay.plugin.clear()

    host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")
    try:
        assert not host.managed_execution_enabled()
        host.ensure_session({"session_id": "native-no-plugins"})
    finally:
        host.shutdown()
        relay_runtime._reset_for_tests()

    assert not (atof_dir / "events.jsonl").exists()


def test_real_binding_layers_project_config_after_explicit_opt_in(
    tmp_path,
    monkeypatch,
):
    relay = pytest.importorskip("nemo_relay")
    if getattr(relay, "_native", None) is None:
        pytest.skip("NeMo Relay native binding is unavailable on this platform")

    project_root = tmp_path / "project"
    working_directory = project_root / "workspace"
    config_directory = project_root / ".nemo-relay"
    selected_directory = tmp_path / "selected-config"
    atof_dir = tmp_path / "atof"
    working_directory.mkdir(parents=True)
    config_directory.mkdir()
    selected_directory.mkdir()
    (config_directory / "plugins.toml").write_text(
        f"""
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 3

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = "{atof_dir}"
filename = "events.jsonl"
mode = "overwrite"
""".strip(),
        encoding="utf-8",
    )
    selected_config = selected_directory / "plugins.toml"
    selected_config.write_text("version = 1", encoding="utf-8")
    xdg_config_home = tmp_path / "xdg"
    xdg_config_home.mkdir()
    monkeypatch.chdir(working_directory)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_home))
    monkeypatch.setenv(
        relay_runtime.RELAY_PLUGINS_CONFIG_ENV,
        str(selected_config),
    )
    relay.plugin.clear()

    host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")
    try:
        assert host.managed_execution_enabled()
        host.ensure_session({"session_id": "native-layered-plugins"})
    finally:
        host.shutdown()
        relay_runtime._reset_for_tests()

    assert (atof_dir / "events.jsonl").is_file()


def test_real_binding_loads_explicit_config_and_exports_native_activity(
    tmp_path,
    monkeypatch,
):
    relay = pytest.importorskip("nemo_relay")
    if getattr(relay, "_native", None) is None:
        pytest.skip("NeMo Relay native binding is unavailable on this platform")
    from agent import relay_llm, relay_tools

    working_directory = tmp_path / "project" / "workspace"
    config_directory = tmp_path / "selected-config"
    atof_dir = tmp_path / "atof"
    atif_dir = tmp_path / "atif"
    working_directory.mkdir(parents=True)
    config_directory.mkdir()
    config_path = config_directory / "plugins.toml"
    config_path.write_text(
        f"""
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 3

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = "{atof_dir}"
filename = "events.jsonl"
mode = "overwrite"

[components.config.atif]
enabled = true
output_directory = "{atif_dir}"
filename_template = "trajectory-{{session_id}}.json"
agent_name = "Hermes Native Test"
agent_version = "test"
""".strip(),
        encoding="utf-8",
    )
    xdg_config_home = tmp_path / "xdg"
    xdg_config_home.mkdir()
    monkeypatch.chdir(working_directory)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_home))
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(config_path))
    monkeypatch.setattr(relay_runtime, "_load_nemo_relay", lambda: relay)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay.plugin.clear()

    profile_key = relay_runtime.current_profile_key()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=profile_key,
        session_id="native-export",
        platform="cli",
        model="test-model",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="turn-1",
        task_id="task-1",
    )
    try:
        assert lease.host.managed_execution_enabled()
        relay_llm.execute(
            {"model": "test-model", "messages": []},
            lambda _request: {
                "id": "response-1",
                "model": "test-model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
            session_id="native-export",
            name="test-provider",
            model_name="test-model",
            metadata={
                "api_mode": "chat_completions",
                "api_request_id": "request-1",
            },
        )
        relay_tools.execute(
            "terminal",
            {"command": "true"},
            lambda _args: {"output": "ok"},
            session_id="native-export",
            metadata={"tool_call_id": "tool-1"},
        )
    finally:
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime.SESSION_COORDINATOR.finalize_conversation(
            profile_key=profile_key,
            session_id="native-export",
        )
        relay_runtime._reset_for_tests()

    assert (atof_dir / "events.jsonl").is_file()
    trajectories = list(atif_dir.glob("trajectory-*.json"))
    assert len(trajectories) == 1
    trajectory = json.loads(trajectories[0].read_text(encoding="utf-8"))
    observed_categories = {
        event["category"]
        for event in trajectory["extra"]["observed_events"]
        if event["kind"] == "scope"
    }
    assert {"agent", "llm", "tool"} <= observed_categories
