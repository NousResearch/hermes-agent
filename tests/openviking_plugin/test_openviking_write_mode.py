from unittest.mock import Mock

import plugins.memory.openviking as openviking_plugin
from plugins.memory.openviking import OpenVikingMemoryProvider


MUTATING_TOOLS = {
    "viking_remember",
    "viking_forget",
    "viking_add_resource",
}
READ_TOOLS = {"viking_search", "viking_read", "viking_browse"}
READ_ONLY_ERROR = "openviking_write_mode_read_only"


class NoMutationClient:
    def __init__(self):
        self.calls = []

    def get(self, path, params=None, **kwargs):
        self.calls.append(("get", path, params or {}))
        if path == "/api/v1/fs/ls":
            return {"result": [{"uri": "viking://resources"}]}
        return {"result": {}}

    def post(self, path, payload=None, **kwargs):
        raise AssertionError(f"unexpected POST {path} {payload}")

    def delete(self, path, **kwargs):
        raise AssertionError(f"unexpected DELETE {path} {kwargs}")

    def upload_temp_file(self, file_path):
        raise AssertionError(f"unexpected upload {file_path}")


class VersionedHealthClient:
    observed_version = "v0.4.9"
    raise_health_error = False

    def __init__(self, *args, **kwargs):
        pass

    def health_payload(self):
        if self.raise_health_error:
            raise RuntimeError("health unavailable")
        return {"healthy": True, "version": self.observed_version}


def make_provider(mode="read_only"):
    provider = OpenVikingMemoryProvider()
    provider._provider_mode = mode
    provider._client = NoMutationClient()
    provider._endpoint = "http://openviking.test"
    provider._account = "peishaoyang"
    provider._user = "peishaoyang"
    provider._agent = "hermes"
    provider._session_id = "session-old"
    return provider


class TestOpenVikingProviderModeContract:
    def test_mode_and_version_gate_fail_closed(self):
        assert openviking_plugin._resolve_provider_mode(None) == "read_only"
        assert openviking_plugin._resolve_provider_mode("") == "read_only"
        assert openviking_plugin._resolve_provider_mode("unexpected") == "read_only"
        assert openviking_plugin._resolve_provider_mode("native") == "native"

        effective = openviking_plugin._effective_provider_mode
        assert effective("read_only", "legacy_hold", "0.3.22", "v0.3.22") == "read_only"
        assert effective("read_only", "target_clone_acceptance", "0.4.9", "v0.4.9") == "read_only"
        assert effective("native", "legacy_hold", "0.3.22", "v0.3.22") == "read_only"
        assert effective("native", "unexpected", "0.4.9", "v0.4.9") == "read_only"
        assert effective("native", "target_clone_acceptance", "", "v0.4.9") == "read_only"
        assert effective("native", "target_clone_acceptance", "0.4.9", "") == "read_only"
        assert effective("native", "target_clone_acceptance", "0.4.9", "v0.4.8") == "read_only"
        assert effective("native", "target_clone_acceptance", "0.4.9", "v0.4.9") == "native"
        assert effective("native", "native_only", "", "v0.4.9") == "read_only"
        assert effective("native", "native_only", "0.4.9", "v0.4.8") == "read_only"
        assert effective("native", "native_only", "0.4.9", "v0.4.9") == "native"

    def test_initialize_activates_native_only_after_exact_health_match(
        self, monkeypatch
    ):
        monkeypatch.setattr(openviking_plugin, "_VikingClient", VersionedHealthClient)
        monkeypatch.setattr(
            openviking_plugin,
            "_load_hermes_openviking_config",
            lambda: {},
        )
        monkeypatch.setattr(
            openviking_plugin,
            "_resolve_connection_settings",
            lambda config: {
                "endpoint": "http://openviking.test",
                "api_key": "",
                "account": "peishaoyang",
                "user": "peishaoyang",
                "agent": "hermes",
            },
        )
        monkeypatch.setenv("OPENVIKING_PROVIDER_MODE", "read_only")
        monkeypatch.setenv(
            "OPENVIKING_COMPATIBILITY_PHASE", "target_clone_acceptance"
        )
        monkeypatch.setenv("OPENVIKING_EXPECTED_SERVER_VERSION", "0.4.9")

        VersionedHealthClient.raise_health_error = False
        VersionedHealthClient.observed_version = "v0.4.9"
        requested_read_only = OpenVikingMemoryProvider()
        requested_read_only.initialize("requested-read-only")
        assert requested_read_only._provider_mode == "read_only"
        requested_read_only.shutdown()

        monkeypatch.setenv("OPENVIKING_PROVIDER_MODE", "native")
        matching = OpenVikingMemoryProvider()
        matching.initialize("matching")
        assert matching._provider_mode == "native"
        matching.shutdown()

        VersionedHealthClient.observed_version = "v0.4.8"
        mismatched = OpenVikingMemoryProvider()
        mismatched.initialize("mismatched")
        assert mismatched._provider_mode == "read_only"
        mismatched.shutdown()

        VersionedHealthClient.observed_version = ""
        missing = OpenVikingMemoryProvider()
        missing.initialize("missing")
        assert missing._provider_mode == "read_only"
        missing.shutdown()

        VersionedHealthClient.raise_health_error = True
        unavailable = OpenVikingMemoryProvider()
        unavailable.initialize("unavailable")
        assert unavailable._provider_mode == "read_only"
        unavailable.shutdown()

        VersionedHealthClient.raise_health_error = False
        VersionedHealthClient.observed_version = "v0.3.22"
        monkeypatch.setenv("OPENVIKING_COMPATIBILITY_PHASE", "legacy_hold")
        monkeypatch.setenv("OPENVIKING_EXPECTED_SERVER_VERSION", "0.3.22")
        legacy = OpenVikingMemoryProvider()
        legacy.initialize("legacy")
        assert legacy._provider_mode == "read_only"
        legacy.shutdown()

    def test_read_only_exposes_only_recall_tool_schemas(self):
        provider = make_provider()

        names = {schema["name"] for schema in provider.get_tool_schemas()}

        assert names == READ_TOOLS
        assert names.isdisjoint(MUTATING_TOOLS)

    def test_dispatcher_blocks_every_mutation_before_transport(self):
        provider = make_provider()
        client = provider._client
        calls = {
            "viking_remember": {"content": "test"},
            "viking_forget": {
                "uri": "viking://user/peers/hermes/memories/events/test.md"
            },
            "viking_add_resource": {"url": "https://example.com/doc"},
        }

        for tool_name, args in calls.items():
            result = provider.handle_tool_call(tool_name, args)
            assert READ_ONLY_ERROR in result

        assert client.calls == []

    def test_private_mutation_handlers_also_block_before_transport(self):
        provider = make_provider()
        client = provider._client

        results = [
            provider._tool_remember({"content": "test"}),
            provider._tool_forget(
                {"uri": "viking://user/peers/hermes/memories/events/test.md"}
            ),
            provider._tool_add_resource({"url": "https://example.com/doc"}),
        ]

        assert all(READ_ONLY_ERROR in result for result in results)
        assert client.calls == []

    def test_lifecycle_callbacks_rotate_local_state_with_zero_transport(
        self, monkeypatch
    ):
        provider = make_provider()
        client = provider._client
        provider._turn_count = 2
        provider._spawn_writer = Mock(side_effect=AssertionError("writer spawned"))
        provider._drain_writers = Mock(side_effect=AssertionError("drain called"))
        provider._finalize_session_async = Mock(
            side_effect=AssertionError("finalizer spawned")
        )
        thread_ctor = Mock(side_effect=AssertionError("thread spawned"))
        monkeypatch.setattr(openviking_plugin.threading, "Thread", thread_ctor)

        provider.sync_turn("user", "assistant", session_id="session-old")
        provider.on_session_end([])
        provider.on_memory_write("add", "memory", "content")
        provider.on_session_switch("session-new")

        assert provider._session_id == "session-new"
        assert provider._turn_count == 0
        assert client.calls == []
        provider._spawn_writer.assert_not_called()
        provider._drain_writers.assert_not_called()
        provider._finalize_session_async.assert_not_called()
        thread_ctor.assert_not_called()

    def test_read_only_prompt_does_not_advertise_mutating_tools(self):
        provider = make_provider()

        prompt = provider.system_prompt_block()

        assert "read-only" in prompt.lower()
        assert "viking_search" in prompt
        assert "viking_read" in prompt
        assert "viking_browse" in prompt
        assert "viking_remember" not in prompt
        assert "viking_forget" not in prompt
        assert "viking_add_resource" not in prompt

    def test_native_mode_retains_pinned_provider_surface(self):
        provider = make_provider("native")

        names = {schema["name"] for schema in provider.get_tool_schemas()}

        assert names == READ_TOOLS | MUTATING_TOOLS
