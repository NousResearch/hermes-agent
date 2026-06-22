"""Tests for plugins/memory/openviking/__init__.py — URI normalization and payload handling."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import plugins.memory.openviking as openviking_plugin
from plugins.memory.openviking import OpenVikingMemoryProvider


def _write_skill(skills_dir, name, body="Do the thing."):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}\n---\n\n# {name}\n\n{body}\n"
    )
    return skill_dir


def _write_bundle(bundles_dir, slug, skills):
    bundles_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"name: {slug}", "skills:"]
    lines.extend(f"  - {skill}" for skill in skills)
    (bundles_dir / f"{slug}.yaml").write_text("\n".join(lines) + "\n")


class FakeVikingClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, path, params=None, **kwargs):
        self.calls.append((path, params or {}))
        response = self.responses[(path, tuple(sorted((params or {}).items())))]
        if isinstance(response, Exception):
            raise response
        return response

    def post(self, path, payload=None, **kwargs):
        self.calls.append((path, payload or {}))
        response = self.responses.get((path, tuple(sorted((payload or {}).items()))), {})
        if isinstance(response, Exception):
            raise response
        return response


class RecordingVikingClient:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    def post(self, path, payload=None, **kwargs):
        self.calls.append((path, payload or {}))
        return {"result": {"memories": [], "resources": []}}


def _recall_context_key(value):
    if isinstance(value, list):
        return tuple(value)
    return value


class FakeRecallClient:
    calls = []
    responses = {}

    def __init__(self, *args, **kwargs):
        pass

    def post(self, path, payload=None, **kwargs):
        payload = payload or {}
        self.__class__.calls.append(("post", path, dict(payload)))
        context_type = _recall_context_key(payload.get("context_type"))
        key = (path, context_type, payload.get("query"), payload.get("session_id"))
        if key not in self.__class__.responses:
            key = (path, context_type, payload.get("query"))
        if key not in self.__class__.responses:
            key = (path, context_type)
        response = self.__class__.responses[key]
        if isinstance(response, Exception):
            raise response
        return response

    def get(self, path, params=None, **kwargs):
        params = params or {}
        self.__class__.calls.append(("get", path, dict(params)))
        response = self.__class__.responses[(path, params.get("uri"))]
        if isinstance(response, Exception):
            raise response
        return response


def make_prefetch_provider(monkeypatch, responses, **env):
    monkeypatch.setattr(openviking_plugin, "_VikingClient", FakeRecallClient)
    FakeRecallClient.calls = []
    FakeRecallClient.responses = responses
    for key in (
        "OPENVIKING_RECALL_LIMIT",
        "OPENVIKING_RECALL_SCORE_THRESHOLD",
        "OPENVIKING_RECALL_MAX_INJECTED_CHARS",
        "OPENVIKING_RECALL_TIMEOUT_SECONDS",
        "OPENVIKING_RECALL_REQUEST_TIMEOUT_SECONDS",
        "OPENVIKING_RECALL_FULL_READ_LIMIT",
        "OPENVIKING_RECALL_PREFER_ABSTRACT",
        "OPENVIKING_RECALL_RESOURCES",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))

    provider = OpenVikingMemoryProvider()
    provider._client = object()
    provider._endpoint = "http://openviking.test"
    provider._account = "default"
    provider._user = "default"
    provider._agent = "hermes"
    provider._session_id = "session-test"
    return provider


def wait_prefetch(provider, query="What should we recall?", session_id="session-test"):
    return provider.prefetch(query, session_id=session_id)


class TestOpenVikingSummaryUriNormalization:
    def test_normalize_summary_uri_maps_pseudo_files_to_parent_directory(self):
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://user/hermes/.overview.md") == "viking://user/hermes"
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://resources/.abstract.md") == "viking://resources"
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://") == "viking://"
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://user/hermes/memories/profile.md") == "viking://user/hermes/memories/profile.md"

class TestOpenVikingSkillQuerySafety:
    def test_derive_returns_empty_string_for_non_string_input(self):
        assert openviking_plugin._derive_openviking_user_text(None) == ""
        assert openviking_plugin._derive_openviking_user_text(123) == ""
        assert openviking_plugin._derive_openviking_user_text([{"text": "hi"}]) == ""

    def test_derive_passes_through_non_skill_content(self):
        assert (
            openviking_plugin._derive_openviking_user_text("regular user message")
            == "regular user message"
        )

    def test_derive_returns_empty_for_skill_scaffolding_with_no_instruction(self):
        skill_message = (
            '[IMPORTANT: The user has invoked the "example" skill, indicating they want '
            "you to follow its instructions. The full skill content is loaded below.]\n\n"
            "# Example\n\n"
            "Skill body only, no instruction."
        )

        assert openviking_plugin._derive_openviking_user_text(skill_message) == ""

    def test_skill_markers_match_hermes_scaffolding(self, tmp_path, monkeypatch):
        import agent.skill_bundles as skill_bundles
        import agent.skill_commands as skill_commands
        import tools.skills_tool as skills_tool

        skills_dir = tmp_path / "skills"
        bundles_dir = tmp_path / "skill-bundles"
        _write_skill(skills_dir, "example")
        _write_bundle(bundles_dir, "demo", ["example"])

        monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
        monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
        monkeypatch.setattr(skill_commands, "_skill_commands", {})
        monkeypatch.setattr(skill_commands, "_skill_commands_platform", None)
        monkeypatch.setattr(skill_bundles, "_bundles_cache", {})
        monkeypatch.setattr(skill_bundles, "_bundles_cache_mtime", None)

        skill_commands.scan_skill_commands()
        single = skill_commands.build_skill_invocation_message(
            "/example",
            user_instruction="hello",
            runtime_note="runtime detail",
        )
        assert single is not None
        assert skill_commands._SKILL_INVOCATION_PREFIX in single
        assert skill_commands._SINGLE_SKILL_MARKER in single
        assert skill_commands._SINGLE_SKILL_INSTRUCTION in single
        assert skill_commands._RUNTIME_NOTE in single

        skill_bundles.scan_bundles()
        bundle_result = skill_bundles.build_bundle_invocation_message(
            "/demo",
            user_instruction="hello",
        )
        assert bundle_result is not None
        bundle, _, _ = bundle_result
        assert skill_commands._BUNDLE_MARKER in bundle
        assert skill_commands._BUNDLE_USER_INSTRUCTION in bundle
        assert skill_commands._BUNDLE_FIRST_SKILL_BLOCK in bundle

    def test_prefetch_searches_only_slash_skill_user_instruction(self, monkeypatch):
        RecordingVikingClient.calls = []
        monkeypatch.setattr(openviking_plugin, "_VikingClient", RecordingVikingClient)
        provider = OpenVikingMemoryProvider()
        provider._client = cast(Any, object())
        provider._endpoint = "http://openviking.test"
        provider._api_key = ""
        provider._account = "default"
        provider._user = "default"
        provider._agent = "hermes"
        skill_message = (
            '[IMPORTANT: The user has invoked the "skill-creator" skill, indicating they want '
            "you to follow its instructions. The full skill content is loaded below.]\n\n"
            "# Skill Creator\n\n"
            "Large skill body that must not be searched or embedded.\n\n"
            "The user has provided the following instruction alongside the skill invocation: "
            "make a skill for release triage"
        )

        provider.prefetch(skill_message)

        assert RecordingVikingClient.calls == [
            (
                "/api/v1/search/find",
                {
                    "query": "make a skill for release triage",
                    "limit": 24,
                    "score_threshold": 0,
                    "context_type": "memory",
                },
            ),
        ]

    def test_prefetch_searches_only_skill_bundle_user_instruction(self, monkeypatch):
        RecordingVikingClient.calls = []
        monkeypatch.setattr(openviking_plugin, "_VikingClient", RecordingVikingClient)
        provider = OpenVikingMemoryProvider()
        provider._client = cast(Any, object())
        provider._endpoint = "http://openviking.test"
        provider._api_key = ""
        provider._account = "default"
        provider._user = "default"
        provider._agent = "hermes"
        skill_message = (
            '[IMPORTANT: The user has invoked the "backend-dev" skill bundle, '
            "loading 2 skills together. Treat every skill below as active guidance for this turn.]\n\n"
            "Bundle: backend-dev\n"
            "Skills loaded: test-driven-development, code-review\n\n"
            "User instruction: fix the failing retrieval test\n\n"
            '[Loaded as part of the "backend-dev" skill bundle.]\n\n'
            "Large bundled skill body that must not be searched or embedded."
        )

        provider.prefetch(skill_message)

        assert RecordingVikingClient.calls == [
            (
                "/api/v1/search/find",
                {
                    "query": "fix the failing retrieval test",
                    "limit": 24,
                    "score_threshold": 0,
                    "context_type": "memory",
                },
            ),
        ]

    def test_prefetch_skips_slash_skill_without_user_instruction(self, monkeypatch):
        RecordingVikingClient.calls = []
        monkeypatch.setattr(openviking_plugin, "_VikingClient", RecordingVikingClient)
        provider = OpenVikingMemoryProvider()
        provider._client = cast(Any, object())
        skill_message = (
            '[IMPORTANT: The user has invoked the "skill-creator" skill, indicating they want '
            "you to follow its instructions. The full skill content is loaded below.]\n\n"
            "# Skill Creator\n\n"
            "Large skill body that must not be searched or embedded."
        )

        assert provider.prefetch(skill_message) == ""

        assert RecordingVikingClient.calls == []

    def test_sync_turn_stores_only_slash_skill_user_instruction(self, monkeypatch):
        RecordingVikingClient.calls = []
        monkeypatch.setattr(openviking_plugin, "_VikingClient", RecordingVikingClient)
        provider = OpenVikingMemoryProvider()
        provider._client = cast(Any, object())
        provider._endpoint = "http://openviking.test"
        provider._api_key = ""
        provider._account = "default"
        provider._user = "default"
        provider._agent = "hermes"
        provider._session_id = "session-1"
        skill_message = (
            '[IMPORTANT: The user has invoked the "skill-creator" skill, indicating they want '
            "you to follow its instructions. The full skill content is loaded below.]\n\n"
            "# Skill Creator\n\n"
            "Large skill body that must not be stored as user content.\n\n"
            "The user has provided the following instruction alongside the skill invocation: "
            "make a skill for release triage"
        )

        provider.sync_turn(skill_message, "Done.")
        assert provider._drain_writers("session-1", timeout=5.0)

        assert RecordingVikingClient.calls == [
            (
                "/api/v1/sessions/session-1/messages/batch",
                {
                    "messages": [
                        {
                            "role": "user",
                            "parts": [
                                {"type": "text", "text": "make a skill for release triage"},
                            ],
                        },
                        {
                            "role": "assistant",
                            "parts": [{"type": "text", "text": "Done."}],
                            "peer_id": "hermes",
                        },
                    ]
                },
            ),
        ]

    def test_sync_turn_skips_slash_skill_without_user_instruction(self, monkeypatch):
        RecordingVikingClient.calls = []
        monkeypatch.setattr(openviking_plugin, "_VikingClient", RecordingVikingClient)
        provider = OpenVikingMemoryProvider()
        provider._client = cast(Any, object())
        skill_message = (
            '[IMPORTANT: The user has invoked the "skill-creator" skill, indicating they want '
            "you to follow its instructions. The full skill content is loaded below.]\n\n"
            "# Skill Creator\n\n"
            "Large skill body that must not be stored as user content."
        )

        provider.sync_turn(skill_message, "Done.")

        assert provider._turn_count == 0
        assert provider._inflight_writers == {}
        assert RecordingVikingClient.calls == []


class TestOpenVikingConfigSchema:
    def test_recall_policy_options_are_exposed_in_setup_schema(self):
        provider = OpenVikingMemoryProvider()

        schema = provider.get_config_schema()
        env_vars = {entry.get("env_var") for entry in schema}

        assert "OPENVIKING_RECALL_LIMIT" in env_vars
        assert "OPENVIKING_RECALL_SCORE_THRESHOLD" in env_vars
        assert "OPENVIKING_RECALL_MAX_INJECTED_CHARS" in env_vars
        assert "OPENVIKING_RECALL_TIMEOUT_SECONDS" in env_vars
        assert "OPENVIKING_RECALL_REQUEST_TIMEOUT_SECONDS" in env_vars
        assert "OPENVIKING_RECALL_FULL_READ_LIMIT" in env_vars
        assert "OPENVIKING_RECALL_PREFER_ABSTRACT" in env_vars
        assert "OPENVIKING_RECALL_RESOURCES" in env_vars
        assert provider._recall_config() == {
            "limit": 6,
            "score_threshold": 0.15,
            "max_injected_chars": 4000,
            "timeout_seconds": 4.0,
            "request_timeout_seconds": 3.0,
            "full_read_limit": 2,
            "prefer_abstract": False,
            "resources": False,
        }


class TestOpenVikingTurnConversion:
    def test_extract_current_turn_anchors_on_latest_matching_user_and_assistant(self):
        messages = [
            {"role": "user", "content": "Please inspect the repository for assemble hooks."},
            {"role": "assistant", "content": "Earlier answer."},
            {"role": "user", "content": "Please inspect the repository for assemble hooks."},
            {
                "role": "assistant",
                "content": "I will search the codebase.",
                "tool_calls": [
                    {
                        "id": "call_rg_1",
                        "type": "function",
                        "function": {
                            "name": "shell_command",
                            "arguments": json.dumps({"command": "rg assemble"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_rg_1",
                "name": "shell_command",
                "content": "agent/context_engine.py: no preassemble hook",
            },
            {"role": "assistant", "content": "The current main does not expose assemble."},
        ]

        turn = OpenVikingMemoryProvider._extract_current_turn_messages(
            messages,
            "Please inspect the repository for assemble hooks.",
            "The current main does not expose assemble.",
        )

        assert turn == messages[2:]

    def test_messages_to_openviking_batch_coalesces_tool_results(self):
        turn = [
            {"role": "user", "content": "Please inspect the repository for assemble hooks."},
            {
                "role": "assistant",
                "content": "I will search the codebase.",
                "tool_calls": [
                    {
                        "id": "call_rg_1",
                        "type": "function",
                        "function": {
                            "name": "shell_command",
                            "arguments": json.dumps({"command": "rg assemble"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_rg_1",
                "name": "shell_command",
                "content": "agent/context_engine.py: no preassemble hook",
            },
            {"role": "assistant", "content": "The current main does not expose assemble."},
        ]

        batch = OpenVikingMemoryProvider._messages_to_openviking_batch(turn)

        assert [message["role"] for message in batch] == ["user", "assistant", "assistant", "assistant"]
        assert batch[0]["parts"] == [
            {"type": "text", "text": "Please inspect the repository for assemble hooks."}
        ]
        assert batch[1]["parts"] == [
            {"type": "text", "text": "I will search the codebase."}
        ]
        assert batch[2]["parts"] == [
            {
                "type": "tool",
                "tool_id": "call_rg_1",
                "tool_name": "shell_command",
                "tool_input": {"command": "rg assemble"},
                "tool_output": "agent/context_engine.py: no preassemble hook",
                "tool_status": "completed",
            }
        ]
        assert batch[3]["parts"] == [
            {"type": "text", "text": "The current main does not expose assemble."}
        ]

    def test_messages_to_openviking_batch_marks_json_tool_error_results(self):
        turn = [
            {"role": "user", "content": "Check the file."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_read_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "missing.md"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_read_1",
                "name": "read_file",
                "content": json.dumps({"error": "File not found", "exit_code": 1}),
            },
        ]

        batch = OpenVikingMemoryProvider._messages_to_openviking_batch(turn)

        assert batch[1]["role"] == "assistant"
        assert batch[1]["parts"] == [
            {
                "type": "tool",
                "tool_id": "call_read_1",
                "tool_name": "read_file",
                "tool_input": {"path": "missing.md"},
                "tool_output": json.dumps({"error": "File not found", "exit_code": 1}),
                "tool_status": "error",
            }
        ]

    def test_messages_to_openviking_batch_keeps_pending_tool_call_without_result(self):
        turn = [
            {"role": "user", "content": "Start a long running check."},
            {
                "role": "assistant",
                "content": "Starting it now.",
                "tool_calls": [
                    {
                        "id": "call_long_1",
                        "type": "function",
                        "function": {
                            "name": "long_check",
                            "arguments": json.dumps({"target": "repo"}),
                        },
                    }
                ],
            },
        ]

        batch = OpenVikingMemoryProvider._messages_to_openviking_batch(turn)

        assert batch[1]["parts"] == [
            {"type": "text", "text": "Starting it now."},
            {
                "type": "tool",
                "tool_id": "call_long_1",
                "tool_name": "long_check",
                "tool_input": {"target": "repo"},
                "tool_status": "pending",
            },
        ]

    def test_messages_to_openviking_batch_coalesces_adjacent_tool_results(self):
        turn = [
            {"role": "user", "content": "Run both tools."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {
                            "name": "first_tool",
                            "arguments": json.dumps({"x": 1}),
                        },
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {
                            "name": "second_tool",
                            "arguments": json.dumps({"y": 2}),
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "name": "first_tool", "content": "a"},
            {"role": "tool", "tool_call_id": "call_b", "name": "second_tool", "content": "b"},
            {"role": "assistant", "content": "Done."},
        ]

        batch = OpenVikingMemoryProvider._messages_to_openviking_batch(turn)

        assert [message["role"] for message in batch] == ["user", "assistant", "assistant"]
        assert batch[1]["parts"] == [
            {
                "type": "tool",
                "tool_id": "call_a",
                "tool_name": "first_tool",
                "tool_input": {"x": 1},
                "tool_output": "a",
                "tool_status": "completed",
            },
            {
                "type": "tool",
                "tool_id": "call_b",
                "tool_name": "second_tool",
                "tool_input": {"y": 2},
                "tool_output": "b",
                "tool_status": "completed",
            },
        ]

    def test_messages_to_openviking_batch_skips_openviking_recall_tool_results(self):
        for recall_tool_name in ("viking_search", "viking_read", "viking_browse"):
            turn = [
                {"role": "user", "content": "What did we decide about context assembly?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_recall_1",
                            "type": "function",
                            "function": {
                                "name": recall_tool_name,
                                "arguments": json.dumps({"query": "context assembly decision"}),
                            },
                        },
                        {
                            "id": "call_shell_1",
                            "type": "function",
                            "function": {
                                "name": "shell_command",
                                "arguments": json.dumps({"command": "rg preassemble"}),
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_recall_1",
                    "name": recall_tool_name,
                    "content": json.dumps({
                        "results": [
                            {
                                "uri": "viking://user/hermes/memories/context",
                                "abstract": "Old OpenViking memory content",
                            }
                        ]
                    }),
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_shell_1",
                    "name": "shell_command",
                    "content": "plugins/memory/openviking/__init__.py",
                },
                {"role": "assistant", "content": "We decided to keep sync_turn scoped to ingestion."},
            ]

            batch = OpenVikingMemoryProvider._messages_to_openviking_batch(turn)

            assert [message["role"] for message in batch] == ["user", "assistant", "assistant"]
            assert batch[1]["parts"] == [
                {
                    "type": "tool",
                    "tool_id": "call_shell_1",
                    "tool_name": "shell_command",
                    "tool_input": {"command": "rg preassemble"},
                    "tool_output": "plugins/memory/openviking/__init__.py",
                    "tool_status": "completed",
                }
            ]
            batch_text = json.dumps(batch)
            assert recall_tool_name not in batch_text
            assert "Old OpenViking memory content" not in batch_text

    def test_messages_to_openviking_batch_empty_tool_id_does_not_drop_other_results(self):
        # A recall tool result that arrives with an empty tool_call_id must not
        # poison the skip set with "" and silently drop unrelated tool results
        # that also lack an id. Empty tool_call_id is reachable in the canonical
        # transcript (agent_runtime_helpers defaults it to "").
        turn = [
            {"role": "user", "content": "What did we decide?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "",
                        "type": "function",
                        "function": {
                            "name": "viking_search",
                            "arguments": json.dumps({"query": "decision"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "",
                "name": "viking_search",
                "content": json.dumps({"results": ["recall stuff"]}),
            },
            {
                "role": "tool",
                "tool_call_id": "",
                "name": "shell_command",
                "content": "important shell output",
            },
            {"role": "assistant", "content": "done"},
        ]

        batch = OpenVikingMemoryProvider._messages_to_openviking_batch(turn)

        batch_text = json.dumps(batch)
        # The unrelated (empty-id) shell result must survive.
        assert "important shell output" in batch_text
        # The recall tool result must still be excluded.
        assert "recall stuff" not in batch_text
        assert "viking_search" not in batch_text

    def test_messages_to_openviking_batch_preserves_responses_text_parts(self):
        turn = [
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
        ]

        batch = OpenVikingMemoryProvider._messages_to_openviking_batch(turn)

        assert batch == [
            {"role": "user", "parts": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": "answer"}]},
        ]

    def test_messages_to_openviking_batch_adds_assistant_peer_id_when_requested(self):
        turn = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "answer"},
        ]

        batch = OpenVikingMemoryProvider._messages_to_openviking_batch(
            turn,
            assistant_peer_id="hermes",
        )

        assert batch == [
            {"role": "user", "parts": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": "answer"}], "peer_id": "hermes"},
        ]


class TestOpenVikingRead:
    def test_overview_read_normalizes_uri_and_unwraps_result(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/overview",
                    (("uri", "viking://user/hermes"),),
                ): {"result": {"content": "overview text"}},
            }
        )

        result = json.loads(provider._tool_read({"uri": "viking://user/hermes/.overview.md", "level": "overview"}))

        assert result["uri"] == "viking://user/hermes/.overview.md"
        assert result["resolved_uri"] == "viking://user/hermes"
        assert result["level"] == "overview"
        assert result["content"] == "overview text"
        assert provider._client.calls == [(
            "/api/v1/content/overview",
            {"uri": "viking://user/hermes"},
        )]

    def test_full_read_keeps_original_uri(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/read",
                    (("uri", "viking://user/hermes/memories/profile.md"),),
                ): {"result": "full text"},
            }
        )

        result = json.loads(provider._tool_read({"uri": "viking://user/hermes/memories/profile.md", "level": "full"}))

        assert result["uri"] == "viking://user/hermes/memories/profile.md"
        assert result["resolved_uri"] == "viking://user/hermes/memories/profile.md"
        assert result["level"] == "full"
        assert result["content"] == "full text"
        assert provider._client.calls == [(
            "/api/v1/content/read",
            {"uri": "viking://user/hermes/memories/profile.md"},
        )]

    def test_read_accepts_uri_batch_and_caps_batch_full_content(self):
        provider = OpenVikingMemoryProvider()
        uris = [
            "viking://user/hermes/memories/a.md",
            "viking://user/hermes/memories/b.md",
            "viking://user/hermes/memories/c.md",
            "viking://user/hermes/memories/d.md",
        ]
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/read",
                    (("uri", uris[0]),),
                ): {"result": {"content": "a" * 3000}},
                (
                    "/api/v1/content/read",
                    (("uri", uris[1]),),
                ): {"result": {"content": "b content"}},
                (
                    "/api/v1/content/read",
                    (("uri", uris[2]),),
                ): {"result": {"content": "c content"}},
            }
        )

        result = json.loads(provider._tool_read({"uris": uris, "level": "full"}))

        assert result["requested"] == 4
        assert result["returned"] == 3
        assert result["truncated"] is True
        assert [entry["uri"] for entry in result["results"]] == uris[:3]
        assert result["results"][0]["content"].endswith(
            "[... truncated, use a more specific URI or full level]"
        )
        assert len(result["results"][0]["content"]) < 2700
        assert provider._client.calls == [
            ("/api/v1/content/read", {"uri": uris[0]}),
            ("/api/v1/content/read", {"uri": uris[1]}),
            ("/api/v1/content/read", {"uri": uris[2]}),
        ]

    def test_read_deduplicates_uri_batch_and_keeps_errors_per_uri(self):
        provider = OpenVikingMemoryProvider()
        ok_uri = "viking://user/hermes/memories/ok.md"
        bad_uri = "viking://user/hermes/memories/bad.md"
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/read",
                    (("uri", ok_uri),),
                ): {"result": {"content": "ok content"}},
                (
                    "/api/v1/content/read",
                    (("uri", bad_uri),),
                ): RuntimeError("read failed"),
            }
        )

        result = json.loads(
            provider._tool_read({"uris": [ok_uri, ok_uri, bad_uri], "level": "full"})
        )

        assert result["requested"] == 2
        assert result["returned"] == 2
        assert result["truncated"] is False
        assert result["results"][0]["content"] == "ok content"
        assert result["results"][1] == {
            "uri": bad_uri,
            "level": "full",
            "error": "read failed",
        }

    def test_overview_file_uri_routes_straight_to_content_read_via_stat_probe(self):
        """Pre-check via fs/stat: file URIs skip the directory-only endpoint entirely."""
        provider = OpenVikingMemoryProvider()
        file_uri = "viking://user/hermes/memories/entities/mem_abc.md"
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/stat",
                    (("uri", file_uri),),
                ): {"result": {"isDir": False}},
                (
                    "/api/v1/content/read",
                    (("uri", file_uri),),
                ): {"result": {"content": "full content"}},
            }
        )

        result = json.loads(provider._tool_read({"uri": file_uri, "level": "overview"}))

        assert result["uri"] == file_uri
        assert result["resolved_uri"] == file_uri
        assert result["level"] == "overview"
        assert result["fallback"] == "content/read"
        assert result["content"] == "full content"
        assert provider._client.calls == [
            ("/api/v1/fs/stat", {"uri": file_uri}),
            ("/api/v1/content/read", {"uri": file_uri}),
        ]

    def test_overview_dir_uri_skips_stat_when_pseudo_summary(self):
        """Pseudo-URI path already resolves to dir, so no stat probe needed."""
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/overview",
                    (("uri", "viking://user/hermes"),),
                ): {"result": "overview"},
            }
        )

        result = json.loads(provider._tool_read({"uri": "viking://user/hermes/.overview.md", "level": "overview"}))

        assert result["content"] == "overview"
        # No fs/stat call — normalization already determined it's a directory.
        assert provider._client.calls == [
            ("/api/v1/content/overview", {"uri": "viking://user/hermes"}),
        ]

    def test_overview_directory_uri_uses_stat_probe_then_overview(self):
        """Non-pseudo directory URI: stat → isDir=True → summary endpoint."""
        provider = OpenVikingMemoryProvider()
        dir_uri = "viking://user/hermes/memories"
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/stat",
                    (("uri", dir_uri),),
                ): {"result": {"isDir": True}},
                (
                    "/api/v1/content/overview",
                    (("uri", dir_uri),),
                ): {"result": "dir overview"},
            }
        )

        result = json.loads(provider._tool_read({"uri": dir_uri, "level": "overview"}))

        assert result["content"] == "dir overview"
        assert "fallback" not in result
        assert provider._client.calls == [
            ("/api/v1/fs/stat", {"uri": dir_uri}),
            ("/api/v1/content/overview", {"uri": dir_uri}),
        ]

    def test_overview_file_uri_falls_back_via_exception_when_stat_indeterminate(self):
        """If fs/stat raises or returns unknown shape, legacy exception fallback still kicks in."""
        provider = OpenVikingMemoryProvider()
        file_uri = "viking://user/hermes/memories/entities/mem_abc.md"
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/stat",
                    (("uri", file_uri),),
                ): RuntimeError("stat unavailable"),
                (
                    "/api/v1/content/overview",
                    (("uri", file_uri),),
                ): RuntimeError("500 Internal Server Error"),
                (
                    "/api/v1/content/read",
                    (("uri", file_uri),),
                ): {"result": {"content": "fallback full content"}},
            }
        )

        result = json.loads(provider._tool_read({"uri": file_uri, "level": "overview"}))

        assert result["uri"] == file_uri
        assert result["level"] == "overview"
        assert result["fallback"] == "content/read"
        assert result["content"] == "fallback full content"
        assert provider._client.calls == [
            ("/api/v1/fs/stat", {"uri": file_uri}),
            ("/api/v1/content/overview", {"uri": file_uri}),
            ("/api/v1/content/read", {"uri": file_uri}),
        ]

    def test_summary_uri_error_does_not_fallback_and_raises(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/overview",
                    (("uri", "viking://user/hermes"),),
                ): RuntimeError("500 Internal Server Error"),
            }
        )

        try:
            provider._tool_read({"uri": "viking://user/hermes/.overview.md", "level": "overview"})
            assert False, "Expected summary endpoint error to be raised"
        except RuntimeError:
            pass

        assert provider._client.calls == [
            ("/api/v1/content/overview", {"uri": "viking://user/hermes"}),
        ]


class TestOpenVikingAutoRecallPrefetch:
    def test_prefetch_e2e_sends_limit_and_reads_l2_content(self, monkeypatch):
        records = {"searches": [], "reads": [], "headers": []}

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/health":
                    self._send_json({"healthy": True})
                    return
                if parsed.path == "/api/v1/content/read":
                    query = parse_qs(parsed.query)
                    uri = query.get("uri", [""])[0]
                    records["reads"].append(uri)
                    self._send_json({"result": {"content": "E2E full L2 memory content."}})
                    return
                self.send_error(404)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0") or "0")
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                records["headers"].append(dict(self.headers))
                if self.path == "/api/v1/search/search":
                    records["searches"].append(payload)
                    if payload.get("context_type") == "memory":
                        self._send_json({
                            "result": {
                                "memories": [
                                    {
                                        "uri": "viking://user/peers/hermes/memories/e2e-full.md",
                                        "score": 0.9,
                                        "level": 2,
                                        "category": "events",
                                        "abstract": "E2E abstract should not be injected.",
                                    }
                                ],
                                "resources": [],
                            }
                        })
                    else:
                        self._send_json({"result": {"memories": [], "resources": []}})
                    return
                self.send_error(404)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f"http://127.0.0.1:{server.server_port}"

        for key in (
            "OPENVIKING_RECALL_LIMIT",
            "OPENVIKING_RECALL_SCORE_THRESHOLD",
            "OPENVIKING_RECALL_MAX_INJECTED_CHARS",
            "OPENVIKING_RECALL_PREFER_ABSTRACT",
            "OPENVIKING_RECALL_RESOURCES",
            "OPENVIKING_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENVIKING_ENDPOINT", endpoint)
        monkeypatch.setenv("OPENVIKING_ACCOUNT", "acct")
        monkeypatch.setenv("OPENVIKING_USER", "user")
        monkeypatch.setenv("OPENVIKING_AGENT", "hermes")

        provider = OpenVikingMemoryProvider()
        try:
            provider.initialize("e2e-session")
            block = provider.prefetch("What should we recall?", session_id="e2e-session")
        finally:
            provider.shutdown()
            server.shutdown()
            server.server_close()
            thread.join(timeout=3.0)

        assert block.startswith("## OpenViking Context\n")
        assert "E2E full L2 memory content." in block
        assert "E2E abstract should not be injected." not in block
        assert records["reads"] == ["viking://user/peers/hermes/memories/e2e-full.md"]
        assert len(records["searches"]) == 1
        assert records["searches"][0]["context_type"] == "memory"
        assert records["searches"][0]["session_id"] == "e2e-session"
        assert "target_uri" not in records["searches"][0]
        assert all(payload["limit"] == 24 for payload in records["searches"])
        assert all("top_k" not in payload for payload in records["searches"])
        assert all("mode" not in payload for payload in records["searches"])
        assert all(payload["score_threshold"] == 0 for payload in records["searches"])
        normalized_headers = [
            {key.lower(): value for key, value in headers.items()}
            for headers in records["headers"]
        ]
        assert all(headers.get("x-openviking-actor-peer") == "hermes" for headers in normalized_headers)
        assert all(headers.get("x-openviking-account") == "acct" for headers in normalized_headers)
        assert all(headers.get("x-openviking-user") == "user" for headers in normalized_headers)

    def test_prefetch_searches_current_query_when_no_background_result(self, monkeypatch):
        responses = {
            (
                "/api/v1/search/search",
                "memory",
                "Who is Caroline?",
                "session-test",
            ): {
                "result": {
                    "memories": [
                        {
                            "uri": "viking://user/peers/hermes/memories/caroline.md",
                            "score": 0.9,
                            "level": 1,
                            "category": "profile",
                            "abstract": "Caroline is a transgender woman.",
                        }
                    ]
                }
            },
        }
        provider = make_prefetch_provider(monkeypatch, responses)

        block = provider.prefetch("Who is Caroline?", session_id="session-test")

        assert "Caroline is a transgender woman." in block

    def test_prefetch_does_not_consume_other_session_query_result(self, monkeypatch):
        responses = {
            (
                "/api/v1/search/search",
                "memory",
                "Who is Caroline?",
                "session-a",
            ): {
                "result": {
                    "memories": [
                        {
                            "uri": "viking://user/peers/hermes/memories/caroline.md",
                            "score": 0.9,
                            "level": 1,
                            "category": "profile",
                            "abstract": "Caroline context should stay scoped.",
                        }
                    ]
                }
            },
            (
                "/api/v1/search/search",
                "memory",
                "When did Melanie run a charity race?",
                "session-b",
            ): {
                "result": {
                    "memories": [
                        {
                            "uri": "viking://user/peers/hermes/memories/melanie-race.md",
                            "score": 0.9,
                            "level": 1,
                            "category": "events",
                            "abstract": "Melanie ran the charity race on May 20.",
                        }
                    ]
                }
            },
        }
        provider = make_prefetch_provider(monkeypatch, responses)

        first_block = provider.prefetch("Who is Caroline?", session_id="session-a")
        block = provider.prefetch(
            "When did Melanie run a charity race?",
            session_id="session-b",
        )

        assert "Caroline context should stay scoped." in first_block
        assert "Melanie ran the charity race on May 20." in block
        assert "Caroline context should stay scoped." not in block

    def test_prefetch_filters_low_score_items_with_local_threshold(self, monkeypatch):
        responses = {
            ("/api/v1/search/search", "memory", "What should we recall?", "session-test"): {
                "result": {
                    "memories": [
                        {
                            "uri": "viking://user/peers/hermes/memories/keep.md",
                            "score": 0.22,
                            "level": 1,
                            "category": "preferences",
                            "abstract": "Keep this relevant memory.",
                        },
                        {
                            "uri": "viking://user/peers/hermes/memories/drop.md",
                            "score": 0.12,
                            "level": 1,
                            "category": "preferences",
                            "abstract": "Drop this weak memory.",
                        },
                    ]
                }
            },
        }
        provider = make_prefetch_provider(monkeypatch, responses)

        block = wait_prefetch(provider)

        assert block.startswith("## OpenViking Context\n")
        assert "Keep this relevant memory." in block
        assert "Drop this weak memory." not in block
        search_payloads = [call[2] for call in FakeRecallClient.calls if call[:2] == ("post", "/api/v1/search/search")]
        assert len(search_payloads) == 1
        assert search_payloads[0]["context_type"] == "memory"
        assert "target_uri" not in search_payloads[0]
        assert all(payload["limit"] == 24 for payload in search_payloads)
        assert all("top_k" not in payload for payload in search_payloads)
        assert all("mode" not in payload for payload in search_payloads)
        assert all(payload["score_threshold"] == 0 for payload in search_payloads)

    def test_prefetch_skips_complete_entries_that_do_not_fit_budget(self, monkeypatch):
        long_memory = "X" * 120
        responses = {
            ("/api/v1/search/search", "memory", "What should we recall?", "session-test"): {
                "result": {
                    "memories": [
                        {
                            "uri": "viking://user/peers/hermes/memories/too-large.md",
                            "score": 0.9,
                            "level": 1,
                            "category": "memory",
                            "abstract": long_memory,
                        },
                        {
                            "uri": "viking://user/peers/hermes/memories/small.md",
                            "score": 0.8,
                            "level": 1,
                            "category": "memory",
                            "abstract": "Small memory fits.",
                        },
                    ]
                }
            },
        }
        provider = make_prefetch_provider(
            monkeypatch,
            responses,
            OPENVIKING_RECALL_MAX_INJECTED_CHARS="90",
        )

        block = wait_prefetch(provider)

        assert "Small memory fits." in block
        assert long_memory not in block
        assert "XXX" not in block

    def test_prefetch_reads_full_l2_content_by_default(self, monkeypatch):
        responses = {
            ("/api/v1/search/search", "memory", "What should we recall?", "session-test"): {
                "result": {
                    "memories": [
                        {
                            "uri": "viking://user/peers/hermes/memories/full.md",
                            "score": 0.9,
                            "level": 2,
                            "category": "events",
                            "abstract": "Abstract only.",
                        }
                    ]
                }
            },
            ("/api/v1/content/read", "viking://user/peers/hermes/memories/full.md"): {
                "result": {"content": "Full L2 memory content."}
            },
        }
        provider = make_prefetch_provider(monkeypatch, responses)

        block = wait_prefetch(provider)

        assert "Full L2 memory content." in block
        assert "Abstract only." not in block
        assert (
            "get",
            "/api/v1/content/read",
            {"uri": "viking://user/peers/hermes/memories/full.md"},
        ) in FakeRecallClient.calls

    def test_prefetch_prefer_abstract_does_not_read_l2_content(self, monkeypatch):
        responses = {
            ("/api/v1/search/search", "memory", "What should we recall?", "session-test"): {
                "result": {
                    "memories": [
                        {
                            "uri": "viking://user/peers/hermes/memories/full.md",
                            "score": 0.9,
                            "level": 2,
                            "category": "events",
                            "abstract": "Use the abstract.",
                        }
                    ]
                }
            },
        }
        provider = make_prefetch_provider(
            monkeypatch,
            responses,
            OPENVIKING_RECALL_PREFER_ABSTRACT="true",
        )

        block = wait_prefetch(provider)

        assert "Use the abstract." in block
        assert not any(call[:2] == ("get", "/api/v1/content/read") for call in FakeRecallClient.calls)

    def test_prefetch_honors_configured_limit_candidate_limit_and_resources(self, monkeypatch):
        responses = {
            ("/api/v1/search/search", ("memory", "resource"), "What should we recall?", "session-test"): {
                "result": {
                    "memories": [],
                    "resources": [
                        {
                            "uri": "viking://resources/doc.md",
                            "score": 0.9,
                            "level": 1,
                            "category": "resource",
                            "abstract": "Resource recall enabled.",
                        }
                    ]
                }
            },
        }
        provider = make_prefetch_provider(
            monkeypatch,
            responses,
            OPENVIKING_RECALL_LIMIT="2",
            OPENVIKING_RECALL_RESOURCES="true",
        )

        block = wait_prefetch(provider)

        assert "Resource recall enabled." in block
        search_payloads = [call[2] for call in FakeRecallClient.calls if call[:2] == ("post", "/api/v1/search/search")]
        assert len(search_payloads) == 1
        assert search_payloads[0]["context_type"] == ["memory", "resource"]
        assert "target_uri" not in search_payloads[0]
        assert all(payload["limit"] == 20 for payload in search_payloads)
        assert all("top_k" not in payload for payload in search_payloads)
        assert all("mode" not in payload for payload in search_payloads)

    def test_queue_prefetch_is_noop_for_openviking_recall(self, monkeypatch):
        provider = make_prefetch_provider(monkeypatch, {})

        provider.queue_prefetch("What should we recall?", session_id="session-test")

        assert FakeRecallClient.calls == []


class TestOpenVikingBrowse:
    def test_list_browse_unwraps_and_normalizes_entry_shapes(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/ls",
                    (("uri", "viking://user/hermes"),),
                ): {
                    "result": {
                        "entries": [
                            {"name": "memories", "uri": "viking://user/hermes/memories", "type": "dir"},
                            {"rel_path": "profile.md", "uri": "viking://user/hermes/memories/profile.md", "isDir": False, "abstract": "Profile"},
                        ]
                    }
                },
            }
        )

        result = json.loads(provider._tool_browse({"action": "list", "path": "viking://user/hermes"}))

        assert result["path"] == "viking://user/hermes"
        assert result["entries"] == [
            {"name": "memories", "uri": "viking://user/hermes/memories", "type": "dir", "abstract": ""},
            {"name": "profile.md", "uri": "viking://user/hermes/memories/profile.md", "type": "file", "abstract": "Profile"},
        ]
        assert provider._client.calls == [(
            "/api/v1/fs/ls",
            {"uri": "viking://user/hermes"},
        )]


class TestOpenVikingMemoryUriBuilder:
    """Regression tests for _build_memory_uri — fixes #36969.

    OpenViking's current memory layout stores peer-scoped memories under
    viking://user/peers/{peer_id}/...
    """

    def _make_provider(self, user="alice", agent="coder"):
        p = OpenVikingMemoryProvider.__new__(OpenVikingMemoryProvider)
        p._user = user
        p._agent = agent
        return p

    def test_uri_layout_includes_peer_segment(self):
        """URI must contain /peers/{peer_id}/ between user and memories."""
        p = self._make_provider(user="alice", agent="coder")
        uri = p._build_memory_uri("preferences")
        assert uri.startswith("viking://user/peers/coder/memories/preferences/mem_")
        assert uri.endswith(".md")

    def test_uri_uses_configured_peer_not_default(self):
        """_agent value is the OpenViking actor peer ID, not hardcoded to 'hermes'."""
        p = self._make_provider(user="alice", agent="research-bot")
        uri = p._build_memory_uri("entities")
        assert "/peers/research-bot/" in uri
        assert "/peers/hermes/" not in uri

    def test_uri_slug_is_twelve_hex_chars_and_unique(self):
        """Slug must be 12 hex chars and differ between calls."""
        import re
        p = self._make_provider()
        uri1 = p._build_memory_uri("preferences")
        uri2 = p._build_memory_uri("preferences")
        slug1 = uri1.split("/mem_")[1].replace(".md", "")
        slug2 = uri2.split("/mem_")[1].replace(".md", "")
        assert re.fullmatch(r"[0-9a-f]{12}", slug1)
        assert re.fullmatch(r"[0-9a-f]{12}", slug2)
        assert slug1 != slug2

    def test_uri_subdir_placed_correctly_for_all_categories(self):
        """All five category subdirs must appear between memories/ and slug."""
        p = self._make_provider(user="u", agent="a")
        subdirs = ["preferences", "entities", "events", "cases", "patterns"]
        for subdir in subdirs:
            uri = p._build_memory_uri(subdir)
            assert f"/memories/{subdir}/mem_" in uri, (
                f"subdir '{subdir}' not placed correctly in URI: {uri}"
            )


# ===================================================================
# Issue #21130 — OPENVIKING_* not reloaded after /reload
# ===================================================================


class TestEnsureClientReloadsEnv:
    """Verify /reload picks up new OPENVIKING_* values without a restart (#21130)."""

    def test_ensure_client_rebuilds_when_api_key_changes(self, monkeypatch):
        constructions = []

        class _StubClient:
            def __init__(self, endpoint, api_key, account="", user="", agent="hermes"):
                constructions.append({"endpoint": endpoint, "api_key": api_key,
                                      "account": account, "user": user, "agent": agent})
                self.endpoint, self.api_key = endpoint, api_key
                self.account, self.user, self.agent = account, user, agent

            def health(self):
                return True

        monkeypatch.setattr("plugins.memory.openviking._VikingClient", _StubClient)
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://srv:31933")
        monkeypatch.setenv("OPENVIKING_API_KEY", "")

        provider = OpenVikingMemoryProvider()
        provider._env_refresh_enabled = True
        first = provider._ensure_client()
        assert first is not None
        assert first.api_key == ""
        assert len(constructions) == 1

        # Same env on second call — must reuse cached client (no rebuild).
        assert provider._ensure_client() is first
        assert len(constructions) == 1

        # Simulate /reload: env now carries the new API key.
        monkeypatch.setenv("OPENVIKING_API_KEY", "sk-fresh")
        rebuilt = provider._ensure_client()
        assert rebuilt is not None
        assert rebuilt is not first
        assert rebuilt.api_key == "sk-fresh"
        assert len(constructions) == 2

    def test_ensure_client_rebuilds_when_endpoint_changes(self, monkeypatch):
        builds = []

        class _StubClient:
            def __init__(self, endpoint, api_key, **kw):
                builds.append(endpoint)
                self.endpoint = endpoint

            def health(self):
                return True

        monkeypatch.setattr("plugins.memory.openviking._VikingClient", _StubClient)
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://a")
        monkeypatch.setenv("OPENVIKING_API_KEY", "key")

        provider = OpenVikingMemoryProvider()
        provider._env_refresh_enabled = True
        provider._ensure_client()
        provider._ensure_client()  # cached
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://b")
        provider._ensure_client()  # rebuilds
        assert builds == ["http://a", "http://b"]

    def test_ensure_client_returns_none_when_health_fails(self, monkeypatch):
        class _StubClient:
            def __init__(self, *a, **kw):
                pass

            def health(self):
                return False

        monkeypatch.setattr("plugins.memory.openviking._VikingClient", _StubClient)
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://dead")
        monkeypatch.setenv("OPENVIKING_API_KEY", "")

        provider = OpenVikingMemoryProvider()
        provider._env_refresh_enabled = True
        assert provider._ensure_client() is None
        assert provider._client is None

    def test_handle_tool_call_uses_ensure_client(self, monkeypatch):
        provider = OpenVikingMemoryProvider()
        provider._env_refresh_enabled = True

        class _StubClient:
            def __init__(self, *a, **kw):
                pass

            def health(self):
                return False

        monkeypatch.setattr("plugins.memory.openviking._VikingClient", _StubClient)

        out = provider.handle_tool_call("viking_search", {"query": "x"})
        assert "not connected" in out.lower()

    def test_prefetch_goes_through_ensure_client(self, monkeypatch):
        # Pre-turn recall must refresh from env too, not just tool calls: a
        # /reload before the first recall of a turn should reach the new client.
        provider = OpenVikingMemoryProvider()
        provider._env_refresh_enabled = True
        calls = {"n": 0}
        monkeypatch.setattr(provider, "_ensure_client", lambda: (calls.__setitem__("n", calls["n"] + 1) or None))
        # Long-enough query so the length guard doesn't short-circuit first.
        assert provider.prefetch("remember my project deadlines please") == ""
        assert calls["n"] == 1

    def test_on_session_switch_goes_through_ensure_client(self, monkeypatch):
        # Session rotation (/new, /branch, /resume) is a live client-gated path;
        # it must refresh from env rather than read a stale snapshot.
        provider = OpenVikingMemoryProvider()
        provider._env_refresh_enabled = True
        calls = {"n": 0}
        monkeypatch.setattr(provider, "_ensure_client", lambda: (calls.__setitem__("n", calls["n"] + 1) or None))
        provider.on_session_switch("new-sid-123")
        assert calls["n"] == 1

    def test_unreachable_local_endpoint_triggers_recovery(self, monkeypatch):
        # A newly-resolved LOCAL endpoint that isn't up must preserve
        # initialize()'s recovery: (re)start the server + attach in the
        # background — not silently disable memory (sweeper feedback on #21130).
        class _StubClient:
            def __init__(self, *a, **kw):
                pass

            def health(self):
                return False  # -> classified "unreachable"

        monkeypatch.setattr("plugins.memory.openviking._VikingClient", _StubClient)
        monkeypatch.setattr("plugins.memory.openviking._is_local_openviking_url", lambda _u: True)
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://127.0.0.1:31933")
        monkeypatch.setenv("OPENVIKING_API_KEY", "")

        provider = OpenVikingMemoryProvider()
        provider._env_refresh_enabled = True
        recovered = {"n": 0}
        monkeypatch.setattr(
            provider,
            "_handle_runtime_openviking_unreachable",
            lambda *a, **k: recovered.__setitem__("n", recovered["n"] + 1),
        )
        assert provider._ensure_client() is None
        assert recovered["n"] == 1
