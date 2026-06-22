"""Tests for plugins/memory/openviking/__init__.py — URI normalization and payload handling."""

import json
from typing import Any, cast

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

    def test_queue_prefetch_searches_only_slash_skill_user_instruction(self, monkeypatch):
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

        provider.queue_prefetch(skill_message)
        assert provider._prefetch_thread is not None
        provider._prefetch_thread.join(timeout=5.0)

        assert RecordingVikingClient.calls == [
            (
                "/api/v1/search/find",
                {"query": "make a skill for release triage", "limit": 5},
            )
        ]

    def test_queue_prefetch_searches_only_skill_bundle_user_instruction(self, monkeypatch):
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

        provider.queue_prefetch(skill_message)
        assert provider._prefetch_thread is not None
        provider._prefetch_thread.join(timeout=5.0)

        assert RecordingVikingClient.calls == [
            (
                "/api/v1/search/find",
                {"query": "fix the failing retrieval test", "limit": 5},
            )
        ]

    def test_queue_prefetch_skips_slash_skill_without_user_instruction(self, monkeypatch):
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

        provider.queue_prefetch(skill_message)

        assert provider._prefetch_thread is None
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

    OpenViking's current memory layout stores agent-scoped memories under
    viking://user/{agent}/...
    """

    def _make_provider(self, user="alice", agent="coder"):
        p = OpenVikingMemoryProvider.__new__(OpenVikingMemoryProvider)
        p._user = user
        p._agent = agent
        return p

    def test_uri_layout_includes_agent_segment(self):
        """URI must contain /{agent}/ between user and memories."""
        p = self._make_provider(user="alice", agent="coder")
        uri = p._build_memory_uri("preferences")
        assert uri.startswith("viking://user/coder/memories/preferences/mem_")
        assert uri.endswith(".md")

    def test_uri_uses_configured_agent_not_default(self):
        """_agent value is the OpenViking actor ID, not hardcoded to 'hermes'."""
        p = self._make_provider(user="alice", agent="research-bot")
        uri = p._build_memory_uri("entities")
        assert "/research-bot/" in uri
        assert "/hermes/" not in uri

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
