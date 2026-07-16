"""Tests for CodexEventProjector — codex item/* events → Hermes messages list.

Drives projection against fixture notifications captured from codex 0.130.0
plus synthetic ones for item types we couldn't auth-test live."""

from __future__ import annotations

import json

import pytest

from agent.transports.codex_event_projector import (
    CodexEventProjector,
    _deterministic_call_id,
    _dynamic_tool_call_id_type,
    _format_tool_args,
)


# --- Fixture: real `commandExecution` notification captured from codex 0.130.0
COMMAND_EXEC_COMPLETED = {
    "method": "item/completed",
    "params": {
        "item": {
            "type": "commandExecution",
            "id": "f8a75c66-a89e-4fd7-8bcf-2d58e664fa9e",
            "command": "/bin/bash -lc 'echo hello && ls /tmp | head -3'",
            "cwd": "/tmp",
            "processId": None,
            "source": "userShell",
            "status": "completed",
            "commandActions": [
                {"type": "listFiles", "command": "ls /tmp", "path": "tmp"}
            ],
            "aggregatedOutput": "hello\naa_lang.json\n",
            "exitCode": 0,
            "durationMs": 10,
        },
        "threadId": "019e1a94-352b-71e1-b214-e5c67c9ec190",
        "turnId": "019e1a94-3553-7940-8af3-4ca57142deb7",
        "completedAtMs": 1778562381151,
    },
}


class TestProjectionInvariants:
    """Universal invariants that must hold across all projection paths."""

    def test_streaming_deltas_dont_materialize(self) -> None:
        p = CodexEventProjector()
        for delta_method in (
            "item/commandExecution/outputDelta",
            "item/agentMessage/delta",
            "item/reasoning/delta",
        ):
            r = p.project({"method": delta_method, "params": {"delta": "x"}})
            assert r.messages == [], (
                f"{delta_method} should NOT produce messages — only "
                f"item/completed materializes"
            )
            assert r.is_tool_iteration is False
            assert r.final_text is None

    def test_turn_started_and_completed_are_silent(self) -> None:
        p = CodexEventProjector()
        for method in ("turn/started", "turn/completed", "thread/started"):
            r = p.project({"method": method, "params": {}})
            assert r.messages == []

    def test_unknown_method_silent(self) -> None:
        p = CodexEventProjector()
        r = p.project({"method": "totally/unknown", "params": {}})
        assert r.messages == []

    def test_mcp_tool_identity_encoding_is_collision_free(self) -> None:
        def projected_call_id(server: str, tool: str) -> str:
            projected = CodexEventProjector().project(
                {
                    "method": "item/completed",
                    "params": {
                        "item": {
                            "type": "mcpToolCall",
                            "id": "same-provider-id",
                            "server": server,
                            "tool": tool,
                        }
                    },
                }
            )
            return projected.messages[0]["tool_calls"][0]["id"]

        assert projected_call_id("a__b", "c") != projected_call_id("a", "b__c")


class TestCommandExecutionProjection:
    """Real captured notification → assistant tool_call + tool result."""

    def test_command_completed_produces_two_messages(self) -> None:
        p = CodexEventProjector()
        r = p.project(COMMAND_EXEC_COMPLETED)
        assert len(r.messages) == 2
        assert r.is_tool_iteration is True

    def test_first_message_is_assistant_tool_call(self) -> None:
        p = CodexEventProjector()
        msgs = p.project(COMMAND_EXEC_COMPLETED).messages
        assistant = msgs[0]
        assert assistant["role"] == "assistant"
        assert assistant["content"] is None
        assert len(assistant["tool_calls"]) == 1
        tc = assistant["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "exec_command"
        args = json.loads(tc["function"]["arguments"])
        assert "echo hello" in args["command"]
        assert args["cwd"] == "/tmp"

    def test_second_message_is_tool_result_correlating_by_id(self) -> None:
        p = CodexEventProjector()
        msgs = p.project(COMMAND_EXEC_COMPLETED).messages
        assistant, tool = msgs
        assert tool["role"] == "tool"
        assert tool["tool_call_id"] == assistant["tool_calls"][0]["id"]
        assert "hello" in tool["content"]

    def test_nonzero_exit_code_annotated_in_tool_result(self) -> None:
        item = {**COMMAND_EXEC_COMPLETED["params"]["item"], "exitCode": 2,
                "aggregatedOutput": "boom"}
        notif = {
            "method": "item/completed",
            "params": {**COMMAND_EXEC_COMPLETED["params"], "item": item},
        }
        p = CodexEventProjector()
        msgs = p.project(notif).messages
        assert "[exit 2]" in msgs[1]["content"]
        assert "boom" in msgs[1]["content"]

    def test_deterministic_call_id_across_replay(self) -> None:
        # Same item id → same call_id (prefix cache must stay valid).
        p1 = CodexEventProjector()
        p2 = CodexEventProjector()
        a = p1.project(COMMAND_EXEC_COMPLETED).messages
        b = p2.project(COMMAND_EXEC_COMPLETED).messages
        assert a[0]["tool_calls"][0]["id"] == b[0]["tool_calls"][0]["id"]

    def test_completed_item_replay_is_not_projected_twice(self) -> None:
        projector = CodexEventProjector()

        first = projector.project(COMMAND_EXEC_COMPLETED)
        replay = projector.project(COMMAND_EXEC_COMPLETED)

        assert len(first.messages) == 2
        assert first.is_tool_iteration is True
        assert replay.messages == []
        assert replay.is_tool_iteration is False

    def test_idless_completed_items_are_dropped_without_colliding(self) -> None:
        projector = CodexEventProjector()

        notifications = [
            {"type": "reasoning", "summary": ["same"]},
            {"type": "reasoning", "summary": ["same"]},
            {"type": "commandExecution", "command": "first"},
            {"type": "commandExecution", "command": "second"},
        ]

        for item in notifications:
            result = projector.project(
                {"method": "item/completed", "params": {"item": item}}
            )
            assert result.messages == []
            assert result.is_tool_iteration is False

    @pytest.mark.parametrize("item_type", ["", " ", "\t"])
    def test_blank_item_type_does_not_consume_item_id(self, item_type: str) -> None:
        projector = CodexEventProjector()
        malformed = projector.project({
            "method": "item/completed",
            "params": {"item": {"type": item_type, "id": "shared-blank-type"}},
        })
        valid = projector.project({
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "commandExecution",
                    "id": "shared-blank-type",
                    "command": "pwd",
                    "cwd": "/tmp",
                    "status": "completed",
                    "aggregatedOutput": "/tmp\n",
                    "exitCode": 0,
                }
            },
        })

        assert malformed.messages == []
        assert malformed.is_tool_iteration is False
        assert len(valid.messages) == 2
        assert valid.is_tool_iteration is True

    def test_conflicting_same_id_completion_preserves_started_identity(self) -> None:
        projector = CodexEventProjector()
        projector.project({
            "method": "item/started",
            "params": {
                "item": {
                    "type": "commandExecution",
                    "id": "shared-1",
                    "command": "pwd",
                    "cwd": "/tmp",
                }
            },
        })

        conflict = projector.project({
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "dynamicToolCall",
                    "id": "shared-1",
                    "namespace": "hermes",
                    "tool": "memory",
                    "arguments": {"action": "add"},
                    "success": True,
                }
            },
        })
        valid = projector.project({
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "commandExecution",
                    "id": "shared-1",
                    "command": "pwd",
                    "cwd": "/tmp",
                    "aggregatedOutput": "/tmp\n",
                    "exitCode": 0,
                }
            },
        })

        assert conflict.messages == []
        assert conflict.is_tool_iteration is False
        assert len(valid.messages) == 2
        assert valid.is_tool_iteration is True
        assert valid.messages[0]["tool_calls"][0]["id"] == "codex_4_exec_shared-1"

    @pytest.mark.parametrize(
        "conflicting_item",
        [
            {
                "type": "dynamicToolCall",
                "id": "shared-start",
                "namespace": "hermes",
                "tool": "memory",
            },
            {
                "type": "mcpToolCall",
                "id": "shared-start",
                "server": "honcho",
                "tool": "search",
            },
        ],
    )
    def test_conflicting_tool_start_does_not_mutate_existing_identity(
        self, conflicting_item: dict
    ) -> None:
        projector = CodexEventProjector()
        projector.project({
            "method": "item/started",
            "params": {"item": {"type": "plan", "id": "shared-start"}},
        })

        conflict = projector.project({
            "method": "item/started",
            "params": {"item": conflicting_item},
        })
        valid = projector.project({
            "method": "item/completed",
            "params": {
                "item": {"type": "plan", "id": "shared-start", "text": "valid"}
            },
        })

        assert conflict.messages == []
        assert projector._item_types_by_id == {"shared-start": "plan"}
        assert projector._item_tool_identities_by_id == {}
        assert len(valid.messages) == 1
        assert "[codex plan]" in valid.messages[0]["content"]

    @pytest.mark.parametrize(
        "started, conflicting, valid",
        [
            (
                {"type": "dynamicToolCall", "id": "same-dyn", "namespace": "hermes", "tool": "memory"},
                {"type": "dynamicToolCall", "id": "same-dyn", "namespace": "hermes", "tool": "terminal"},
                {"type": "dynamicToolCall", "id": "same-dyn", "namespace": "hermes", "tool": "memory"},
            ),
            (
                {
                    "type": "mcpToolCall",
                    "id": "same-mcp",
                    "server": "honcho",
                    "tool": "search",
                },
                {
                    "type": "mcpToolCall",
                    "id": "same-mcp",
                    "server": "honcho",
                    "tool": "conclude",
                },
                {
                    "type": "mcpToolCall",
                    "id": "same-mcp",
                    "server": "honcho",
                    "tool": "search",
                },
            ),
        ],
    )
    def test_same_type_conflicting_tool_name_does_not_consume_identity(
        self, started: dict, conflicting: dict, valid: dict
    ) -> None:
        projector = CodexEventProjector()
        projector.project({"method": "item/started", "params": {"item": started}})

        conflict = projector.project({
            "method": "item/completed",
            "params": {"item": {**conflicting, "success": True}},
        })
        accepted = projector.project({
            "method": "item/completed",
            "params": {"item": {**valid, "success": True}},
        })

        assert conflict.messages == []
        assert conflict.is_tool_iteration is False
        assert len(accepted.messages) == 2
        assert accepted.is_tool_iteration is True

    def test_dynamic_namespace_conflict_does_not_consume_identity(self) -> None:
        projector = CodexEventProjector()
        started = {
            "type": "dynamicToolCall",
            "id": "same-dyn-namespace",
            "namespace": "hermes",
            "tool": "memory",
        }
        projector.project({"method": "item/started", "params": {"item": started}})

        conflict = projector.project({
            "method": "item/completed",
            "params": {
                "item": {
                    **started,
                    "namespace": "foreign",
                    "success": True,
                }
            },
        })
        accepted = projector.project({
            "method": "item/completed",
            "params": {"item": {**started, "success": True}},
        })

        assert conflict.messages == []
        assert conflict.is_tool_iteration is False
        assert len(accepted.messages) == 2
        assert accepted.is_tool_iteration is True


class TestAgentMessageProjection:
    """assistant text → final_text + assistant message."""

    def test_agent_message_projects_to_assistant(self) -> None:
        p = CodexEventProjector()
        r = p.project({
            "method": "item/completed",
            "params": {"item": {"type": "agentMessage", "id": "x",
                                "text": "hi there"}},
        })
        assert r.final_text == "hi there"
        assert r.messages == [{"role": "assistant", "content": "hi there"}]
        assert r.is_tool_iteration is False

    def test_pending_reasoning_attaches_to_next_assistant_message(self) -> None:
        p = CodexEventProjector()
        # First a reasoning item lands
        r1 = p.project({
            "method": "item/completed",
            "params": {"item": {"type": "reasoning", "id": "r1",
                                "summary": ["thinking..."],
                                "content": ["step 1", "step 2"]}},
        })
        assert r1.messages == []  # reasoning alone produces no message
        # Then the assistant message
        r2 = p.project({
            "method": "item/completed",
            "params": {"item": {"type": "agentMessage", "id": "a1",
                                "text": "ok"}},
        })
        assistant = r2.messages[0]
        assert "reasoning" in assistant
        assert "thinking" in assistant["reasoning"]
        assert "step 1" in assistant["reasoning"]

    def test_reasoning_consumed_after_attaching(self) -> None:
        p = CodexEventProjector()
        p.project({"method": "item/completed", "params": {"item": {
            "type": "reasoning", "id": "r1", "summary": ["once"], "content": []}}})
        first = p.project({"method": "item/completed", "params": {"item": {
            "type": "agentMessage", "id": "a", "text": "first"}}}).messages[0]
        second = p.project({"method": "item/completed", "params": {"item": {
            "type": "agentMessage", "id": "b", "text": "second"}}}).messages[0]
        assert "reasoning" in first
        assert "reasoning" not in second


class TestFileChangeProjection:
    def test_file_change_summary_no_inlined_content(self) -> None:
        item = {
            "type": "fileChange",
            "id": "fc1",
            "status": "applied",
            "changes": [
                {"kind": {"type": "add"}, "path": "/tmp/new.py"},
                {"kind": {"type": "update"}, "path": "/tmp/old.py"},
            ],
        }
        p = CodexEventProjector()
        msgs = p.project({"method": "item/completed",
                          "params": {"item": item}}).messages
        assert len(msgs) == 2
        tc = msgs[0]["tool_calls"][0]
        assert tc["function"]["name"] == "apply_patch"
        args = json.loads(tc["function"]["arguments"])
        assert len(args["changes"]) == 2
        assert all("kind" in c and "path" in c for c in args["changes"])
        assert "applied" in msgs[1]["content"]


class TestMcpToolCallProjection:
    def test_mcp_tool_call_namespaced(self) -> None:
        item = {
            "type": "mcpToolCall",
            "id": "m1",
            "server": "obsidian",
            "tool": "search_notes",
            "status": "completed",
            "arguments": {"query": "hermes"},
            "result": {"content": [{"text": "found"}]},
            "error": None,
        }
        msgs = CodexEventProjector().project(
            {"method": "item/completed", "params": {"item": item}}
        ).messages
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "mcp.obsidian.search_notes"
        assert "found" in msgs[1]["content"]

    def test_mcp_error_surfaced(self) -> None:
        item = {
            "type": "mcpToolCall", "id": "m2",
            "server": "x", "tool": "y", "status": "failed",
            "arguments": {}, "result": None,
            "error": {"code": -1, "message": "no"},
        }
        msgs = CodexEventProjector().project(
            {"method": "item/completed", "params": {"item": item}}
        ).messages
        assert "error" in msgs[1]["content"]


class TestUserAndOpaqueProjection:
    def test_user_message_text_fragments_only(self) -> None:
        item = {
            "type": "userMessage", "id": "u1",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "url": "http://x/y"},
                {"type": "text", "text": "world"},
            ],
        }
        msgs = CodexEventProjector().project(
            {"method": "item/completed", "params": {"item": item}}
        ).messages
        assert msgs[0]["role"] == "user"
        assert "hello" in msgs[0]["content"]
        assert "world" in msgs[0]["content"]

    def test_opaque_item_recorded_without_fabricated_tool_calls(self) -> None:
        item = {"type": "plan", "id": "p1", "text": "do the thing"}
        msgs = CodexEventProjector().project(
            {"method": "item/completed", "params": {"item": item}}
        ).messages
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        assert "plan" in msgs[0]["content"].lower()
        assert "tool_calls" not in msgs[0]


class TestHelpers:
    def test_deterministic_call_id_stable(self) -> None:
        assert _deterministic_call_id("exec", "abc") == _deterministic_call_id("exec", "abc")
        assert _deterministic_call_id("exec", "abc") != _deterministic_call_id("exec", "xyz")

    def test_deterministic_call_id_rejects_missing_id(self) -> None:
        with pytest.raises(ValueError, match="item_id"):
            _deterministic_call_id("exec", "")

    def test_deterministic_call_id_is_unambiguous_across_underscore_boundaries(self) -> None:
        assert _deterministic_call_id("dyn_a_b", "c") != _deterministic_call_id(
            "dyn_a", "b_c"
        )

    def test_dynamic_identity_type_is_unambiguous_across_components(self) -> None:
        assert _dynamic_tool_call_id_type("a_b", "c") != _dynamic_tool_call_id_type(
            "a", "b_c"
        )

    def test_format_tool_args_sorted_keys(self) -> None:
        # Sorted keys = deterministic across replays = prefix cache stays valid
        a = _format_tool_args({"b": 1, "a": 2})
        b = _format_tool_args({"a": 2, "b": 1})
        assert a == b


class TestDynamicToolProjection:
    @pytest.mark.parametrize(
        ("item_type", "incomplete_identity", "valid_identity"),
        [
            ("mcpToolCall", {"server": "srv"}, {"server": "srv", "tool": "search"}),
            (
                "dynamicToolCall",
                {"namespace": "hermes"},
                {"namespace": "hermes", "tool": "memory"},
            ),
            (
                "dynamicToolCall",
                {"tool": "memory"},
                {"namespace": "hermes", "tool": "memory"},
            ),
        ],
    )
    @pytest.mark.parametrize("start_incomplete", [False, True])
    def test_incomplete_tool_identity_does_not_materialize_or_consume_item_id(
        self,
        item_type: str,
        incomplete_identity: dict,
        valid_identity: dict,
        start_incomplete: bool,
    ) -> None:
        projector = CodexEventProjector()
        item_id = f"incomplete-{item_type}-{start_incomplete}"
        incomplete_item = {
            "type": item_type,
            "id": item_id,
            **incomplete_identity,
        }
        if start_incomplete:
            projector.project(
                {"method": "item/started", "params": {"item": incomplete_item}}
            )
            assert projector._item_types_by_id == {}
            assert projector._item_tool_identities_by_id == {}
            assert projector._completed_item_keys == set()

        rejected = projector.project(
            {"method": "item/completed", "params": {"item": incomplete_item}}
        )
        accepted = projector.project(
            {
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": item_type,
                        "id": item_id,
                        **valid_identity,
                        "arguments": {},
                        "result": None,
                        "error": None,
                        "contentItems": [],
                        "success": True,
                    }
                },
            }
        )

        assert rejected.messages == []
        assert rejected.is_tool_iteration is False
        assert len(accepted.messages) == 2
        assert accepted.is_tool_iteration is True

    @pytest.mark.parametrize(
        "malformed_identity",
        [
            {"tool": "memory"},
            {"namespace": "", "tool": "memory"},
            {"namespace": "   ", "tool": "memory"},
            {"namespace": "\t", "tool": "memory"},
            {"namespace": "hermes"},
            {"namespace": "hermes", "tool": ""},
        ],
    )
    def test_malformed_dynamic_start_does_not_bind_item_type(
        self, malformed_identity: dict
    ) -> None:
        projector = CodexEventProjector()
        item_id = "reusable-after-malformed-dynamic-start"

        projector.project({
            "method": "item/started",
            "params": {
                "item": {
                    "type": "dynamicToolCall",
                    "id": item_id,
                    **malformed_identity,
                }
            },
        })
        accepted = projector.project({
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "commandExecution",
                    "id": item_id,
                    "command": "pwd",
                    "cwd": "/tmp",
                    "aggregatedOutput": "/tmp\n",
                    "exitCode": 0,
                }
            },
        })

        assert len(accepted.messages) == 2
        assert accepted.is_tool_iteration is True

    def test_input_text_content_is_projected_without_transport_envelope(self) -> None:
        item = {
            "type": "dynamicToolCall",
            "id": "d-text",
            "namespace": "hermes",
            "tool": "todo",
            "arguments": {},
            "status": "completed",
            "contentItems": [
                {"type": "inputText", "text": '{"todos": []}'},
            ],
            "success": True,
        }

        messages = CodexEventProjector().project(
            {"method": "item/completed", "params": {"item": item}}
        ).messages

        assert messages[1]["content"] == '{"todos": []}'


class TestRoleAlternationInvariant:
    """The project must never emit two assistant messages back-to-back from
    one item — that breaks Hermes' message alternation invariant."""

    @pytest.mark.parametrize(
        "item",
        [
            {"type": "commandExecution", "id": "c1", "command": "x",
             "cwd": "/", "status": "completed", "aggregatedOutput": "",
             "exitCode": 0, "commandActions": []},
            {"type": "fileChange", "id": "f1", "status": "applied",
             "changes": []},
            {"type": "mcpToolCall", "id": "m1", "server": "s", "tool": "t",
             "status": "completed", "arguments": {}, "result": None,
             "error": None},
            {"type": "dynamicToolCall", "id": "d1", "namespace": "hermes", "tool": "x",
             "arguments": {}, "status": "completed",
             "contentItems": [], "success": True},
        ],
    )
    def test_tool_items_emit_assistant_then_tool(self, item) -> None:
        msgs = CodexEventProjector().project(
            {"method": "item/completed", "params": {"item": item}}
        ).messages
        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert msgs[1]["role"] == "tool"
        assert msgs[1]["tool_call_id"] == msgs[0]["tool_calls"][0]["id"]
