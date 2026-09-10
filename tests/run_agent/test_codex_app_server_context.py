"""Hermes context must survive the boundary to a native Codex thread."""

import json
import os
from contextlib import ExitStack, closing
from pathlib import Path

import pytest
import yaml

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from agent.transports.codex_app_server import CodexAppServerError
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.agent.transports.test_codex_app_server_session import FakeClient


@pytest.fixture
def codex_runtime(tmp_path, monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"context_length": 256000},
        "auxiliary": {"title_generation": {"enabled": False}},
    }), encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    clients, failures, threads, canonical_instructions = [], {}, {}, {}

    def client_factory(**kwargs):
        client = FakeClient(**kwargs)
        loaded_threads = set()
        clients.append(client)
        number = len(clients)

        def request(method, params):
            if method in failures:
                raise CodexAppServerError(-32602, failures[method])
            if method in {"thread/start", "thread/resume"}:
                thread_id = params.get("threadId", f"codex-thread-{number}")
                # Codex replays existing developer items even when resume supplies new instructions.
                threads.setdefault(thread_id, params.get("developerInstructions", ""))
                if thread_id not in loaded_threads:
                    canonical_instructions[thread_id] = params.get("developerInstructions", "")
                    loaded_threads.add(thread_id)
                return {
                    "thread": {"id": thread_id}, "cwd": str(tmp_path),
                    "model": params.get("model", "codex-config-model"),
                    "modelProvider": "openai", "approvalPolicy": "never",
                    "approvalsReviewer": "user", "sandbox": {"type": "readOnly"},
                }
            if method == "turn/start":
                client.instructions_seen.append(threads[params["threadId"]])
                turn_id = f"turn-{number}-{len(client.requests)}"
                scope = {"threadId": params["threadId"], "turnId": turn_id}
                client.queue_notification("item/completed", **scope, item={
                    "type": "userMessage", "id": f"user-{turn_id}", "content": params["input"],
                })
                client.queue_notification("item/completed", **scope, item={
                    "type": "agentMessage", "id": f"answer-{turn_id}", "text": "Acknowledged.",
                })
                turn = {"id": turn_id, "items": [], "status": "completed", "error": None}
                client.queue_notification("turn/completed", threadId=params["threadId"], turn=turn)
                return {"turn": {**turn, "status": "inProgress"}}
            if method == "thread/inject_items":
                for item in params["items"]:
                    if item.get("role") == "developer":
                        threads[params["threadId"]] = "\n".join(part["text"] for part in item["content"])
                return {}
            if method == "thread/compact/start":
                thread_id, turn_id = params["threadId"], f"compact-{number}"
                threads[thread_id] = canonical_instructions[thread_id]
                client.queue_notification("turn/started", threadId=thread_id, turn={"id": turn_id})
                client.queue_notification("turn/completed", threadId=thread_id, turn={
                    "id": turn_id, "status": "completed", "error": None,
                })
                return {}
            raise AssertionError(f"Unexpected Codex RPC: {method}")

        client._request_handler = request
        return client

    monkeypatch.setattr(
        "agent.transports.codex_app_server_session.CodexAppServerClient", client_factory,
    )
    with ExitStack() as resources:
        def make_agent(db, session_id="hermes-session", *, api_mode="codex_app_server", model="gpt-5.4", **kwargs):
            agent = AIAgent(
                api_key="test-key", base_url="https://stub.invalid", provider="openai",
                api_mode=api_mode, model=model, enabled_toolsets=[],
                quiet_mode=True, skip_context_files=True, skip_memory=True,
                save_trajectories=False, session_db=db, session_id=session_id,
                **kwargs,
            )
            setattr(agent, "_end_session_on_close", False)
            resources.callback(agent.close)
            return agent

        yield make_agent, clients, failures


def test_codex_runtime_selected_settings_reach_each_turn(codex_runtime, tmp_path):
    make_agent, clients, _ = codex_runtime
    with closing(SessionDB(tmp_path / "state.db")) as db:
        agent = make_agent(db, reasoning_config={"enabled": True, "effort": "high"}, service_tier="fast")
        first = agent.run_conversation("Explain the release checklist.")
        agent.reasoning_config = {"enabled": True, "effort": "low"}
        agent.service_tier = "default"
        second = agent.run_conversation("Summarize the next step.", conversation_history=first["messages"])

        assert first["completed"] and second["completed"]
        starts = [params for method, params in clients[0].requests if method == "thread/start"]
        turns = [params for method, params in clients[0].requests if method == "turn/start"]
        assert len(starts) == 1
        assert starts[0]["model"] == agent.model
        assert [(p["model"], p["effort"], p["serviceTier"]) for p in turns] == [
            (agent.model, "high", "fast"), (agent.model, "low", None),
        ]
        assert turns[0]["threadId"] == turns[1]["threadId"]
        assert all("sandbox" not in p and "approvalPolicy" not in p for p in starts)
        assert all("sandboxPolicy" not in p and "approvalPolicy" not in p for p in turns)
        agent.close()


def test_codex_runtime_instructions_and_turn_context_reach_codex(codex_runtime, tmp_path, monkeypatch):
    class Memory(MemoryProvider):
        name = "test-memory"

        def is_available(self):
            return True

        def initialize(self, session_id, **kwargs):
            pass

        def get_tool_schemas(self):
            return []

        def prefetch(self, query, *, session_id=""):
            return f"RECALL: {query}"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda hook, **kw: (
        [{"context": f"PLUGIN: {kw['user_message']}"}] if hook == "pre_llm_call" else []
    ))
    make_agent, clients, _ = codex_runtime
    with closing(SessionDB(tmp_path / "state.db")) as db:
        agent = make_agent(db, ephemeral_system_prompt="EPHEMERAL_POLICY")
        agent._memory_manager = MemoryManager()
        agent._memory_manager.add_provider(Memory())
        first = agent.run_conversation("Inspect the first document.", system_message="STATIC_POLICY")
        second = agent.run_conversation("Inspect the second document.", conversation_history=first["messages"])

        assert first["completed"] and second["completed"]
        starts = [params for method, params in clients[0].requests if method == "thread/start"]
        turns = [params for method, params in clients[0].requests if method == "turn/start"]
        assert len(starts) == 1
        instructions = starts[0]["developerInstructions"]
        assert "STATIC_POLICY" in instructions and "EPHEMERAL_POLICY" in instructions
        assert "RECALL:" not in instructions and "PLUGIN:" not in instructions
        for params, text in zip(turns, ["Inspect the first document.", "Inspect the second document."]):
            sent = "\n".join(part["text"] for part in params["input"] if part["type"] == "text")
            assert f"RECALL: {text}" in sent and f"PLUGIN: {text}" in sent
        assert "Inspect the first document." not in turns[1]["input"][0]["text"]
        users = [row["content"] for row in db.get_messages_as_conversation(agent.session_id) if row["role"] == "user"]
        assert users == ["Inspect the first document.", "Inspect the second document."]
        agent.close()


def test_codex_runtime_reconstructs_native_thread_from_reopened_db(codex_runtime, tmp_path):
    make_agent, clients, _ = codex_runtime
    db_path = tmp_path / "state.db"
    with closing(SessionDB(db_path)) as db:
        first_agent = make_agent(db)
        first = first_agent.run_conversation("Remember the marker: copper-orbit.")
        assert first["completed"]
        first_agent.close()
    with closing(SessionDB(db_path)) as db:
        resumed = make_agent(db)
        history = db.get_messages_as_conversation(resumed.session_id)
        second = resumed.run_conversation("What was the marker?", conversation_history=history)
        separate = make_agent(db, session_id="hermes-new-session")
        fresh = separate.run_conversation("Start a separate conversation.")

        assert second["completed"] and fresh["completed"]
        assert second["codex_thread_id"] == first["codex_thread_id"]
        assert fresh["codex_thread_id"] != first["codex_thread_id"]
        assert clients[1].requests[0][0] == "thread/resume"
        assert clients[1].requests[0][1]["threadId"] == first["codex_thread_id"]
        assert all(method not in {"thread/start", "thread/inject_items"} for method, _ in clients[1].requests)
        assert clients[2].requests[0][0] == "thread/start"
        users = [row["content"] for row in db.get_messages_as_conversation(resumed.session_id) if row["role"] == "user"]
        assert users == ["Remember the marker: copper-orbit.", "What was the marker?"]
        resumed.close()
        separate.close()


def test_codex_runtime_imports_legacy_history_once(codex_runtime, tmp_path):
    make_agent, clients, _ = codex_runtime
    history = [
        {"role": "user", "content": "The marker is copper-orbit."},
        {"role": "assistant", "content": "I will remember copper-orbit."},
    ]
    with closing(SessionDB(tmp_path / "state.db")) as db:
        agent = make_agent(db)
        first = agent.run_conversation("Repeat the marker.", conversation_history=history)
        second = agent.run_conversation("Repeat it again.", conversation_history=first["messages"])

        assert first["completed"] and second["completed"]
        methods = [method for method, _ in clients[0].requests]
        assert methods == ["thread/start", "thread/inject_items", "turn/start", "turn/start"]
        imported = clients[0].requests[1][1]
        assert imported["threadId"] == first["codex_thread_id"]
        assert [
            (item["type"], item["role"], [(part["type"], part["text"]) for part in item["content"]])
            for item in imported["items"]
        ] == [
            ("message", "user", [("input_text", history[0]["content"])]),
            ("message", "assistant", [("output_text", history[1]["content"])]),
        ]
        assert "Repeat the marker." not in json.dumps(imported["items"])
        agent.close()


@pytest.mark.parametrize("failed_method", ["thread/resume", "thread/inject_items"])
def test_codex_runtime_context_restore_failure_stops_turn(codex_runtime, tmp_path, failed_method):
    make_agent, clients, failures = codex_runtime
    with closing(SessionDB(tmp_path / "state.db")) as db:
        if failed_method == "thread/resume":
            first_agent = make_agent(db)
            first = first_agent.run_conversation("Remember copper-orbit.")
            assert first["completed"]
            history = db.get_messages_as_conversation(first_agent.session_id)
            first_agent.close()
        else:
            history = [{"role": "user", "content": "Remember copper-orbit."}]
        failures[failed_method] = "Stored context is unavailable"
        agent = make_agent(db)
        result = agent.run_conversation("Recall the earlier marker.", conversation_history=history)

        assert result["completed"] is False
        assert "Stored context is unavailable" in result["error"]
        methods = [method for method, _ in clients[-1].requests]
        assert methods[-1] == failed_method
        assert "turn/start" not in methods
        if failed_method == "thread/resume":
            assert "thread/start" not in methods
        agent.close()


def test_codex_runtime_rebinds_after_reset(codex_runtime, tmp_path):
    make_agent, clients, _ = codex_runtime
    with closing(SessionDB(tmp_path / "state.db")) as db:
        agent = make_agent(db)
        original_session_id = agent.session_id
        first = agent.run_conversation("Remember the private marker: copper-orbit.")
        db.create_session(session_id="new-hermes-session", source="cli", model=agent.model)

        agent.session_id = "new-hermes-session"
        agent.reset_session_state(previous_messages=first["messages"], old_session_id=original_session_id)
        fresh = agent.run_conversation("Begin an unrelated conversation.")
        agent.session_id = original_session_id
        agent.reset_session_state(previous_messages=fresh["messages"], old_session_id="new-hermes-session")
        resumed = agent.run_conversation(
            "Recall my private marker.", conversation_history=db.get_messages_as_conversation(original_session_id),
        )

        assert first["completed"] and fresh["completed"] and resumed["completed"]
        assert fresh["codex_thread_id"] != first["codex_thread_id"]
        assert resumed["codex_thread_id"] == first["codex_thread_id"]
        resumes = [params for client in clients for method, params in client.requests if method == "thread/resume"]
        assert any(params["threadId"] == first["codex_thread_id"] for params in resumes)
        agent.close()


@pytest.mark.parametrize("reconstruct", [False, True], ids=["warm", "reconstructed"])
def test_codex_runtime_clearing_settings_resets_native_overrides(codex_runtime, tmp_path, reconstruct):
    make_agent, clients, _ = codex_runtime
    with closing(SessionDB(tmp_path / "state.db")) as db:
        agent = make_agent(db, reasoning_config={"enabled": True, "effort": "high"}, service_tier="priority")
        first = agent.run_conversation("Explain the release checklist.")
        if reconstruct:
            agent.close()
            agent = make_agent(db, reasoning_config={"enabled": False}, service_tier=None)
        else:
            agent.reasoning_config = {"enabled": False}
            agent.service_tier = None
        second = agent.run_conversation(
            "Summarize the next step.", conversation_history=db.get_messages_as_conversation(agent.session_id),
        )

        assert first["completed"] and second["completed"]
        turns = [params for client in clients for method, params in client.requests if method == "turn/start"]
        assert turns[0]["serviceTier"] == "fast" and turns[0]["effort"] == "high"
        assert turns[1]["effort"] == "none"
        assert turns[1]["serviceTier"] is None
        assert second["codex_thread_id"] == first["codex_thread_id"]
        agent.close()


def test_codex_runtime_round_trip_imports_intervening_hermes_turn(codex_runtime, tmp_path, monkeypatch):
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

    model_requests = []

    def create(_client, **kwargs):
        model_requests.append(kwargs)
        common = {"id": "hermes-response", "created": 0, "model": kwargs["model"]}
        reply = "Hermes recorded the intermediate marker: bronze-signal."
        if kwargs.get("stream"):
            return iter([
                ChatCompletionChunk.model_validate({**common, "object": "chat.completion.chunk", "choices": [{
                    "index": 0, "delta": {"role": "assistant", "content": reply}, "finish_reason": None,
                }]}),
                ChatCompletionChunk.model_validate({**common, "object": "chat.completion.chunk", "choices": [{
                    "index": 0, "delta": {}, "finish_reason": "stop",
                }]}),
            ])
        return ChatCompletion.model_validate({**common, "object": "chat.completion", "choices": [{
            "index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop",
        }]})

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", create)
    make_agent, clients, _ = codex_runtime
    with closing(SessionDB(tmp_path / "state.db")) as db:
        first_agent = make_agent(db)
        first = first_agent.run_conversation("Remember the first marker: copper-orbit.")
        assert first["completed"]
        first_agent.close()

        normal = make_agent(db, api_mode="chat_completions", model="gpt-4.1")
        middle = normal.run_conversation(
            "The intermediate marker is bronze-signal.",
            conversation_history=db.get_messages_as_conversation(normal.session_id),
        )
        assert middle["completed"]
        assert middle["final_response"] == "Hermes recorded the intermediate marker: bronze-signal."
        assert len(model_requests) == 1
        assert "copper-orbit" in json.dumps(model_requests[0]["messages"])
        normal.close()

        resumed = make_agent(db)
        last = resumed.run_conversation(
            "Recall both markers.", conversation_history=db.get_messages_as_conversation(resumed.session_id),
        )
        assert last["completed"]
        imported = [params["items"] for method, params in clients[-1].requests if method == "thread/inject_items"]
        assert imported, "The native thread never received the intervening Hermes turn"
        assert "The intermediate marker is bronze-signal." in json.dumps(imported)
        assert middle["final_response"] in json.dumps(imported)
        resumed.close()


def test_codex_runtime_warm_turn_does_not_reencode_imported_history(codex_runtime, tmp_path, monkeypatch):
    from agent import conversation_loop
    from agent.turn_context import build_api_messages

    make_agent, clients, _ = codex_runtime
    with closing(SessionDB(tmp_path / "state.db")) as db:
        agent = make_agent(db)
        history = [
            {"role": role, "content": f"Archived {role} message {index}."}
            for index in range(128) for role in ("user", "assistant")
        ]
        first = agent.run_conversation("Continue the imported conversation.", conversation_history=history)
        assert first["completed"]
        cloned_messages = []
        real_clone = conversation_loop._clone_message_for_send

        def counted_clone(value):
            if isinstance(value, dict) and value.get("role") in {"user", "assistant", "tool"}:
                cloned_messages.append(value.get("content"))
            return real_clone(value)

        monkeypatch.setattr(conversation_loop, "_clone_message_for_send", counted_clone)
        calibration = {"role": "user", "content": "Calibration input."}
        build_api_messages(
            agent, [calibration], current_turn_user_idx=0, ext_prefetch_cache="", plugin_user_context="",
            moa_config=None, active_system_prompt=agent._cached_system_prompt,
        )
        assert cloned_messages == [calibration["content"]]
        cloned_messages.clear()

        last = agent.run_conversation("Process only this new input.", conversation_history=first["messages"])
        assert last["completed"]
        assert len(cloned_messages) <= 1, f"Re-encoded {len(cloned_messages)} messages for one new input"
        assert clients[0].requests[-1][1]["input"] == [{"type": "text", "text": "Process only this new input."}]
        agent.close()


@pytest.mark.parametrize("reconstruct", [False, True], ids=["warm", "reconstructed"])
def test_codex_runtime_changing_ephemeral_instructions_preserves_native_thread(codex_runtime, tmp_path, reconstruct):
    make_agent, clients, _ = codex_runtime
    with closing(SessionDB(tmp_path / "state.db")) as db:
        agent = make_agent(db, ephemeral_system_prompt="EPHEMERAL_ALPHA")
        first = agent.run_conversation("Start the review.", system_message="SESSION_POLICY")
        history = first["messages"]
        results = [first]
        for instructions in ("EPHEMERAL_BETA", None, None):
            if reconstruct:
                agent.close()
                agent = make_agent(db)
            agent.ephemeral_system_prompt = instructions
            result = agent.run_conversation("Continue the review.", conversation_history=history)
            results.append(result)
            history = result["messages"]
            if instructions == "EPHEMERAL_BETA":
                compacted = agent._codex_session.compact_thread()
                assert compacted.error is None and not compacted.interrupted
                result = agent.run_conversation("Continue after compaction.", conversation_history=history)
                results.append(result)
                history = result["messages"]

        assert all(result["completed"] for result in results)
        assert {result["codex_thread_id"] for result in results} == {first["codex_thread_id"]}
        seen = [instructions for client in clients for instructions in client.instructions_seen]
        assert "EPHEMERAL_ALPHA" in seen[0]
        assert all("EPHEMERAL_BETA" in instructions and "EPHEMERAL_ALPHA" not in instructions for instructions in seen[1:3])
        assert all("EPHEMERAL_" not in instructions for instructions in seen[3:])
        assert all(agent._cached_system_prompt in instructions for instructions in seen)
        updates = [params for client in clients for method, params in client.requests if method == "thread/inject_items"]
        assert len(updates) == 2, "Unchanged instructions must not grow the native history"
        assert all("replace" in params["items"][0]["content"][0]["text"].lower() for params in updates)
        agent.close()


def test_codex_runtime_instruction_update_failure_stops_turn(codex_runtime, tmp_path):
    make_agent, clients, failures = codex_runtime
    with closing(SessionDB(tmp_path / "state.db")) as db:
        agent = make_agent(db, ephemeral_system_prompt="EPHEMERAL_ALPHA")
        first = agent.run_conversation("Start the review.")
        assert first["completed"]
        failures["thread/inject_items"] = "Could not update instructions"
        agent.ephemeral_system_prompt = "EPHEMERAL_BETA"
        failed = agent.run_conversation("Continue the review.", conversation_history=first["messages"])

        assert failed["completed"] is False
        assert "Could not update instructions" in failed["error"]
        assert sum(method == "turn/start" for client in clients for method, _ in client.requests) == 1
        agent.close()


def test_codex_runtime_hard_stop_survives_instruction_restart(codex_runtime, tmp_path, monkeypatch):
    make_agent, clients, _ = codex_runtime
    with closing(SessionDB(tmp_path / "state.db")) as db:
        agent = make_agent(db, ephemeral_system_prompt="EPHEMERAL_ALPHA")
        first = agent.run_conversation("Start the review.")
        assert first["completed"]
        close = clients[0].close

        def stop_during_close():
            agent.hard_interrupt()
            close()

        monkeypatch.setattr(clients[0], "close", stop_during_close)
        agent.ephemeral_system_prompt = "EPHEMERAL_BETA"
        stopped = agent.run_conversation("Continue the review.", conversation_history=first["messages"])

        assert stopped["interrupted"] and not stopped["completed"]
        assert sum(method == "turn/start" for client in clients for method, _ in client.requests) == 1
        agent.close()


@pytest.mark.parametrize(("mode", "window", "expected"), [
    ("auto", 60, ["fast", "fast"]),
    ("cold", 60, ["fast", None]),
    ("auto", 0, [None, None]),
])
def test_codex_runtime_resolves_bounded_fast_mode(codex_runtime, tmp_path, mode, window, expected):
    make_agent, clients, _ = codex_runtime
    with closing(SessionDB(tmp_path / "state.db")) as db:
        agent = make_agent(db, service_tier=mode)
        agent.base_url = "https://api.openai.com/v1"
        agent.fast_auto_seconds = window
        first = agent.run_conversation("Start the review.")
        second = agent.run_conversation("Continue the review.", conversation_history=first["messages"])

        assert first["completed"] and second["completed"]
        tiers = [params["serviceTier"] for client in clients for method, params in client.requests if method == "turn/start"]
        assert tiers == expected
        agent.close()
