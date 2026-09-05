import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.bot_chain import (
    BOT_CHAIN_USAGE,
    BOT_CHAIN_CONVERSATION_PREFIX,
    BotChainCancelled,
    BotChainControl,
    BotRuntimeUnavailable,
    BotChainRunner,
    BotChainSyntaxError,
    FallbackBotTurnExecutor,
    HermesProfileTurnExecutor,
    HermesSessionRpcTurnExecutor,
    BotTurnError,
    _SessionRPCClient,
    default_bot_turn_executor,
    format_bot_chain_result,
    parse_bot_chain_message,
    publish_bot_chain_history,
    recover_durable_step_output,
)
from hermes_cli.bot_profiles import BotProfile
from hermes_state import SessionDB


def _profile(name: str) -> BotProfile:
    return BotProfile(
        name=name,
        path=Path("/tmp") / name,
        model=f"model-{name}",
        provider="test",
        system_prompt=f"You are {name}",
    )


def _temp_profile(tmp_path: Path, name: str) -> BotProfile:
    profile_home = tmp_path / ".hermes" / "profiles" / name
    profile_home.mkdir(parents=True, exist_ok=True)
    return BotProfile(
        name=name,
        path=profile_home,
        model=f"model-{name}",
        provider="test",
        system_prompt=f"You are {name}",
    )


def _persist_and_publish_turn(
    profile: BotProfile,
    conversation_name: str,
    prompt: str,
    output: str,
    *,
    session_id: str,
) -> None:
    db = SessionDB(Path(profile.path) / "state.db")
    try:
        db.create_session(
            session_id,
            source="cli",
            model=profile.model,
            model_config={"follow_profile_config": True},
            profile_name=profile.name,
        )
        db.set_session_title(session_id, conversation_name)
        db.set_session_hidden(session_id, True)
        db.append_messages_batch(
            session_id,
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output},
            ],
        )
    finally:
        db.close()
    publish_bot_chain_history(profile, conversation_name)


def _event(session_id: str, payload: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "type": "message.complete",
            "session_id": session_id,
            "payload": payload,
        },
    }


def _fake_rpc_server(prompt_handler):
    """Behavioral fake around the real transport binding contract."""
    from tui_gateway.transport import current_transport

    proof = object()
    calls = []
    state = {
        "transport": None,
        "closed": 0,
        "interrupted": 0,
        "running": False,
        "status_sequence": [],
    }

    def create(rid, params):
        calls.append(("session.create", dict(params)))
        state["transport"] = current_transport()
        return {"jsonrpc": "2.0", "id": rid, "result": {"session_id": "rpc-1"}}

    def submit(rid, params):
        calls.append(("prompt.submit", dict(params)))
        assert current_transport() is state["transport"]
        return prompt_handler(rid, params, state)

    def interrupt(rid, params):
        calls.append(("session.interrupt", dict(params)))
        state["interrupted"] += 1
        return {"jsonrpc": "2.0", "id": rid, "result": {"status": "interrupted"}}

    def status(rid, params):
        calls.append(("session.status", dict(params)))
        sequence = state["status_sequence"]
        running = sequence.pop(0) if sequence else state["running"]
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {"running": running, "turn_settled": not running},
        }

    def compress(rid, params):
        calls.append(("session.compress", dict(params)))
        result = state.get("compression_result") or {"status": "compressed"}
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def close(rid, params):
        calls.append(("session.close", dict(params)))
        state["closed"] += 1
        return {"jsonrpc": "2.0", "id": rid, "result": {"closed": True}}

    server = SimpleNamespace(
        _IN_PROCESS_SINGLE_QUERY_PROOF=proof,
        _methods={
            "session.create": create,
            "session.compress": compress,
            "prompt.submit": submit,
            "session.status": status,
            "session.interrupt": interrupt,
            "session.close": close,
        },
    )
    return server, calls, state, proof


def test_parse_leading_bot_chain_and_prompt():
    request = parse_bot_chain_message(
        "  $Writer $Reviewer complete the task\nwith all relevant details"
    )

    assert request is not None
    assert request.names == ("Writer", "Reviewer")
    assert request.prompt == "complete the task\nwith all relevant details"


def test_parse_returns_none_for_ordinary_chat():
    assert parse_bot_chain_message("Explain $PATH") is None


@pytest.mark.parametrize("message", ["$", "$DeepSeek", "$DeepSeek   ", "$bad,name task"])
def test_parse_invalid_or_empty_chain_shows_usage(message):
    with pytest.raises(BotChainSyntaxError, match="Usage") as exc:
        parse_bot_chain_message(message)
    assert str(exc.value) == BOT_CHAIN_USAGE


def test_runner_preserves_order_and_hands_output_to_next_bot():
    calls = []

    def execute(profile, prompt, control, *, conversation_name):
        calls.append((profile.name, prompt, conversation_name))
        return f"output-{profile.name}"

    profiles = [_profile("first"), _profile("second"), _profile("third")]
    result = BotChainRunner(turn_executor=execute).run(profiles, "original task")

    assert [name for name, _prompt, _conversation in calls] == [
        "first",
        "second",
        "third",
    ]
    assert calls[0][1] == "original task"
    assert "Original user request:\noriginal task" in calls[1][1]
    assert "Previous bot ($first) output:\noutput-first" in calls[1][1]
    assert "Previous bot ($second) output:\noutput-second" in calls[2][1]
    assert len({conversation for _name, _prompt, conversation in calls}) == 1
    assert calls[0][2].startswith(BOT_CHAIN_CONVERSATION_PREFIX)
    assert calls[0][2] != "Bot Chat"
    assert result.final_output == "output-third"
    assert format_bot_chain_result(result).endswith("$third (final):\noutput-third")


def test_runner_stops_before_downstream_bot_after_failure():
    calls = []

    def execute(profile, prompt, control, *, conversation_name):
        calls.append(profile.name)
        if profile.name == "second":
            raise BotTurnError(profile.name, "provider unavailable", reason="server")
        return f"output-{profile.name}"

    profiles = [_profile("first"), _profile("second"), _profile("third")]

    with pytest.raises(BotTurnError, match=r"\$second failed"):
        BotChainRunner(turn_executor=execute).run(profiles, "task")

    assert calls == ["first", "second"]


def test_runner_honors_preexisting_cancellation():
    control = BotChainControl()
    control.interrupt("stop")

    with pytest.raises(BotChainCancelled, match="stopped"):
        BotChainRunner(turn_executor=lambda *_args, **_kwargs: "unused").run(
            [_profile("first")], "task", control=control
        )


@pytest.mark.parametrize("revoke", ["profile", "roster"])
def test_live_policy_stops_next_step_after_resolution(tmp_path, revoke):
    first = _temp_profile(tmp_path, "first")
    second = _temp_profile(tmp_path, "second")
    home = first.path.parent.parent
    control = BotChainControl()
    control.source_home = home
    calls = []
    def execute(profile, *args, **kwargs):
        calls.append(profile.name)
        if revoke == "profile":
            (second.path / "profile.yaml").write_text("bot:\n  enabled: false\n", encoding="utf-8")
        else:
            (home / "config.yaml").write_text("agent:\n  bot_mode:\n    roster: []\n", encoding="utf-8")
        return "first result"
    with pytest.raises(BotTurnError):
        BotChainRunner(execute).run([first, second], "task", control=control)
    assert calls == ["first"]


def test_parallel_chain_runs_receive_distinct_conversations():
    conversations = []
    conversations_lock = threading.Lock()
    both_running = threading.Barrier(2)

    def execute(profile, prompt, control, *, conversation_name):
        with conversations_lock:
            conversations.append(conversation_name)
        both_running.wait(timeout=5)
        return f"output-{profile.name}"

    runner = BotChainRunner(turn_executor=execute)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda prompt: runner.run([_profile("worker")], prompt),
                ("first task", "second task"),
            )
        )

    assert [result.final_output for result in results] == [
        "output-worker",
        "output-worker",
    ]
    assert len(conversations) == 2
    assert len(set(conversations)) == 2
    assert all(
        conversation.startswith(BOT_CHAIN_CONVERSATION_PREFIX)
        for conversation in conversations
    )


def test_rpc_executor_uses_warm_session_transport_and_closes_runtime(monkeypatch):
    def submit(rid, params, state):
        state["transport"].write(
            _event(params["session_id"], {"status": "complete", "text": "warm result"})
        )
        return {"jsonrpc": "2.0", "id": rid, "result": {"status": "streaming"}}

    server, calls, state, proof = _fake_rpc_server(submit)
    executor = HermesSessionRpcTurnExecutor(_SessionRPCClient(server))
    monkeypatch.setattr(
        "tools.bot_relay.acquire_turn_lock",
        lambda *_args, **_kwargs: pytest.fail("RPC turns must not take the profile lock"),
    )

    output = executor(
        _profile("worker"),
        "do it",
        BotChainControl(),
        conversation_name="Bot Chain exact",
    )

    assert output == "warm result"
    assert [method for method, _params in calls] == [
        "session.create",
        "prompt.submit",
        "session.status",
        "session.close",
    ]
    create_params = calls[0][1]
    assert create_params["profile"] == "worker"
    assert create_params["title"] == "Bot Chain exact"
    assert create_params["hidden"] is True
    assert create_params["follow_profile_config"] is True
    assert create_params["_single_query_proof"] is proof
    assert state["closed"] == 1


def test_rpc_executor_retries_transient_error_in_same_runtime_session():
    attempts = []

    def submit(rid, params, state):
        attempts.append(params["session_id"])
        payload = (
            {
                "status": "error",
                "text": "",
                "error": "Error code: 429 - rate limit exceeded",
                "error_surface": {"code": "rate_limit"},
            }
            if len(attempts) == 1
            else {"status": "complete", "text": "recovered"}
        )
        state["transport"].write(_event(params["session_id"], payload))
        return {"jsonrpc": "2.0", "id": rid, "result": {"status": "streaming"}}

    server, _calls, _state, _proof = _fake_rpc_server(submit)
    executor = HermesSessionRpcTurnExecutor(_SessionRPCClient(server))

    assert executor(
        _profile("worker"),
        "do it",
        BotChainControl(),
        conversation_name="Bot Chain retry",
    ) == "recovered"
    assert attempts == ["rpc-1", "rpc-1"]


def test_rpc_executor_compresses_context_before_same_session_retry():
    attempts = []

    def submit(rid, params, state):
        attempts.append(params["session_id"])
        payload = (
            {
                "status": "error",
                "error": "maximum context length exceeded",
                "error_surface": {"code": "context_overflow"},
            }
            if len(attempts) == 1
            else {"status": "complete", "text": "compressed result"}
        )
        state["transport"].write(_event(params["session_id"], payload))
        return {"jsonrpc": "2.0", "id": rid, "result": {"status": "streaming"}}

    server, calls, _state, _proof = _fake_rpc_server(submit)
    executor = HermesSessionRpcTurnExecutor(_SessionRPCClient(server))

    assert executor(
        _profile("worker"),
        "do it",
        BotChainControl(),
        conversation_name="Bot Chain context",
    ) == "compressed result"
    assert attempts == ["rpc-1", "rpc-1"]
    methods = [method for method, _params in calls]
    assert methods.index("session.compress") < methods.index("prompt.submit", 2)


def test_rpc_executor_does_not_retry_unchanged_context_after_aborted_compression():
    attempts = 0

    def submit(rid, params, state):
        nonlocal attempts
        attempts += 1
        state["compression_result"] = {
            "status": "aborted",
            "message": "not enough history to compress",
        }
        state["transport"].write(
            _event(
                params["session_id"],
                {
                    "status": "error",
                    "error": "maximum context length exceeded",
                    "error_surface": {"code": "context_overflow"},
                },
            )
        )
        return {"jsonrpc": "2.0", "id": rid, "result": {"status": "streaming"}}

    server, calls, _state, _proof = _fake_rpc_server(submit)
    executor = HermesSessionRpcTurnExecutor(_SessionRPCClient(server))

    with pytest.raises(BotTurnError) as exc:
        executor(
            _profile("worker"),
            "do it",
            BotChainControl(),
            conversation_name="Bot Chain context-abort",
        )

    assert exc.value.reason == "context_overflow"
    assert attempts == 1
    assert [method for method, _params in calls].count("session.compress") == 1


def test_rpc_executor_waits_for_terminal_turn_to_settle_before_close():
    def submit(rid, params, state):
        state["status_sequence"][:] = [True, False]
        state["transport"].write(
            _event(params["session_id"], {"status": "complete", "text": "settled"})
        )
        return {"jsonrpc": "2.0", "id": rid, "result": {"status": "streaming"}}

    server, calls, _state, _proof = _fake_rpc_server(submit)
    executor = HermesSessionRpcTurnExecutor(_SessionRPCClient(server))

    assert executor(
        _profile("worker"),
        "do it",
        BotChainControl(),
        conversation_name="Bot Chain settle",
    ) == "settled"
    assert [method for method, _params in calls].count("session.status") == 2
    assert [method for method, _params in calls][-1] == "session.close"


def test_rpc_session_owner_refusal_is_typed_and_never_replayed():
    def submit(rid, _params, _state):
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "error": {
                "code": 4090,
                "message": "Session already has a live owner",
                "data": {"reason": "SESSION_NOT_OWNED"},
            },
        }

    server, calls, state, _proof = _fake_rpc_server(submit)
    executor = HermesSessionRpcTurnExecutor(_SessionRPCClient(server))

    with pytest.raises(BotTurnError) as exc:
        executor(
            _profile("worker"),
            "do it",
            BotChainControl(),
            conversation_name="Bot Chain busy",
        )

    assert exc.value.reason == "session_busy"
    assert [method for method, _params in calls].count("prompt.submit") == 1
    assert state["closed"] == 1


def test_rpc_executor_interrupts_and_closes_cancelled_runtime():
    started = threading.Event()

    def submit(rid, _params, _state):
        started.set()
        return {"jsonrpc": "2.0", "id": rid, "result": {"status": "streaming"}}

    server, _calls, state, _proof = _fake_rpc_server(submit)
    executor = HermesSessionRpcTurnExecutor(_SessionRPCClient(server))
    control = BotChainControl()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            executor,
            _profile("worker"),
            "do it",
            control,
            conversation_name="Bot Chain cancel",
        )
        assert started.wait(timeout=2)
        control.interrupt("stop")
        with pytest.raises(BotChainCancelled):
            future.result(timeout=3)

    assert state["interrupted"] == 1
    assert state["closed"] == 1


def test_fallback_runs_only_for_pre_admission_runtime_unavailable():
    calls = []

    def primary(*_args, **_kwargs):
        calls.append("primary")
        raise BotRuntimeUnavailable("RPC missing")

    def fallback(*_args, **_kwargs):
        calls.append("fallback")
        return "legacy result"

    executor = FallbackBotTurnExecutor(primary, fallback)
    assert executor(
        _profile("worker"),
        "do it",
        BotChainControl(),
        conversation_name="Bot Chain fallback",
    ) == "legacy result"
    assert calls == ["primary", "fallback"]


def test_fallback_never_replays_an_admitted_turn_failure():
    fallback_called = False

    def primary(*_args, **_kwargs):
        raise BotTurnError("worker", "provider failed")

    def fallback(*_args, **_kwargs):
        nonlocal fallback_called
        fallback_called = True
        return "must not run"

    executor = FallbackBotTurnExecutor(primary, fallback)
    with pytest.raises(BotTurnError):
        executor(
            _profile("worker"),
            "do it",
            BotChainControl(),
            conversation_name="Bot Chain admitted",
        )
    assert fallback_called is False


def test_default_executor_is_process_wide_and_rpc_first():
    first = default_bot_turn_executor()
    second = default_bot_turn_executor()

    assert first is second
    assert isinstance(first, FallbackBotTurnExecutor)
    assert isinstance(first.primary, HermesSessionRpcTurnExecutor)
    assert isinstance(first.fallback, HermesProfileTurnExecutor)
    assert first.history_publisher is publish_bot_chain_history


@pytest.mark.parametrize("canonical_exists", [False, True])
def test_runner_recovers_completed_exact_turn_without_reexecution_or_republish(
    tmp_path, canonical_exists
):
    """The same chain identity is an idempotency key for model and history."""
    profile = _temp_profile(tmp_path, "worker")
    if canonical_exists:
        db = SessionDB(Path(profile.path) / "state.db")
        try:
            db.create_session(
                "canonical",
                source="desktop",
                model=profile.model,
                profile_name=profile.name,
            )
            db.set_session_title("canonical", "Bot Chat")
            db.set_session_hidden("canonical", True)
        finally:
            db.close()

    calls = []
    conversation_name = "Bot Chain recover-completed"

    def execute(profile, prompt, control, *, conversation_name):
        calls.append((profile.name, prompt, conversation_name))
        if len(calls) > 1:
            raise AssertionError("completed durable turn executed twice")
        _persist_and_publish_turn(
            profile,
            conversation_name,
            prompt,
            "durable answer",
            session_id="completed-turn",
        )
        return "durable answer"

    runner = BotChainRunner(turn_executor=execute)
    first = runner.run(
        [profile],
        "do the task",
        conversation_name=conversation_name,
    )
    db = SessionDB(Path(profile.path) / "state.db")
    try:
        canonical = db.get_session_by_title("Bot Chat")
        assert canonical is not None
        db.append_messages_batch(
            canonical["id"],
            [
                {"role": "user", "content": "unrelated later turn"},
                {"role": "assistant", "content": "unrelated later answer"},
            ],
        )
    finally:
        db.close()
    recovered = runner.run(
        [profile],
        "do the task",
        conversation_name=conversation_name,
    )

    assert first.final_output == "durable answer"
    assert recovered.final_output == "durable answer"
    assert len(calls) == 1
    db = SessionDB(Path(profile.path) / "state.db")
    try:
        canonical = db.get_session_by_title("Bot Chat")
        assert canonical is not None
        assert [
            (message["role"], message["content"])
            for message in db.get_messages_as_conversation(canonical["id"])
        ] == [
            ("user", "do the task"),
            ("assistant", "durable answer"),
            ("user", "unrelated later turn"),
            ("assistant", "unrelated later answer"),
        ]
    finally:
        db.close()


@pytest.mark.parametrize("canonical_exists", [False, True])
def test_runner_executes_fresh_chain_when_prompt_repeats_older_bot_chat_turn(
    tmp_path, canonical_exists
):
    """Chain identity, never prompt text, is the idempotency key.

    Bot Chat already holds an older chain's user/assistant pair for the
    exact same prompt. A fresh chain with a new conversation_name must
    execute its own model turn instead of recovering the stale answer,
    while a redelivery of the original identity still recovers without
    re-executing.
    """
    profile = _temp_profile(tmp_path, "worker")
    if canonical_exists:
        db = SessionDB(Path(profile.path) / "state.db")
        try:
            db.create_session(
                "canonical",
                source="desktop",
                model=profile.model,
                profile_name=profile.name,
            )
            db.set_session_title("canonical", "Bot Chat")
            db.set_session_hidden("canonical", True)
        finally:
            db.close()

    calls = []

    def execute(profile, prompt, control, *, conversation_name):
        calls.append(conversation_name)
        output = f"answer from {conversation_name}"
        _persist_and_publish_turn(
            profile,
            conversation_name,
            prompt,
            output,
            session_id=f"turn-{len(calls)}",
        )
        return output

    runner = BotChainRunner(turn_executor=execute)
    old = runner.run([profile], "do the task", conversation_name="Bot Chain old")
    fresh = runner.run([profile], "do the task", conversation_name="Bot Chain new")

    assert old.final_output == "answer from Bot Chain old"
    assert fresh.final_output == "answer from Bot Chain new"
    assert calls == ["Bot Chain old", "Bot Chain new"]

    redelivered = runner.run(
        [profile], "do the task", conversation_name="Bot Chain old"
    )
    assert redelivered.final_output == "answer from Bot Chain old"
    assert calls == ["Bot Chain old", "Bot Chain new"]

    db = SessionDB(Path(profile.path) / "state.db")
    try:
        canonical = db.get_session_by_title("Bot Chat")
        assert canonical is not None
        assert [
            (
                message["role"],
                message["content"],
                (message.get("display_metadata") or {})
                .get("bot_chain", {})
                .get("chain"),
            )
            for message in db.get_messages_as_conversation(canonical["id"])
        ] == [
            ("user", "do the task", "Bot Chain old"),
            ("assistant", "answer from Bot Chain old", "Bot Chain old"),
            ("user", "do the task", "Bot Chain new"),
            ("assistant", "answer from Bot Chain new", "Bot Chain new"),
        ]
    finally:
        db.close()


def test_unreadable_durable_step_never_replays_model(tmp_path, monkeypatch):
    from agent.bot_chain import BotChainRecoveryUnavailable

    profile = _temp_profile(tmp_path, "worker")
    with_db = SessionDB(profile.path / "state.db")
    with_db.close()
    def unreadable(*args, **kwargs):
        raise OSError("state store unavailable")
    monkeypatch.setattr(SessionDB, "get_session_by_title", unreadable)
    def no_execution(*args, **kwargs):
        pytest.fail("cannot prove missing step, must not execute")
    with pytest.raises(BotChainRecoveryUnavailable):
        BotChainRunner(turn_executor=no_execution).run([profile], "task", conversation_name="Bot Chain old")


def test_runner_resumes_multibot_chain_after_last_durable_step(tmp_path):
    """A crash before the next side effect resumes after, not before, it."""

    class _CrashBeforeEffect(BaseException):
        pass

    writer = _temp_profile(tmp_path, "writer")
    reviewer = _temp_profile(tmp_path, "reviewer")
    conversation_name = "Bot Chain partial-resume"
    attempts = []
    side_effects = []

    def execute(profile, prompt, control, *, conversation_name):
        attempts.append((profile.name, prompt))
        if profile.name == "reviewer" and not any(
            name == "reviewer" for name, _output in side_effects
        ):
            reviewer_attempts = sum(
                name == "reviewer" for name, _prompt in attempts
            )
            if reviewer_attempts == 1:
                raise _CrashBeforeEffect()
        if any(name == profile.name for name, _output in side_effects):
            raise AssertionError(f"${profile.name} side effect executed twice")
        output = "draft" if profile.name == "writer" else "final answer"
        _persist_and_publish_turn(
            profile,
            conversation_name,
            prompt,
            output,
            session_id=f"{profile.name}-turn",
        )
        side_effects.append((profile.name, output))
        return output

    runner = BotChainRunner(turn_executor=execute)
    with pytest.raises(_CrashBeforeEffect):
        runner.run(
            [writer, reviewer],
            "ship it",
            conversation_name=conversation_name,
        )

    result = runner.run(
        [writer, reviewer],
        "ship it",
        conversation_name=conversation_name,
    )

    assert side_effects == [
        ("writer", "draft"),
        ("reviewer", "final answer"),
    ]
    assert [name for name, _prompt in attempts] == [
        "writer",
        "reviewer",
        "reviewer",
    ]
    assert "Previous bot ($writer) output:\ndraft" in attempts[-1][1]
    assert result.final_output == "final answer"
    assert [step.output for step in result.steps] == ["draft", "final answer"]


@pytest.mark.parametrize("existing_canonical", [False, True])
@pytest.mark.parametrize("same_database", [False, True])
def test_history_publication_refuses_reclaimed_owner_before_heartbeat(tmp_path, existing_canonical, same_database):
    from functools import partial
    from hermes_state import SessionDB

    profile_home = tmp_path / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    target = SessionDB(profile_home / "state.db")
    ingress = target if same_database else SessionDB(tmp_path / "state.db")
    try:
        target.create_session("isolated", source="cli")
        target.set_session_title("isolated", "Bot Chain fence")
        target.append_messages_batch("isolated", [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "durable answer"},
        ])
        if existing_canonical:
            target.create_session("canonical", source="desktop")
            target.set_session_title("canonical", "Bot Chat")
        ingress.admit_bot_chain_delivery("gateway", "event", "Bot Chain fence")
        old_token = ingress.mark_bot_chain_delivery_running("gateway", "event")
        control = BotChainControl()
        control.publication_guard = partial(ingress.bot_chain_publication_guard, "gateway", "event", old_token)
        ingress.release_bot_chain_delivery_claim("gateway", "event", old_token)
        new_token = ingress.mark_bot_chain_delivery_running("gateway", "event")
        assert not control.cancel_event.is_set()  # no heartbeat has observed the loss
        profile = BotProfile(name="worker", path=profile_home, model="test", provider="test", system_prompt="Work")
        with pytest.raises(BotChainCancelled):
            publish_bot_chain_history(profile, "Bot Chain fence", control=control)
        assert target.get_session_by_title("Bot Chain fence") is not None
        if existing_canonical:
            assert target.get_messages_as_conversation("canonical") == []
        else:
            assert target.get_session_by_title("Bot Chat") is None
        control = BotChainControl()
        control.publication_guard = partial(ingress.bot_chain_publication_guard, "gateway", "event", new_token)
        canonical_id = publish_bot_chain_history(profile, "Bot Chain fence", control=control)
        assert any(m.get("content") == "durable answer" for m in target.get_messages_as_conversation(canonical_id))
    finally:
        if ingress is not target:
            ingress.close()
        target.close()


def test_history_projection_promotes_first_isolated_turn_to_bot_chat(tmp_path):
    from hermes_state import SessionDB

    profile_dir = tmp_path / "hermes" / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    db = SessionDB(profile_dir / "state.db")
    try:
        db.create_session(
            "chain-first",
            source="cli",
            model="test/model",
            model_config={"follow_profile_config": True},
            profile_name="worker",
        )
        db.set_session_title("chain-first", "Bot Chain first")
        db.set_session_hidden("chain-first", True)
        db.append_messages_batch(
            "chain-first",
            [
                {"role": "user", "content": "first prompt"},
                {"role": "assistant", "content": "first answer"},
            ],
        )
    finally:
        db.close()

    canonical_id = publish_bot_chain_history(
        BotProfile(
            name="worker",
            path=profile_dir,
            model="test/model",
            provider="test",
            system_prompt="Work",
        ),
        "Bot Chain first",
    )

    db = SessionDB(profile_dir / "state.db")
    try:
        canonical = db.get_session_by_title("Bot Chat")
        assert canonical is not None
        assert canonical["id"] == "chain-first"
        assert canonical["hidden"] == 1
        assert canonical_id == "chain-first"
        assert [
            (message["role"], message["content"])
            for message in db.get_messages_as_conversation("chain-first")
        ] == [
            ("user", "first prompt"),
            ("assistant", "first answer"),
        ]
        assert db.get_session_by_title("Bot Chain first") is None
    finally:
        db.close()


def test_history_projection_appends_to_existing_bot_chat_despite_desktop_owner(
    tmp_path,
):
    from hermes_cli.active_sessions import try_acquire_active_session
    from hermes_state import SessionDB

    profile_dir = tmp_path / "hermes" / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    db = SessionDB(profile_dir / "state.db")
    try:
        for session_id, title in (
            ("canonical", "Bot Chat"),
            ("chain-next", "Bot Chain next"),
        ):
            db.create_session(
                session_id,
                source="desktop" if session_id == "canonical" else "cli",
                model="test/model",
                model_config={"follow_profile_config": True},
                profile_name="worker",
            )
            db.set_session_title(session_id, title)
            db.set_session_hidden(session_id, True)
        db.append_messages_batch(
            "canonical",
            [
                {"role": "user", "content": "older prompt"},
                {"role": "assistant", "content": "older answer"},
            ],
        )
        db.append_messages_batch(
            "chain-next",
            [
                {"role": "user", "content": "new prompt"},
                {"role": "assistant", "content": "new answer"},
            ],
        )
    finally:
        db.close()

    desktop_lease, refusal = try_acquire_active_session(
        session_id="canonical",
        surface="desktop",
        config={},
        metadata={"live_session_id": "desktop-owner"},
        registry_home=profile_dir,
    )
    assert desktop_lease is not None and refusal is None
    try:
        canonical_id = publish_bot_chain_history(
            BotProfile(
                name="worker",
                path=profile_dir,
                model="test/model",
                provider="test",
                system_prompt="Work",
            ),
            "Bot Chain next",
        )
    finally:
        desktop_lease.release()

    db = SessionDB(profile_dir / "state.db")
    try:
        assert canonical_id == "canonical"
        assert [
            (message["role"], message["content"])
            for message in db.get_messages_as_conversation("canonical")
        ] == [
            ("user", "older prompt"),
            ("assistant", "older answer"),
            ("user", "new prompt"),
            ("assistant", "new answer"),
        ]
        assert db.get_session_by_title("Bot Chain next")["id"] == "chain-next"
    finally:
        db.close()


@pytest.mark.parametrize("canonical_exists", [False, True])
def test_recovery_finishes_pending_publication_once(tmp_path, canonical_exists):
    from agent.bot_chain import BotChainRecoveryUnavailable

    profile = _temp_profile(tmp_path, "worker")
    db = SessionDB(profile.path / "state.db")
    if canonical_exists:
        db.create_session("canonical", source="desktop")
        db.set_session_title("canonical", "Bot Chat")
    calls = []
    def primary(profile, prompt, control, *, conversation_name):
        calls.append(prompt)
        db.create_session("isolated", source="cli")
        db.set_session_title("isolated", conversation_name)
        db.append_messages_batch("isolated", [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "durable result"},
        ])
        return "durable result"
    def no_fallback(*args, **kwargs):
        pytest.fail("publication failures must not replay inference")
    def unavailable(*args, **kwargs):
        raise OSError("projection temporarily unavailable")
    executor = FallbackBotTurnExecutor(primary, no_fallback, history_publisher=unavailable)
    runner = BotChainRunner(executor)
    try:
        with pytest.raises(BotChainRecoveryUnavailable):
            runner.run([profile], "task", conversation_name="Bot Chain resume-publish")
        executor.history_publisher = publish_bot_chain_history
        for _ in range(2):
            result = runner.run([profile], "task", conversation_name="Bot Chain resume-publish")
            assert result.final_output == "durable result"
        assert calls == ["task"]
        canonical = db.get_session_by_title("Bot Chat")
        assert [m["content"] for m in db.get_messages_as_conversation(canonical["id"])] == ["task", "durable result"]
    finally:
        db.close()


def test_history_projection_failure_does_not_replay_completed_turn():
    from agent.bot_chain import BotChainRecoveryUnavailable
    published = []

    def primary(*_args, **_kwargs):
        return "completed once"

    def fallback(*_args, **_kwargs):
        raise AssertionError("completed primary turn must not be replayed")

    def publisher(*_args, **_kwargs):
        published.append("attempted")
        raise RuntimeError("projection unavailable")

    executor = FallbackBotTurnExecutor(
        primary,
        fallback,
        history_publisher=publisher,
    )

    with pytest.raises(BotChainRecoveryUnavailable, match="publication is pending"):
        executor(
            _profile("worker"),
            "do it",
            BotChainControl(),
            conversation_name="Bot Chain completed",
        )
    assert published == ["attempted"]


def test_live_bot_chat_owner_does_not_block_isolated_chain_turn(
    tmp_path,
    monkeypatch,
):
    from hermes_cli.active_sessions import try_acquire_active_session
    from hermes_cli.main import (
        _create_titled_session,
        _resolve_session_by_name_or_id,
    )

    root = tmp_path / "hermes"
    profile_dir = root / "profiles" / "test1"
    profile_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))

    bot_chat_session_id = _create_titled_session("Bot Chat")
    assert bot_chat_session_id
    desktop_lease, error = try_acquire_active_session(
        session_id=bot_chat_session_id,
        surface="desktop",
        config={},
        metadata={"live_session_id": "desktop-owner"},
    )
    assert desktop_lease is not None and error is None

    attempted_conversations = []

    def fake_run_once(self, argv, control):
        conversation_name = argv[argv.index("-c") + 1]
        attempted_conversations.append(conversation_name)
        session_id = _resolve_session_by_name_or_id(conversation_name)
        if session_id is None and "--create-if-missing" in argv:
            session_id = _create_titled_session(conversation_name)
        assert session_id
        chain_lease, refusal = try_acquire_active_session(
            session_id=session_id,
            surface="cli",
            config={},
            metadata={"live_session_id": "chain-owner"},
        )
        if chain_lease is None:
            return subprocess.CompletedProcess(
                argv,
                1,
                stdout="",
                stderr=str(refusal),
            )
        chain_lease.release()
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="isolated result",
            stderr="",
        )

    monkeypatch.setattr(HermesProfileTurnExecutor, "_run_once", fake_run_once)
    profile = BotProfile(
        name="test1",
        path=profile_dir,
        model="test/model",
        provider="test",
        system_prompt="Work",
    )

    try:
        result = BotChainRunner(turn_executor=HermesProfileTurnExecutor()).run(
            [profile], "task"
        )
    finally:
        desktop_lease.release()

    assert result.final_output == "isolated result"
    assert len(attempted_conversations) == 1
    assert attempted_conversations[0].startswith(BOT_CHAIN_CONVERSATION_PREFIX)
    assert attempted_conversations[0] != "Bot Chat"


def test_local_delivery_default_remains_canonical_bot_chat():
    from tools.bot_relay import local_delivery_command

    argv = local_delivery_command("test1", "/tmp/query.txt")

    assert argv[argv.index("-c") + 1] == "Bot Chat"


@pytest.mark.parametrize("publication", ["pending", "promoted", "appended"])
@pytest.mark.parametrize("rotate", [False, True])
def test_recovery_and_publication_survive_compression(tmp_path, publication, rotate):
    profile = _temp_profile(tmp_path, "worker")
    title = "Bot Chain compressed"
    db = SessionDB(profile.path / "state.db")
    try:
        if publication == "appended":
            db.create_session("canonical", source="desktop")
            db.set_session_title("canonical", "Bot Chat")
        db.create_session("isolated", source="cli")
        db.set_session_title("isolated", title)
        db.append_message("isolated", "user", "task")
        if publication != "pending":
            db.append_message("isolated", "assistant", "durable result")
            session_id = publish_bot_chain_history(profile, title)
        else:
            session_id = "isolated"

        summary = [
            {"role": "user", "content": "compressed context"},
            {"role": "assistant", "content": "summary, not a completed turn", "_compressed_summary": True},
        ]
        assert db.try_acquire_compression_lock(session_id, "test-compressor")
        if rotate:
            db.publish_compression_child(
                parent_session_id=session_id, child_session_id="child", source="cli",
                messages=summary, compression_lock_holder="test-compressor",
            )
            tip = "child"
        else:
            db.archive_and_compact(session_id, summary, lock_holder="test-compressor")
            tip = session_id
        db.release_compression_lock(session_id, "test-compressor")
        if publication == "pending":
            assert recover_durable_step_output(profile, title) is None
            db.append_message(tip, "assistant", "durable result")

        def no_execution(*args, **kwargs):
            pytest.fail("durably completed inference must not be replayed")

        executor = FallbackBotTurnExecutor(no_execution, no_execution, history_publisher=publish_bot_chain_history)
        for _ in range(2):
            assert recover_durable_step_output(profile, title) == "durable result"
            result = BotChainRunner(executor).run([profile], "task", conversation_name=title)
            assert result.final_output == "durable result"
        canonical = db.get_session_by_title("Bot Chat")
        rows = [message for sid in db.get_compression_chain(canonical["id"])
                for message in db.get_messages(sid, include_inactive=True)]
        assert sum(m["content"] == "durable result" for m in rows) == 1
    finally:
        db.close()


@pytest.mark.parametrize("existing_canonical", [False, True])
def test_repeated_profile_has_distinct_recoverable_steps(tmp_path, existing_canonical):
    writer, reviewer = _temp_profile(tmp_path, "writer"), _temp_profile(tmp_path, "reviewer")
    if existing_canonical:
        for profile in (writer, reviewer):
            db = SessionDB(profile.path / "state.db")
            db.create_session("canonical", source="desktop")
            db.set_session_title("canonical", "Bot Chat")
            db.close()
    calls = []
    def execute(profile, prompt, control, *, conversation_name):
        calls.append((profile.name, prompt, conversation_name))
        output = f"answer-{len(calls)}"
        db = SessionDB(profile.path / "state.db")
        try:
            session_id = f"turn-{len(calls)}"
            db.create_session(session_id, source="cli")
            db.set_session_title(session_id, conversation_name)
            db.append_messages_batch(session_id, [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output},
            ])
        finally:
            db.close()
        return output

    executor = FallbackBotTurnExecutor(execute, execute, history_publisher=publish_bot_chain_history)
    runner = BotChainRunner(executor)
    def crash_after_second_step(step, index, total):
        if index == 1:
            raise RuntimeError("simulated restart")
    with pytest.raises(RuntimeError, match="simulated restart"):
        runner.run([writer, reviewer, writer], "revise", conversation_name="Bot Chain repeated",
                   on_step=crash_after_second_step)
    for _ in range(2):
        result = runner.run([writer, reviewer, writer], "revise", conversation_name="Bot Chain repeated")
        assert [s.output for s in result.steps] == ["answer-1", "answer-2", "answer-3"]
    assert [call[0] for call in calls] == ["writer", "reviewer", "writer"]
    assert "answer-2" in calls[2][1]
    assert calls[0][2] != calls[2][2]
    db = SessionDB(writer.path / "state.db")
    try:
        canonical = db.get_session_by_title("Bot Chat")
        answers = [m["content"] for m in db.get_messages(canonical["id"]) if m["role"] == "assistant"]
        assert answers == ["answer-1", "answer-3"]
    finally:
        db.close()


def test_runner_accepts_caller_supplied_conversation_name():
    """The gateway binds the durable admission receipt to the chain identity,
    so the runner must execute under the exact caller-supplied name."""
    calls = []

    def execute(profile, prompt, control, *, conversation_name):
        calls.append(conversation_name)
        return "ok"

    result = BotChainRunner(turn_executor=execute).run(
        [_profile("first")],
        "task",
        conversation_name="Bot Chain deadbeef",
    )

    assert calls == ["Bot Chain deadbeef"]
    assert result.final_output == "ok"


def test_runner_rejects_conversation_name_without_chain_prefix():
    def execute(profile, prompt, control, *, conversation_name):
        raise AssertionError("must not execute")

    with pytest.raises(ValueError, match="prefix"):
        BotChainRunner(turn_executor=execute).run(
            [_profile("first")],
            "task",
            conversation_name="Bot Chat",
        )
