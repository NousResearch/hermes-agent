"""Native images from earlier rounds settle before the authoritative terminal final write."""
from copy import deepcopy
from pathlib import Path

import pytest

from hermes_state import SessionDB
from tests.agent.native_responses_image_support import image_item, media_paths
from tests.agent.native_responses_image_support import make_agent as native_agent_factory, response


@pytest.fixture
def make_agent(monkeypatch, tmp_path):
    return native_agent_factory.__wrapped__(monkeypatch, tmp_path)


def test_pending_image_terminal_400_settles_before_final_write(make_agent, monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = make_agent(native=True)
    seen, calls, flushes = [], [], []

    def transform(**kw):
        seen.append(kw["response_text"])

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kw: (
        [transform(**kw)] if name == "transform_llm_output" else []
    ))
    monkeypatch.setattr("agent.conversation_loop.held_stream_transforms", lambda: ())
    original = agent._persist_session

    def persist(messages, history=None):
        flushes.append((len(seen), deepcopy(messages)))
        return original(messages, history)

    monkeypatch.setattr(agent, "_persist_session", persist)

    def send(kwargs, **kw):
        calls.append(kwargs)
        if len(calls) == 1:
            return response(image_item(), status="incomplete")
        assert len(calls) == 2, "Terminal 400 must not generate again"
        raise ProviderFailure(400)

    monkeypatch.setattr(agent, "_interruptible_api_call", send)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", send)
    agent._session_db = SessionDB(tmp_path / "state.db")
    agent._ensure_db_session()
    try:
        result = agent.run_conversation("Produce one offline image.")
        stored = agent._session_db.get_messages_as_conversation(agent.session_id)
        assert media_paths(result["final_response"]), "Completed undelivered image missing from terminal final"
        assert len(seen) == 1 and "MEDIA:" in seen[0]
        assert stored[-1]["content"] == result["final_response"]
        assert stored[-1].get("display_kind") != "hidden"
        assert result["failed"] and not result["completed"]
        assert result["failure_reason"] == "format_error"
        assert "/new" in result["final_response"]
        assert result["response_transformed"] is False
        assert result["pre_transform_response"] is None
        assert len(calls) == 2
        assert all(path.is_file() for path in media_paths(result["final_response"]))
        final_flushes = [n for n, rows in flushes if rows[-1].get("content") == result["final_response"]]
        assert final_flushes and all(n == 1 for n in final_flushes)
    finally:
        agent._session_db.close()


class ProviderFailure(RuntimeError):
    def __init__(self, status, message="synthetic provider failure"):
        super().__init__(f"{status} {message}")
        self.status_code = status


def terminal_response(kind, agent):
    """Only the provider/compressor boundary is scripted; error consumers stay real."""
    from types import SimpleNamespace as NS
    from agent.conversation_compression import mark_context_compression_timed_out
    from tests.agent.native_responses_image_support import text_item

    if kind == "complete":
        return response(text_item("Finished."), response_id="resp_2")
    if kind == "invalid":
        return None
    if kind == "refusal":
        raw = response(text_item("Synthetic refusal."), status="incomplete", response_id="resp_2")
        raw.incomplete_details = NS(reason="content_filter")
        return raw
    if kind == "truncated_tool":
        return response(NS(type="function_call", id="fc_bad", call_id="call_bad", name="terminal",
                           status="completed", arguments='{"path":'), response_id="resp_2")
    if kind == "shutdown":
        raise RuntimeError("cannot schedule new futures after interpreter shutdown")
    if kind == "compression_disabled":
        raise ProviderFailure(413, "request payload too large")
    if kind == "error_interrupt":
        agent._interrupt_requested = True
        raise ProviderFailure(400)
    if kind == "timeout":
        raise TimeoutError("synthetic request timed out")
    if kind == "interrupt":
        agent._interrupt_requested = True
        raise InterruptedError("synthetic stop")
    if kind == "missing":
        for artifact in agent._responses_image_artifacts.values():
            Path(artifact["path"]).unlink()
        raise ProviderFailure(400)
    if kind.startswith("overflow"):
        agent.compression_enabled = True
        agent.max_compression_attempts = 0 if kind == "overflow_exhausted" else 3

        def compress(messages, system, **kwargs):
            if kind == "overflow_timeout":
                mark_context_compression_timed_out(agent)
            elif kind == "overflow_lock":
                agent._compression_skipped_due_to_lock = "offline holder"
            elif kind == "overflow_transient":
                agent._compression_blocked_transient = "offline cooldown"
            return messages, agent._cached_system_prompt

        agent._compress_context = compress
        raise ProviderFailure(413 if kind != "overflow_timeout" else 400, (
            "maximum context length is 100 tokens; requested 1000 tokens" if kind == "overflow_timeout"
            else "request payload too large"
        ))
    raise ProviderFailure(int(kind))


@pytest.fixture
def terminal_turn(make_agent, monkeypatch, tmp_path):
    import socket
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **kw: pytest.fail("network disabled"))

    def run(kind="400", *, held=False, transform="noop", artifact="pending"):
        from tests.agent.native_responses_image_support import text_item
        agent = make_agent(native=True, tools=artifact == "tool_pending")
        monkeypatch.setattr("model_tools.handle_function_call", lambda *args, **kw: '{"ok": true}')
        agent.max_iterations = 5
        agent._api_max_retries = 1
        agent._auto_recovery_cycles = 0
        seen, delivered, calls, flushes, interim_stored = [], [], [], [], []
        original_persist = agent._persist_session

        def rewrite(**kw):
            seen.append(kw["response_text"])
            if transform == "failure":
                raise RuntimeError("synthetic transform failure")
            return "REVIEWED FINAL" if transform == "replace" else None

        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kw: (
            [rewrite(**kw)] if name == "transform_llm_output" else []
        ))
        monkeypatch.setattr("agent.conversation_loop.held_stream_transforms", lambda: (rewrite,) if held else ())
        if artifact == "delivered":
            agent.interim_assistant_callback = lambda text, **kw: delivered.append(text)
        if kind == "rate_guard":
            monkeypatch.setattr("agent.nous_rate_guard.nous_rate_limit_remaining", lambda **kw: 60)
            monkeypatch.setattr("hermes_cli.anon_auth.is_anonymous_agent", lambda agent: False)
            monkeypatch.setattr("hermes_cli.anon_auth.apply_model_switch", lambda agent: None)

        def persist(messages, history=None):
            flushes.append((len(seen), deepcopy(messages)))
            value = original_persist(messages, history)
            if not seen:
                interim_stored.append(agent._session_db.get_messages_as_conversation(agent.session_id))
            return value

        monkeypatch.setattr(agent, "_persist_session", persist)

        def send(kwargs, **kw):
            calls.append(deepcopy(kwargs))
            assert len(calls) <= 2, "No image regeneration or hidden provider retry"
            if len(calls) == 1:
                item = text_item("Partial text.") if artifact == "none" else image_item()
                if artifact == "none":
                    item.status = "incomplete"
                if artifact == "invalid":
                    item.result = "not base64!"
                if kind == "rate_guard":
                    agent.provider = "nous"
                if artifact == "tool_pending":
                    from types import SimpleNamespace as NS
                    return response(item, NS(type="function_call", id="fc_ok", call_id="call_ok", name="terminal",
                                             status="completed", arguments="{}"))
                return response(item, status="incomplete")
            return terminal_response(kind, agent)

        monkeypatch.setattr(agent, "_interruptible_api_call", send)
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", send)
        agent._session_db = SessionDB(tmp_path / "matrix.db")
        agent._ensure_db_session()
        try:
            result = agent.run_conversation("Produce one offline image.")
            stored = agent._session_db.get_messages_as_conversation(agent.session_id)
            for earlier in interim_stored:
                assert stored[:len(earlier)] == earlier, "Settlement must not rewrite persisted interim rows"
            assert agent._cached_system_prompt == "You are helpful."
            # An extra flush may not duplicate the new final or rewrite earlier rows.
            snapshot = deepcopy(stored)
            agent._persist_session(result["messages"], None)
            assert agent._session_db.get_messages_as_conversation(agent.session_id) == snapshot
            return NS(agent=agent, result=result, stored=stored, seen=seen, calls=calls,
                      delivered=delivered, flushes=flushes, artifacts=list(agent._responses_image_artifacts.values()))
        finally:
            agent._session_db.close()

    from types import SimpleNamespace as NS
    return run


def assert_terminal_publication(run, calls=2):
    final_flushes = [count for count, rows in run.flushes if rows and rows[-1].get("content") == run.result["final_response"]]
    if run.result.get("failure_reason") == "content_policy_blocked":
        # The owner retains this final as a display-only notice. Neither the clean
        # in-memory model history nor an ordinary durable row may carry the refusal.
        assert run.result["messages"] == []
        assert not final_flushes
        assert all(row.get("content") != run.result["final_response"] for row in run.stored)
    else:
        assert run.stored[-1]["role"] == "assistant"
        assert run.stored[-1]["content"] == run.result["final_response"]
        assert run.stored[-1].get("display_kind") != "hidden"
        assert run.result["messages"][-1]["content"] == run.result["final_response"]
        assert final_flushes and all(count == 1 for count in final_flushes)
    assert len(run.calls) == calls


def test_held_compression_replacement_retains_error_mirror(terminal_turn):
    run = terminal_turn("overflow_exhausted", held=True, transform="replace")
    assert_terminal_publication(run)
    assert run.result["final_response"] == "REVIEWED FINAL"
    assert run.result["error"] == run.result["final_response"]
    assert run.result["failure_reason"] == "context_overflow"
    assert run.result["compression_exhausted"]


@pytest.mark.parametrize("kind", [
    "400", "401", "403", "404", "402", "429", "500", "timeout", "invalid", "refusal",
    "overflow_exhausted", "overflow_timeout", "overflow_lock", "overflow_transient",
])
@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "held"])
def test_terminal_siblings_match_no_image_error_contract(terminal_turn, kind, held):
    control = terminal_turn(kind, artifact="none")
    run = terminal_turn(kind, held=held)
    assert_terminal_publication(run)
    assert len(run.seen) == 1 and "MEDIA:" in run.seen[0]
    assert len(media_paths(run.result["final_response"])) == 1
    assert all(Path(a["path"]).is_file() for a in run.artifacts)
    assert not any(a.get("delivered") for a in run.artifacts)
    assert run.result["response_transformed"] is False
    assert run.result["pre_transform_response"] is None
    # Preserve the provider's classifier, retry verdict, guidance, and compression disposition.
    for key in ("failed", "completed", "failure_reason", "failure_retryable", "compression_exhausted", "compression_deferred"):
        assert run.result.get(key) == control.result.get(key), key
    assert run.result["final_response"].split("\nMEDIA:", 1)[0] == control.result["final_response"]
    assert not control.artifacts and not control.seen
    assert not media_paths(control.result["final_response"])
    assert "response_transformed" not in control.result
    if kind in ("overflow_lock", "overflow_transient"):
        assert run.result.get("compression_deferred") and not run.result.get("failed")
    if kind == "overflow_timeout":
        assert run.result["turn_exit_reason"] == "context_compression_timeout"


@pytest.mark.parametrize("kind,reason", [
    ("400", "format_error"), ("401", "auth"), ("timeout", "timeout"),
    ("refusal", "content_policy_blocked"), ("overflow_exhausted", "context_overflow"),
])
@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "held"])
@pytest.mark.parametrize("transform", ["noop", "replace", "failure"])
def test_terminal_transform_receipt_precedes_native_final_persistence(terminal_turn, kind, reason, held, transform):
    from agent.turn_finalizer import HELD_TRANSFORM_FAILED_REPLY
    run = terminal_turn(kind, held=held, transform=transform)
    assert_terminal_publication(run)
    assert len(run.seen) == 1 and "MEDIA:" in run.seen[0]
    assert len(run.artifacts) == 1 and Path(run.artifacts[0]["path"]).is_file()
    if transform == "replace" or (held and transform == "failure"):
        expected = HELD_TRANSFORM_FAILED_REPLY if transform == "failure" else "REVIEWED FINAL"
        assert run.result["final_response"] == expected
        assert run.result["response_transformed"] is True
        assert not media_paths(run.result["final_response"])
        if held:
            assert run.result["pre_transform_response"] is None
            protected = run.stored if kind == "refusal" else run.stored[:-1]
            assert all(row.get("display_kind") == "hidden" for row in protected if "MEDIA:" in str(row.get("content")))
        else:
            assert run.result["pre_transform_response"] == run.seen[0]
    else:
        assert len(media_paths(run.result["final_response"])) == 1
        assert run.result["response_transformed"] is False
        assert run.result["pre_transform_response"] is None
    receipt = run.agent._llm_output_transform
    assert receipt[0] == run.agent._current_turn_id
    assert receipt[1] == run.result["response_transformed"]
    assert run.result["failed"] and run.result["failure_reason"] == reason


@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "held"])
def test_already_settled_final_is_not_transformed_or_reattached(terminal_turn, held):
    from agent.turn_finalizer import settle_held_final
    run = terminal_turn("complete", held=held, transform="replace")
    assert_terminal_publication(run)
    assert run.result["completed"]
    assert run.result["final_response"] == "REVIEWED FINAL"
    before = deepcopy(run.result)
    settle_held_final(run.agent, run.result)
    assert run.result == before
    assert len(run.seen) == 1 and not media_paths(run.result["final_response"])


def test_delivered_original_receipt_prevents_final_image_duplicate(terminal_turn):
    run = terminal_turn(artifact="delivered")
    assert len(run.calls) == 2
    assert len(run.artifacts) == 1 and run.artifacts[0]["delivered"] is True
    assert len([text for text in run.delivered if "MEDIA:" in text]) == 1
    assert not media_paths(run.result["final_response"])
    assert not run.seen and "response_transformed" not in run.result
    assert Path(run.artifacts[0]["path"]).is_file()


@pytest.mark.parametrize("kind,artifact", [("missing", "pending"), ("400", "invalid")])
def test_missing_or_invalid_artifact_never_fabricates_media(terminal_turn, kind, artifact):
    run = terminal_turn(kind, artifact=artifact)
    assert_terminal_publication(run)
    assert not media_paths(run.result["final_response"])
    assert "Image output error:" in run.result["final_response"]
    assert len(run.seen) == 1 and "Image output error:" in run.seen[0]
    assert all(not a.get("path") for a in run.artifacts)


@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "held"])
def test_interrupt_policy_preserves_image_without_ordinary_transform(terminal_turn, held):
    run = terminal_turn("interrupt", held=held, transform="replace")
    assert run.result["interrupted"] and not run.result["completed"]
    assert len(run.calls) == 2
    assert Path(run.artifacts[0]["path"]).is_file()
    if held:
        assert run.result["final_response"] == "REVIEWED FINAL"
        assert len(run.seen) == 1 and "MEDIA:" in run.seen[0]
        assert run.result["pre_transform_response"] is None
    else:
        assert not run.seen
        assert len(media_paths(run.result["final_response"])) == 1
        assert not run.result["response_transformed"]
    assert run.stored[-1]["content"] == run.result["final_response"]


@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "held"])
def test_rate_guard_direct_return_settles_prior_image_without_sending(terminal_turn, held):
    run = terminal_turn("rate_guard", held=held)
    assert_terminal_publication(run, calls=1)
    assert len(run.seen) == 1 and "MEDIA:" in run.seen[0]
    assert len(media_paths(run.result["final_response"])) == 1
    assert run.result["failure_reason"] == "rate_limit"
    assert run.result["failed"] and not run.result["completed"]
    assert run.result["response_transformed"] is False


@pytest.mark.parametrize("kind", ["truncated_tool", "error_interrupt"])
@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "held"])
def test_terminal_tool_tail_closer_settles_before_its_first_final_flush(terminal_turn, kind, held):
    run = terminal_turn(kind, held=held, transform="replace", artifact="tool_pending")
    assert len(run.calls) == 2
    assert len(run.artifacts) == 1 and Path(run.artifacts[0]["path"]).is_file()
    assert run.stored[-1]["content"] == run.result["final_response"]
    expected_calls = 0 if kind == "error_interrupt" and not held else 1
    assert len(run.seen) == expected_calls
    if kind == "error_interrupt":
        assert run.stored[-1].get("display_metadata", {}).get("interrupted") is True
    final_flushes = []
    for count, rows in run.flushes:
        tool_indices = [i for i, row in enumerate(rows) if row.get("role") == "tool"]
        if tool_indices:
            finals = [row for row in rows[tool_indices[-1] + 1:] if row.get("role") == "assistant" and row.get("content")]
            for row in finals:
                assert row["content"] == run.result["final_response"], "Early closer flushed an unsettled final"
                assert row.get("display_kind") != "hidden"
                assert count == expected_calls
                final_flushes.append(row)
    assert final_flushes
    if expected_calls:
        assert run.result["final_response"] == "REVIEWED FINAL"
        assert run.result["response_transformed"]
    else:
        assert run.result["interrupted"] and media_paths(run.result["final_response"])


@pytest.mark.parametrize("kind", ["shutdown", "compression_disabled"])
@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "held"])
def test_additional_terminal_direct_returns_keep_native_final(terminal_turn, kind, held):
    run = terminal_turn(kind, held=held)
    assert_terminal_publication(run)
    assert len(run.seen) == 1 and "MEDIA:" in run.seen[0]
    assert len(media_paths(run.result["final_response"])) == 1
    assert run.result["failed"] and not run.result["completed"]
    if kind == "shutdown":
        assert run.result["failure_reason"] == "interpreter_shutdown"
    else:
        assert run.result["failure_reason"] == "context_overflow" and run.result["compaction_disabled"]
