"""Authoritative native recovery publication is independent of attachment text equality."""
from copy import deepcopy
from pathlib import Path
import socket
from types import SimpleNamespace as NS

import pytest

from hermes_state import SessionDB
from tests.agent.native_responses_image_support import image_item, media_paths
from tests.agent.native_responses_image_support import make_agent as native_agent_factory, response, text_item
from tests.agent.test_native_image_terminal_settlement import terminal_response


@pytest.fixture
def publication_turn(monkeypatch, tmp_path, request):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **kw: pytest.fail("network disabled"))
    factory = native_agent_factory.__wrapped__(monkeypatch, tmp_path)

    def run(kind="incomplete", *, held=False, mode="strip", artifact=True, delivered=False):
        agent = factory(native=True, tools=kind in {"function", "error_interrupt"})
        agent.max_iterations = 8
        agent._api_max_retries = 1
        agent._auto_recovery_cycles = 0
        monkeypatch.setattr("model_tools.handle_function_call", lambda *a, **kw: '{"ok": true}')
        seen, calls, flushes, interims, deliveries = [], [], [], [], []
        if delivered:
            agent.interim_assistant_callback = lambda text, **kw: deliveries.append(text)

        def rewrite(**kw):
            seen.append(kw["response_text"])
            if mode == "failure":
                raise RuntimeError("offline held transform failure")
            if mode == "replace":
                return "REVIEWED FINAL"
            stripped = "\n".join(line for line in kw["response_text"].splitlines() if not line.startswith("MEDIA:"))
            return stripped + ("\nREVIEWED TERMINAL" if mode == "suffix" else "")

        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kw: (
            [rewrite(**kw)] if name == "transform_llm_output" else []
        ))
        monkeypatch.setattr("agent.conversation_loop.held_stream_transforms", lambda: (rewrite,) if held else ())
        agent._session_db = SessionDB(tmp_path / "publication.db")
        agent._ensure_db_session()
        request.addfinalizer(agent._session_db.close)
        original = agent._persist_session

        def persist(messages, history=None):
            flushes.append((len(seen), deepcopy(messages)))
            value = original(messages, history)
            if not seen:
                interims.append(agent._session_db.get_messages_as_conversation(agent.session_id))
            return value

        def send(kwargs, **kw):
            calls.append(deepcopy(kwargs))
            expected = {"complete": 1, "incomplete": 3, "function": 5}.get(kind, 2)
            assert len(calls) <= expected, "No regeneration or hidden provider retry"
            item = image_item() if artifact else text_item("INCOMPLETE FIRST ROUND")
            if not artifact:
                item.status = "incomplete"
            if kind == "complete":
                return response(text_item("NORMAL FINAL TEXT"), *([item] if artifact else []))
            if kind == "function":
                raw = response(item, NS(type="function_call", id="fc_bad", call_id="call_bad", name="terminal",
                                        status="completed", arguments='{"x":'), status="incomplete")
                raw.incomplete_details = NS(reason="max_output_tokens")
                return raw
            if len(calls) == 1 or kind == "incomplete":
                if kind == "error_interrupt":
                    return response(item, NS(type="function_call", id="fc_ok", call_id="call_ok", name="terminal",
                                             status="completed", arguments="{}"))
                return response(item, status="incomplete")
            return terminal_response(kind, agent)

        monkeypatch.setattr(agent, "_persist_session", persist)
        monkeypatch.setattr(agent, "_interruptible_api_call", send)
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", send)
        result = agent.run_conversation("Produce one offline image.")
        stored = agent._session_db.get_messages_as_conversation(agent.session_id)
        return NS(agent=agent, result=result, stored=stored, seen=seen, calls=calls,
                  flushes=flushes, interims=interims, deliveries=deliveries,
                  artifacts=list(agent._responses_image_artifacts.values()))

    return run


def assert_publication(run):
    final = run.result["final_response"]
    assert run.stored[-1]["content"] == final
    assert run.stored[-1].get("display_kind") != "hidden"
    assert run.result["messages"][-1]["content"] == final
    final_flushes = [(count, rows[-1]) for count, rows in run.flushes if rows[-1].get("content") == final]
    assert final_flushes and all(count == len(run.seen) for count, _ in final_flushes)
    for earlier in run.interims:
        assert run.stored[:len(earlier)] == earlier, "Never rewrite previously durable interim history"
    assert run.agent._cached_system_prompt == "You are helpful."
    assert len(run.artifacts) == 1
    path = Path(run.artifacts[0]["path"])
    assert path.is_file()
    assert list(path.parent.glob("*.png")) == [path], "Repeated image identity must materialize only once"


@pytest.mark.parametrize("kind,calls", [("incomplete", 3), ("function", 5)])
@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "held"])
@pytest.mark.parametrize("mode", ["strip", "suffix"])
def test_recovery_final_survives_remove_only_equality_and_suffix_control(publication_turn, kind, calls, held, mode):
    run = publication_turn(kind, held=held, mode=mode)
    assert len(run.calls) == calls and len(run.seen) == 1
    assert run.result["partial"] is True and not run.result["completed"]
    assert run.result["response_transformed"] is True
    assert run.result["pre_transform_response"] == (None if held else run.seen[0])
    assert not media_paths(run.result["final_response"])
    assert_publication(run)


@pytest.mark.parametrize("mode", ["strip", "suffix", "replace", "failure"])
@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "held"])
def test_provider_interrupt_publishes_stopped_final_with_correct_receipt(publication_turn, held, mode):
    from agent.turn_finalizer import HELD_TRANSFORM_FAILED_REPLY
    run = publication_turn("interrupt", held=held, mode=mode)
    assert len(run.calls) == 2
    assert run.result["interrupted"] and not run.result["completed"]
    assert run.result["pre_transform_response"] is None
    assert len(run.seen) == int(held)
    assert run.result["response_transformed"] is held
    assert bool(media_paths(run.result["final_response"])) is not held
    if held and mode in {"replace", "failure"}:
        assert run.result["final_response"] == (HELD_TRANSFORM_FAILED_REPLY if mode == "failure" else "REVIEWED FINAL")
    assert_publication(run)
    assert run.stored[-1].get("display_metadata", {}).get("interrupted") is True
    first = next(rows[-1] for _, rows in run.flushes if rows[-1].get("content") == run.result["final_response"])
    assert first.get("display_metadata", {}).get("interrupted") is True


@pytest.mark.parametrize("publisher", ["recovery", "terminal"])
@pytest.mark.parametrize("fresh_stopped_tail", [False, True])
def test_transform_receipt_does_not_replace_first_native_publication(monkeypatch, tmp_path, publisher, fresh_stopped_tail):
    from agent.responses_images import preserve_recovery_response_images
    from agent.turn_finalizer import apply_llm_output_transform, settle_held_final
    from tests.agent.native_responses_image_support import intake
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    factory = native_agent_factory.__wrapped__(monkeypatch, tmp_path)
    agent = factory(native=True)
    agent._current_turn_id = "offline-current-turn"
    seen = []

    def rewrite(name, **kw):
        if name != "transform_llm_output":
            return []
        seen.append(kw["response_text"])
        return ["REVIEWED FINAL"]

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", rewrite)
    agent._session_db = SessionDB(tmp_path / "receipt.db")
    agent._ensure_db_session()
    try:
        intake(agent, response(image_item()))
        messages = [{"role": "user", "content": "Produce an image."}]
        agent._hold_stream = fresh_stopped_tail
        final, transformed, _ = apply_llm_output_transform(agent, "Recovery notice.", turn_id=agent._current_turn_id)
        assert transformed and len(seen) == 1 and media_paths(seen[0])
        if fresh_stopped_tail:
            messages.append({"role": "assistant", "content": final})
        finish_reason = "interrupted" if fresh_stopped_tail else "incomplete"
        if publisher == "terminal":
            result = {"final_response": final, "messages": messages, "interrupted": fresh_stopped_tail}
            settle_held_final(agent, result)
            published, receipt = result["final_response"], result.get("response_transformed")
        else:
            published, receipt, _ = preserve_recovery_response_images(agent, messages, final, finish_reason)
        if fresh_stopped_tail:
            assert messages[-1].get("display_metadata", {}).get("interrupted") is True
        agent._persist_session(messages, None)
        stored = agent._session_db.get_messages_as_conversation(agent.session_id)
        assert stored[-1]["role"] == "assistant" and stored[-1]["content"] == published
        assert receipt is True and published == "REVIEWED FINAL" and len(seen) == 1
        assert not media_paths(published)
        before = deepcopy(stored)
        preserve_recovery_response_images(agent, messages, published, "incomplete")
        agent._persist_session(messages, None)
        assert agent._session_db.get_messages_as_conversation(agent.session_id) == before
    finally:
        agent._session_db.close()


@pytest.mark.parametrize("held,mode", [(True, "strip"), (True, "replace"), (False, "strip")])
def test_error_interrupt_exact_notice_is_marked_before_first_final_write(publication_turn, held, mode):
    run = publication_turn("error_interrupt", held=held, mode=mode)
    assert run.result["interrupted"] and not run.result["completed"]
    assert run.result["pre_transform_response"] is None
    assert run.result["response_transformed"] is held
    assert len(run.seen) == int(held) and len(run.calls) == 2
    assert_publication(run)
    first = next(rows[-1] for _, rows in run.flushes if rows[-1].get("content") == run.result["final_response"])
    assert first.get("display_metadata", {}).get("interrupted") is True
    assert run.stored[-1].get("display_metadata", {}).get("interrupted") is True


@pytest.mark.parametrize("kind", ["complete", "400"])
@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "held"])
@pytest.mark.parametrize("mode", ["strip", "suffix"])
def test_normal_and_error_controls_settle_once_without_regeneration(publication_turn, kind, held, mode):
    from agent.turn_finalizer import settle_held_final
    run = publication_turn(kind, held=held, mode=mode)
    assert_publication(run)
    assert len(run.calls) == (1 if kind == "complete" else 2) and len(run.seen) == 1
    assert run.result["response_transformed"] is True
    assert not media_paths(run.result["final_response"])
    assert run.result["pre_transform_response"] == (None if held else run.seen[0])
    if kind == "complete":
        assert run.result["completed"] and not run.result["failed"]
    else:
        assert run.result["failed"] and not run.result["completed"]
        assert run.result["failure_reason"] == "format_error" and run.result["failure_retryable"] is False
        assert "/new" in run.result["final_response"]
    before = deepcopy(run.result)
    settle_held_final(run.agent, run.result)
    run.agent._persist_session(run.result["messages"], None)
    assert run.result == before and len(run.seen) == 1
    assert run.agent._session_db.get_messages_as_conversation(run.agent.session_id) == run.stored


@pytest.mark.parametrize("kind", ["complete", "400", "interrupt"])
@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "held"])
def test_no_image_controls_keep_native_error_and_interrupt_classification(publication_turn, kind, held):
    run = publication_turn(kind, held=held, artifact=False)
    assert not run.artifacts and not media_paths(run.result["final_response"])
    if kind == "complete":
        assert run.result["completed"] and len(run.seen) == 1
    elif kind == "400":
        assert run.result["failure_reason"] == "format_error" and run.result["failure_retryable"] is False
        assert run.result["failed"] and not run.result["completed"]
        assert len(run.seen) == int(held)
    else:
        assert run.result["interrupted"] and not run.result["completed"]
        assert len(run.seen) == int(held)
    if held:
        assert run.result["pre_transform_response"] is None


def test_delivered_image_receipt_is_not_reattached_or_regenerated(publication_turn):
    run = publication_turn("400", delivered=True)
    assert len(run.calls) == 2 and not run.seen
    assert len(run.artifacts) == 1 and run.artifacts[0]["delivered"] is True
    assert len([text for text in run.deliveries if media_paths(text)]) == 1
    assert Path(run.artifacts[0]["path"]).is_file()
    assert not media_paths(run.result["final_response"])
    assert not run.result.get("response_transformed")
    assert run.result["failure_reason"] == "format_error"


def test_remove_only_media_incomplete_publishes_authoritative_final(publication_turn):
    run = publication_turn()
    assert len(run.calls) == 3 and len(run.seen) == 1
    assert media_paths(run.seen[0]) and not media_paths(run.result["final_response"])
    assert run.result["response_transformed"] is True
    assert_publication(run)
