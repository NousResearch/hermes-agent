"""Completed native images must survive refusal without poisoning model history."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from hermes_state import SessionDB
from hermes_state_rewind import retain_refused_turn_for_display
from agent.turn_failure_copy import CONTENT_REFUSED_METADATA_KEY, content_policy_retention_notice
from tests.agent.native_responses_image_support import image_item, media_paths, response, text_item

pytest_plugins = ("tests.agent.native_responses_image_support",)


class PolicyBlocked(RuntimeError):
    status_code = 400


def refusal(*items):
    raw = response(*items, status="incomplete", response_id="refused")
    raw.incomplete_details = NS(reason="content_filter")
    return raw


@pytest.mark.parametrize("native", [False, True], ids=["custom", "native"])
@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "held"])
@pytest.mark.parametrize("wire", ["response", "exception"])
@pytest.mark.parametrize("fallback", [False, True], ids=["no-fallback", "fallback"])
def test_prior_image_refusal_returns_only_clean_history(make_agent, monkeypatch, tmp_path, native, held, wire, fallback):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = make_agent(native=native)
    seen, requests, activations = [], [], []
    if fallback:
        def activate(**kwargs):
            activations.append(kwargs)
            return len(activations) == 1

        monkeypatch.setattr(agent, "_has_pending_fallback", lambda: not activations)
        monkeypatch.setattr(agent, "_try_activate_fallback", activate)

    def transform(**kwargs):
        seen.append(kwargs["response_text"])

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kw: (
        [transform(**kw)] if name == "transform_llm_output" else []
    ))
    monkeypatch.setattr("agent.conversation_loop.held_stream_transforms", lambda: (transform,) if held else ())
    raw = iter([response(text_item("Accepted reply."), response_id="accepted"),
                response(image_item(), status="incomplete", response_id="partial"),
                refusal(text_item("REFUSAL_EXPLANATION_MARKER")),
                response(text_item("Benign follow-up."), response_id="followup")])

    def send(kwargs, **callbacks):
        requests.append(deepcopy(kwargs))
        item = next(raw)
        if wire == "exception" and getattr(item, "id", None) == "refused":
            raise PolicyBlocked("400 content_filter: REFUSAL_EXPLANATION_MARKER")
        return item

    monkeypatch.setattr(agent, "_interruptible_api_call", send)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", send)
    agent._session_db = SessionDB(tmp_path / "refusal.db")
    agent._ensure_db_session()
    try:
        accepted = agent.run_conversation("Earlier accepted ask.")
        clean = deepcopy(accepted["messages"])
        seen.clear()
        result = agent.run_conversation("REFUSED_REQUEST_MARKER", conversation_history=clean)
        assert result["failed"] and not result["completed"]
        assert result["failure_reason"] == "content_policy_blocked"
        assert result["failure_retryable"] is False
        assert len(media_paths(result["final_response"])) == 1
        assert all(p.is_file() for p in media_paths(result["final_response"]))
        assert len(seen) == 1 and "MEDIA:" in seen[0]
        assert result["messages"] == clean, "Refused final must not enter CLI/ACP model history"
        assert len(requests) == 3
        assert not activations, "Completed image output must survive a refusal without a paid fallback retry"
        assert retain_refused_turn_for_display(
            agent._session_db, agent.session_id, notice=content_policy_retention_notice(result),
            boundary_row_id=result.get("refused_turn_boundary_row_id"),
        )
        model_rows = agent._session_db.get_messages_as_conversation(agent.session_id)
        assert [(r["role"], r["content"]) for r in model_rows] == [(r["role"], r["content"]) for r in clean]
        display = agent._session_db.get_messages_as_conversation(agent.session_id, include_compacted=True)
        assert display[-1]["display_metadata"][CONTENT_REFUSED_METADATA_KEY]["notice"] == result["final_response"]
        agent.run_conversation("A different benign query.", conversation_history=result["messages"])
        followup = json.dumps(requests[-1])
        assert "REFUSED_REQUEST_MARKER" not in followup
        assert "REFUSAL_EXPLANATION_MARKER" not in followup
        assert "MEDIA:" not in followup
    finally:
        agent._session_db.close()


@pytest.mark.parametrize("native", [False, True], ids=["custom", "native"])
@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "held"])
@pytest.mark.parametrize("timing", ["current", "prior"])
@pytest.mark.parametrize("artifacts", ["valid", "invalid", "mixed"])
def test_completed_image_refusal_never_restarts_fallback(
    make_agent, monkeypatch, native, held, timing, artifacts,
):
    agent = make_agent(native=native)
    requests, activations = [], []
    monkeypatch.setattr("agent.conversation_loop.held_stream_transforms", lambda: ())
    if held:
        monkeypatch.setattr("agent.conversation_loop.held_stream_transforms", lambda: (lambda **kw: None,))
    images = [] if artifacts == "invalid" else [image_item()]
    if artifacts != "valid":
        images.append(image_item("invalid", result="invalid base64"))
    blocked = refusal(text_item("REFUSAL_EXPLANATION_MARKER"), *(images if timing == "current" else []))
    raw = iter(([response(*images, status="incomplete", response_id="partial")] if timing == "prior" else [])
               + [blocked, response(text_item("Regenerated by fallback."), response_id="fallback")])

    def activate(**kwargs):
        activations.append(kwargs)
        return len(activations) == 1

    monkeypatch.setattr(agent, "_has_pending_fallback", lambda: not activations)
    monkeypatch.setattr(agent, "_try_activate_fallback", activate)

    def send(kwargs, **callbacks):
        requests.append(deepcopy(kwargs))
        return next(raw)

    monkeypatch.setattr(agent, "_interruptible_api_call", send)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", send)
    result = agent.run_conversation("REFUSED_REQUEST_MARKER")
    assert len(agent._responses_image_artifacts) == len(images), "Refusal carrier must be materialized before exit"
    assert result["failure_reason"] == "content_policy_blocked"
    assert result["failed"] and not result["completed"]
    assert result["failure_retryable"] is False
    assert not result["messages"], "Only pre-refusal history may leave the loop"
    assert not activations, "A completed native image must not be regenerated by fallback"
    assert len(requests) == (1 if timing == "current" else 2)
    paths = media_paths(result["final_response"])
    assert len(paths) == (0 if artifacts == "invalid" else 1)
    assert all(path.is_file() for path in paths)
    if artifacts != "valid":
        assert "Image output error:" in result["final_response"]
    assert "Regenerated by fallback." not in result["final_response"]


@pytest.mark.parametrize("wire", ["response", "exception"])
def test_no_image_refusal_still_uses_configured_fallback(make_agent, monkeypatch, wire):
    agent = make_agent(native=True)
    requests, activations = [], []
    raw = iter([refusal(text_item("Synthetic refusal.")), response(text_item("Fallback answered."))])

    def activate(**kwargs):
        activations.append(kwargs)
        return len(activations) == 1

    def send(kwargs, **callbacks):
        requests.append(deepcopy(kwargs))
        item = next(raw)
        if wire == "exception" and getattr(item, "id", None) == "refused":
            raise PolicyBlocked("400 content_filter: synthetic no-image refusal")
        return item

    monkeypatch.setattr(agent, "_has_pending_fallback", lambda: not activations)
    monkeypatch.setattr(agent, "_try_activate_fallback", activate)
    monkeypatch.setattr(agent, "_interruptible_api_call", send)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", send)
    result = agent.run_conversation("An unrelated no-image request.")
    assert result["completed"] and not result.get("failed")
    assert result["final_response"] == "Fallback answered."
    assert len(requests) == 2 and len(activations) == 1
