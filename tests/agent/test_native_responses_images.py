"""Offline native Responses image delivery through the real conversation loop."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS


import pytest
from PIL import Image
from run_agent import AIAgent


def png_b64() -> str:
    out = io.BytesIO()
    Image.new("RGB", (1, 1), (25, 50, 75)).save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


def image_item(item_id: str | None = "img_1", **overrides):
    return NS(**{"type": "image_generation_call", "id": item_id, "status": "completed",
                 "result": png_b64(), "output_format": "png", **overrides})


def response(*items, status="completed", response_id: str | None = "resp_1"):
    return NS(id=response_id, status=status, output=list(items), output_text="", model="gpt-5-codex",
              usage=NS(input_tokens=17, output_tokens=3, total_tokens=20))


def text_item(text="Done."):
    return NS(type="message", id="msg_1", status="completed", phase="final_answer",
              role="assistant", content=[NS(type="output_text", text=text)])


@pytest.fixture
def make_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **kw: NS())
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *args, **kw: None)

    def make(native=False, tools=False):
        if tools:
            monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kw: [{
                "type": "function", "function": {"name": "terminal", "description": "Offline fixture",
                "parameters": {"type": "object", "properties": {}}},
            }])
        agent = AIAgent(model="gpt-5-codex", provider="openai-codex" if native else "custom",
                        base_url="https://chatgpt.com/backend-api/codex" if native else "http://localhost:9999/v1",
                        api_mode="codex_responses", api_key="offline-fixture", quiet_mode=True,
                        max_iterations=3, skip_context_files=True, skip_memory=True,
                        save_trajectories=False)
        agent._cached_system_prompt = "You are helpful."
        agent.compression_enabled = False

        agent._cleanup_task_resources = lambda *args: None
        return agent
    return make


def run_turn(agent, monkeypatch, *responses, history=None):
    pending = iter(responses)
    requests = []

    def fake_call(kwargs, **callbacks):
        requests.append(kwargs)
        # A recovery request is a bug here; never allow a real provider fallback.
        return next(pending)

    monkeypatch.setattr(agent, "_interruptible_api_call", fake_call)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", fake_call)
    result = agent.run_conversation("Produce the requested output.", conversation_history=history)
    return result, requests


def media_paths(text):
    return [Path(line.removeprefix("MEDIA:").strip()) for line in text.splitlines() if line.startswith("MEDIA:")]


@pytest.mark.parametrize("native", [False, True])
def test_image_only_turn_delivers_two_files_without_recovery(make_agent, monkeypatch, native):
    agent = make_agent(native=native)
    raw = response(NS(type="reasoning", id="rs_1", encrypted_content="opaque-fixture", summary=[]),
                   image_item("img_1"), image_item("img_2"))
    result, requests = run_turn(agent, monkeypatch, raw)
    paths = media_paths(result["final_response"])
    assert len(paths) == 2
    assert len(set(paths)) == 2  # distinct items with identical bytes are distinct results
    assert result["completed"] is True
    assert len(requests) == 1
    assert not any(m.get("tool_calls") for m in result["messages"])
    for path in paths:
        assert path.parent.name == "images"
        assert path.parent.parent.name == "generated"
        with Image.open(path) as image:
            assert image.size == (1, 1)
    assert png_b64() not in json.dumps(result["messages"])


@pytest.mark.parametrize("native", [False, True])
def test_text_and_images_replay_refs_without_shadowing(make_agent, monkeypatch, native):
    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    agent = make_agent(native=native)
    raw = response(NS(type="reasoning", id="rs_1", encrypted_content="opaque-fixture", summary=[]),
                   text_item("Original text."), image_item())
    result, requests = run_turn(agent, monkeypatch, raw)
    paths = media_paths(result["final_response"])
    assert len(paths) == 1
    assert result["final_response"].startswith("Original text.\nMEDIA:")
    assert result["completed"] and len(requests) == 1
    replay = _chat_messages_to_responses_input(result["messages"])
    assert f"MEDIA:{paths[0]}" in json.dumps(replay)
    assert json.dumps(replay).count("Original text.") == 1
    assert png_b64() not in json.dumps(replay)


def test_images_beside_function_calls_survive_final_text(make_agent, monkeypatch):
    from unittest.mock import Mock
    agent = make_agent(tools=True)
    execute = Mock(return_value='{"success": true}')
    monkeypatch.setattr("model_tools.handle_function_call", execute)
    raw = response(text_item("Preparing."), image_item(),
                   NS(type="function_call", id="fc_1", call_id="call_1", name="terminal",
                      status="completed", arguments="{}"))
    result, requests = run_turn(agent, monkeypatch, raw, response(text_item("Finished."), response_id="resp_2"))
    assert execute.call_count == 1
    assert execute.call_args.args[0] == "terminal"
    assert len(requests) == 2
    assert result["completed"] is True
    paths = media_paths(result["final_response"])
    assert len(paths) == 1
    assert paths[0].is_file()
    assert result["final_response"].startswith("Finished.\nMEDIA:")
    assert str(paths[0]) in json.dumps(requests[1]["input"])
    assert png_b64() not in json.dumps(requests)


def intake(agent, raw):
    from agent.turn_response_intake import normalize_model_response
    return normalize_model_response(
        agent, response=raw, messages=[{"role": "user", "content": "fixture"}], api_messages=[],
        conversation_history=None, api_call_count=1, api_duration=0, api_start_time=0,
        api_request_id="offline", effective_task_id="offline", turn_id="offline",
    )


def test_repeated_intake_deduplicates_response_item_identity(make_agent):
    agent = make_agent()
    raw = response(image_item("img_1"), image_item("img_1"), image_item("img_2"))
    first = intake(agent, raw).assistant_message
    second = intake(agent, raw).assistant_message
    assert first.content == second.content
    paths = media_paths(first.content)
    assert len(paths) == 2
    assert len(list(paths[0].parent.glob("*.png"))) == 2
    assert "result" not in json.dumps(second.provider_data)
    # Consuming an already-materialized normalized response is also idempotent.
    from agent.responses_images import materialize_response_images
    materialize_response_images(agent, first)
    assert first.content == second.content


@pytest.mark.parametrize("item_done", [False, True])
def test_stream_completed_images_materialize_once(make_agent, monkeypatch, item_done):
    from agent.codex_runtime import _consume_codex_event_stream
    item = image_item()
    events = [NS(type="response.output_item.done", item=item, output_index=0)] if item_done else []
    events.append(NS(type="response.completed", response=response(item)))
    raw = _consume_codex_event_stream(events, model="gpt-5-codex")
    assert len(raw.output) == 1
    result, requests = run_turn(make_agent(), monkeypatch, raw)
    assert len(media_paths(result["final_response"])) == 1
    assert len(requests) == 1


@pytest.mark.parametrize("status", ["in_progress", "queued", "incomplete"])
def test_progress_images_are_not_final_artifacts(make_agent, status):
    from agent.transports.codex import ResponsesApiTransport
    raw = response(image_item(status=status))
    normalized = ResponsesApiTransport().normalize_response(raw)
    assert not normalized.responses_image_outputs
    assert not normalized.tool_calls


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_failed_response_images_are_not_materialized(make_agent, status):
    from agent.transports.codex import ResponsesApiTransport
    with pytest.raises(RuntimeError):
        ResponsesApiTransport().normalize_response(response(image_item(), status=status))


def test_genuinely_incomplete_image_retains_continuation_policy(make_agent, monkeypatch):
    agent = make_agent(native=True)
    raw = response(image_item(), status="incomplete")
    result, requests = run_turn(agent, monkeypatch, raw, response(text_item("Completed."), response_id="resp_2"))
    assert len(requests) == 2
    assert result["completed"]
    assert len(media_paths(result["final_response"])) == 1
    assert png_b64() not in json.dumps(requests)


@pytest.mark.parametrize("bad", [
    {"result": None}, {"result": ""}, {"result": "not base64!"},
    {"result": base64.b64encode(b"not an image").decode()},
    {"output_format": "gif"}, {"output_format": "../png"}, {"output_format": "jpeg"},
    {"result": base64.b64encode(b"\x89PNG\r\n\x1a\ntruncated").decode()},
])
def test_invalid_completed_image_returns_explicit_error_without_retry(make_agent, monkeypatch, bad):
    agent = make_agent()
    result, requests = run_turn(agent, monkeypatch, response(image_item(**bad)))
    assert len(requests) == 1
    assert result.get("failed") is True
    assert result["completed"] is False
    assert result["failure_reason"] == "image_artifact_error"
    assert result["failure_retryable"] is False
    assert "image" in result["error"].lower()
    assert not media_paths(result["final_response"])
    assert "(empty)" not in result["final_response"]


def test_partial_image_failure_keeps_successful_attachment(make_agent, monkeypatch):
    agent = make_agent()
    result, requests = run_turn(agent, monkeypatch, response(image_item("ok"), image_item("bad", result=None)))
    assert len(requests) == 1
    assert result.get("partial") is True
    assert result["completed"] is False
    assert len(media_paths(result["final_response"])) == 1
    assert media_paths(result["final_response"])[0].is_file()
    assert "image" in result["error"].lower()


@pytest.mark.parametrize("format,extension", [("PNG", ".png"), ("JPEG", ".jpg"), ("WEBP", ".webp")])
def test_explicit_and_inferred_image_formats(make_agent, format, extension):
    out = io.BytesIO()
    Image.new("RGB", (1, 1)).save(out, format=format)
    encoded = base64.b64encode(out.getvalue()).decode()
    agent = make_agent()
    for index, supplied_format in enumerate([format.lower(), None]):
        verdict = intake(agent, response(image_item(result=encoded, output_format=supplied_format), response_id=f"resp_{index}"))
        path = media_paths(verdict.assistant_message.content)[0]
        assert path.suffix == extension
        with Image.open(path) as image:
            assert image.format == format


def test_oversize_image_rejected_before_decode(make_agent, monkeypatch):
    monkeypatch.setattr("agent.image_gen_provider._MAX_NATIVE_IMAGE_BYTES", 8, raising=False)
    result, requests = run_turn(make_agent(), monkeypatch, response(image_item()))
    assert len(requests) == 1
    assert result.get("failed")
    assert "size" in result["error"].lower()
    assert not media_paths(result["final_response"])


def test_atomic_write_failure_leaves_no_attachment(make_agent, monkeypatch, tmp_path):
    agent = make_agent()

    def fail_replace(*args, **kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr("os.replace", fail_replace)
    result, requests = run_turn(agent, monkeypatch, response(image_item()))
    assert len(requests) == 1
    assert result.get("failed")
    assert not media_paths(result["final_response"])
    assert not list((tmp_path / "cache" / "generated" / "images").glob("*"))


def test_save_reload_resume_does_not_resend_old_images(make_agent, monkeypatch, tmp_path):
    from hermes_state import SessionDB
    first = make_agent(native=True)
    first._session_db = SessionDB(tmp_path / "state.db")
    first._ensure_db_session()
    result, requests = run_turn(first, monkeypatch, response(text_item("Created."), image_item("one"), image_item("two")))
    paths = media_paths(result["final_response"])
    assert len(paths) == 2
    session_id = first.session_id
    first._session_db.close()
    db = SessionDB(tmp_path / "state.db")
    history = db.get_messages_as_conversation(session_id)
    assert all(str(path) in json.dumps(history) and path.is_file() for path in paths)
    assert png_b64() not in json.dumps(db.get_messages(session_id))
    resumed = make_agent(native=True)
    resumed.session_id = session_id
    resumed._session_db = db
    resumed._session_db_created = True
    next_result, next_requests = run_turn(resumed, monkeypatch, response(text_item("Next turn."), response_id="resp_next"), history=history)
    assert next_result["completed"] and len(next_requests) == 1
    assert next_result["final_response"] == "Next turn."
    assert not media_paths(next_result["final_response"])
    assert all(str(path) in json.dumps(next_requests[0]["input"]) for path in paths)
    assert png_b64() not in json.dumps(next_requests)
    db.close()


def test_hook_facing_normalized_output_has_no_image_bytes(make_agent, monkeypatch):
    from dataclasses import asdict
    captured = []
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: name == "post_api_request")
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kw: captured.append(kw) if name == "post_api_request" else None)
    verdict = intake(make_agent(), response(image_item()))
    assert verdict.action == "fallthrough"
    assert len(captured) == 1
    normalized = captured[0]["assistant_message"]
    assert png_b64() not in json.dumps(asdict(normalized))
    assert png_b64() not in repr(normalized)
    assert png_b64() not in json.dumps(captured[0]["response"])
    assert media_paths(normalized.content)


def test_sdk_response_image_only_is_pure_until_intake(make_agent, tmp_path):
    from openai.types.responses import Response
    from agent.transports.codex import ResponsesApiTransport
    sdk = Response.model_validate({
        "id": "resp_sdk", "created_at": 0, "object": "response", "status": "completed",
        "model": "gpt-5-codex", "parallel_tool_calls": True, "tools": [], "tool_choice": "auto",
        "output": [vars(image_item())],
    })
    normalized = ResponsesApiTransport().normalize_response(sdk, issuer_kind="codex_backend")
    assert normalized.finish_reason == "stop"
    assert len(normalized.responses_image_outputs) == 1
    assert not (tmp_path / "cache" / "generated").exists()
    verdict = intake(make_agent(native=True), sdk)
    assert len(media_paths(verdict.assistant_message.content)) == 1


def test_deleted_artifact_reports_error_instead_of_regenerating(make_agent, monkeypatch):
    agent = make_agent()
    raw = response(image_item())
    first = intake(agent, raw)
    path = media_paths(first.assistant_message.content)[0]
    path.unlink()
    second = intake(agent, raw)
    assert not media_paths(second.assistant_message.content)
    assert "missing" in second.assistant_message.content.lower()
    assert not path.exists()


@pytest.mark.parametrize("commentary", [False, True])
def test_mixed_tool_interim_and_final_deliver_each_image_once(make_agent, monkeypatch, commentary):
    from unittest.mock import Mock
    agent = make_agent(tools=True)
    delivered = []
    agent.interim_assistant_callback = lambda text, **kw: delivered.append(text)
    monkeypatch.setattr("model_tools.handle_function_call", Mock(return_value='{"ok": true}'))
    text = text_item("Preparing.")
    if commentary:
        text.phase = "commentary"
    raw = response(text, image_item(), NS(type="function_call", id="fc_1", call_id="call_1", name="terminal",
                                       status="completed", arguments="{}"))
    result, requests = run_turn(agent, monkeypatch, raw, response(text_item("Finished."), response_id="resp_2"))
    paths = [path for content in [*delivered, result["final_response"]] for path in media_paths(content)]
    assert len(paths) == 1
    assert paths[0].is_file()
    assert len(requests) == 2


@pytest.mark.parametrize("status", ["queued", "in_progress"])
def test_nonterminal_response_does_not_deliver_images(make_agent, status):
    normalized = make_agent()._get_transport().normalize_response(response(image_item(), status=status))
    assert not normalized.responses_image_outputs
    assert normalized.finish_reason == "incomplete"


def test_completed_image_with_commentary_is_not_reasoning_only(make_agent, monkeypatch):
    agent = make_agent(native=True)
    commentary = text_item("Drawing the image.")
    commentary.phase = "commentary"
    result, requests = run_turn(agent, monkeypatch, response(commentary, image_item()))
    assert len(requests) == 1
    assert result["completed"]
    assert len(media_paths(result["final_response"])) == 1


def test_truncated_jpeg_is_not_delivered_as_a_valid_image(make_agent, monkeypatch):
    out = io.BytesIO()
    Image.new("RGB", (1, 1)).save(out, format="JPEG")
    encoded = base64.b64encode(out.getvalue()[:-2]).decode()
    result, requests = run_turn(make_agent(), monkeypatch, response(image_item(result=encoded, output_format="jpeg")))
    assert len(requests) == 1
    assert result.get("failed")
    assert not media_paths(result["final_response"])


def test_decoded_image_size_cap_is_enforced(make_agent, monkeypatch):
    raw_size = len(base64.b64decode(png_b64()))
    monkeypatch.setattr("agent.image_gen_provider._MAX_NATIVE_IMAGE_BYTES", raw_size - 1)
    result, requests = run_turn(make_agent(), monkeypatch, response(image_item()))
    assert len(requests) == 1
    assert result.get("failed")
    assert "decoded size" in result["error"].lower()


def test_pixel_size_cap_is_enforced(make_agent, monkeypatch):
    monkeypatch.setattr("agent.image_gen_provider._MAX_NATIVE_IMAGE_PIXELS", 0)
    result, requests = run_turn(make_agent(), monkeypatch, response(image_item()))
    assert len(requests) == 1
    assert result.get("failed")
    assert not media_paths(result["final_response"])


def test_failed_file_write_cleans_temporary_partial_bytes(make_agent, monkeypatch, tmp_path):
    import tempfile
    agent = make_agent()
    real_temporary_file = tempfile.NamedTemporaryFile

    class FailingFile:
        def __init__(self, **kwargs):
            self.handle = real_temporary_file(**kwargs)
            self.name = self.handle.name

        def __enter__(self):
            return self

        def write(self, data):
            self.handle.write(data[:8])
            self.handle.flush()
            raise OSError("simulated partial write failure")

        def __exit__(self, *args):
            self.handle.close()

    monkeypatch.setattr("agent.provider_media.tempfile.NamedTemporaryFile", FailingFile)
    result, requests = run_turn(agent, monkeypatch, response(image_item()))
    assert len(requests) == 1
    assert result.get("failed")
    assert not list((tmp_path / "cache" / "generated" / "images").glob("*"))


def test_image_survives_incomplete_function_call_retry(make_agent, monkeypatch):
    agent = make_agent(native=True, tools=True)
    raw = response(image_item(), NS(type="function_call", id="fc_1", call_id="call_1", name="terminal",
                                   status="completed", arguments="{}"), status="incomplete")
    raw.incomplete_details = NS(reason="max_output_tokens")
    result, requests = run_turn(agent, monkeypatch, raw, response(text_item("Recovered."), response_id="resp_2"))
    assert len(requests) == 2
    assert result["completed"]
    assert len(media_paths(result["final_response"])) == 1


def test_terminal_image_fallback_preserves_provider_item_order():
    from agent.codex_runtime import _consume_codex_event_stream
    one, two = image_item("one"), image_item("two")
    raw = _consume_codex_event_stream([
        NS(type="response.output_item.done", item=two, output_index=1),
        NS(type="response.completed", response=response(one, two)),
    ], model="gpt-5-codex")
    assert [item.id for item in raw.output] == ["one", "two"]


def test_stream_image_without_provider_ids_deduplicates_terminal_copy(make_agent, monkeypatch):
    from agent.codex_runtime import _consume_codex_event_stream
    item = image_item(item_id=None)
    raw = _consume_codex_event_stream([
        NS(type="response.output_item.done", item=item),
        NS(type="response.completed", response=response(item, response_id=None)),
    ], model="gpt-5-codex")
    result, requests = run_turn(make_agent(), monkeypatch, raw)
    assert len(media_paths(result["final_response"])) == 1
    assert len(requests) == 1


@pytest.mark.parametrize("done_positions", [(1,), (2,), (2, 1), (0, 2)])
@pytest.mark.parametrize("same_bytes", [False, True])
def test_idless_terminal_siblings_match_partial_done_occurrences(make_agent, monkeypatch, done_positions, same_bytes):
    from agent.codex_runtime import _consume_codex_event_stream
    siblings = [image_item(None) for _ in range(3)]
    if not same_bytes:
        for index, item in enumerate(siblings):
            out = io.BytesIO()
            Image.new("RGB", (1, 1), (index, 50, 75)).save(out, format="PNG")
            item.result = base64.b64encode(out.getvalue()).decode()
    raw = _consume_codex_event_stream([
        *(NS(type="response.output_item.done", item=NS(**vars(siblings[index]))) for index in done_positions),
        NS(type="response.completed", response=response(*siblings)),
    ], model="gpt-5-codex")
    assert [item.result for item in raw.output] == [item.result for item in siblings]
    result, requests = run_turn(make_agent(), monkeypatch, raw)
    paths = media_paths(result["final_response"])
    assert len(paths) == len(set(paths)) == 3
    assert len(requests) == 1
    assert [base64.b64encode(path.read_bytes()).decode() for path in paths] == [item.result for item in siblings]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("done_positions", [(1,), (1, 0)])
def test_pending_function_settlement_keeps_terminal_image_attachment_order(make_agent, monkeypatch, native, done_positions):
    from unittest.mock import Mock
    from agent.codex_runtime import _consume_codex_event_stream

    siblings = [image_item(None) for _ in range(2)]
    colors = [(10, 50, 75), (20, 50, 75)]
    for item, color in zip(siblings, colors):
        out = io.BytesIO()
        Image.new("RGB", (1, 1), color).save(out, format="PNG")
        item.result = base64.b64encode(out.getvalue()).decode()
    function = NS(type="function_call", id="fc_1", call_id="call_1", name="terminal", arguments="{}")
    commentary = text_item("Preparing.")
    commentary.phase = "commentary"
    raw = _consume_codex_event_stream([
        NS(type="response.output_item.added", item=function),
        NS(type="response.output_item.done", item=commentary),
        *(NS(type="response.output_item.done", item=NS(**vars(siblings[index]))) for index in done_positions),
        NS(type="response.completed", response=response(function, commentary, *siblings)),
    ], model="gpt-5-codex")
    assert [item.type for item in raw.output if item.type != "image_generation_call"] == ["function_call", "message"]
    execute = Mock(return_value='{"ok": true}')
    monkeypatch.setattr("model_tools.handle_function_call", execute)
    result, requests = run_turn(make_agent(native=native, tools=True), monkeypatch, raw,
                                response(text_item("Finished."), response_id="resp_2"))
    assert execute.call_count == 1
    assert len(requests) == 2
    assert result["completed"] is True
    paths = media_paths(result["final_response"])
    assert len(paths) == len(set(paths)) == 2
    actual_colors = []
    for path in paths:
        with Image.open(path) as image:
            actual_colors.append(image.getpixel((0, 0)))
    assert actual_colors == colors
    assert png_b64() not in json.dumps(requests)


@pytest.mark.parametrize("exhaustion", ["function", "text"])
@pytest.mark.parametrize("images", ["valid", "invalid", "mixed"])
def test_incomplete_exhaustion_preserves_native_image_outcome(make_agent, monkeypatch, tmp_path, exhaustion, images):
    from unittest.mock import Mock
    from hermes_state import SessionDB
    agent = make_agent(native=True, tools=exhaustion == "function")
    agent.max_iterations = 10
    agent._session_db = SessionDB(tmp_path / "exhaustion.db")
    agent._ensure_db_session()
    execute = Mock(return_value='{"ok": true}')
    monkeypatch.setattr("model_tools.handle_function_call", execute)
    siblings = []
    if images != "invalid":
        siblings.append(image_item("valid"))
    if images != "valid":
        siblings.append(image_item("invalid", result=None))
    if exhaustion == "function":
        siblings.append(NS(type="function_call", id="fc_1", call_id="call_1", name="terminal",
                           status="completed", arguments="{\"unfinished\":"))
    raw = response(*siblings, status="incomplete")
    raw.incomplete_details = NS(reason="max_output_tokens")
    expected_calls = 5 if exhaustion == "function" else 3
    result, requests = run_turn(agent, monkeypatch, *([raw] * expected_calls))
    assert len(requests) == expected_calls
    assert execute.call_count == 0
    assert result["completed"] is False
    assert result.get("partial") is True
    paths = media_paths(result["final_response"])
    assert len(paths) == (0 if images == "invalid" else 1)
    assert all(path.is_file() for path in paths)
    if images != "valid":
        assert "Image output error:" in result["final_response"]
        assert "Image output error:" in result["error"]
    assert "incomplete" in result["error"].lower() or "truncat" in result["error"].lower()
    stored = agent._session_db.get_messages_as_conversation(agent.session_id)
    terminal = stored[-1]
    assert terminal["role"] == "assistant"
    assert all(str(path) in terminal["content"] for path in paths)
    if images != "valid":
        assert "Image output error:" in terminal["content"]
    assert png_b64() not in json.dumps(stored)
    assert png_b64() not in json.dumps(requests)
    agent._session_db.close()


def test_delivered_native_image_completes_clean_empty_tool_followup(make_agent, monkeypatch):
    from unittest.mock import Mock
    agent = make_agent(native=True, tools=True)
    delivered = []
    agent.interim_assistant_callback = lambda text, **kw: delivered.append(text)
    execute = Mock(return_value='{"ok": true}')
    monkeypatch.setattr("model_tools.handle_function_call", execute)
    raw = response(image_item(), NS(type="function_call", id="fc_1", call_id="call_1", name="terminal",
                                   status="completed", arguments="{}"))
    result, requests = run_turn(agent, monkeypatch, raw, response(text_item(""), response_id="resp_2"))
    assert len(requests) == 2
    assert execute.call_count == 1
    assert result["completed"] is True
    assert result["final_response"] == ""  # no fabricated provider text or reattachment
    paths = [path for text in delivered for path in media_paths(text)]
    assert len(paths) == 1 and paths[0].is_file()
    assert str(paths[0]) in json.dumps(result["messages"])
    assert not any(message.get("_empty_recovery_synthetic") for message in result["messages"])


@pytest.mark.parametrize("exhaustion", ["function", "text"])
@pytest.mark.parametrize("invalid_sibling", [False, True])
def test_delivered_image_is_not_reattached_on_incomplete_exhaustion(make_agent, monkeypatch, tmp_path, exhaustion, invalid_sibling):
    from unittest.mock import Mock
    from hermes_state import SessionDB
    agent = make_agent(native=True, tools=True)
    agent.max_iterations = 10
    agent._session_db = SessionDB(tmp_path / "delivered-exhaustion.db")
    agent._ensure_db_session()
    delivered = []
    agent.interim_assistant_callback = lambda text, **kw: delivered.append(text)
    execute = Mock(return_value='{"ok": true}')
    monkeypatch.setattr("model_tools.handle_function_call", execute)
    first = response(image_item("valid"), *([image_item("invalid", result=None)] if invalid_sibling else []),
                     NS(type="function_call", id="fc_1", call_id="call_1", name="terminal",
                        status="completed", arguments="{}"))
    tail = (NS(type="function_call", id="fc_2", call_id="call_2", name="terminal",
               status="completed", arguments="{\"unfinished\":") if exhaustion == "function" else text_item("Partial."))
    if exhaustion == "text":
        tail.status = "incomplete"
    raw = response(tail, status="incomplete", response_id="resp_2")
    raw.incomplete_details = NS(reason="max_output_tokens")
    continuations = 5 if exhaustion == "function" else 3
    result, requests = run_turn(agent, monkeypatch, first, *([raw] * continuations))
    assert len(requests) == continuations + 1
    assert execute.call_count == 1  # outstanding truncated calls never execute
    assert not result["completed"] and result.get("partial")
    paths = [path for text in delivered for path in media_paths(text)]
    assert len(paths) == 1 and paths[0].is_file()
    assert not media_paths(result["final_response"])
    stored = agent._session_db.get_messages_as_conversation(agent.session_id)
    assert str(paths[0]) in json.dumps(stored)
    if invalid_sibling:
        assert "Image output error:" in result["final_response"]
        assert "Image output error:" in stored[-1]["content"]
    assert png_b64() not in json.dumps(stored)
    agent._session_db.close()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("delivered", [False, True])
@pytest.mark.parametrize("images", ["valid", "invalid", "mixed"])
@pytest.mark.parametrize("summary_succeeds", [False, True])
def test_budget_finalization_preserves_pending_image_outcome(make_agent, monkeypatch, tmp_path, native, delivered, images, summary_succeeds):
    from hermes_state import SessionDB

    agent = make_agent(native=native)
    agent.max_iterations = 1
    agent._session_db = SessionDB(tmp_path / "budget-image.db")
    agent._ensure_db_session()
    callbacks = []
    if delivered:
        agent.interim_assistant_callback = lambda text, **kw: callbacks.append(text)
    siblings = []
    if images != "invalid":
        siblings.append(image_item("valid"))
    if images != "valid":
        siblings.append(image_item("invalid", result=None))
    raw = response(*siblings, status="incomplete")
    raw.incomplete_details = NS(reason="max_output_tokens")
    followup = [response(text_item("Budget summary."), response_id="resp_2")] if summary_succeeds else []
    result, requests = run_turn(agent, monkeypatch, raw, *followup)
    assert len(requests) == 2  # incomplete attempt plus the real budget-summary path
    assert result["completed"] is False
    assert result["failed"] is False  # existing resumable budget boundary, not an image-only success
    assert result["turn_exit_reason"] == "max_iterations_reached(1/1)"
    paths = media_paths(result["final_response"])
    assert len(paths) == (1 if images != "invalid" and not delivered else 0)
    assert all(path.is_file() for path in paths)
    callback_paths = [path for text in callbacks for path in media_paths(text)]
    assert len(callback_paths) == (1 if images != "invalid" and delivered else 0)
    if images != "valid":
        assert "Image output error:" in result["final_response"]
    stored = agent._session_db.get_messages_as_conversation(agent.session_id)
    assert stored[-1]["role"] == "assistant"
    assert stored[-1]["content"] == result["final_response"]
    if summary_succeeds:
        assert result["final_response"].startswith("Budget summary.")
        assert sum("Budget summary." in (message.get("content") or "") for message in stored) == 1
    assert png_b64() not in json.dumps(stored)
    assert png_b64() not in json.dumps(requests)
    agent._session_db.close()


@pytest.mark.parametrize("native", [False, True])
def test_pending_images_survive_interrupted_budget_summary_without_success(make_agent, monkeypatch, tmp_path, native):
    from hermes_state import SessionDB

    agent = make_agent(native=native)
    agent.max_iterations = 1
    agent._session_db = SessionDB(tmp_path / "interrupted-budget-image.db")
    agent._ensure_db_session()
    requests = []
    raw = response(image_item("valid"), image_item("invalid", result=None), status="incomplete")
    raw.incomplete_details = NS(reason="max_output_tokens")

    def fake_call(kwargs, **callbacks):
        requests.append(kwargs)
        if len(requests) == 1:
            return raw
        raise InterruptedError("offline summary interruption")

    monkeypatch.setattr(agent, "_interruptible_api_call", fake_call)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", fake_call)
    result = agent.run_conversation("Produce the requested output.")
    assert len(requests) == 2
    assert result["interrupted"] is True
    assert result["completed"] is False
    assert result["failed"] is False
    assert result["turn_exit_reason"].startswith("interrupted")
    paths = media_paths(result["final_response"])
    assert len(paths) == 1 and paths[0].is_file()
    assert "Image output error:" in result["final_response"]
    stored = agent._session_db.get_messages_as_conversation(agent.session_id)
    assert stored[-1]["role"] == "assistant"
    assert stored[-1]["content"] == result["final_response"]
    assert png_b64() not in json.dumps(stored)
    agent._session_db.close()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("images", ["valid", "invalid", "mixed"])
def test_scratchpad_exhaustion_preserves_image_outcome(make_agent, monkeypatch, tmp_path, native, images):
    from hermes_state import SessionDB

    agent = make_agent(native=native)
    db_path = tmp_path / "scratchpad-exhaustion.db"
    agent._session_db = SessionDB(db_path)
    agent._ensure_db_session()
    siblings = []
    if images != "invalid":
        siblings.append(image_item("valid"))
    if images != "valid":
        siblings.append(image_item("invalid", result=None))
    raw = response(text_item("<REASONING_SCRATCHPAD>Unfinished reasoning."), *siblings)
    result, requests = run_turn(agent, monkeypatch, raw, raw, raw)
    assert len(requests) == result["api_calls"] == 3  # initial attempt plus two bounded retries
    assert agent._incomplete_scratchpad_retries == 0
    assert result["completed"] is False
    assert result.get("partial") is True
    assert not result.get("failed")
    assert result["final_response"].startswith("Incomplete REASONING_SCRATCHPAD after 2 retries")
    assert result["error"] == result["final_response"]
    paths = media_paths(result["final_response"])
    assert len(paths) == (0 if images == "invalid" else 1)
    assert all(path.is_file() for path in paths)
    if images != "valid":
        assert "Image output error:" in result["final_response"]
    terminal = result["messages"][-1]
    assert terminal["role"] == "assistant"
    assert terminal["content"] == result["final_response"]
    assert "Unfinished reasoning." not in json.dumps(result["messages"])
    session_id = agent.session_id
    agent._session_db.close()
    with SessionDB(db_path) as db:
        stored = db.get_messages_as_conversation(session_id)
        assert stored[-1]["role"] == "assistant"
        assert stored[-1]["content"] == result["final_response"]
        assert sum(message.get("content") == result["final_response"] for message in stored) == 1
        assert all(str(path) in json.dumps(stored) for path in paths)
        assert png_b64() not in json.dumps(db.get_messages(session_id))
    assert png_b64() not in json.dumps(result)
    assert png_b64() not in json.dumps(requests)


@pytest.mark.parametrize("native", [False, True])
def test_delivered_image_is_not_reattached_on_scratchpad_exhaustion(make_agent, monkeypatch, tmp_path, native):
    from unittest.mock import Mock
    from hermes_state import SessionDB

    agent = make_agent(native=native, tools=True)
    agent.max_iterations = 4
    db_path = tmp_path / "delivered-scratchpad.db"
    agent._session_db = SessionDB(db_path)
    agent._ensure_db_session()
    delivered = []
    agent.interim_assistant_callback = lambda text, **kw: delivered.append(text)
    execute = Mock(return_value='{"ok": true}')
    monkeypatch.setattr("model_tools.handle_function_call", execute)
    first = response(image_item("delivered"),
                     NS(type="function_call", id="fc_1", call_id="call_1", name="terminal",
                        status="completed", arguments="{}"))
    raw = response(text_item("<REASONING_SCRATCHPAD>Unfinished reasoning."), response_id="resp_2")
    result, requests = run_turn(agent, monkeypatch, first, raw, raw, raw)
    assert len(requests) == result["api_calls"] == 4
    assert execute.call_count == 1
    assert agent._incomplete_scratchpad_retries == 0
    assert result["completed"] is False and result.get("partial") is True
    assert not result.get("failed")
    assert result["final_response"] == result["error"] == "Incomplete REASONING_SCRATCHPAD after 2 retries"
    paths = [path for text in delivered for path in media_paths(text)]
    assert len(paths) == 1 and paths[0].is_file()
    assert not media_paths(result["final_response"])
    assert "Unfinished reasoning." not in json.dumps(result["messages"])
    session_id = agent.session_id
    agent._session_db.close()
    with SessionDB(db_path) as db:
        stored = db.get_messages_as_conversation(session_id)
        assert sum(f"MEDIA:{paths[0]}" in (message.get("content") or "") for message in stored) == 1
        assert "Unfinished reasoning." not in json.dumps(stored)
        assert png_b64() not in json.dumps(db.get_messages(session_id))
    assert png_b64() not in json.dumps(requests)


def test_invalid_image_does_not_block_ordinary_function_execution(make_agent, monkeypatch):
    from unittest.mock import Mock
    agent = make_agent(native=True, tools=True)
    execute = Mock(return_value='{"ok": true}')
    monkeypatch.setattr("model_tools.handle_function_call", execute)
    raw = response(image_item(result=None),
                   NS(type="function_call", id="fc_1", call_id="call_1", name="terminal",
                      status="completed", arguments="{}"))
    result, requests = run_turn(agent, monkeypatch, raw, response(text_item("Finished."), response_id="resp_2"))
    assert execute.call_count == 1
    assert len(requests) == 2
    assert result["failure_reason"] == "image_artifact_error"
    assert result["failure_retryable"] is False
    assert "Finished." in result["final_response"]
    assert not media_paths(result["final_response"])
