"""Final markers need text; explicit exhaustion differs from Azure's soft envelope."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from agent.codex_responses_adapter import _normalize_codex_response


def message(text, phase, status="completed"):
    return NS(type="message", role="assistant", id="msg_" + phase, phase=phase, status=status,
              content=[NS(type="output_text", text=text)])


@pytest.mark.parametrize("phase", ["final_answer", "final"])
@pytest.mark.parametrize("empty", ["", " \n "])
@pytest.mark.parametrize("progress_phase", ["commentary", "analysis"])
def test_empty_final_marker_cannot_promote_aggregate_progress(phase, empty, progress_phase):
    response = NS(status="completed", output=[message("Still checking.", progress_phase), message(empty, phase)], output_text="Still checking.")
    original = deepcopy(response)
    result, finish = _normalize_codex_response(response, issuer_kind="codex_backend")
    assert finish == "incomplete"
    assert result.content == ""
    assert result.codex_message_items[0]["phase"] == progress_phase
    assert response == original


@pytest.mark.parametrize("reason", ["max_output_tokens", "length"])
@pytest.mark.parametrize("details_as_dict", [False, True])
@pytest.mark.parametrize("issuer", [None, "codex_backend"])
def test_explicit_output_exhaustion_is_not_overridden_by_a_completed_message(reason, details_as_dict, issuer):
    details = {"reason": reason} if details_as_dict else NS(reason=reason)
    response = NS(status="incomplete", incomplete_details=details, output=[message("Partial result.", "final_answer")])
    original = deepcopy(response)
    result, finish = _normalize_codex_response(response, issuer_kind=issuer)
    assert finish == "incomplete"
    assert result.content == "Partial result."
    assert response == original


@pytest.mark.parametrize("details", [None, NS(reason=None), NS(reason="unknown_vendor_detail")])
def test_substantive_final_keeps_azure_soft_incomplete_compatibility(details):
    response = NS(status="incomplete", incomplete_details=details, output=[message("The result is 42.", "final_answer")])
    result, finish = _normalize_codex_response(response)
    assert finish == "stop"
    assert result.content == "The result is 42."


def test_nonempty_final_is_authoritative_over_aggregate_commentary():
    response = NS(status="completed", output=[message("Checking.", "commentary"), message("42", "final_answer")], output_text="Checking.42")
    result, finish = _normalize_codex_response(response, issuer_kind="codex_backend")
    assert finish == "stop"
    assert result.content == "42"


def test_explicit_content_filter_still_takes_precedence():
    response = NS(status="incomplete", incomplete_details={"reason": "content_filter"}, output=[message("Visible explanation.", "final_answer")])
    result, finish = _normalize_codex_response(response)
    assert finish == "content_filter"
    assert result.content == "Visible explanation."


def test_incomplete_final_item_is_not_treated_as_complete():
    result, finish = _normalize_codex_response(NS(status="completed", output=[message("Partial", "final_answer", "incomplete")]))
    assert finish == "incomplete"
    assert result.content == "Partial"


def test_completed_tool_call_is_not_discarded_by_output_exhaustion():
    tool = NS(type="function_call", id="fc_1", call_id="call_1", name="terminal", arguments="{}", status="completed")
    result, finish = _normalize_codex_response(NS(status="incomplete", incomplete_details={"reason": "max_output_tokens"}, output=[tool]))
    assert finish == "tool_calls"
    assert result.tool_calls[0].function.name == "terminal"


def test_unphased_output_text_backfill_still_works():
    result, finish = _normalize_codex_response(NS(status="completed", output=[], output_text="Complete response."))
    assert finish == "stop"
    assert result.content == "Complete response."
