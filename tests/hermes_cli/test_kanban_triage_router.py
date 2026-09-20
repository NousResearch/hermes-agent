"""Tests for kanban_triage_router — the Jev pre-check ahead of the full
`hermes kanban specify` LLM call.

Every test patches agent.auxiliary_client's config/call_llm at their source
module (kanban_triage_router imports them lazily, per-call, same pattern as
kanban_specify.py's _call_aux and tools/approval_smart.py's _smart_approve —
patching the source module works because the import happens at call time,
not at module load).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_triage_router as router


def _task(title="fix a typo in the README", body=None) -> kb.Task:
    return kb.Task(
        id="t_test", title=title, body=body, assignee=None, status="triage",
        priority=0, created_by=None, created_at=0, started_at=None,
        completed_at=None, workspace_kind="scratch", workspace_path=None,
        claim_lock=None, claim_expires=None, tenant=None,
    )


def _fake_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _configured(model="typesafe/jev-latest", max_body_chars=300):
    return patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        lambda task: {"model": model, "max_body_chars": max_body_chars} if task == "triage_router" else {},
    )


def _unconfigured():
    return patch("agent.auxiliary_client._get_auxiliary_task_config", lambda task: {})


# ---------------------------------------------------------------------------
# router_configured / eligible
# ---------------------------------------------------------------------------


def test_router_configured_false_when_model_unset():
    with _unconfigured():
        assert router.router_configured() is False
        assert router.configured_model() is None


def test_router_configured_false_when_config_load_fails():
    # When _get_auxiliary_task_config raises, config load should fail gracefully
    with patch("agent.auxiliary_client._get_auxiliary_task_config", side_effect=RuntimeError("config error")):
        assert router.router_configured() is False
        assert router.configured_model() is None


def test_router_configured_true_when_model_set():
    with _configured(model="typesafe/jev-latest"):
        assert router.router_configured() is True
        assert router.configured_model() == "typesafe/jev-latest"


def test_eligible_true_under_threshold():
    with _configured(max_body_chars=300):
        assert router.eligible(_task(title="short", body="also short")) is True


def test_eligible_false_over_threshold():
    with _configured(max_body_chars=10):
        assert router.eligible(_task(title="a title longer than ten chars")) is False


# ---------------------------------------------------------------------------
# is_trivial — the guardrail surface
# ---------------------------------------------------------------------------


def test_is_trivial_false_when_unconfigured_never_calls_llm():
    mock_llm = MagicMock()
    with _unconfigured(), patch("agent.auxiliary_client.call_llm", mock_llm):
        assert router.is_trivial(_task()) is False
    mock_llm.assert_not_called()


def test_is_trivial_false_when_ineligible_never_calls_llm():
    mock_llm = MagicMock()
    with _configured(max_body_chars=5), patch("agent.auxiliary_client.call_llm", mock_llm):
        assert router.is_trivial(_task(title="a much longer than five char title")) is False
    mock_llm.assert_not_called()


def test_is_trivial_true_on_clean_trivial_answer():
    mock_llm = MagicMock(return_value=_fake_response("TRIVIAL"))
    with _configured(), patch("agent.auxiliary_client.call_llm", mock_llm):
        assert router.is_trivial(_task()) is True


def test_is_trivial_normalizes_whitespace_and_case():
    mock_llm = MagicMock(return_value=_fake_response("  trivial \n"))
    with _configured(), patch("agent.auxiliary_client.call_llm", mock_llm):
        assert router.is_trivial(_task()) is True


@pytest.mark.parametrize("reply", ["NEEDS_SHAPE", "UNSURE", "Trivial.", "Answer: TRIVIAL", "", "garbage"])
def test_is_trivial_false_on_anything_but_clean_trivial(reply):
    mock_llm = MagicMock(return_value=_fake_response(reply))
    with _configured(), patch("agent.auxiliary_client.call_llm", mock_llm):
        assert router.is_trivial(_task()) is False


def test_is_trivial_false_when_call_llm_raises():
    mock_llm = MagicMock(side_effect=RuntimeError("provider down"))
    with _configured(), patch("agent.auxiliary_client.call_llm", mock_llm):
        assert router.is_trivial(_task()) is False


def test_is_trivial_false_when_response_shape_is_unexpected():
    mock_llm = MagicMock(return_value=object())  # no .choices at all
    with _configured(), patch("agent.auxiliary_client.call_llm", mock_llm):
        assert router.is_trivial(_task()) is False


def test_ask_jev_passes_task_name_to_call_llm():
    mock_llm = MagicMock(return_value=_fake_response("TRIVIAL"))
    with _configured(), patch("agent.auxiliary_client.call_llm", mock_llm):
        router.is_trivial(_task())
    assert mock_llm.call_args.kwargs["task"] == "triage_router"


def test_is_trivial_false_when_auxiliary_client_import_fails():
    with _configured():
        # Create a fake module without the required attributes
        # When Python tries to `from agent.auxiliary_client import _get_task_timeout, call_llm`,
        # it will raise ImportError because these attributes don't exist in the fake module
        fake_aux = types.ModuleType('agent.auxiliary_client')

        with patch.dict('sys.modules', {'agent.auxiliary_client': fake_aux}):
            assert router.is_trivial(_task()) is False


# ---------------------------------------------------------------------------
# build_minimal_spec
# ---------------------------------------------------------------------------


def test_build_minimal_spec_includes_goal_and_scope_safety_instruction():
    title, body = router.build_minimal_spec(_task(title="Fix a typo in the README"))
    assert title == "Fix a typo in the README"
    assert "**Goal**" in body
    assert "Fix a typo in the README" in body
    assert "**Approach**" in body
    assert "**Acceptance criteria**" in body
    assert "request-changes" in body  # the off-ramp instruction from the guardrail design


def test_build_minimal_spec_truncates_long_title_to_80_chars():
    long_title = "x" * 200
    title, _ = router.build_minimal_spec(_task(title=long_title))
    assert len(title) <= 80
    assert title.endswith("…")
