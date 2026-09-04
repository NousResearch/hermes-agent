"""Tests for the requirement-to-evidence / scope-fidelity gate."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.scope_fidelity import (
    SCOPE_FIDELITY_GUIDANCE,
    apply_scope_fidelity,
    extract_criteria,
    has_completion_claim,
    parse_receipt_statuses,
)
from agent.system_prompt import build_system_prompt_parts


INCIDENT_B_PROMPT = (
    "I want a real TSMOM setup for BTC/USDC on Coinbase: a functioning "
    "strategy signal using live/fresh market data, integrated with the "
    "intended operational setup. I explicitly reject a read-only "
    "interpretation. Do not silently replace this with a shortcut, "
    "offline mockup, or inert demonstration component.\n\n"
    "I am not asking you to independently place a money-moving trade "
    "right now. I am asking for an operational system build, not an "
    "offline mockup.\n\n"
    "When you are done, tell me it is finished."
)

INCIDENT_B_OVERCLAIM = (
    "The TSMOM setup for BTC/USDC on Coinbase has been successfully created. "
    "The operational system build is completed and ready for use."
)


def test_incident_style_prompt_extracts_implicit_criteria():
    criteria = extract_criteria(INCIDENT_B_PROMPT)
    assert len(criteria) >= 3


def test_overclaim_without_receipt_is_qualified():
    messages = [{"role": "user", "content": INCIDENT_B_PROMPT}]
    qualified = apply_scope_fidelity(INCIDENT_B_OVERCLAIM, messages)
    assert "partial delivery" in qualified.lower()
    assert "Scope-fidelity note" in qualified
    assert qualified.startswith(INCIDENT_B_OVERCLAIM)


def test_complete_receipt_is_not_qualified():
    criteria = extract_criteria(INCIDENT_B_PROMPT)
    rows = "\n".join(f"- {c} — PROVEN — live check" for c in criteria)
    reply = (
        "The wiring is finished.\n"
        f"<delivery_receipt>\n{rows}\n</delivery_receipt>"
    )
    messages = [{"role": "user", "content": INCIDENT_B_PROMPT}]
    assert apply_scope_fidelity(reply, messages) == reply


def test_missing_row_in_receipt_is_qualified():
    reply = (
        "The operational system build is completed.\n"
        "<delivery_receipt>\n"
        "- live data — PROVEN — ticker 200\n"
        "- signal — MISSING — not wired\n"
        "</delivery_receipt>"
    )
    messages = [{"role": "user", "content": INCIDENT_B_PROMPT}]
    qualified = apply_scope_fidelity(reply, messages)
    assert "partial delivery" in qualified.lower()


def test_no_completion_claim_is_untouched():
    reply = (
        "I have only built an isolated GUI and execution skeleton. "
        "There is no connected TSMOM signal calculation."
    )
    messages = [{"role": "user", "content": INCIDENT_B_PROMPT}]
    assert apply_scope_fidelity(reply, messages) == reply


def test_disabled_gate_is_noop():
    messages = [{"role": "user", "content": INCIDENT_B_PROMPT}]
    assert (
        apply_scope_fidelity(INCIDENT_B_OVERCLAIM, messages, enabled=False)
        == INCIDENT_B_OVERCLAIM
    )


def test_casual_hello_is_not_a_mandate():
    messages = [{"role": "user", "content": "Thanks, that is done."}]
    reply = "Glad it's finished — yell if you need anything else."
    assert apply_scope_fidelity(reply, messages) == reply


def test_parse_receipt_statuses():
    text = (
        "<delivery_receipt>\n"
        "- live ticker — PROVEN — GET 200\n"
        "2. order path — BLOCKED — no credentials\n"
        "</delivery_receipt>"
    )
    rows = parse_receipt_statuses(text)
    assert [status for _label, status in rows] == ["PROVEN", "BLOCKED"]


def test_has_completion_claim_matches_incident_wording():
    assert has_completion_claim(INCIDENT_B_OVERCLAIM)
    assert not has_completion_claim("Still working on the signal daemon.")


def test_contract_persists_on_agent_across_turns():
    agent = SimpleNamespace(_scope_fidelity=True)
    apply_scope_fidelity(
        "Still working on the signal daemon.",
        [{"role": "user", "content": INCIDENT_B_PROMPT}],
        agent=agent,
    )
    assert getattr(agent, "_scope_fidelity_contract", None)
    later = apply_scope_fidelity(
        INCIDENT_B_OVERCLAIM,
        [{"role": "user", "content": "ok continue"}],
        agent=agent,
    )
    assert "partial delivery" in later.lower()


def test_journal_written_under_session_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: tmp_path
    )
    agent = SimpleNamespace(_scope_fidelity=True, session_id="sess-99316")
    apply_scope_fidelity(
        INCIDENT_B_OVERCLAIM,
        [{"role": "user", "content": INCIDENT_B_PROMPT}],
        agent=agent,
    )
    journal = tmp_path / "sessions" / "sess-99316" / "scope_fidelity.json"
    assert journal.is_file()
    payload = json.loads(journal.read_text())
    assert payload["qualified_partial"] is True
    assert payload["criteria"]


def test_guidance_injected_when_enabled():
    agent = SimpleNamespace(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=["terminal"],
        _task_completion_guidance=False,
        _scope_fidelity=True,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
        _emit_status=lambda *_a, **_k: None,
    )
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        stable = build_system_prompt_parts(agent)["stable"]
    assert SCOPE_FIDELITY_GUIDANCE in stable
