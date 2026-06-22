"""Unit tests for the autopilot council gate (the goal-completion judge).

All network is mocked via the ``_council_run`` / ``_aux_call`` seams. One
integration test exercises the REAL Hermes Council offline lane when the package
is present (deterministic, no network).
"""

import os
from pathlib import Path

import pytest

from agent.autopilot import council_gate as cg
from agent.autopilot.council_gate import CompletionVerdict


# --------------------------------------------------------------------------- #
# Helpers / parsing                                                            #
# --------------------------------------------------------------------------- #
def test_extract_json_plain():
    assert cg._extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert cg._extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_inline_prefix():
    assert cg._extract_json('sure, here: {"a": 1} done')["a"] == 1


def test_extract_json_none_on_garbage():
    assert cg._extract_json("not json at all") is None


def test_trunc_caps_length():
    out = cg._trunc("x" * 100, 10)
    assert out.startswith("x" * 10) and "truncated" in out


def test_compose_directive_uses_wrong_point_and_checks():
    arb = {"most_likely_wrong_point": "tests are missing", "required_checks": ["run pytest", "verify Y"]}
    d = cg._compose_directive(arb, [])
    assert "tests are missing" in d and "run pytest" in d


def test_compose_directive_falls_back_to_critic():
    delibs = [{"key_points": ["validate inputs"], "claim": "c"}]
    d = cg._compose_directive({}, delibs)
    assert "validate inputs" in d


# --------------------------------------------------------------------------- #
# judge_completion — council path (mocked seam)                                #
# --------------------------------------------------------------------------- #
def test_judge_completion_council_allow(monkeypatch):
    monkeypatch.setattr(cg, "ensure_council_importable", lambda *a, **k: True)
    monkeypatch.setattr(
        cg, "_council_run",
        lambda q, *, mode, max_tokens: {
            "verdict": "allow", "confidence": 0.82, "arbiter": {}, "deliberations": [],
            "sycophancy": {"overall": 0.0}, "meta": {"panel": "fast"},
        },
    )
    v = cg.judge_completion("goal", "work", "final")
    assert isinstance(v, CompletionVerdict)
    assert v.complete and v.source == "council" and v.verdict == "allow"


def test_judge_completion_council_deny_yields_directive(monkeypatch):
    monkeypatch.setattr(cg, "ensure_council_importable", lambda *a, **k: True)
    monkeypatch.setattr(
        cg, "_council_run",
        lambda q, *, mode, max_tokens: {
            "verdict": "deny", "confidence": 0.6,
            "arbiter": {"most_likely_wrong_point": "no tests", "required_checks": ["write tests"]},
            "deliberations": [], "sycophancy": {"overall": 0.2}, "meta": {"panel": "fast"},
        },
    )
    v = cg.judge_completion("goal", "work", "final")
    assert not v.complete and v.verdict == "deny" and "write tests" in v.directive


def test_judge_completion_conditional_is_not_complete(monkeypatch):
    monkeypatch.setattr(cg, "ensure_council_importable", lambda *a, **k: True)
    monkeypatch.setattr(
        cg, "_council_run",
        lambda q, *, mode, max_tokens: {
            "verdict": "conditional", "confidence": 0.5,
            "arbiter": {"required_checks": ["verify build"]}, "deliberations": [],
            "sycophancy": {}, "meta": {"panel": "fast"},
        },
    )
    v = cg.judge_completion("goal", "work", "final")
    assert not v.complete and "verify build" in v.directive


def test_judge_completion_council_failure_falls_back_to_aux(monkeypatch):
    monkeypatch.setattr(cg, "ensure_council_importable", lambda *a, **k: True)

    def boom(*a, **k):
        raise RuntimeError("council exploded")

    monkeypatch.setattr(cg, "_council_run", boom)
    monkeypatch.setattr(
        cg, "_aux_call",
        lambda msgs, *, model, max_tokens, timeout: '{"complete": true, "confidence": 0.7, "next_action": "", "reason": "done"}',
    )
    v = cg.judge_completion("goal", "work", "final")
    assert v.complete and v.source == "aux"


# --------------------------------------------------------------------------- #
# judge_completion — aux fallback path (mocked seam)                           #
# --------------------------------------------------------------------------- #
def test_judge_completion_aux_incomplete(monkeypatch):
    monkeypatch.setattr(cg, "ensure_council_importable", lambda *a, **k: False)
    monkeypatch.setattr(
        cg, "_aux_call",
        lambda msgs, *, model, max_tokens, timeout: '{"complete": false, "confidence": 0.4, "next_action": "do Z", "reason": "x"}',
    )
    v = cg.judge_completion("goal", "work", "final")
    assert not v.complete and v.source == "aux" and "do Z" in v.directive


def test_judge_completion_total_failure_fails_open(monkeypatch):
    monkeypatch.setattr(cg, "ensure_council_importable", lambda *a, **k: False)

    def boom(*a, **k):
        raise RuntimeError("no aux backend")

    monkeypatch.setattr(cg, "_aux_call", boom)
    v = cg.judge_completion("goal", "work", "final")
    # Total judge failure must fail OPEN (deliver) so we never loop blindly.
    assert v.complete and v.source == "fallback"


# --------------------------------------------------------------------------- #
# Integration: real Hermes Council offline lane (deterministic, no network)    #
# --------------------------------------------------------------------------- #
def _council_src():
    for c in (os.environ.get("COUNCIL_SRC", ""), "/path/to/council/src"):
        if c and (Path(c) / "libs" / "hermes_council" / "deliberation.py").exists():
            return c
    return None


@pytest.mark.skipif(_council_src() is None, reason="hermes_council package not present")
def test_real_offline_council_round_trip(monkeypatch):
    monkeypatch.setenv("COUNCIL_PROVIDER", "offline")
    monkeypatch.setenv("AUTOPILOT_COUNCIL_SRC", _council_src())
    # reset the module-level import cache so the env takes effect
    cg._COUNCIL_READY = None
    cg._COUNCIL_SRC = None
    v = cg.judge_completion("Add a composite index", "ran migration", "I added the index")
    assert v.source == "council"
    assert v.verdict in {"allow", "deny", "conditional"}
    assert 0.0 <= v.confidence <= 1.0


# --------------------------------------------------------------------------- #
# choose_answer — clarify auto-answer (Seam A)                                 #
# --------------------------------------------------------------------------- #
def test_match_option_single():
    assert cg._match_option("we should use Postgres here", ["Postgres", "SQLite"]) == "Postgres"


def test_match_option_ambiguous_returns_empty():
    assert cg._match_option("Postgres or SQLite both work", ["Postgres", "SQLite"]) == ""


def test_choose_answer_council_direct_match(monkeypatch):
    monkeypatch.setattr(cg, "ensure_council_importable", lambda *a, **k: True)
    monkeypatch.setattr(cg, "_council_decision",
                        lambda options, ctx: {"arbiter": {"safest_reversible_path": "Choose Postgres for durability."}})
    assert cg.choose_answer("Which DB?", ["Postgres", "SQLite"]) == "Postgres"


def test_choose_answer_council_then_aux_pick(monkeypatch):
    monkeypatch.setattr(cg, "ensure_council_importable", lambda *a, **k: True)
    monkeypatch.setattr(cg, "_council_decision",
                        lambda options, ctx: {"arbiter": {"safest_reversible_path": "It depends on tradeoffs."}})
    monkeypatch.setattr(cg, "_aux_call",
                        lambda msgs, *, model, max_tokens, timeout: '{"choice": "SQLite", "rationale": "simpler"}')
    assert cg.choose_answer("Which DB?", ["Postgres", "SQLite"]) == "SQLite"


def test_choose_answer_open_ended_council(monkeypatch):
    monkeypatch.setattr(cg, "ensure_council_importable", lambda *a, **k: True)
    monkeypatch.setattr(cg, "_council_run",
                        lambda q, *, mode, max_tokens: {"arbiter": {"safest_reversible_path": "Back up first, then migrate."}})
    assert cg.choose_answer("What should I do?") == "Back up first, then migrate."


def test_choose_answer_aux_fallback_no_council(monkeypatch):
    monkeypatch.setattr(cg, "ensure_council_importable", lambda *a, **k: False)
    monkeypatch.setattr(cg, "_aux_call",
                        lambda msgs, *, model, max_tokens, timeout: '{"choice": "Postgres"}')
    assert cg.choose_answer("Which DB?", ["Postgres", "SQLite"]) == "Postgres"
