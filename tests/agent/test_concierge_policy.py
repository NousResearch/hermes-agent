"""Tests for agent.concierge_policy.classify_request.

Coverage targets (PRD §11.1 + design review §7.1):
- All 4 recommendations: MAIN, WORKER_LANE, STEER, CONTROL
- All STOP tokens (EN + KO), including trailing punctuation
- All ACK tokens (EN + KO), including smileys
- Whole-body strictness: partial matches fall to STEER (not STOP/ACK)
- Steer prefix detection (direct and post-comma)
- Single worker anchor → WORKER_LANE (artifact, research, code_edit)
- Multi-anchor confidence: HIGH
- Explicit overrides (worker and main)
- Status queries
- Noise / empty
- Determinism (INV-6)
- Side-effect freeness (no forbidden imports)
- to_dict() JSON round-trip
- Property helpers (should_delegate, is_control, is_stop, is_ack, has_korean)
- Korean corpus parity (parametrized from YAML)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from agent.concierge_policy import (
    ACK_TOKENS_EN,
    ACK_TOKENS_KO,
    STOP_TOKENS_EN,
    STOP_TOKENS_KO,
    ConciergeConfidence,
    ConciergePolicyDecision,
    ConciergeRecommendation,
    ConciergeSignal,
    classify_request,
    fingerprint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_CORPUS_PATH = Path(__file__).parent / "data" / "frontdesk_intents_ko.yaml"


def _corpus() -> list[dict]:
    return yaml.safe_load(_CORPUS_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CONTROL recommendation — STOP tokens
# ---------------------------------------------------------------------------
class TestStopTokens:
    """All STOP_TOKENS_EN entries produce CONTROL+STOP (whole-body equality)."""

    @pytest.mark.parametrize("token", STOP_TOKENS_EN)
    def test_stop_en_whole_body_returns_control(self, token: str):
        d = classify_request(token)
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.STOP in d.signals

    @pytest.mark.parametrize("token", STOP_TOKENS_KO)
    def test_stop_ko_whole_body_returns_control(self, token: str):
        d = classify_request(token)
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.STOP in d.signals

    def test_stop_en_with_trailing_exclamation(self):
        d = classify_request("Stop!")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.STOP in d.signals

    def test_stop_en_with_trailing_question(self):
        d = classify_request("stop?")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.STOP in d.signals

    def test_stop_ko_with_trailing_period(self):
        d = classify_request("그만.")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.STOP in d.signals

    def test_stop_slash_stop_variant(self):
        d = classify_request("/stop")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.STOP in d.signals

    def test_stop_en_uppercase(self):
        d = classify_request("STOP")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.STOP in d.signals

    def test_stop_en_mixed_case(self):
        d = classify_request("Cancel")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.STOP in d.signals

    def test_stop_en_with_whitespace_padding(self):
        d = classify_request("  abort  ")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.STOP in d.signals

    def test_stop_partial_phrase_is_not_stop(self):
        """'stop, and then do X' must NOT produce STOP — it falls through."""
        d = classify_request("stop, and then reformat the code")
        assert ConciergeSignal.STOP not in d.signals

    def test_stop_ko_partial_phrase_is_not_stop(self):
        """'그만, 근데 Y' must NOT produce STOP — it steers instead."""
        d = classify_request("그만, 근데 이거도 봐줘")
        assert ConciergeSignal.STOP not in d.signals
        assert d.recommendation is ConciergeRecommendation.STEER

    def test_stop_ko_with_korean_signal(self):
        d = classify_request("그만")
        assert ConciergeSignal.KOREAN in d.signals
        assert d.has_korean is True


# ---------------------------------------------------------------------------
# CONTROL recommendation — ACK tokens
# ---------------------------------------------------------------------------
class TestAckTokens:
    """All ACK_TOKENS_EN/KO entries produce CONTROL+ACK (whole-body equality)."""

    @pytest.mark.parametrize("token", ACK_TOKENS_EN)
    def test_ack_en_whole_body_returns_control(self, token: str):
        d = classify_request(token)
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.ACK in d.signals

    @pytest.mark.parametrize("token", ACK_TOKENS_KO)
    def test_ack_ko_whole_body_returns_control(self, token: str):
        d = classify_request(token)
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.ACK in d.signals

    def test_ack_en_with_smiley(self):
        d = classify_request("thanks :)")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.ACK in d.signals

    def test_ack_en_with_double_exclamation(self):
        d = classify_request("thanks!!")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.ACK in d.signals

    def test_ack_ko_with_korean_laughter(self):
        d = classify_request("고마워 ㅎㅎ")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.ACK in d.signals

    def test_ack_ko_with_kkk_laughter(self):
        d = classify_request("고마워ㅋㅋ")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.ACK in d.signals

    def test_ack_ko_formal_with_punctuation(self):
        d = classify_request("감사합니다!")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.ACK in d.signals

    def test_ack_en_uppercase(self):
        d = classify_request("THANKS")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.ACK in d.signals

    def test_ack_partial_phrase_is_not_ack(self):
        """'thanks, also do X' must NOT produce ACK — it steers instead."""
        d = classify_request("thanks, also do the zanu comparison")
        assert ConciergeSignal.ACK not in d.signals

    def test_ack_ko_partial_phrase_is_not_ack(self):
        """'고마워, 근데 Z' must NOT produce ACK."""
        d = classify_request("고마워, 근데 zanu도 같이 봐줘")
        assert ConciergeSignal.ACK not in d.signals

    def test_ack_ko_has_korean_signal(self):
        d = classify_request("고마워")
        assert ConciergeSignal.KOREAN in d.signals
        assert d.has_korean is True
        assert d.is_ack is True


# ---------------------------------------------------------------------------
# WORKER_LANE recommendation
# ---------------------------------------------------------------------------
class TestWorkerLaneRecommendation:
    """Single-anchor cases produce WORKER_LANE."""

    def test_artifact_anchor_only_draft_report(self):
        d = classify_request("draft a report.md")
        assert d.recommendation is ConciergeRecommendation.WORKER_LANE
        assert ConciergeSignal.ARTIFACT in d.signals

    def test_artifact_anchor_only_write_summary(self):
        d = classify_request("write a summary of the findings")
        assert d.recommendation is ConciergeRecommendation.WORKER_LANE
        assert ConciergeSignal.ARTIFACT in d.signals

    def test_artifact_anchor_only_compose(self):
        d = classify_request("compose the final report")
        assert d.recommendation is ConciergeRecommendation.WORKER_LANE
        assert ConciergeSignal.ARTIFACT in d.signals

    def test_research_anchor_only_investigate(self):
        d = classify_request("investigate the regression")
        assert d.recommendation is ConciergeRecommendation.WORKER_LANE
        assert ConciergeSignal.RESEARCH in d.signals

    def test_research_anchor_only_search_for(self):
        d = classify_request("search for the root cause of the issue")
        assert d.recommendation is ConciergeRecommendation.WORKER_LANE
        assert ConciergeSignal.RESEARCH in d.signals

    def test_research_anchor_only_audit(self):
        d = classify_request("audit the dependency versions")
        assert d.recommendation is ConciergeRecommendation.WORKER_LANE
        assert ConciergeSignal.RESEARCH in d.signals

    def test_code_edit_anchor_only_refactor(self):
        d = classify_request("refactor the module")
        assert d.recommendation is ConciergeRecommendation.WORKER_LANE
        assert ConciergeSignal.CODE_EDIT in d.signals

    def test_code_edit_anchor_only_implement(self):
        d = classify_request("implement the new endpoint")
        assert d.recommendation is ConciergeRecommendation.WORKER_LANE
        assert ConciergeSignal.CODE_EDIT in d.signals

    def test_code_edit_anchor_only_patch(self):
        d = classify_request("patch the broken handler")
        assert d.recommendation is ConciergeRecommendation.WORKER_LANE
        assert ConciergeSignal.CODE_EDIT in d.signals

    def test_multi_anchor_research_and_artifact_gives_high_confidence(self):
        d = classify_request("investigate and write report.md")
        assert d.recommendation is ConciergeRecommendation.WORKER_LANE
        assert d.confidence is ConciergeConfidence.HIGH
        assert ConciergeSignal.RESEARCH in d.signals
        assert ConciergeSignal.ARTIFACT in d.signals

    def test_multi_anchor_code_and_artifact_gives_high_confidence(self):
        d = classify_request("implement the feature and produce a summary.md")
        assert d.recommendation is ConciergeRecommendation.WORKER_LANE
        assert d.confidence is ConciergeConfidence.HIGH

    def test_explicit_worker_override_en_background(self):
        d = classify_request("run this in the background")
        assert d.recommendation is ConciergeRecommendation.WORKER_LANE
        assert ConciergeSignal.EXPLICIT_WORKER_REQ in d.signals

    def test_explicit_worker_override_ko_background(self):
        d = classify_request("백그라운드로 돌려")
        assert d.recommendation is ConciergeRecommendation.WORKER_LANE
        assert ConciergeSignal.EXPLICIT_WORKER_REQ in d.signals

    def test_explicit_worker_override_ko_delegate(self):
        d = classify_request("워커에 맡겨")
        assert d.recommendation is ConciergeRecommendation.WORKER_LANE
        assert ConciergeSignal.EXPLICIT_WORKER_REQ in d.signals

    def test_explicit_main_beats_explicit_worker_in_tie_breaker(self):
        """EXPLICIT_MAIN_REQ wins over EXPLICIT_WORKER_REQ — step 9 checks MAIN first."""
        d = classify_request("직접 해줘, 백그라운드로")
        # Both signals fire, but EXPLICIT_MAIN_REQ is checked first in step 9.
        assert d.recommendation is ConciergeRecommendation.MAIN
        assert ConciergeSignal.EXPLICIT_MAIN_REQ in d.signals
        assert ConciergeSignal.EXPLICIT_WORKER_REQ in d.signals

    def test_should_delegate_property_true_for_worker_lane(self):
        d = classify_request("investigate the regression")
        assert d.should_delegate is True

    def test_is_control_false_for_worker_lane(self):
        d = classify_request("draft a report.md")
        assert d.is_control is False


# ---------------------------------------------------------------------------
# MAIN recommendation
# ---------------------------------------------------------------------------
class TestMainRecommendation:
    """Status queries and explicit-main overrides stay on MAIN."""

    def test_status_query_en(self):
        d = classify_request("status?")
        assert d.recommendation is ConciergeRecommendation.MAIN
        assert ConciergeSignal.STATUS in d.signals

    def test_status_query_ko(self):
        d = classify_request("지금 뭐 해?")
        assert d.recommendation is ConciergeRecommendation.MAIN
        assert ConciergeSignal.STATUS in d.signals

    def test_status_query_ko_eodickkaji(self):
        d = classify_request("어디까지 갔어?")
        assert d.recommendation is ConciergeRecommendation.MAIN
        assert ConciergeSignal.STATUS in d.signals

    def test_explicit_main_override_ko_direct(self):
        d = classify_request("직접 해")
        assert d.recommendation is ConciergeRecommendation.MAIN
        assert ConciergeSignal.EXPLICIT_MAIN_REQ in d.signals

    def test_explicit_main_override_ko_compact(self):
        d = classify_request("직접해")
        assert d.recommendation is ConciergeRecommendation.MAIN
        assert ConciergeSignal.EXPLICIT_MAIN_REQ in d.signals

    def test_explicit_main_override_beats_worker_anchors(self):
        """직접 해 + worker anchor still stays on MAIN."""
        d = classify_request("이 모듈 직접 해서 리팩해줘")
        assert d.recommendation is ConciergeRecommendation.MAIN
        assert ConciergeSignal.EXPLICIT_MAIN_REQ in d.signals

    def test_default_foreground_request_stays_main(self):
        d = classify_request("이 파일 읽어줘")
        assert d.recommendation is ConciergeRecommendation.MAIN

    def test_explanation_request_stays_main(self):
        d = classify_request("방금 결과 설명해줘")
        assert d.recommendation is ConciergeRecommendation.MAIN

    def test_should_delegate_false_for_main(self):
        d = classify_request("status?")
        assert d.should_delegate is False

    def test_is_control_false_for_main(self):
        d = classify_request("이 파일 읽어줘")
        assert d.is_control is False


# ---------------------------------------------------------------------------
# STEER recommendation
# ---------------------------------------------------------------------------
class TestSteerRecommendation:
    """Steer prefix detection — direct and post-comma."""

    def test_steer_prefix_also_en(self):
        d = classify_request("also check the test file")
        assert d.recommendation is ConciergeRecommendation.STEER
        assert ConciergeSignal.STEER in d.signals

    def test_steer_prefix_and_also_en(self):
        d = classify_request("and also update the README")
        assert d.recommendation is ConciergeRecommendation.STEER
        assert ConciergeSignal.STEER in d.signals

    def test_steer_prefix_geunde_ko(self):
        d = classify_request("근데 이거도 봐줘")
        assert d.recommendation is ConciergeRecommendation.STEER
        assert ConciergeSignal.STEER in d.signals

    def test_steer_prefix_geurigo_ko(self):
        d = classify_request("그리고 저것도 확인해줘")
        assert d.recommendation is ConciergeRecommendation.STEER
        assert ConciergeSignal.STEER in d.signals

    def test_steer_prefix_chuga_ko(self):
        d = classify_request("추가로 이 파일도 봐줘")
        assert d.recommendation is ConciergeRecommendation.STEER

    def test_steer_post_comma_stop_then_steer_ko(self):
        """'그만, 근데 X' — stop-leading with post-comma steer prefix."""
        d = classify_request("그만, 근데 이거 봐줘")
        assert d.recommendation is ConciergeRecommendation.STEER
        assert ConciergeSignal.STEER in d.signals
        assert ConciergeSignal.STOP not in d.signals

    def test_steer_post_comma_ack_then_steer_ko(self):
        """'고마워, 근데 X' — ack-leading with post-comma steer prefix, no anchors."""
        d = classify_request("고마워, 근데 zanu도 같이 봐줘")
        assert d.recommendation is ConciergeRecommendation.STEER
        assert ConciergeSignal.ACK not in d.signals

    def test_steer_post_comma_thanks_then_also_en(self):
        """'thanks, also do X' — must NOT be ACK."""
        d = classify_request("thanks, also update the config")
        assert ConciergeSignal.ACK not in d.signals
        assert ConciergeSignal.STOP not in d.signals

    def test_steer_post_comma_stop_en_then_also(self):
        d = classify_request("stop, also fix the test")
        assert ConciergeSignal.STOP not in d.signals

    def test_is_control_false_for_steer(self):
        d = classify_request("근데 이거도 봐줘")
        assert d.is_control is False

    def test_should_delegate_false_for_steer(self):
        d = classify_request("근데 이거도 봐줘")
        assert d.should_delegate is False


# ---------------------------------------------------------------------------
# CONTROL recommendation — NOISE
# ---------------------------------------------------------------------------
class TestNoiseRecommendation:
    """Empty and whitespace-only input produces CONTROL+NOISE."""

    def test_empty_string_is_noise(self):
        d = classify_request("")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.NOISE in d.signals

    def test_whitespace_only_is_noise(self):
        d = classify_request("   ")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.NOISE in d.signals

    def test_newline_only_is_noise(self):
        d = classify_request("\n")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.NOISE in d.signals

    def test_tab_only_is_noise(self):
        d = classify_request("\t")
        assert d.recommendation is ConciergeRecommendation.CONTROL
        assert ConciergeSignal.NOISE in d.signals

    def test_noise_is_control(self):
        d = classify_request("")
        assert d.is_control is True

    def test_noise_confidence_high(self):
        d = classify_request("")
        assert d.confidence is ConciergeConfidence.HIGH

    def test_noise_is_not_stop(self):
        d = classify_request("")
        assert d.is_stop is False

    def test_noise_is_not_ack(self):
        d = classify_request("")
        assert d.is_ack is False


# ---------------------------------------------------------------------------
# Determinism (INV-6)
# ---------------------------------------------------------------------------
class TestDeterminism:
    """Same (text, mode) always returns equal decision objects."""

    def test_same_text_returns_equal_decisions(self):
        text = "investigate the regression"
        d1 = classify_request(text)
        d2 = classify_request(text)
        assert d1 == d2

    def test_same_text_returns_identical_fingerprint_policy_level(self):
        text = "draft a report.md"
        fp1 = fingerprint(text, concierge_mode_active=False)
        fp2 = fingerprint(text, concierge_mode_active=False)
        assert fp1 == fp2

    def test_fingerprint_differs_across_mode_bits(self):
        text = "write a summary.md"
        fp_off = fingerprint(text, concierge_mode_active=False)
        fp_on = fingerprint(text, concierge_mode_active=True)
        assert fp_off != fp_on

    def test_fingerprint_differs_for_different_texts(self):
        fp1 = fingerprint("stop", concierge_mode_active=False)
        fp2 = fingerprint("cancel", concierge_mode_active=False)
        assert fp1 != fp2

    def test_multiple_calls_to_stop_are_equal(self):
        d1 = classify_request("그만")
        d2 = classify_request("그만")
        assert d1 == d2
        assert d1.recommendation == d2.recommendation
        assert d1.signals == d2.signals

    def test_decision_is_hashable_as_frozen_dataclass(self):
        d1 = classify_request("status?")
        d2 = classify_request("그만")
        s = {d1, d2}
        assert len(s) == 2


# ---------------------------------------------------------------------------
# Side-effect freeness
# ---------------------------------------------------------------------------
class TestSideEffectFreeness:
    """Classifier does not import cli, gateway, ui-tui or agent.run_agent in its own source."""

    def test_no_forbidden_imports_in_module_source(self):
        """Parse concierge_policy's AST to confirm it contains no forbidden imports.

        Checking sys.modules at test time is unreliable because the test runner
        itself may load gateway/cli modules for other tests.  Inspecting the
        module's own import statements is the correct scope for this invariant.
        """
        import ast
        import importlib.util
        import agent.concierge_policy

        spec = importlib.util.find_spec("agent.concierge_policy")
        assert spec is not None, "Cannot locate agent.concierge_policy module"
        source = spec.origin
        tree = ast.parse(open(source).read())

        imported_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_names.append(node.module)

        forbidden_prefixes = ("cli", "gateway", "ui_tui", "ui-tui", "agent.run_agent")
        violations = [
            n for n in imported_names
            if any(n == p or n.startswith(p + ".") for p in forbidden_prefixes)
        ]
        assert violations == [], f"Forbidden imports in concierge_policy source: {violations}"


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------
class TestJsonRoundTrip:
    """to_dict() must produce JSON-serializable output."""

    def test_to_dict_is_json_safe_for_stop(self):
        d = classify_request("그만")
        j = json.dumps(d.to_dict())
        data = json.loads(j)
        assert data["recommendation"] == "control"
        assert "stop" in data["signals"]

    def test_to_dict_is_json_safe_for_worker(self):
        d = classify_request("investigate the regression and write report.md")
        j = json.dumps(d.to_dict())
        data = json.loads(j)
        assert data["recommendation"] == "worker_lane"

    def test_to_dict_contains_required_keys(self):
        d = classify_request("status?")
        data = d.to_dict()
        required = {"recommendation", "confidence", "signals", "debug_label", "raw_text", "notes"}
        assert required.issubset(data.keys())

    def test_to_dict_signals_are_sorted(self):
        d = classify_request("investigate and write report.md")
        data = d.to_dict()
        assert data["signals"] == sorted(data["signals"])

    def test_to_dict_raw_text_preserved(self):
        text = "draft a report.md with the audit"
        d = classify_request(text)
        assert d.to_dict()["raw_text"] == text


# ---------------------------------------------------------------------------
# Property helpers
# ---------------------------------------------------------------------------
class TestPropertyHelpers:
    """should_delegate, is_control, is_stop, is_ack, has_korean."""

    def test_should_delegate_true_for_worker_lane(self):
        d = classify_request("investigate the regression")
        assert d.should_delegate is True

    def test_should_delegate_false_for_main(self):
        d = classify_request("what is the status?")
        assert d.should_delegate is False

    def test_should_delegate_false_for_control(self):
        d = classify_request("stop")
        assert d.should_delegate is False

    def test_is_control_true_for_stop(self):
        d = classify_request("cancel")
        assert d.is_control is True

    def test_is_control_true_for_ack(self):
        d = classify_request("thanks")
        assert d.is_control is True

    def test_is_control_true_for_noise(self):
        d = classify_request("")
        assert d.is_control is True

    def test_is_stop_true_for_stop_signal(self):
        d = classify_request("abort")
        assert d.is_stop is True

    def test_is_stop_false_for_ack(self):
        d = classify_request("thanks")
        assert d.is_stop is False

    def test_is_ack_true_for_ack_signal(self):
        d = classify_request("고마워")
        assert d.is_ack is True

    def test_is_ack_false_for_stop(self):
        d = classify_request("그만")
        assert d.is_ack is False

    def test_has_korean_true_for_hangul_text(self):
        d = classify_request("근데 이거도 봐줘")
        assert d.has_korean is True

    def test_has_korean_false_for_ascii_text(self):
        d = classify_request("investigate the regression")
        assert d.has_korean is False


# ---------------------------------------------------------------------------
# Korean corpus parity (parametrized from YAML)
# ---------------------------------------------------------------------------
def _corpus_params():
    rows = _corpus()
    params = []
    for row in rows:
        # DUPLICATE rows are labelled with current classifier output; surface-level
        # dedup is documented in notes, not xfailed here.
        params.append(
            pytest.param(
                row,
                id=f"row{row['id']}",
            )
        )
    return params


@pytest.mark.parametrize("row", _corpus_params())
def test_korean_corpus_row_matches_classifier(row: dict):
    """Each corpus row's expected_intent and expected_recommendation must match the classifier.

    For NEW_TASK_WORKER rows, classify() is called with concierge_mode_active=True.
    All other rows use concierge_mode_active=False (the recommendation column
    reflects the effective output under those conditions).
    """
    from agent.control_plane import classify

    mode = row["expected_intent"] in {"NEW_TASK_WORKER", "STEER"} or row["expected_recommendation"] in {"WORKER_LANE", "STEER"}
    d = classify(row["text"], concierge_mode_active=mode)
    assert d.intent.name == row["expected_intent"], (
        f"id={row['id']} text={row['text']!r} mode={mode} "
        f"expected_intent={row['expected_intent']!r} got={d.intent.name!r}"
    )
    assert d.recommendation.name == row["expected_recommendation"], (
        f"id={row['id']} text={row['text']!r} mode={mode} "
        f"expected_rec={row['expected_recommendation']!r} got={d.recommendation.name!r}"
    )
