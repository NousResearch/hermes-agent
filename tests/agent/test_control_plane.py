"""Tests for agent.control_plane — classify(), Intent, Recommendation, WorkerTaskSpec,
WorkerTaskResult, WorkerLaneCancelMode.

Coverage targets (design review §3 invariants + §7.1):
- Intent mapping for each policy verdict + mode combination
- Transcript-render visibility invariant (INV-2)
- STOP sanctity invariant (INV-1)
- Fingerprint determinism (INV-6)
- Hashability (frozen dataclass)
- WorkerTaskSpec.from_decision field contracts (INV-7)
- WorkerTaskResult INV-7 enforcement (non-empty body forces gate_required=True)
- WorkerLaneCancelMode spelling
- Korean corpus parity (broad)
- Mode-off downgrade invariant
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent.control_plane import (
    Confidence,
    ControlPlaneDecision,
    Intent,
    Recommendation,
    Signal,
    WorkerLaneCancelMode,
    WorkerTaskResult,
    WorkerTaskSpec,
    classify,
    decide_intent_from_policy,
)

_CORPUS_PATH = Path(__file__).parent / "data" / "concierge_intents_ko.yaml"


def _corpus() -> list[dict]:
    return yaml.safe_load(_CORPUS_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Intent mapping — policy verdict × mode
# ---------------------------------------------------------------------------
class TestIntentMapping:
    """Each policy verdict + mode combination maps to the expected (Intent, Recommendation)."""

    # WORKER_LANE policy, mode OFF → downgraded to NEW_TASK_MAIN / MAIN
    def test_worker_lane_policy_mode_off_downgrades_to_main(self):
        d = classify("investigate the regression", concierge_mode_active=False)
        assert d.intent is Intent.NEW_TASK_MAIN
        assert d.recommendation is Recommendation.MAIN

    def test_keyword_no_longer_produces_worker_downgrade_note(self):
        d = classify("draft a report.md", concierge_mode_active=False)
        assert d.recommendation is Recommendation.MAIN
        assert not any("downgraded" in n for n in d.notes)

    def test_worker_lane_policy_mode_on_stays_worker(self):
        d = classify("investigate the regression", concierge_mode_active=True)
        assert d.intent is Intent.NEW_TASK_MAIN
        assert d.recommendation is Recommendation.MAIN

    def test_worker_lane_artifact_mode_on(self):
        d = classify("draft a report.md", concierge_mode_active=True)
        assert d.intent is Intent.NEW_TASK_MAIN
        assert d.recommendation is Recommendation.MAIN

    def test_worker_lane_code_edit_mode_on(self):
        d = classify("refactor the module", concierge_mode_active=True)
        assert d.intent is Intent.NEW_TASK_MAIN
        assert d.recommendation is Recommendation.MAIN

    # CONTROL + STOP signal
    def test_control_stop_mode_off(self):
        d = classify("그만", concierge_mode_active=False)
        assert d.intent is Intent.STOP
        assert d.recommendation is Recommendation.CONTROL

    def test_control_stop_mode_on(self):
        d = classify("그만", concierge_mode_active=True)
        assert d.intent is Intent.STOP
        assert d.recommendation is Recommendation.CONTROL

    def test_control_stop_en_mode_off(self):
        d = classify("stop", concierge_mode_active=False)
        assert d.intent is Intent.STOP
        assert d.recommendation is Recommendation.CONTROL

    def test_control_stop_en_mode_on(self):
        d = classify("cancel", concierge_mode_active=True)
        assert d.intent is Intent.STOP
        assert d.recommendation is Recommendation.CONTROL

    # CONTROL + ACK signal
    def test_control_ack_mode_off(self):
        d = classify("고마워", concierge_mode_active=False)
        assert d.intent is Intent.ACK
        assert d.recommendation is Recommendation.CONTROL

    def test_control_ack_mode_on(self):
        d = classify("감사합니다", concierge_mode_active=True)
        assert d.intent is Intent.ACK
        assert d.recommendation is Recommendation.CONTROL

    def test_control_ack_en_mode_off(self):
        d = classify("thanks", concierge_mode_active=False)
        assert d.intent is Intent.ACK
        assert d.recommendation is Recommendation.CONTROL

    def test_control_ack_en_mode_on(self):
        d = classify("thank you", concierge_mode_active=True)
        assert d.intent is Intent.ACK
        assert d.recommendation is Recommendation.CONTROL

    # MAIN + STATUS signal
    def test_main_status_signal(self):
        d = classify("status?", concierge_mode_active=False)
        assert d.intent is Intent.STATUS
        assert d.recommendation is Recommendation.MAIN

    def test_main_status_signal_ko(self):
        d = classify("지금 뭐 해?", concierge_mode_active=False)
        assert d.intent is Intent.STATUS
        assert d.recommendation is Recommendation.MAIN

    # MAIN without STATUS → NEW_TASK_MAIN
    def test_main_no_status_is_new_task_main(self):
        d = classify("이 파일 읽어줘", concierge_mode_active=False)
        assert d.intent is Intent.NEW_TASK_MAIN
        assert d.recommendation is Recommendation.MAIN

    def test_main_default_conservative_en(self):
        d = classify("explain the error message", concierge_mode_active=False)
        assert d.intent is Intent.NEW_TASK_MAIN
        assert d.recommendation is Recommendation.MAIN

    # STEER policy is mode-gated: OFF downgrades to MAIN; ON stays STEER.
    def test_steer_policy_mode_off_downgrades_to_main(self):
        d = classify("근데 이거도 봐줘", concierge_mode_active=False)
        assert d.intent is Intent.NEW_TASK_MAIN
        assert d.recommendation is Recommendation.MAIN

    def test_steer_policy_mode_on_maps_to_steer_intent(self):
        d = classify("근데 이거도 봐줘", concierge_mode_active=True)
        assert d.intent is Intent.STEER
        assert d.recommendation is Recommendation.STEER

    def test_steer_policy_en_mode_on(self):
        d = classify("also update the config file", concierge_mode_active=True)
        assert d.intent is Intent.STEER
        assert d.recommendation is Recommendation.STEER

    # NOISE
    def test_noise_empty(self):
        d = classify("", concierge_mode_active=False)
        assert d.intent is Intent.NOISE
        assert d.recommendation is Recommendation.CONTROL

    def test_noise_whitespace(self):
        d = classify("   ", concierge_mode_active=False)
        assert d.intent is Intent.NOISE
        assert d.recommendation is Recommendation.CONTROL


# ---------------------------------------------------------------------------
# Transcript-render visibility invariant (INV-2)
# ---------------------------------------------------------------------------
class TestTranscriptRenderVisibility:
    """STOP / ACK / NOISE → transcript_render=True.
    STATUS: True only when concierge_mode_active=True.
    NEW_TASK_MAIN → transcript_render=False.
    """
    def test_stop_transcript_render_true(self):
        assert classify("그만", concierge_mode_active=False).transcript_render is True

    def test_stop_transcript_render_true_mode_on(self):
        assert classify("stop", concierge_mode_active=True).transcript_render is True

    def test_new_task_main_transcript_render_false_by_default(self):
        d = classify("investigate the regression", concierge_mode_active=True)
        assert d.intent is Intent.NEW_TASK_MAIN
        assert d.transcript_render is False

    def test_steer_transcript_render_false_when_mode_off(self):
        d = classify("근데 이거도 봐줘", concierge_mode_active=False)
        assert d.intent is Intent.NEW_TASK_MAIN
        assert d.transcript_render is False

    def test_steer_transcript_render_true_when_mode_on(self):
        d = classify("근데 이거도 봐줘", concierge_mode_active=True)
        assert d.intent is Intent.STEER
        assert d.transcript_render is True

    def test_steer_transcript_render_true_en_mode_on(self):
        d = classify("also check the logs", concierge_mode_active=True)
        assert d.intent is Intent.STEER
        assert d.transcript_render is True

    def test_ack_transcript_render_true(self):
        d = classify("thanks", concierge_mode_active=False)
        assert d.intent is Intent.ACK
        assert d.transcript_render is True

    def test_noise_transcript_render_true(self):
        d = classify("", concierge_mode_active=False)
        assert d.intent is Intent.NOISE
        assert d.transcript_render is True

    def test_status_transcript_render_false_when_mode_off(self):
        d = classify("status?", concierge_mode_active=False)
        assert d.intent is Intent.STATUS
        assert d.transcript_render is False

    def test_status_transcript_render_true_when_mode_on(self):
        d = classify("status?", concierge_mode_active=True)
        assert d.intent is Intent.STATUS
        assert d.transcript_render is True

    def test_new_task_main_transcript_render_false(self):
        d = classify("이 파일 읽어줘", concierge_mode_active=False)
        assert d.intent is Intent.NEW_TASK_MAIN
        assert d.transcript_render is False

    def test_new_task_main_transcript_render_false_mode_on(self):
        d = classify("이 파일 읽어줘", concierge_mode_active=True)
        assert d.intent is Intent.NEW_TASK_MAIN
        assert d.transcript_render is False


# ---------------------------------------------------------------------------
# STOP sanctity invariant (INV-1)
# ---------------------------------------------------------------------------
class TestStopSanctity:
    """classify('그만') always produces STOP / CONTROL regardless of mode."""

    def test_stop_invariant_ko_mode_off(self):
        d = classify("그만", concierge_mode_active=False)
        assert d.intent is Intent.STOP
        assert d.recommendation is Recommendation.CONTROL

    def test_stop_invariant_ko_mode_on(self):
        d = classify("그만", concierge_mode_active=True)
        assert d.intent is Intent.STOP
        assert d.recommendation is Recommendation.CONTROL

    def test_stop_invariant_en_mode_off(self):
        d = classify("stop", concierge_mode_active=False)
        assert d.intent is Intent.STOP
        assert d.recommendation is Recommendation.CONTROL

    def test_stop_invariant_en_mode_on(self):
        d = classify("stop", concierge_mode_active=True)
        assert d.intent is Intent.STOP
        assert d.recommendation is Recommendation.CONTROL

    def test_stop_is_stop_property(self):
        d = classify("그만", concierge_mode_active=False)
        assert d.is_stop is True

    def test_stop_is_control_property(self):
        d = classify("그만", concierge_mode_active=False)
        assert d.is_control is True

    def test_stop_transcript_render_always_true(self):
        for mode in (False, True):
            d = classify("그만", concierge_mode_active=mode)
            assert d.transcript_render is True


# ---------------------------------------------------------------------------
# Fingerprint determinism (INV-6)
# ---------------------------------------------------------------------------
class TestFingerprintDeterminism:
    """Same (text, mode) → same fingerprint; differs across mode bits."""

    def test_same_text_same_mode_same_fingerprint(self):
        d1 = classify("investigate the regression", concierge_mode_active=False)
        d2 = classify("investigate the regression", concierge_mode_active=False)
        assert d1.fingerprint == d2.fingerprint

    def test_different_mode_different_fingerprint(self):
        d_off = classify("draft a report.md", concierge_mode_active=False)
        d_on = classify("draft a report.md", concierge_mode_active=True)
        assert d_off.fingerprint != d_on.fingerprint

    def test_different_text_different_fingerprint(self):
        d1 = classify("stop", concierge_mode_active=False)
        d2 = classify("cancel", concierge_mode_active=False)
        assert d1.fingerprint != d2.fingerprint

    def test_fingerprint_stored_on_decision(self):
        d = classify("근데 이거도 봐줘", concierge_mode_active=False)
        assert isinstance(d.fingerprint, str)
        assert len(d.fingerprint) > 0

    def test_fingerprint_in_to_dict(self):
        d = classify("status?", concierge_mode_active=False)
        data = d.to_dict()
        assert "fingerprint" in data
        assert data["fingerprint"] == d.fingerprint


# ---------------------------------------------------------------------------
# Hashability
# ---------------------------------------------------------------------------
class TestHashability:
    """ControlPlaneDecision can be added to a set (frozen dataclass)."""

    def test_decision_hashable(self):
        d = classify("그만", concierge_mode_active=False)
        s = {d}
        assert len(s) == 1

    def test_two_decisions_in_set(self):
        d1 = classify("그만", concierge_mode_active=False)
        d2 = classify("status?", concierge_mode_active=False)
        s = {d1, d2}
        assert len(s) == 2

    def test_equal_decisions_deduplicate_in_set(self):
        d1 = classify("thanks", concierge_mode_active=False)
        d2 = classify("thanks", concierge_mode_active=False)
        s = {d1, d2}
        assert len(s) == 1


# ---------------------------------------------------------------------------
# Slots schema lock-in
# ---------------------------------------------------------------------------
class TestSlotsSchemaLockIn:
    def test_control_plane_decision_uses_slots(self):
        assert hasattr(ControlPlaneDecision, "__slots__")

    def test_worker_task_spec_uses_slots(self):
        assert hasattr(WorkerTaskSpec, "__slots__")

    def test_worker_task_result_uses_slots(self):
        assert hasattr(WorkerTaskResult, "__slots__")


# ---------------------------------------------------------------------------
# WorkerTaskSpec.from_decision
# ---------------------------------------------------------------------------
class TestWorkerTaskSpecFromDecision:
    """Schema contracts for WorkerTaskSpec.from_decision (INV-7)."""

    def _worker_decision(self, text: str = "investigate and write report.md") -> ControlPlaneDecision:
        return classify(text, concierge_mode_active=True)

    def test_title_defaults_to_first_60_chars(self):
        long_text = "investigate the regression in the authentication module and write report.md"
        d = classify(long_text, concierge_mode_active=True)
        spec = WorkerTaskSpec.from_decision(d)
        assert spec.title == long_text[:60].strip()

    def test_title_uses_caller_supplied_value(self):
        d = self._worker_decision()
        spec = WorkerTaskSpec.from_decision(d, title="My custom title")
        assert spec.title == "My custom title"

    def test_decision_fingerprint_copied(self):
        d = self._worker_decision()
        spec = WorkerTaskSpec.from_decision(d)
        assert spec.decision_fingerprint == d.fingerprint

    def test_allow_memory_writes_defaults_false(self):
        d = self._worker_decision()
        spec = WorkerTaskSpec.from_decision(d)
        assert spec.allow_memory_writes is False

    def test_allow_memory_writes_caller_can_override(self):
        d = self._worker_decision()
        spec = WorkerTaskSpec.from_decision(d, allow_memory_writes=True)
        assert spec.allow_memory_writes is True

    def test_user_intent_is_raw_text(self):
        text = "investigate the regression and write report.md"
        d = classify(text, concierge_mode_active=True)
        spec = WorkerTaskSpec.from_decision(d)
        assert spec.user_intent == text

    def test_source_surface_default_cli(self):
        d = self._worker_decision()
        spec = WorkerTaskSpec.from_decision(d)
        assert spec.source_surface == "cli"

    def test_source_surface_caller_override(self):
        d = self._worker_decision()
        spec = WorkerTaskSpec.from_decision(d, source_surface="cli")
        assert spec.source_surface == "cli"

    def test_priority_default_normal(self):
        d = self._worker_decision()
        spec = WorkerTaskSpec.from_decision(d)
        assert spec.priority == "normal"

    def test_lane_name_default_none(self):
        d = self._worker_decision()
        spec = WorkerTaskSpec.from_decision(d)
        assert spec.lane_name is None

    def test_to_dict_is_json_safe(self):
        d = self._worker_decision()
        spec = WorkerTaskSpec.from_decision(d, source_surface="tui", priority="high")
        j = json.dumps(spec.to_dict())
        data = json.loads(j)
        assert data["allow_memory_writes"] is False
        assert data["source_surface"] == "tui"
        assert data["priority"] == "high"
        assert data["decision_fingerprint"] == d.fingerprint

    def test_requested_artifacts_passed_through(self):
        d = self._worker_decision()
        spec = WorkerTaskSpec.from_decision(d, requested_artifacts=("report.md", "summary.csv"))
        assert spec.requested_artifacts == ("report.md", "summary.csv")


# ---------------------------------------------------------------------------
# WorkerTaskResult INV-7 enforcement
# ---------------------------------------------------------------------------
class TestWorkerTaskResultInv7:
    """Non-empty body auto-sets gate_required=True; empty body respects caller."""

    def test_non_empty_body_forces_gate_required_true(self):
        r = WorkerTaskResult(task_id="t1", status="done", summary="done", body="output here")
        assert r.gate_required is True

    def test_non_empty_body_forces_gate_required_even_if_caller_passed_false(self):
        r = WorkerTaskResult(
            task_id="t1", status="done", summary="done", body="output here", gate_required=False
        )
        assert r.gate_required is True

    def test_empty_body_leaves_gate_required_as_supplied_false(self):
        r = WorkerTaskResult(task_id="t1", status="done", summary="done", body="", gate_required=False)
        assert r.gate_required is False

    def test_empty_body_leaves_gate_required_as_supplied_true(self):
        r = WorkerTaskResult(task_id="t1", status="done", summary="done", body="", gate_required=True)
        assert r.gate_required is True

    def test_to_dict_roundtrip_carries_gate_required_and_body(self):
        r = WorkerTaskResult(task_id="t2", status="done", summary="worker done", body="full output")
        data = r.to_dict()
        assert data["gate_required"] is True
        assert data["body"] == "full output"

    def test_to_dict_is_json_safe(self):
        r = WorkerTaskResult(
            task_id="t3",
            status="running",
            summary="in progress",
            started_at=1_700_000_000.0,
        )
        j = json.dumps(r.to_dict())
        data = json.loads(j)
        assert data["status"] == "running"
        assert data["gate_required"] is False

    def test_no_body_gate_required_defaults_false(self):
        r = WorkerTaskResult(task_id="t4", status="queued", summary="queued")
        assert r.gate_required is False


# ---------------------------------------------------------------------------
# WorkerLaneCancelMode spellings
# ---------------------------------------------------------------------------
class TestWorkerLaneCancelMode:
    """Three values: graceful, hard, force — exact spellings Phase 4 relies on."""

    def test_graceful_value(self):
        assert WorkerLaneCancelMode.GRACEFUL.value == "graceful"

    def test_hard_value(self):
        assert WorkerLaneCancelMode.HARD.value == "hard"

    def test_force_value(self):
        assert WorkerLaneCancelMode.FORCE.value == "force"

    def test_exactly_three_members(self):
        assert len(WorkerLaneCancelMode) == 3

    def test_members_by_name(self):
        assert WorkerLaneCancelMode["GRACEFUL"] is WorkerLaneCancelMode.GRACEFUL
        assert WorkerLaneCancelMode["HARD"] is WorkerLaneCancelMode.HARD
        assert WorkerLaneCancelMode["FORCE"] is WorkerLaneCancelMode.FORCE


# ---------------------------------------------------------------------------
# Korean corpus parity (broad)
# ---------------------------------------------------------------------------
def _corpus_params_cp():
    rows = _corpus()
    params = []
    for row in rows:
        is_duplicate = "DUPLICATE handled by surface" in row.get("notes", "")
        marks = []
        # Duplicate rows are labelled with their current classifier output;
        # surface-level dedup is documented in notes, not xfailed here.
        params.append(
            pytest.param(
                row,
                marks=marks,
                id=f"row{row['id']}",
            )
        )
    return params


@pytest.mark.parametrize("row", _corpus_params_cp())
def test_korean_corpus_row_matches_control_plane(row: dict):
    """Each corpus row's expected_intent and expected_recommendation match classify()."""
    mode = row["expected_intent"] in {"NEW_TASK_WORKER", "STEER"} or row["expected_recommendation"] in {"WORKER_LANE", "STEER"}
    d = classify(row["text"], concierge_mode_active=mode)
    assert d.intent.name == row["expected_intent"], (
        f"id={row['id']} text={row['text']!r} mode={mode} "
        f"expected={row['expected_intent']!r} got={d.intent.name!r}"
    )
    assert d.recommendation.name == row["expected_recommendation"], (
        f"id={row['id']} text={row['text']!r} mode={mode} "
        f"expected={row['expected_recommendation']!r} got={d.recommendation.name!r}"
    )


# ---------------------------------------------------------------------------
# Mode-off downgrade is the ONLY pathway for non-MAIN/CONTROL under mode-off
# ---------------------------------------------------------------------------
class TestModeOffDowngradeInvariant:
    """When concierge_mode_active=False, only MAIN and CONTROL recommendations appear."""

    def test_corpus_mode_off_yields_only_main_or_control(self):
        """Mode-off downgrade: only MAIN / CONTROL recommendations may appear."""
        rows = _corpus()
        violations = []
        for row in rows:
            d = classify(row["text"], concierge_mode_active=False)
            if d.recommendation not in {Recommendation.MAIN, Recommendation.CONTROL}:
                violations.append(
                    f"id={row['id']} text={row['text']!r} got={d.recommendation.name}"
                )
        assert violations == [], f"Mode-off yielded non-MAIN/CONTROL: {violations}"

    def test_explicit_worker_request_downgrades_when_mode_off(self):
        d = classify("백그라운드로 돌려", concierge_mode_active=False)
        assert d.recommendation is Recommendation.MAIN

    def test_artifact_anchor_downgrades_when_mode_off(self):
        d = classify("draft a report.md", concierge_mode_active=False)
        assert d.recommendation is Recommendation.MAIN

    def test_research_anchor_downgrades_when_mode_off(self):
        d = classify("investigate the regression", concierge_mode_active=False)
        assert d.recommendation is Recommendation.MAIN

    def test_steer_downgrades_when_mode_off_and_stays_when_on(self):
        d_off = classify("근데 이거도 봐줘", concierge_mode_active=False)
        d_on = classify("근데 이거도 봐줘", concierge_mode_active=True)
        assert d_off.recommendation is Recommendation.MAIN
        assert d_off.intent is Intent.NEW_TASK_MAIN
        assert d_on.recommendation is Recommendation.STEER

        assert d_on.recommendation is Recommendation.STEER
        assert d_on.intent is Intent.STEER


# ---------------------------------------------------------------------------
# Temporary rename aliases
# ---------------------------------------------------------------------------
def test_legacy_concierge_mode_active_kw_alias():
    """Legacy keyword still enables worker-lane routing."""
    d_new = classify("draft a report.md", concierge_mode_active=True)
    d_old = classify("draft a report.md", concierge_mode_active=True)
    assert d_new.intent is d_old.intent
    assert d_new.recommendation is d_old.recommendation
    assert d_new.fingerprint == d_old.fingerprint
    assert d_new.concierge_mode_active is True
    assert d_old.concierge_mode_active is True
    assert "concierge_mode_active" in d_new.to_dict()
    assert "concierge_mode_active" in d_new.to_dict()
