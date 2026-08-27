#!/usr/bin/env python3
"""RED/GREEN coverage for Kensei's optional council feature set."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import council


def member(name: str) -> council.CouncilMember:
    return council.CouncilMember(provider="test", model=name)


def critique(
    label: str,
    verdict: str = "APPROVED",
) -> council.CouncilCritique:
    return council.CouncilCritique(
        member_label=label,
        verdict=verdict,
        completeness="complete",
        feasibility="feasible",
        risks="none" if verdict == "APPROVED" else "material risk",
        scope_creep="none",
        missing_ac="none" if verdict == "APPROVED" else "rollback proof",
        simpler_alternatives="none",
        overall="solid" if verdict == "APPROVED" else "revise first",
        raw_response="{}",
    )


def config(protocol: str = "deliberate", **overrides) -> council.CouncilConfig:
    cfg = council.CouncilConfig(
        panel=[member("a"), member("b"), member("c")],
        chairman=member("chair"),
        token_cap=100_000,
        timeout_seconds=100,
        member_timeout_seconds=80,
        quorum_min=2,
        fallback_pool=[],
        compose=True,
        cross_examination=True,
        cascade_breaker=True,
        minority_report=True,
        evidence_labels=True,
        html_report=True,
        protocol=protocol,
        adaptive_stopping=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def write_artifacts(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "prd.md").write_text("## Problem\nBuild the right thing.\n")
    (root / "spec.md").write_text("## Architecture\nUse a safe design.\n")


def patch_config(monkeypatch: pytest.MonkeyPatch, cfg: council.CouncilConfig) -> None:
    monkeypatch.setattr("hermes_cli.config.get_council_config", lambda: cfg)


# ---------------------------------------------------------------------------
# Config and prompts
# ---------------------------------------------------------------------------


def test_config_loads_all_eight_optional_feature_controls() -> None:
    cfg = council.CouncilConfig.from_config(
        {
            "panel": [
                {"provider": "test", "model": "a"},
                {"provider": "test", "model": "b"},
            ],
            "chairman": {"provider": "test", "model": "chair"},
            "compose": True,
            "cross_examination": True,
            "cascade_breaker": True,
            "minority_report": True,
            "evidence_labels": True,
            "html_report": True,
            "protocol": "vote",
            "adaptive_stopping": True,
        }
    )

    assert cfg.compose is True
    assert cfg.cross_examination is True
    assert cfg.cascade_breaker is True
    assert cfg.minority_report is True
    assert cfg.evidence_labels is True
    assert cfg.html_report is True
    assert cfg.protocol == "vote"
    assert cfg.adaptive_stopping is True


def test_phase_zero_is_task_specific_and_fences_prd_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []

    def fake_call(_member, messages, *_args, **_kwargs):
        captured.extend(messages)
        return (
            json.dumps(
                [
                    {"name": "A", "expertise": "security"},
                    {"name": "B", "expertise": "delivery"},
                ]
            ),
            5,
        )

    monkeypatch.setattr(council, "_call_llm_with_fallback", fake_call)
    personas, tokens = council._run_phase_0(
        member("chair"),
        "PRD-MARKER: user outcome",
        "SPEC-MARKER: system shape",
        n=2,
        member_timeout=30,
        token_cap=1000,
        current_tokens=0,
    )

    assert tokens == 5
    assert personas and len(personas) == 2
    prompt = "\n".join(item["content"] for item in captured)
    assert "PRD-MARKER" in prompt
    assert "SPEC-MARKER" in prompt
    assert "<<<UNTRUSTED_DOCUMENT" in prompt


def test_phase_one_parses_numeric_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        {
            "verdict": "APPROVED",
            "completeness": "complete",
            "feasibility": "yes",
            "risks": "none",
            "scope_creep": "none",
            "missing_ac": "none",
            "simpler_alternatives": "none",
            "overall": "good",
            "confidence": 0.83,
        }
    )
    monkeypatch.setattr(
        council,
        "_call_llm_with_fallback",
        lambda *_a, **_k: (payload, 3),
    )

    critiques, tokens, _active = council._run_phase_1(
        [member("a")],
        "prd",
        "spec",
        30,
        1000,
        total_timeout=30,
    )

    assert tokens == 3
    assert critiques[0].confidence == pytest.approx(0.83)


def test_cascade_breaker_prompt_does_not_receive_panel_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []

    def fake_call(_member, messages, *_args, **_kwargs):
        captured.extend(messages)
        return (
            json.dumps(
                {
                    "independent_verdict": "REVISE",
                    "shortcut_cascade_risk": "high",
                    "confidence": 0.9,
                }
            ),
            4,
        )

    monkeypatch.setattr(council, "_call_llm_with_fallback", fake_call)
    result, tokens = council._run_cascade_breaker(
        member("chair"),
        [critique("secret-panel-signal")],
        member_timeout=30,
        token_cap=1000,
        current_tokens=0,
    )

    assert tokens == 4
    assert result and result["independent_verdict"] == "REVISE"
    prompt = "\n".join(item["content"] for item in captured)
    assert "secret-panel-signal" not in prompt


def test_minority_report_uses_originating_member_after_parallel_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    low = critique("Member 3", "REVISE")
    low.confidence = 0.1
    low.member_index = 2
    high = critique("Member 1", "APPROVED")
    high.confidence = 0.9
    high.member_index = 0
    used: list[str] = []

    def fake_call(selected, *_args, **_kwargs):
        used.append(selected.model)
        return (
            json.dumps(
                {
                    "minority_position": "dissent",
                    "dissent_reasoning": "reason",
                    "risk_if_ignored": "risk",
                    "confidence_in_dissent": 0.8,
                }
            ),
            2,
        )

    monkeypatch.setattr(council, "_call_llm_with_fallback", fake_call)
    report, tokens = council._run_minority_report(
        [member("a"), member("b"), member("c")],
        [low, high],
        member_timeout=30,
        token_cap=1000,
        current_tokens=0,
    )

    assert report and tokens == 2
    assert used == ["c"]


def test_chairman_fences_all_prior_model_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []

    def fake_call(_member, messages, *_args, **_kwargs):
        captured.extend(messages)
        return (
            json.dumps(
                {
                    "verdict": "APPROVED",
                    "rationale": "ok",
                    "issues": [],
                    "dissents": [],
                }
            ),
            2,
        )

    monkeypatch.setattr(council, "_call_llm_with_fallback", fake_call)
    result, tokens = council._run_phase_3(
        member("chair"),
        "prd",
        "spec",
        [critique("Member 1")],
        "RANKING-MODEL-OUTPUT",
        member_timeout=30,
        token_cap=1000,
        current_tokens=0,
        cascade_breaker_output={"risk": "CASCADE-MODEL-OUTPUT"},
    )

    assert result["verdict"] == "APPROVED" and tokens == 2
    prompt = next(item["content"] for item in captured if item["role"] == "user")
    assert 'name="Phase 1 Reviews"' in prompt
    assert 'name="Phase 2 Rankings"' in prompt
    assert 'name="Cascade Breaker"' in prompt


# ---------------------------------------------------------------------------
# Protocol routing
# ---------------------------------------------------------------------------


def install_phase_spies(
    monkeypatch: pytest.MonkeyPatch,
    phase_calls: list[tuple],
) -> None:
    reviews = [
        critique("Member 1", "APPROVED"),
        critique("Member 2", "REVISE"),
        critique("Member 3", "REVISE"),
    ]

    def phase0(*_args, **_kwargs):
        phase_calls.append(("compose",))
        return ([{"name": "A"}, {"name": "B"}, {"name": "C"}], 1)

    def phase1(*_args, **_kwargs):
        phase_calls.append(("independent",))
        return reviews, 3, set()

    def cross(*_args, **_kwargs):
        phase_calls.append(("cross",))
        return {"round": [{"updated_confidence": 0.5}] * 3}, 2

    def phase2(*_args, **_kwargs):
        phase_calls.append(("ranking",))
        return "rankings", 2

    def cascade(*_args, **_kwargs):
        phase_calls.append(("cascade",))
        return {"independent_verdict": "REVISE"}, 1

    def phase3(*_args, **_kwargs):
        phase_calls.append(("chairman",))
        return {
            "verdict": "APPROVED",
            "rationale": "chairman approved",
            "issues": [],
            "dissents": [],
        }, 2

    def minority(*_args, **_kwargs):
        phase_calls.append(("minority",))
        return {"minority_position": "dissent"}, 1

    monkeypatch.setattr(council, "_run_phase_0", phase0)
    monkeypatch.setattr(council, "_run_phase_1", phase1)
    monkeypatch.setattr(council, "_run_cross_examination", cross)
    monkeypatch.setattr(council, "_run_phase_2", phase2)
    monkeypatch.setattr(council, "_run_cascade_breaker", cascade)
    monkeypatch.setattr(council, "_run_phase_3", phase3)
    monkeypatch.setattr(council, "_run_minority_report", minority)
    monkeypatch.setattr(
        council,
        "_generate_html_report",
        lambda *_a, **_k: phase_calls.append(("html",)),
    )


@pytest.mark.parametrize(
    ("protocol", "expected_calls", "expected_verdict"),
    [
        (
            "deliberate",
            [
                ("compose",),
                ("independent",),
                ("cross",),
                ("ranking",),
                ("cascade",),
                ("chairman",),
                ("minority",),
                ("html",),
            ],
            "APPROVED",
        ),
        (
            "vote",
            [
                ("compose",),
                ("independent",),
                ("ranking",),
                ("minority",),
                ("html",),
            ],
            "REVISE",
        ),
        (
            "synthesize",
            [
                ("compose",),
                ("independent",),
                ("chairman",),
                ("minority",),
                ("html",),
            ],
            "APPROVED",
        ),
    ],
)
def test_protocols_execute_real_distinct_subsets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    protocol: str,
    expected_calls: list[tuple],
    expected_verdict: str,
) -> None:
    cfg = config(protocol)
    patch_config(monkeypatch, cfg)
    write_artifacts(tmp_path)
    calls: list[tuple] = []
    install_phase_spies(monkeypatch, calls)

    verdict = council.deliberate("task", str(tmp_path))

    assert calls == expected_calls
    assert verdict.verdict == expected_verdict


def test_adaptive_stopping_is_wired_and_skips_ranking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = config(
        "deliberate",
        compose=False,
        cascade_breaker=False,
        minority_report=False,
        html_report=False,
        adaptive_stopping=True,
    )
    patch_config(monkeypatch, cfg)
    write_artifacts(tmp_path)
    reviews = [critique("Member 1"), critique("Member 2"), critique("Member 3")]

    monkeypatch.setattr(council, "_run_phase_1", lambda *_a, **_k: (reviews, 3, set()))
    monkeypatch.setattr(
        council,
        "_run_cross_examination",
        lambda *_a, **_k: ({"round": [{"updated_confidence": 0.5}] * 3}, 2),
    )
    monkeypatch.setattr(council, "_should_stop_adaptive", lambda *_a, **_k: True)
    monkeypatch.setattr(
        council,
        "_run_phase_2",
        lambda *_a, **_k: pytest.fail("ranking should have been skipped"),
    )
    monkeypatch.setattr(
        council,
        "_run_phase_3",
        lambda *_a, **_k: (
            {"verdict": "APPROVED", "rationale": "stable", "issues": [], "dissents": []},
            1,
        ),
    )

    verdict = council.deliberate("task", str(tmp_path))

    assert verdict.verdict == "APPROVED"
    assert "adaptive" in verdict.rankings_snapshot.lower()


# ---------------------------------------------------------------------------
# Time and token budgets
# ---------------------------------------------------------------------------


def test_provider_call_rejects_post_call_token_cap_overrun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
        usage=SimpleNamespace(total_tokens=6),
    )
    monkeypatch.setattr("agent.auxiliary_client.call_llm", lambda **_kwargs: response)

    with pytest.raises(RuntimeError, match="token cap"):
        council._call_llm_with_fallback(
            member("a"),
            [{"role": "user", "content": "test"}],
            timeout=30,
            token_cap=10,
            current_total_tokens=5,
        )


def test_provider_call_releases_model_reservation_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
        usage=SimpleNamespace(total_tokens=2),
    )
    monkeypatch.setattr("agent.auxiliary_client.call_llm", lambda **_kwargs: response)
    active_models = {"already-active"}

    content, tokens = council._call_llm_with_fallback(
        member("a"),
        [{"role": "user", "content": "test"}],
        timeout=30,
        token_cap=100,
        current_total_tokens=0,
        active_models=active_models,
    )

    assert (content, tokens) == ("{}", 2)
    assert active_models == {"already-active"}


def test_remaining_overall_timeout_is_passed_to_cross_examination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = config(
        "deliberate",
        cascade_breaker=False,
        minority_report=False,
        html_report=False,
    )
    patch_config(monkeypatch, cfg)
    write_artifacts(tmp_path)
    reviews = [critique("Member 1"), critique("Member 2"), critique("Member 3")]
    observed: dict[str, float] = {}

    tick = {"value": -10.0}

    def monotonic() -> float:
        tick["value"] += 10.0
        return tick["value"]

    monkeypatch.setattr(council.time, "monotonic", monotonic)
    monkeypatch.setattr(council, "_run_phase_0", lambda *_a, **_k: ([], 0))
    monkeypatch.setattr(council, "_run_phase_1", lambda *_a, **_k: (reviews, 3, set()))

    def cross(*_args, **kwargs):
        observed["total_timeout"] = kwargs["total_timeout"]
        return {"round": []}, 0

    monkeypatch.setattr(council, "_run_cross_examination", cross)
    monkeypatch.setattr(council, "_run_phase_2", lambda *_a, **_k: ("rankings", 0))
    monkeypatch.setattr(
        council,
        "_run_phase_3",
        lambda *_a, **_k: (
            {"verdict": "APPROVED", "rationale": "ok", "issues": [], "dissents": []},
            0,
        ),
    )

    council.deliberate("task", str(tmp_path))

    assert 0 < observed["total_timeout"] < cfg.timeout_seconds
