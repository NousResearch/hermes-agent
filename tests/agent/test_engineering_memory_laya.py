from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.engineering_memory_laya import (
    LayaDecisionEngine,
    LayaRuntimeConfig,
    calibrate_laya,
    download_laya,
)


class FakeRuntime:
    model_revision = "rev-1"
    device = "cpu"

    def __init__(self, payload=None):
        self.payload = payload or {"decisions": [{"question_id": "relevance", "choice": "yes", "probability": 0.95}]}

    def decide(self, state, questions):
        return self.payload


def test_laya_returns_typed_decisions_and_metadata() -> None:
    config = LayaRuntimeConfig(model_repo="convaiinnovations/laya", revision="rev-1")
    evaluation = LayaDecisionEngine(FakeRuntime(), config).evaluate({"component": "runner"}, [{"id": "relevance", "choices": ["yes", "no"]}])

    assert evaluation.diagnostic == "ok"
    assert evaluation.decisions[0].question_id == "relevance"
    assert evaluation.metadata["model_revision"] == "rev-1"


def test_missing_or_low_confidence_runtime_falls_back_without_blocking() -> None:
    config = LayaRuntimeConfig(model_repo="convaiinnovations/laya", revision="rev-1", min_confidence=0.8)
    low = FakeRuntime({"decisions": [{"question_id": "relevance", "choice": "yes", "probability": 0.2}]})
    evaluation = LayaDecisionEngine(low, config).evaluate({}, [])
    assert evaluation.decisions == ()
    assert evaluation.diagnostic == "laya_uncalibrated"

    unavailable = LayaDecisionEngine(None, config).evaluate({}, [])
    assert unavailable.decisions == ()
    assert unavailable.diagnostic == "laya_unavailable"


def test_download_requires_immutable_revision(monkeypatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="immutable revision"):
        download_laya("convaiinnovations/laya", "main", tmp_path)

    called = {}

    def fake_download(**kwargs):
        called.update(kwargs)
        return str(tmp_path / "model")

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", type("Hub", (), {"snapshot_download": staticmethod(fake_download)}))
    result = download_laya("convaiinnovations/laya", "0123456789abcdef0123456789abcdef01234567", tmp_path)
    assert result == tmp_path / "model"
    assert called["revision"].startswith("0123")


def test_calibration_artifact_records_fixture_and_model_metadata(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures.jsonl"
    fixtures.write_text(json.dumps({"state": {}, "questions": [], "expected": {}}) + "\n", encoding="utf-8")
    output = tmp_path / "calibration.json"
    artifact = calibrate_laya(FakeRuntime(), fixtures, output, model_revision="rev-1")

    assert artifact["model_revision"] == "rev-1"
    assert artifact["fixture_count"] == 1
    assert output.exists()
