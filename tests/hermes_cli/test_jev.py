"""Behavior contracts for the ``hermes jev`` Decisions API client."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx

from hermes_cli import jev


def _question_set() -> dict:
    return {
        "ship": {
            "type": "noul",
            "instructions": "Should this change ship?",
            "criteria": {"true": "Safe and complete", "false": "Unsafe or incomplete"},
        },
        "model": {
            "type": "choice",
            "instructions": "Choose the best model.",
            "criteria": {"fast": "Optimize latency", "deep": "Optimize reasoning"},
        },
        "risk": {
            "type": "score",
            "instructions": "Place the change on this risk rubric.",
            "criteria": ["low", "medium", "high"],
        },
    }


def test_cmd_jev_posts_exact_contract_and_preserves_response(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps(_question_set()), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    expected_response = {
        "model": jev.JEV_MODEL,
        "answers": {
            "ship": {"type": "noul", "noul": 0.91},
            "model": {
                "type": "choice",
                "choice": "deep",
                "probabilities": {"fast": 0.2, "deep": 0.8},
                "confidence": 0.6,
            },
            "risk": {
                "type": "score",
                "score": 0.25,
                "probabilities": {"0": 0.75, "1": 0.25, "2": 0.0},
                "confidence": 0.7,
                "legend": {"0": "low", "1": "medium", "2": "high"},
            },
        },
        "usage": {"total_tokens": 42, "cost": 0.0004},
    }
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return expected_response

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(httpx, "post", fake_post)

    from hermes_cli.main import _build_cli_parser

    parser, _ = _build_cli_parser()
    args = parser.parse_args([
        "jev",
        '{"branch":"release","tests_green":true}',
        "--questions",
        str(questions_path),
    ])
    rc = args.func(args)

    assert rc == 0
    assert captured == {
        "url": "https://openrouter.ai/api/alpha/decisions",
        "headers": {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
        "json": {
            "model": "~typesafe/jev-latest",
            "state": {"branch": "release", "tests_green": True},
            "questions": _question_set(),
        },
        "timeout": 60.0,
    }
    assert json.loads(capsys.readouterr().out) == expected_response


def test_question_types_are_validated_before_network_access(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    called = False

    def fake_post(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(httpx, "post", fake_post)
    invalid = {
        "risk": {
            "type": "score",
            "instructions": "Rank the risk.",
            "criteria": {"low": "safe", "high": "dangerous"},
        }
    }

    rc = jev.cmd_jev(SimpleNamespace(state="release candidate", questions=json.dumps(invalid)))

    assert rc == 2
    assert called is False
    assert "ordered list" in capsys.readouterr().err


def test_task_classification_is_advisory_and_escalates_high_risk(monkeypatch):
    captured = {}

    def fake_request(payload, api_key):
        captured["payload"] = payload
        captured["api_key"] = api_key
        return {
            "answers": {
                "risk_level": {"type": "score", "score": 0.8},
                "agent_choice": {"type": "choice", "choice": "both"},
                "needs_review": {"type": "noul", "noul": 0.95},
                "model_class": {"type": "choice", "choice": "premium"},
            },
            "usage": {"cost": 0.0008},
        }

    monkeypatch.setattr(jev, "request_decision", fake_request)
    task = "Refactor the production auth module; security-sensitive and spans multiple files."

    result = jev.classify_task(task, api_key="test-key")

    assert captured == {
        "payload": {
            "model": jev.JEV_MODEL,
            "state": task,
            "questions": jev.TASK_CLASSIFICATION_QUESTIONS,
        },
        "api_key": "test-key",
    }
    assert result["risk_level"]["score"] >= jev.HUMAN_ESCALATION_MIN_RISK
    assert result["needs_review"]["value"] is True
    assert result["recommendation"]["disposition"] == "escalate_to_human"
    assert result["recommendation"]["advisory_only"] is True
    assert "does not approve or execute" in result["recommendation"]["message"]


def test_task_classification_applies_low_risk_thresholds(monkeypatch):
    monkeypatch.setattr(
        jev,
        "request_decision",
        lambda payload, api_key: {
            "answers": {
                "risk_level": {"type": "score", "score": 0.2},
                "agent_choice": {"type": "choice", "choice": "claude"},
                "needs_review": {"type": "noul", "noul": 0.09},
                "model_class": {"type": "choice", "choice": "cheap_fast"},
            }
        },
    )

    result = jev.classify_task("Summarize this pasted document in three sentences.", api_key="k")

    assert result["risk_level"]["score"] <= jev.AUTO_APPROVAL_MAX_RISK
    assert result["needs_review"] == {"noul": 0.09, "value": False}
    assert result["model_class"]["choice"] == "cheap_fast"
    assert (
        result["recommendation"]["disposition"]
        == "eligible_for_coordinator_auto_approval"
    )
