"""Exit-status contract for non-interactive ``hermes -z`` runs."""

from __future__ import annotations

import json

import pytest

from hermes_cli import oneshot


@pytest.mark.parametrize(
    "result",
    [
        {"failed": True, "partial": False, "completed": False},
        {"failed": False, "partial": True, "completed": False},
        {"failed": False, "partial": False, "completed": False},
    ],
)
def test_unsuccessful_result_is_nonzero_even_with_diagnostic_response(
    monkeypatch,
    capsys,
    result,
):
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_args, **_kwargs: ("The requested work was not completed.", result),
    )

    assert oneshot.run_oneshot("do the work") == 2
    assert capsys.readouterr().out == "The requested work was not completed.\n"


def test_completed_result_with_response_remains_successful(monkeypatch, capsys):
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_args, **_kwargs: (
            "done",
            {"failed": False, "partial": False, "completed": True},
        ),
    )

    assert oneshot.run_oneshot("do the work") == 0
    assert capsys.readouterr().out == "done\n"


def test_max_iteration_fallback_with_response_matches_cron_success_contract(
    monkeypatch,
    capsys,
    tmp_path,
):
    usage_path = tmp_path / "usage.json"
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_args, **_kwargs: (
            "Summary produced at the iteration limit.",
            {
                "failed": False,
                "partial": False,
                "completed": False,
                "turn_exit_reason": "max_iterations_reached(60/60)",
            },
        ),
    )

    assert oneshot.run_oneshot("do the work", usage_file=str(usage_path)) == 0
    assert capsys.readouterr().out == "Summary produced at the iteration limit.\n"
    report = json.loads(usage_path.read_text())
    assert report["completed"] is False
    assert report["exit_code"] == 0
    assert report["successful"] is True


def test_billing_failure_exit_matches_usage_report(monkeypatch, tmp_path):
    usage_path = tmp_path / "usage.json"
    billing_result = {
        "completed": False,
        "failed": True,
        "partial": False,
        "failure_reason": "billing",
        "turn_exit_reason": "provider_error",
    }
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_args, **_kwargs: (
            "Billing or credits exhausted: HTTP 402 Payment Required",
            billing_result,
        ),
    )

    assert (
        oneshot.run_oneshot(
            "do the work",
            usage_file=str(usage_path),
        )
        == 2
    )
    report = json.loads(usage_path.read_text())
    assert report["exit_code"] == 2
    assert report["successful"] is False
    assert report["failed"] is True
    assert report["partial"] is False
    assert report["failure_reason"] == "billing"
    assert report["turn_exit_reason"] == "provider_error"


def test_partial_result_preserves_raw_failed_but_reports_unsuccessful(
    monkeypatch,
    tmp_path,
):
    usage_path = tmp_path / "usage.json"
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_args, **_kwargs: (
            "Some work completed before the run stopped.",
            {
                "completed": False,
                "failed": False,
                "partial": True,
            },
        ),
    )

    assert oneshot.run_oneshot("do the work", usage_file=str(usage_path)) == 2
    report = json.loads(usage_path.read_text())
    assert report["exit_code"] == 2
    assert report["successful"] is False
    assert report["failed"] is False
    assert report["partial"] is True
