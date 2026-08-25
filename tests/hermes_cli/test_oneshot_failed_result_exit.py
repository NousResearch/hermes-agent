"""Oneshot provider failures must not report process success."""

from __future__ import annotations

import json

import hermes_cli.oneshot as oneshot


def test_failed_result_with_response_exits_nonzero(monkeypatch, tmp_path, capsys):
    usage_path = tmp_path / "usage.json"
    error = "HTTP 400: model is not supported"
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *args, **kwargs: (
            error,
            {
                "final_response": error,
                "failed": True,
                "partial": False,
                "completed": False,
                "api_calls": 1,
            },
        ),
    )

    assert oneshot.run_oneshot("hello", usage_file=str(usage_path)) == 2
    assert capsys.readouterr().out == f"{error}\n"
    assert json.loads(usage_path.read_text(encoding="utf-8"))["failed"] is True


def test_partial_result_with_response_preserves_success(monkeypatch, capsys):
    response = "Useful partial answer"
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *args, **kwargs: (
            response,
            {
                "final_response": response,
                "failed": False,
                "partial": True,
                "completed": False,
            },
        ),
    )

    assert oneshot.run_oneshot("hello") == 0
    assert capsys.readouterr().out == f"{response}\n"
