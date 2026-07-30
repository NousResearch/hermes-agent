"""Exit-status contract for non-interactive ``hermes -z`` runs."""

from __future__ import annotations

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
