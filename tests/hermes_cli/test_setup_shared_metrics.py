"""Setup-command coverage for consented Relay lifecycle metrics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _args(**overrides):
    return SimpleNamespace(
        non_interactive=False,
        portal=overrides.get("portal", False),
        quick=overrides.get("quick", False),
        reconfigure=False,
        reset=overrides.get("reset", False),
        section=overrides.get("section"),
    )


@pytest.mark.parametrize(
    ("args", "expected_mode"),
    [
        (_args(), "full"),
        (_args(portal=True), "portal"),
        (_args(quick=True), "quick"),
        (_args(reset=True), "reset"),
        (_args(section="model"), "section"),
    ],
)
def test_cmd_setup_records_a_bounded_success_lifecycle(
    monkeypatch,
    args,
    expected_mode,
):
    from hermes_cli import main
    from hermes_cli.observability import relay_shared_metrics

    attempt = object()
    started = []
    finished = []
    monkeypatch.setattr("hermes_cli.setup.run_setup_wizard", lambda _: True)
    monkeypatch.setattr(
        relay_shared_metrics,
        "start_setup_lifecycle",
        lambda mode: started.append(mode) or attempt,
    )
    monkeypatch.setattr(
        relay_shared_metrics,
        "finish_setup_lifecycle",
        lambda value, **kwargs: finished.append((value, kwargs)),
    )

    main.cmd_setup(args)

    assert started == [expected_mode]
    assert finished == [(attempt, {"outcome": "success", "failure_stage": "none"})]


def test_cmd_setup_records_a_returned_failure_without_error_details(monkeypatch):
    from hermes_cli import main
    from hermes_cli.observability import relay_shared_metrics

    attempt = object()
    finished = []
    monkeypatch.setattr("hermes_cli.setup.run_setup_wizard", lambda _: False)
    monkeypatch.setattr(
        relay_shared_metrics, "start_setup_lifecycle", lambda _: attempt
    )
    monkeypatch.setattr(
        relay_shared_metrics,
        "finish_setup_lifecycle",
        lambda value, **kwargs: finished.append((value, kwargs)),
    )

    main.cmd_setup(_args())

    assert finished == [(attempt, {"outcome": "failed", "failure_stage": "unknown"})]


@pytest.mark.parametrize(
    ("error", "expected_outcome"),
    [
        (RuntimeError("privacy-canary"), "failed"),
        (KeyboardInterrupt(), "cancelled"),
        (SystemExit(130), "cancelled"),
    ],
)
def test_cmd_setup_records_terminal_exceptions_and_reraises(
    monkeypatch,
    error,
    expected_outcome,
):
    from hermes_cli import main
    from hermes_cli.observability import relay_shared_metrics

    attempt = object()
    finished = []

    def fail(_):
        raise error

    monkeypatch.setattr("hermes_cli.setup.run_setup_wizard", fail)
    monkeypatch.setattr(
        relay_shared_metrics, "start_setup_lifecycle", lambda _: attempt
    )
    monkeypatch.setattr(
        relay_shared_metrics,
        "finish_setup_lifecycle",
        lambda value, **kwargs: finished.append((value, kwargs)),
    )

    with pytest.raises(type(error)):
        main.cmd_setup(_args())

    assert finished == [
        (attempt, {"outcome": expected_outcome, "failure_stage": "execution"})
    ]
    assert "privacy-canary" not in repr(finished)
