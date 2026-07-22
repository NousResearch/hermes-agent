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
        (_args(), "interactive"),
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


@pytest.mark.parametrize(
    ("completed", "expected_code", "expected_outcome", "expected_stage"),
    [
        (True, 0, "success", "none"),
        (False, 1, "failed", "unknown"),
    ],
)
def test_portal_alias_records_the_same_setup_lifecycle(
    monkeypatch,
    completed,
    expected_code,
    expected_outcome,
    expected_stage,
):
    from hermes_cli import portal_cli
    from hermes_cli.observability import relay_shared_metrics

    attempt = object()
    started = []
    finished = []
    monkeypatch.setattr(portal_cli, "load_config", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.setup._run_portal_one_shot",
        lambda _: completed,
    )
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

    assert portal_cli._cmd_login(SimpleNamespace()) == expected_code
    assert started == ["portal"]
    assert finished == [
        (
            attempt,
            {"outcome": expected_outcome, "failure_stage": expected_stage},
        )
    ]


def test_first_time_quick_setup_reports_provider_failure(monkeypatch, tmp_path):
    from hermes_cli import main, setup

    def fail_provider(_):
        raise RuntimeError("privacy-canary")

    monkeypatch.setattr(main, "_model_flow_nous", fail_provider)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr(setup, "setup_terminal_backend", lambda _: None)
    monkeypatch.setattr(setup, "_apply_default_agent_settings", lambda _: None)
    monkeypatch.setattr(setup, "save_config", lambda _: None)
    monkeypatch.setattr(setup, "prompt_choice", lambda *args, **kwargs: 1)
    monkeypatch.setattr(setup, "_print_setup_summary", lambda *args: None)

    assert setup._run_first_time_quick_setup({}, tmp_path, False) is False
