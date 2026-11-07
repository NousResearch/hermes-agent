"""First-time Quick Setup reaches the first chat with fewer questions: the terminal backend stays
local without a menu, and messaging plus the gateway service move to a closing hint. Full Setup
still asks both, and the shared-metrics consent offer still runs."""

from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import patch

import pytest

QUICK, FULL = 0, 1  # indexes in setup._FIRST_TIME_MODES


def _fake_nous_flow(config):
    """The real flow logs in and saves provider ``nous`` with a Portal model."""
    from hermes_cli.config import save_config
    config["model"] = {"provider": "nous", "default": "Hermes-4-405B"}
    save_config(config)


def _run_first_time_setup(mode: int, **extra_patches):
    """Run ``hermes setup`` on a fresh home, picking *mode* and Enter everywhere else. Returns the
    questions asked through ``prompt_choice`` and the started mocks."""
    from hermes_cli.setup import run_setup_wizard

    questions = []

    def answer(question, choices, default=0, description=None):
        questions.append(question)
        return mode if question == "How would you like to set up Hermes?" else default

    targets = {
        "gateway": "hermes_cli.setup.setup_gateway",
        "service": "hermes_cli.gateway_setup_service.ensure_gateway_service",
        "summary": "hermes_cli.setup._print_setup_summary",
        "consent": "hermes_cli.observability.shared_metrics_consent.offer_consent_if_undecided",
        "recorded": "hermes_cli.observability.shared_metrics_events.record_setup_completed",
        **extra_patches,
    }
    with ExitStack() as stack:
        stack.enter_context(patch("hermes_cli.setup.is_interactive_stdin", return_value=True))
        stack.enter_context(patch("hermes_cli.setup.offer_agent_import", return_value=""))
        stack.enter_context(patch("hermes_cli.setup.prompt_choice", side_effect=answer))
        stack.enter_context(patch("hermes_cli.model_setup_flows._model_flow_nous", side_effect=_fake_nous_flow))
        mocks = {name: stack.enter_context(patch(target)) for name, target in targets.items()}
        run_setup_wizard(Namespace(non_interactive=False, section=None, reset=False,
                                   reconfigure=False, quick=False))
    return questions, mocks


def test_quick_setup_keeps_local_backend_without_asking():
    from hermes_cli.config import load_config

    questions, _ = _run_first_time_setup(QUICK)

    assert "Select terminal backend:" not in questions, questions
    assert load_config()["terminal"]["backend"] == "local"


def test_quick_setup_asks_no_messaging_or_service_question_and_prints_hint(capsys):
    questions, mocks = _run_first_time_setup(QUICK)

    assert not any("messaging" in q.lower() for q in questions), questions
    mocks["gateway"].assert_not_called()
    mocks["service"].assert_not_called()
    out = capsys.readouterr().out
    assert "Connect a chat app:        hermes setup gateway" in out
    assert "hermes setup terminal  (now: local)" in out
    mocks["summary"].assert_called_once()


def test_quick_setup_still_offers_metrics_consent():
    _, mocks = _run_first_time_setup(QUICK)

    mocks["consent"].assert_called_once()
    mocks["recorded"].assert_called_once()


@pytest.mark.parametrize("step", ["terminal", "gateway"])
def test_full_setup_still_asks_terminal_and_messaging(step):
    _, mocks = _run_first_time_setup(FULL, terminal="hermes_cli.setup.setup_terminal_backend",
                                     model="hermes_cli.setup.setup_model_provider",
                                     tools="hermes_cli.setup.setup_tools")

    mocks[step].assert_called_once()
    mocks["consent"].assert_called_once()
