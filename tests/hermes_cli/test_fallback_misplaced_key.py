"""A fallback chain nested under ``model:`` is ignored — it must not be ignored SILENTLY.

Regression for the runtime half of #19691. That issue found the CLI reference documented
``fallback_providers`` "under ``model:``" while every loader reads the TOP-LEVEL key, and
warned that users hand-editing config from those docs "can place fallback settings under the
wrong key and get confusing failover behavior". It closed on the docs fix alone, so configs
already written in the wrong shape still resolve to an empty chain with no diagnostic.
"""

import logging

import pytest

from hermes_cli import fallback_config
from hermes_cli.fallback_config import get_fallback_chain

NOUS_ENTRY = {"provider": "nous", "model": "z-ai/glm-5.3-flash"}


@pytest.fixture(autouse=True)
def _reset_warn_state():
    """The warning is once-per-process; each test needs a clean slate. Tolerates the attribute
    being absent so these fail on the behaviour contract, not on a setup AttributeError."""
    getattr(fallback_config, "_warned_misplaced_fallback", set()).clear()
    yield
    getattr(fallback_config, "_warned_misplaced_fallback", set()).clear()


def test_chain_nested_under_model_is_empty_but_reported(caplog):
    """The contract: still not read (no silent behaviour change), but no longer silent."""
    config = {"model": {"default": "gpt-5.6-sol", "fallback_providers": [NOUS_ENTRY]}}

    with caplog.at_level(logging.WARNING, logger=fallback_config.__name__):
        chain = get_fallback_chain(config)

    assert chain == []
    assert "model.fallback_providers" in caplog.text
    assert "TOP-LEVEL" in caplog.text


def test_warning_names_the_key_without_leaking_entry_values(caplog):
    """Fallback dicts may carry credentials — the diagnostic must never echo them."""
    secret = "sk-must-not-appear-in-logs"
    config = {"model": {"fallback_providers": [{**NOUS_ENTRY, "api_key": secret}]}}

    with caplog.at_level(logging.WARNING, logger=fallback_config.__name__):
        get_fallback_chain(config)

    assert "model.fallback_providers" in caplog.text
    assert secret not in caplog.text


def test_correctly_placed_and_real_nested_keys_never_warn(caplog):
    """No false positives: the top-level key is correct, and delegation.fallback_providers is
    a genuine, read key (config_defaults) that must not be flagged as misplaced."""
    config = {
        "fallback_providers": [NOUS_ENTRY],
        "delegation": {"model": "", "fallback_providers": [NOUS_ENTRY]},
        "model": {"default": "gpt-5.6-sol", "fallback_providers": []},
    }

    with caplog.at_level(logging.WARNING, logger=fallback_config.__name__):
        chain = get_fallback_chain(config)

    assert [entry["provider"] for entry in chain] == ["nous"]
    assert caplog.text == ""


def test_warning_does_not_repeat_per_call(caplog):
    """get_fallback_chain runs per turn on some surfaces; the notice must not spam the log."""
    config = {"model": {"fallback_providers": [NOUS_ENTRY]}}

    with caplog.at_level(logging.WARNING, logger=fallback_config.__name__):
        for _ in range(5):
            get_fallback_chain(config)

    assert caplog.text.count("model.fallback_providers") == 1
