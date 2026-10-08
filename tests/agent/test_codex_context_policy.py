"""The opt-in Codex window remains the default, with a configurable large mode."""
from unittest.mock import patch

import pytest

from agent import model_metadata as mm
from agent.auxiliary_client import _is_codex_gpt54_or_gpt55
from hermes_cli.codex_models import _add_context_variants


@pytest.mark.parametrize("policy, bare_window, bare_autoraise, picker_variant", [
    ("advertised", None, True, True),
    ("large", 900_000, False, False),
])
def test_policy_agrees_across_resolver_picker_and_autoraise(policy, bare_window, bare_autoraise, picker_variant):
    with patch("hermes_cli.config.read_raw_config_readonly", return_value={"model": {"codex_context_policy": policy}}):
        assert mm._verified_codex_ctx_for_slug("gpt-6-sol") == bare_window
        assert mm._verified_codex_ctx_for_slug("gpt-6-sol-900k") == 900_000
        assert _is_codex_gpt54_or_gpt55("gpt-6-sol", "openai-codex") is bare_autoraise
        assert ("gpt-6-sol-900k" in _add_context_variants(["gpt-6-sol"])) is picker_variant
        assert mm.strip_codex_context_variant_suffix("gpt-6-sol-900k") == "gpt-6-sol"
        assert mm._verified_codex_ctx_for_slug("gpt-5.5") is None


def test_default_and_legacy_model_config_preserve_opt_in():
    for model in (None, "gpt-6-sol", {}, {"codex_context_policy": "bogus"}):
        with patch("hermes_cli.config.read_raw_config_readonly", return_value={"model": model}):
            assert mm.codex_context_policy() == "advertised"
            assert mm._verified_codex_ctx_for_slug("gpt-6-sol") is None


def test_profile_config_is_read_from_active_config():
    with patch("hermes_cli.config.read_raw_config_readonly", side_effect=[
        {"model": {"codex_context_policy": "large"}},
        {"model": {"codex_context_policy": "advertised"}},
    ]):
        assert mm.codex_context_policy() == "large"
        assert mm.codex_context_policy() == "advertised"
