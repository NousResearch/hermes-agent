"""gpt-6.1-sol must reach the Codex OAuth picker, offline and via synthesis."""

from hermes_cli.codex_models import (
    DEFAULT_CODEX_MODELS,
    _FORWARD_COMPAT_TEMPLATE_MODELS,
    _finalize_codex_models,
)


def test_gpt61_sol_heads_the_curated_offline_fallback() -> None:
    """The curated list is newest-first, so the newest Sol tier leads it."""
    assert DEFAULT_CODEX_MODELS[0] == "gpt-6.1-sol"


def test_gpt61_sol_is_synthesized_from_the_previous_sol_tier() -> None:
    """Live discovery returning only gpt-6-sol still yields the newer tier."""
    assert "gpt-6.1-sol" in _finalize_codex_models(["gpt-6-sol"])


def test_gpt61_sol_templates_follow_the_two_previous_tiers_convention() -> None:
    """Every entry in this table names the tiers it can be synthesized from;
    a new one that skipped them would never surface from live discovery.
    """
    templates = dict(_FORWARD_COMPAT_TEMPLATE_MODELS)
    assert templates["gpt-6.1-sol"] == ("gpt-6-sol", "gpt-5.6-sol")
