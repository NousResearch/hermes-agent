"""Tests for the unified model-selection guard registry."""

from unittest.mock import patch

from hermes_cli.model_selection_guards import (
    SelectionWarning,
    combined_message,
    combined_selection_warning,
    selection_warnings,
)


def test_no_guard_fires_on_ordinary_model():
    # No pricing data (no provider), no data-policy rule match.
    assert selection_warnings("some/ordinary-model") == []
    assert combined_selection_warning("some/ordinary-model") is None


def test_data_policy_guard_fires_through_registry():
    warnings = selection_warnings("muse-spark-1.2-contributor", provider="custom")
    kinds = [w.kind for w in warnings]
    assert "data_policy" in kinds
    w = next(w for w in warnings if w.kind == "data_policy")
    assert "train" in w.message.lower()
    assert w.title == "Data-Training Tier Warning"


def test_include_kinds_filters_guards():
    warnings = selection_warnings(
        "muse-spark-1.2-contributor",
        provider="custom",
        include_kinds=["cost"],
    )
    assert all(w.kind == "cost" for w in warnings)
    assert not any(w.kind == "data_policy" for w in warnings)


def test_combined_selection_warning_single():
    w = combined_selection_warning("muse-spark-1.2-contributor")
    assert w is not None
    assert w.kind == "data_policy"


def test_combined_selection_warning_merges_multiple():
    cost = SelectionWarning(
        kind="cost",
        title="Expensive Model Warning",
        model="m",
        provider="p",
        message="COST BLOCK",
    )
    policy = SelectionWarning(
        kind="data_policy",
        title="Data-Training Tier Warning",
        model="m",
        provider="p",
        message="POLICY BLOCK",
    )
    with patch(
        "hermes_cli.model_selection_guards._GUARDS",
        (lambda *a: cost, lambda *a: policy),
    ):
        merged = combined_selection_warning("m")
    assert merged is not None
    assert merged.kind == "multiple"
    assert "COST BLOCK" in merged.message
    assert "POLICY BLOCK" in merged.message


def test_misbehaving_guard_never_breaks_selection():
    def _boom(*args):
        raise RuntimeError("bad guard")

    with patch(
        "hermes_cli.model_selection_guards._GUARDS",
        (_boom,),
    ):
        assert selection_warnings("anything") == []


def test_combined_message_joins_blocks():
    a = SelectionWarning("cost", "t1", "m", "p", "AAA")
    b = SelectionWarning("data_policy", "t2", "m", "p", "BBB")
    assert combined_message([a, b]) == "AAA\n\nBBB"


def test_cost_guard_still_fires_through_registry():
    # The registry must preserve the existing cost-guard behavior; feed it
    # explicit model_info so no network lookup is needed.
    from agent.models_dev import ModelInfo

    info = ModelInfo(
        id="pricey/model",
        name="pricey/model",
        family="",
        provider_id="anthropic",
        cost_input=50.0,
        cost_output=200.0,
    )
    warnings = selection_warnings(
        "pricey/model", provider="anthropic", model_info=info
    )
    assert any(w.kind == "cost" for w in warnings)


# ── persisted interactive data-policy acknowledgement (#102048) ───────────────────────────────────
#
# ``security.allow_data_training_tiers_interactive`` records, per profile, that the user already
# accepted a tier whose vendor trains on prompts. Surfaces pass ``interactive=True`` for a
# user-driven selection so the confirmation stops repeating; an unattended run must keep refusing
# without the separate ``…_noninteractive`` key.

CONTRIBUTOR_MODEL = "muse-spark-1.2-contributor"


class _CostPayload:
    """Minimal duck-typed cost-guard payload: the registry only reads ``.message``."""

    message = "COST BLOCK"


def _home_with_security(monkeypatch, tmp_path, **security) -> None:
    """Point the process at a throwaway HERMES_HOME carrying this ``security`` section."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    lines = ["security:"]
    lines += [f"  {key}: {str(value).lower() if isinstance(value, bool) else value}"
              for key, value in security.items()]
    (tmp_path / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    from hermes_cli.config import invalidate_env_cache

    invalidate_env_cache()


def _acknowledged_home(monkeypatch, tmp_path) -> None:
    _home_with_security(monkeypatch, tmp_path, allow_data_training_tiers_interactive=True)


def test_interactive_ack_drops_only_the_data_policy_warning(monkeypatch, tmp_path):
    """Once acknowledged, an interactive pick stops asking for the data policy — cost still asks."""
    _acknowledged_home(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning", lambda *a, **k: _CostPayload())
    warnings = selection_warnings(CONTRIBUTOR_MODEL, provider="custom", interactive=True)
    assert [w.kind for w in warnings] == ["cost"]
    assert warnings[0].message == "COST BLOCK"


def test_interactive_selection_still_confirms_without_the_acknowledgement(monkeypatch, tmp_path):
    """Default (key absent): every interactive selection of a contributor tier confirms."""
    _home_with_security(monkeypatch, tmp_path)
    warnings = selection_warnings(CONTRIBUTOR_MODEL, provider="custom", interactive=True)
    assert any(w.kind == "data_policy" for w in warnings)


def test_unattended_selection_ignores_the_interactive_acknowledgement(monkeypatch, tmp_path):
    """The interactive ack is not a silent opt-out for unattended runs: only the
    ``…_noninteractive`` key unlocks those (see ``hermes_cli/main.py``)."""
    _acknowledged_home(monkeypatch, tmp_path)
    warnings = selection_warnings(CONTRIBUTOR_MODEL, provider="custom")
    assert any(w.kind == "data_policy" for w in warnings)


def test_combined_warning_is_silent_after_acknowledgement(monkeypatch, tmp_path):
    """The combined (drop-in) entry point used by pickers goes quiet for an acknowledged tier."""
    _acknowledged_home(monkeypatch, tmp_path)
    assert combined_selection_warning(
        CONTRIBUTOR_MODEL, provider="custom", interactive=True) is None
