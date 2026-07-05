"""Tests for the sub-provider drill-down mechanism in the model picker.

Two helpers live behind the feature:

* :func:`hermes_cli.model_switch._load_subprovider_picker_config` — reads the
  opt-in ``model_picker.group_by_subprovider`` user config and normalizes the
  slug list (lowercase, strip whitespace, drop empties).
* :func:`hermes_cli.model_switch._group_provider_models_by_subprovider` —
  groups a flat model catalog by the segment before the first ``/``, applies
  threshold guards (at least 3 slashed entries AND at least 50% of the
  catalog), and returns ``None`` when grouping is not worthwhile.

When both helpers pass, :func:`list_authenticated_providers` rewrites the
provider entry so ``models`` is replaced by sub-provider labels
("openai (12 models)") and the drill-down UI can take over the second step.

These tests pin the relationship between config, threshold, and output shape —
not snapshots of any specific catalog. Per AGENTS.md: behavior contracts,
never change-detector lists.
"""

from __future__ import annotations

import pytest

from hermes_cli import model_switch


# ── _load_subprovider_picker_config ─────────────────────────────────────────


def test_load_config_returns_empty_when_key_absent(monkeypatch):
    """No ``model_picker.group_by_subprovider`` key → empty list (opt-in)."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"provider": "anthropic"}},
    )

    assert model_switch._load_subprovider_picker_config() == []


def test_load_config_returns_empty_when_load_config_raises(monkeypatch):
    """A broken YAML must not crash the picker — it must default to []."""
    def _raise():
        raise RuntimeError("yaml parse error")

    monkeypatch.setattr("hermes_cli.config.load_config", _raise)

    assert model_switch._load_subprovider_picker_config() == []


def test_load_config_returns_empty_when_model_picker_not_a_dict(monkeypatch):
    """A scalar/list at ``model_picker`` must not crash — treat as absent."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model_picker": "openrouter"},
    )

    assert model_switch._load_subprovider_picker_config() == []


def test_load_config_normalizes_case_and_whitespace(monkeypatch):
    """Slugs are matched lowercase against ``ProviderDef.id``; preserve order.

    Duplicate entries are kept verbatim — the gate is membership-based, and
    a user who lists the same provider twice (e.g. once intentionally and
    once by accident) shouldn't get silent dedup at the config layer.
    """
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model_picker": {
                "group_by_subprovider": [" OpenRouter ", "OPENROUTER", " hugginGface  "],
            },
        },
    )

    # Order is preserved (matters for deterministic picker display), casing
    # and surrounding whitespace are stripped.
    assert model_switch._load_subprovider_picker_config() == [
        "openrouter",
        "openrouter",
        "huggingface",
    ]


def test_load_config_drops_empty_strings(monkeypatch):
    """An accidental blank entry must not register as the slug ``\"\"``."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model_picker": {"group_by_subprovider": ["", "  ", "openrouter"]}},
    )

    assert model_switch._load_subprovider_picker_config() == ["openrouter"]


def test_load_config_membership_check_is_lowercase(monkeypatch):
    """Producers of the list use lowercase; casing is normalized in one place.

    Gates downstream against ``ProviderDef.id`` (which is also lowercase), so
    a producer that forgets to lowercase still gets correctly matched.
    """
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model_picker": {"group_by_subprovider": ["OpenRouter"]}},
    )

    result = model_switch._load_subprovider_picker_config()

    assert "openrouter" in result
    assert "OpenRouter" not in result


# ── _group_provider_models_by_subprovider ──────────────────────────────────


def test_group_returns_none_on_empty_catalog():
    """Nothing to group → flat-list fallback (no drill-down)."""
    assert model_switch._group_provider_models_by_subprovider("openrouter", []) is None


def test_group_returns_none_when_below_count_threshold():
    """Fewer than 3 slashed models → no group (the threshold guards UX).

    The 3-absolute threshold prevents the drill-down UI from showing a
    single-sub list that would just be a flatter model picker.
    """
    # 2 slashed entries — strictly under the 3-absolute threshold.
    models = ["openai/gpt-4o", "anthropic/claude-3-5-sonnet"]
    assert model_switch._group_provider_models_by_subprovider("openrouter", models) is None


def test_group_returns_none_when_below_ratio_threshold():
    """Enough slashed in absolute terms, but they are a minority of the catalog.

    The 50% threshold means the provider is "really" a slash aggregator —
    a provider whose slash IDs are mixed with bare IDs drifts back to the
    flat list instead of being mislabelled as an aggregator.
    """
    # 3 slashed out of 7 total → meets the 3-absolute threshold but fails
    # the 50% ratio check.
    models = [
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet",
        "gemini/gemini-1.5-pro",
        "gpt-4o",
        "claude-3-5-sonnet",
        "gemini-1.5-pro",
        "mistral-large",
    ]
    assert model_switch._group_provider_models_by_subprovider("openrouter", models) is None


def test_group_groups_preserving_first_appearance_order():
    """Sub-provider order in the output mirrors first-occurrence in the catalog.

    The picker UI relies on a stable order across runs — alphabetical sort
    would scramble it. ``OrderedDict`` accumulation in the helper pins this.
    """
    models = [
        "google/gemini-1.5-pro",
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet",
        "openai/gpt-4o-mini",
        "google/gemini-1.5-flash",
        "anthropic/claude-3-opus",
        "openai/o1-preview",
    ]
    grouped = model_switch._group_provider_models_by_subprovider("openrouter", models)

    assert grouped is not None
    # First occurrence: google, then openai, then anthropic.
    assert grouped["subproviders"] == ["google", "openai", "anthropic"]
    # The label format used by both CLI and Telegram/Discord pickers.
    assert grouped["sub_labels"] == [
        "google (2 models)",
        "openai (3 models)",
        "anthropic (2 models)",
    ]


def test_group_sub_models_isolated_per_sub_no_cross_contamination():
    """``sub_models[sub]`` must contain only that sub's models — no leakage.

    The drill-down feeds ``sub_models[chosen_sub]`` straight to the model
    picker; any foreign model in there would land users on a 404 model.
    """
    models = [
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet",
        "google/gemini-1.5-pro",
        "openai/gpt-4o-mini",
    ]
    grouped = model_switch._group_provider_models_by_subprovider("openrouter", models)
    assert grouped is not None

    sub_models = grouped["sub_models"]
    assert set(sub_models.keys()) == {"openai", "anthropic", "google"}
    # Cross-contamination guard: ``openai/`` must not appear in another sub.
    assert sub_models["anthropic"] == ["anthropic/claude-3-5-sonnet"]
    assert sub_models["google"] == ["google/gemini-1.5-pro"]
    assert sub_models["openai"] == ["openai/gpt-4o", "openai/gpt-4o-mini"]


def test_group_sets_is_subprovider_picker_flag():
    """The flag is the contract the drill-down UI filters on.

    Without it, the CLI/TUI/Desktop/Telegram/Discord picker would treat
    a sub-labels list as a flat model list — and would crash on spaces.
    """
    models = [
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet",
        "google/gemini-1.5-pro",
    ]
    grouped = model_switch._group_provider_models_by_subprovider("openrouter", models)

    assert grouped is not None
    assert grouped["is_subprovider_picker"] is True


# ── Membership contract (relates config and threshold helpers) ─────────────
# The integration point — the ``if slug in _load_subprovider_picker_config()``
# branch in ``list_authenticated_providers`` — cannot be exercised in
# isolation without mocking the credential store and the models.dev registry,
# so its full coverage stays in ``test_list_picker_providers.py`` (which
# already mocks ``list_authenticated_providers``). What is testable here is
# the membership contract the branch relies on.


def test_config_and_grouping_compose_into_drill_down_payload(monkeypatch):
    """The config + grouping helpers compose end-to-end on raw inputs.

    Pin the contract: with a slug opted-in and a catalog above threshold,
    the two helpers together produce a payload that downstream UIs (CLI,
    TUI, Desktop, Telegram, Discord) consume without further processing.
    """
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model_picker": {"group_by_subprovider": ["huggingface"]}},
    )

    opted_in = model_switch._load_subprovider_picker_config()
    models = [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet",
        "google/gemini-1.5-pro",
    ]

    # Membership check: the slug gating uses ``in`` on the configured list.
    assert "huggingface" in opted_in
    assert "openrouter" not in opted_in

    grouped = model_switch._group_provider_models_by_subprovider("huggingface", models)

    # The grouped payload is the contract the drill-down UI consumes.
    assert grouped is not None
    assert grouped["is_subprovider_picker"] is True
    # ``sub_labels`` is what ``list_authenticated_providers`` writes into
    # ``models`` for the picker to render in stage 2.
    assert grouped["sub_labels"] == [
        "openai (2 models)",
        "anthropic (1 models)",
        "google (1 models)",
    ]
    # Each upstream's catalog is exactly its ``sub/*`` slice — no leakage.
    flat = [m for models in grouped["sub_models"].values() for m in models]
    assert sorted(flat) == sorted(models)
