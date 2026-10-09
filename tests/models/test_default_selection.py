from models import ModelRef
from models.selection import SelectionReason, select_default_model, select_nous_default_model


def _selected(selection):
    return selection.selected.ref if selection.selected is not None else None


def test_preferred_default_wins_when_present():
    selection = select_default_model(
        "openrouter",
        ("vendor/expensive", "z-ai/glm-5.2", "vendor/cheap"),
        preferred_model="z-ai/glm-5.2",
    )
    assert _selected(selection) == ModelRef("openrouter", "z-ai/glm-5.2")
    assert selection.reason == SelectionReason.PREFERRED_MATCH


def test_first_candidate_wins_when_preferred_is_absent():
    selection = select_default_model(
        "anthropic",
        ("claude-sonnet-5", "claude-opus-5"),
        preferred_model="not/listed",
    )
    assert _selected(selection) == ModelRef("anthropic", "claude-sonnet-5")
    assert selection.reason == SelectionReason.POLICY_DEFAULT


def test_empty_universe_has_no_default():
    selection = select_default_model("anthropic", (), preferred_model="not/listed")
    assert selection.selected is None
    assert selection.reason == SelectionReason.NO_ELIGIBLE_CANDIDATE


def test_preferred_may_be_trusted_when_provider_has_no_static_catalog():
    selection = select_default_model(
        "openrouter",
        (),
        preferred_model="z-ai/glm-5.2",
        allow_preferred_without_models=True,
    )
    assert _selected(selection) == ModelRef("openrouter", "z-ai/glm-5.2")
    assert selection.selected.catalogued is False


def test_deprioritized_default_loses_to_ordinary_candidate():
    selection = select_default_model(
        "nous",
        ("openai/gpt-5.4", "free/model"),
        preferred_model="openai/gpt-5.4",
        deprioritized_model_ids=("openai/gpt-5.4",),
    )
    assert _selected(selection) == ModelRef("nous", "free/model")


def test_deprioritized_candidate_is_kept_when_it_is_the_only_choice():
    selection = select_default_model(
        "nous",
        ("openai/gpt-5.4",),
        preferred_model="openai/gpt-5.4",
        deprioritized_model_ids=("openai/gpt-5.4",),
    )
    assert _selected(selection) == ModelRef("nous", "openai/gpt-5.4")


def test_duplicate_ids_are_deduplicated_without_reordering():
    selection = select_default_model(
        "provider-x",
        ("model-b", "model-b", "model-a"),
    )
    assert [candidate.ref.model for candidate in selection.eligible] == [
        "model-b",
        "model-a",
    ]


def test_nous_default_applies_policy_before_choice():
    selection = select_nous_default_model(
        ("vendor/blocked", "vendor/allowed"),
        policy_allowed_ids={"vendor/allowed"},
    )
    assert _selected(selection) == ModelRef("nous", "vendor/allowed")


def test_nous_free_tier_accepts_portal_free_model_missing_from_pricing():
    selection = select_nous_default_model(
        ("vendor/paid",),
        portal_recommended_model_ids=("vendor/new-free",),
        pricing={"vendor/paid": {"prompt": "1", "completion": "1"}},
        free_tier=True,
    )
    assert _selected(selection) == ModelRef("nous", "vendor/new-free")


def test_nous_free_tier_prefers_zero_credit_over_subscription_billed():
    selection = select_nous_default_model(
        ("openai/gpt-5.4", "free/model"),
        pricing={
            "openai/gpt-5.4": {
                "prompt": "1",
                "completion": "1",
                "billing_mode": "subscription",
            },
            "free/model": {"prompt": "0", "completion": "0"},
        },
        free_tier=True,
        preferred_model="openai/gpt-5.4",
    )
    assert _selected(selection) == ModelRef("nous", "free/model")


def test_nous_free_tier_keeps_subscription_model_when_it_is_only_choice():
    selection = select_nous_default_model(
        ("openai/gpt-5.4",),
        pricing={
            "openai/gpt-5.4": {
                "prompt": "1",
                "completion": "1",
                "billing_mode": "subscription",
            },
        },
        free_tier=True,
        preferred_model="openai/gpt-5.4",
    )
    assert _selected(selection) == ModelRef("nous", "openai/gpt-5.4")
