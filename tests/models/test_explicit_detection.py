from models import ModelRef
from models.selection import ExplicitDetectionFacts, select_detected_model


def test_current_live_catalog_match_wins():
    result = select_detected_model(
        "bare-model",
        "copilot",
        ExplicitDetectionFacts(
            current_catalog_model="vendor/bare-model",
            static_candidates=(ModelRef("anthropic", "bare-model"),),
            eligible_providers=("anthropic",),
        ),
    )
    assert result == ModelRef("copilot", "vendor/bare-model")


def test_current_native_vendor_ownership_keeps_provider():
    result = select_detected_model(
        "gpt-6-early",
        "openai-codex",
        ExplicitDetectionFacts(
            current_provider_owns_model=True,
            openrouter_candidate=ModelRef("openrouter", "openai/gpt-6-early"),
            eligible_providers=("openrouter",),
        ),
    )
    assert result == ModelRef("openai-codex", "gpt-6-early")


def test_named_provider_default_precedes_catalog_guesses():
    named = ModelRef("nous", "nous/default")
    result = select_detected_model(
        "nous",
        "openai",
        ExplicitDetectionFacts(
            named_provider_candidate=named,
            static_candidates=(ModelRef("anthropic", "nous"),),
            eligible_providers=("anthropic",),
        ),
    )
    assert result == named


def test_first_eligible_static_candidate_wins():
    facts = ExplicitDetectionFacts(
        static_candidates=(
            ModelRef("openai-api", "gpt-x"),
            ModelRef("openai-codex", "gpt-x"),
        ),
        eligible_providers=("openai-codex",),
    )
    assert select_detected_model("gpt-x", "anthropic", facts) == ModelRef(
        "openai-codex", "gpt-x"
    )


def test_current_static_ownership_stops_after_ineligible_native_guess():
    facts = ExplicitDetectionFacts(
        static_candidates=(ModelRef("openai-api", "gpt-x"),),
        current_static_owns_model=True,
    )
    assert select_detected_model("gpt-x", "copilot", facts) == ModelRef(
        "copilot", "gpt-x"
    )


def test_openrouter_candidate_uses_eligibility():
    candidate = ModelRef("openrouter", "anthropic/model-x")
    facts = ExplicitDetectionFacts(
        openrouter_candidate=candidate,
        eligible_providers=("openrouter",),
    )
    assert select_detected_model("model-x", "openai", facts) == candidate


def test_auto_selection_returns_first_guess_when_nothing_is_eligible():
    first = ModelRef("anthropic", "model-x")
    facts = ExplicitDetectionFacts(
        static_candidates=(first,),
        openrouter_candidate=ModelRef("openrouter", "vendor/model-x"),
        allow_first_guess=True,
    )
    assert select_detected_model("model-x", "auto", facts) == first


def test_declared_provider_prefix_is_selection_even_without_credentials():
    declared = ModelRef("relay", "model-x")
    facts = ExplicitDetectionFacts(declared_provider_candidate=declared)
    assert select_detected_model("relay/model-x", "openai", facts) == declared


def test_no_detection_facts_means_stay_with_caller():
    assert select_detected_model(
        "model-x", "openai", ExplicitDetectionFacts()
    ) is None
