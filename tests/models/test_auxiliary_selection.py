from models import ModelRef
from models.selection import (
    auxiliary_task_prefers_fast_model,
    select_auxiliary_fallback_model,
    select_auxiliary_model,
    select_fast_auxiliary_model,
    select_vision_auxiliary_model,
)


def _selected(selection):
    return selection.selected.ref if selection.selected is not None else None


def test_fast_task_policy_is_owned_by_selection_domain():
    assert auxiliary_task_prefers_fast_model("title_generation", True) is True
    assert auxiliary_task_prefers_fast_model("title_generation", False) is False
    assert auxiliary_task_prefers_fast_model("compression", True) is False
    assert auxiliary_task_prefers_fast_model("vision", True) is False
    assert auxiliary_task_prefers_fast_model(None, True) is False


def test_fast_selection_prefers_rolling_alias():
    selection = select_fast_auxiliary_model(
        "nous",
        (
            "z-ai/glm-5.2",
            "openai/gpt-5.4-mini",
            "~openai/gpt-mini-latest",
            "stepfun/step-3.7-flash:free",
        ),
    )
    assert _selected(selection) == ModelRef("nous", "~openai/gpt-mini-latest")


def test_fast_selection_skips_reasoning_batch_embedding_and_non_chat_siblings():
    selection = select_fast_auxiliary_model(
        "nous",
        (
            "openai/o3-mini",
            "openai/gpt-5.4-mini:batch",
            "sentence-transformers/all-minilm-l6-v2",
            "openai/gpt-4o-mini-tts",
            "openai/gpt-4o-mini-transcribe",
            "openai/gpt-4o-mini-search-preview",
            "google/gemini-3.6-flash",
        ),
    )
    assert _selected(selection) == ModelRef("nous", "google/gemini-3.6-flash")


def test_fast_selection_takes_newest_generation_within_family():
    selection = select_fast_auxiliary_model(
        "nous",
        ("openai/gpt-3.5-mini", "openai/gpt-9-mini", "openai/gpt-10-mini"),
    )
    assert _selected(selection) == ModelRef("nous", "openai/gpt-10-mini")


def test_fast_selection_respects_allowed_ids():
    selection = select_fast_auxiliary_model(
        "nous",
        ("vendor/gpt-5.4-mini", "vendor/gemini-3.6-flash"),
        allowed_model_ids={"vendor/gemini-3.6-flash"},
    )
    assert _selected(selection) == ModelRef("nous", "vendor/gemini-3.6-flash")


def test_auxiliary_fast_ladder_uses_dynamic_then_default_then_main():
    dynamic = select_auxiliary_model(
        "provider-x",
        main_model="main",
        resolved_aux_model="dynamic",
        default_aux_model="default",
        prefer_fast=True,
    )
    ordinary = select_auxiliary_model(
        "provider-x",
        main_model="main",
        resolved_aux_model="dynamic",
        default_aux_model="default",
    )
    assert _selected(dynamic) == ModelRef("provider-x", "dynamic")
    assert _selected(ordinary) == ModelRef("provider-x", "default")


def test_auxiliary_falls_back_to_main_when_provider_has_no_aux_default():
    selection = select_auxiliary_model("openai-codex", main_model="gpt-5.4")
    assert _selected(selection) == ModelRef("openai-codex", "gpt-5.4")


def test_auxiliary_policy_can_reject_provider_default_and_keep_main():
    selection = select_auxiliary_model(
        "nous",
        main_model="allowed/main",
        default_aux_model="blocked/default",
        allowed_model_ids={"allowed/main"},
    )
    assert _selected(selection) == ModelRef("nous", "allowed/main")


def test_vision_selection_precedence_is_explicit_default_main():
    explicit = select_vision_auxiliary_model(
        "zai",
        explicit_model="explicit",
        vision_default="glm-5.3-flash",
        main_model="glm-5.2",
        main_supports_vision=True,
    )
    default = select_vision_auxiliary_model(
        "zai",
        vision_default="glm-5.3-flash",
        main_model="glm-5.2",
        main_supports_vision=True,
    )
    main = select_vision_auxiliary_model(
        "provider-x",
        main_model="vision-main",
        main_supports_vision=None,
    )
    assert _selected(explicit) == ModelRef("zai", "explicit")
    assert _selected(default) == ModelRef("zai", "glm-5.3-flash")
    assert _selected(main) == ModelRef("provider-x", "vision-main")


def test_known_text_only_main_is_not_a_vision_candidate():
    selection = select_vision_auxiliary_model(
        "provider-x",
        main_model="text-only",
        main_supports_vision=False,
    )
    assert selection.selected is None


def test_fallback_selection_prefers_recommendation_then_provider_fallback():
    preferred = select_auxiliary_fallback_model(
        "nous",
        preferred_model="portal/recommended",
        fallback_model="provider/fallback",
    )
    fallback = select_auxiliary_fallback_model(
        "nous",
        preferred_model="stale/model",
        fallback_model="provider/fallback",
        excluded_model="stale/model",
    )
    assert _selected(preferred) == ModelRef("nous", "portal/recommended")
    assert _selected(fallback) == ModelRef("nous", "provider/fallback")


def test_fallback_selection_respects_policy_allowlist():
    selection = select_auxiliary_fallback_model(
        "nous",
        preferred_model="blocked/recommended",
        fallback_model="allowed/fallback",
        allowed_model_ids={"allowed/fallback"},
    )
    assert _selected(selection) == ModelRef("nous", "allowed/fallback")
