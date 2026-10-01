from models import ModelRef
from models.selection import list_picker_candidates, picker_model_ids


def test_picker_candidates_trim_dedupe_and_filter_whitespace():
    refs = list_picker_candidates(
        "openai",
        ("  gpt-5.4  ", "gpt-5.4", "", "bad model", "gpt-5.4-mini"),
    )
    assert refs == (
        ModelRef("openai", "gpt-5.4"),
        ModelRef("openai", "gpt-5.4-mini"),
    )


def test_picker_candidates_can_preserve_self_hosted_whitespace_ids():
    assert picker_model_ids(
        "ollama",
        ("model with spaces", "other"),
        allow_whitespace=True,
    ) == ("model with spaces", "other")


def test_picker_current_model_is_promoted_only_when_present():
    assert picker_model_ids(
        "openai",
        ("a", "b", "c"),
        current_model="b",
    ) == ("b", "a", "c")
    assert picker_model_ids(
        "openai",
        ("a", "b"),
        current_model="missing",
    ) == ("a", "b")


def test_picker_candidates_use_provider_owned_normalization():
    refs = list_picker_candidates(
        "openai-codex",
        ("openai/gpt-5.4", "gpt-5.4"),
    )
    assert refs == (ModelRef("openai-codex", "gpt-5.4"),)
