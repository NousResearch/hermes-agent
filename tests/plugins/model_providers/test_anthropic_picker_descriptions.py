"""Anthropic picker metadata stays identical to the native Messages wire contract."""

from providers import get_provider_profile
from hermes_cli.inventory import _apply_capabilities


MODELS = [
    "claude-fable-5.1",
    "claude-fable-5-1",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-20250514",
    "claude-haiku-4-5-20251001",
    "unrecognized-route",
]


def _descriptions():
    profile = get_provider_profile("anthropic")
    assert profile is not None
    return profile.describe_models(model_ids=MODELS)


def test_aliases_collapse_to_one_friendly_family():
    descriptions = _descriptions()
    dotted = descriptions["claude-fable-5.1"]
    hyphenated = descriptions["claude-fable-5-1"]

    assert dotted["display_name"] == hyphenated["display_name"] == "Claude Fable 5.1"
    assert dotted["family_id"] == hyphenated["family_id"] == "claude-fable-5.1"
    assert "unrecognized-route" not in descriptions


def test_adaptive_families_publish_only_wire_efforts():
    descriptions = _descriptions()

    assert descriptions["claude-opus-5"] == {
        "display_name": "Claude Opus 5",
        "family_id": "claude-opus-5",
        "fast": True,
        "reasoning": True,
        "reasoning_control": "adjustable",
        "can_disable_reasoning": True,
        "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"],
    }
    assert descriptions["claude-opus-4-7"]["reasoning_efforts"] == [
        "low", "medium", "high", "xhigh", "max",
    ]
    assert descriptions["claude-opus-4-6"]["reasoning_efforts"] == [
        "low", "medium", "high", "max",
    ]
    assert descriptions["claude-sonnet-4-6"]["reasoning_efforts"] == [
        "low", "medium", "high", "max",
    ]


def test_mandatory_legacy_and_unsupported_controls_match_runtime():
    descriptions = _descriptions()

    assert descriptions["claude-fable-5.1"]["can_disable_reasoning"] is False
    assert descriptions["claude-opus-4-5-20251101"]["reasoning_efforts"] == [
        "low", "medium", "high", "xhigh",
    ]
    assert descriptions["claude-opus-4-5-20251101"]["can_disable_reasoning"] is True
    assert descriptions["claude-sonnet-4-20250514"]["reasoning_efforts"] == [
        "low", "medium", "high", "xhigh",
    ]
    assert descriptions["claude-haiku-4-5-20251001"] == {
        "display_name": "Claude Haiku 4.5",
        "family_id": "claude-haiku-4-5-20251001",
        "fast": False,
        "reasoning": False,
        "reasoning_control": "unsupported",
        "can_disable_reasoning": False,
        "reasoning_efforts": [],
    }


def test_fast_is_only_published_for_native_supported_families():
    descriptions = _descriptions()

    assert descriptions["claude-opus-5"]["fast"] is True
    assert descriptions["claude-opus-4-8"]["fast"] is True
    assert descriptions["claude-opus-4-7"]["fast"] is False
    assert descriptions["claude-opus-4-6"]["fast"] is False


def test_inventory_forwards_descriptors_and_collapsible_family_ids():
    row = {"slug": "anthropic", "models": list(MODELS[:-1])}

    _apply_capabilities([row])

    capabilities = row["capabilities"]
    assert capabilities["claude-opus-4-6"]["reasoning_efforts"] == [
        "low", "medium", "high", "max",
    ]
    assert capabilities["claude-haiku-4-5-20251001"]["reasoning_control"] == "unsupported"
    assert capabilities["claude-fable-5-1"]["family_id"] == "claude-fable-5.1"
    assert len({capabilities[model]["family_id"] for model in row["models"]}) == len(row["models"]) - 1
