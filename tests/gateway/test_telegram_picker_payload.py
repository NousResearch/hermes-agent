"""Regression tests: exact 3-provider Telegram picker payload."""
from __future__ import annotations

from hermes_cli.telegram_picker_menu import build_three_provider_payload

ZEN = {"slug": "opencode-zen", "name": "OpenCode Zen",
       "models": ["zen-a", "shared-x"], "total_models": 2, "is_current": True}
GO = {"slug": "opencode-go", "name": "OpenCode Go",
      "models": ["go-a"], "total_models": 1}
OR_CURATED = {"slug": "openrouter", "name": "OpenRouter",
              "models": ["or-curated-1"], "total_models": 1}
OR_CUSTOM = {"slug": "custom:openrouter", "name": "OpenRouter",
             "models": ["or-curated-1", "or-live-2", "or-live-3"],
             "total_models": 3, "is_user_defined": True,
             "api_url": "https://openrouter.ai/api/v1"}
MOA = {"slug": "moa", "name": "Mixture of Agents", "models": ["m"], "total_models": 1}
DEEPSEEK = {"slug": "deepseek", "name": "DeepSeek", "models": ["d"], "total_models": 1}

ALL_ROWS = [MOA, GO, OR_CUSTOM, ZEN, OR_CURATED, DEEPSEEK]


def _by_slug(rows):
    return {r["slug"]: r for r in rows}


def test_payload_is_exactly_the_three_providers_in_order():
    out = build_three_provider_payload(ALL_ROWS)
    assert [r["slug"] for r in out] == ["opencode-zen", "opencode-go", "openrouter"]


def test_payload_drops_moa_and_deepseek_and_custom_alias():
    out = build_three_provider_payload(ALL_ROWS)
    assert "moa" not in _by_slug(out)
    assert "deepseek" not in _by_slug(out)
    assert "custom:openrouter" not in _by_slug(out)


def test_payload_merges_alias_rows_without_duplicate_models():
    or_row = _by_slug(build_three_provider_payload(ALL_ROWS))["openrouter"]
    assert or_row["models"] == ["or-curated-1", "or-live-2", "or-live-3"]
    assert or_row["total_models"] == 3
    assert len(or_row["models"]) == len(set(or_row["models"]))


def test_payload_keeps_canonical_runtime_identity_and_current_flag():
    rows = _by_slug(build_three_provider_payload(ALL_ROWS))
    assert rows["opencode-zen"]["is_current"] is True
    assert rows["openrouter"].get("is_user_defined") is False
    # Provider slug must stay the canonical one the runtime switch understands.
    assert rows["openrouter"]["slug"] == "openrouter"


def test_payload_omits_providers_without_models():
    out = build_three_provider_payload([MOA, DEEPSEEK])
    assert out == []


def test_payload_handles_missing_alias_rows():
    out = build_three_provider_payload([ZEN, GO, OR_CURATED])
    assert [r["slug"] for r in out] == ["opencode-zen", "opencode-go", "openrouter"]