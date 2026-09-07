"""Regression test: Discord /model picker must surface ALL models for a
provider whose list exceeds 25 entries (e.g. Nous curated + Portal free
recommendations), not silently truncate the tail at 25 options.

Discord caps a single Select at 25 options and a View at 5 action rows. The
picker keeps 2 rows for Back/Cancel, so it must partition the model list
across up to 3 select menus. This is what makes free-tier ``:free`` Portal
picks (appended after the curated list) appear on Discord — they previously
fell off the 25-option cliff.
"""

from types import SimpleNamespace

from gateway.platforms.base import utf16_len
from plugins.platforms.discord.adapter import ModelPickerView


def _all_options(view: "ModelPickerView"):
    """Flatten every model select menu's options into (label, value).

    Detect selects by their ``model_model_select*`` custom_id (and presence of
    ``.options``) rather than class name — the discord mock in conftest uses a
    ``_FakeSelect`` class, while the real library uses ``discord.ui.Select``.
    """
    out = []
    for child in view.children:
        custom_id = getattr(child, "custom_id", "")
        if isinstance(custom_id, str) and custom_id.startswith("model_model_select"):
            out.extend((opt.label, opt.value) for opt in getattr(child, "options", []))
    return out


def test_nous_free_models_render_across_partitioned_selects():
    # 37 models: 32 curated + 5 free Portal recommendations appended at the tail
    # (the real-world shape that was getting clipped at 25 on Discord).
    models = [
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-4.8",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-haiku-4.5",
        "openai/gpt-5.6-sol",
        "openai/gpt-5.6-sol-pro",
        "openai/gpt-5.6-terra",
        "openai/gpt-5.6-terra-pro",
        "openai/gpt-5.6-luna",
        "openai/gpt-5.6-luna-pro",
        "openai/gpt-5.5",
        "openai/gpt-5.5-pro",
        "openai/gpt-5.4-mini",
        "google/gemini-3-pro-preview",
        "google/gemini-3.1-pro-preview",
        "google/gemini-3.5-flash",
        "x-ai/grok-4.5",
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash",
        "qwen/qwen3.7-max",
        "qwen/qwen3.7-plus",
        "qwen/qwen3.6-35b-a3b",
        "moonshotai/kimi-k2.6",
        "moonshotai/kimi-k2.7-code",
        "minimax/minimax-m3",
        "z-ai/glm-5.2",
        "z-ai/glm-5.1",
        "xiaomi/mimo-v2.5-pro",
        "tencent/hy3",
        "stepfun/step-3.7-flash",
        "nvidia/nemotron-3-super-120b-a12b",
        "sakana/fugu-ultra",
        # --- Portal free recommendations (the tail that was dropped) ---
        "tencent/hy3:free",
        "poolside/laguna-s-2.1:free",
        "inclusionai/ling-3.0-flash:free",
        "stepfun/step-3.7-flash:free",
        "poolside/laguna-xs-2.1:free",
    ]
    assert len(models) == 37

    view = ModelPickerView(
        providers=[
            {
                "slug": "nous",
                "name": "Nous Portal",
                "models": list(models),
                "total_models": len(models),
                "is_current": True,
            }
        ],
        current_model="tencent/hy3",
        current_provider="nous",
        session_key="session-1",
        on_model_selected=lambda *a, **k: None,
        allowed_user_ids={"123"},
    )
    view._selected_provider = "nous"
    view._build_model_select("nous")

    # The tail must render — this is the regression that was failing before
    # multi-select partitioning.
    rendered = [v for _label, v in _all_options(view)]
    assert "tencent/hy3:free" in rendered
    assert "stepfun/step-3.7-flash:free" in rendered
    assert "poolside/laguna-s-2.1:free" in rendered
    assert "inclusionai/ling-3.0-flash:free" in rendered
    assert "poolside/laguna-xs-2.1:free" in rendered

    # Every model must appear exactly once (no truncation, no dupes).
    assert sorted(rendered) == sorted(models)
    assert len(rendered) == 37

    # 37 models => 2 select rows of 25 + 12, plus Back/Cancel = 4 action rows.
    select_rows = [
        c for c in view.children
        if isinstance(getattr(c, "custom_id", ""), str)
        and c.custom_id.startswith("model_model_select")
    ]
    assert len(select_rows) == 2
    assert len(select_rows[0].options) == 25
    assert len(select_rows[1].options) == 12

    # Each option still respects Discord's 100-char utf16 field limit.
    for _label, value in _all_options(view):
        assert utf16_len(value) <= 100

    # No single select exceeds Discord's 25-option hard cap.
    for sel in select_rows:
        assert len(sel.options) <= 25


def test_small_provider_single_select_unchanged():
    """A <25-model provider still renders as a single select menu."""
    models = [f"m/{i}" for i in range(10)]
    view = ModelPickerView(
        providers=[
            {
                "slug": "emoji",
                "name": "Emoji",
                "models": models,
                "total_models": len(models),
                "is_current": False,
            }
        ],
        current_model="m/0",
        current_provider="emoji",
        session_key="session-1",
        on_model_selected=lambda *a, **k: None,
        allowed_user_ids={"123"},
    )
    view._selected_provider = "emoji"
    view._build_model_select("emoji")

    select_rows = [
        c for c in view.children
        if isinstance(getattr(c, "custom_id", ""), str)
        and c.custom_id.startswith("model_model_select")
    ]
    assert len(select_rows) == 1
    assert len(select_rows[0].options) == 10
