"""Regression test: Discord /model picker must surface ALL models for a
provider whose list exceeds 25 entries (e.g. Nous curated + Portal free
recommendations), not silently truncate the tail.

The picker is a paginated button grid: 4 rows × 5 buttons per page (20
models/page) with a persistent 5th row for Prev/Next navigation. A select-menu
picker caps at 3 dropdowns × 25 options (75 models) inside Discord's 5-row
View limit — anything past that clips. Buttons page arbitrarily deep, so even
providers with hundreds of models (NVIDIA NIM: 139) are fully reachable.

The :free Portal tail is the original regression: it used to fall off the
25-option cliff, and the multi-select partition that replaced it capped at 75.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncio

from gateway.platforms.base import utf16_len
from plugins.platforms.discord.adapter import ModelPickerView


def _model_buttons(view: "ModelPickerView"):
    """Return (label, value) for every model button in the grid.

    Model buttons carry their model id in the callback default arg; nav/action
    buttons (row 4) are excluded because they use bound methods instead.
    """
    out = []
    for child in view.children:
        row = getattr(child, "row", None)
        if row is None or row >= 4:
            continue
        cb = getattr(child, "callback", None)
        defaults = getattr(cb, "__defaults__", None) or ()
        if defaults:
            out.append((child.label, defaults[0]))
    return out


def _nav_labels(view: "ModelPickerView"):
    """Labels of the row-4 action buttons (Prev/Next/Free/Back/Cancel)."""
    return [
        child.label
        for child in view.children
        if getattr(child, "row", None) == 4 and getattr(child, "label", None)
    ]


def _make_view(models, *, free_models=None, current="tencent/hy3"):
    view = ModelPickerView(
        providers=[
            {
                "slug": "nous",
                "name": "Nous Portal",
                "models": list(models),
                **({"free_models": list(free_models)} if free_models else {}),
                "total_models": len(models),
                "is_current": True,
            }
        ],
        current_model=current,
        current_provider="nous",
        session_key="session-1",
        on_model_selected=lambda *a, **k: None,
        allowed_user_ids={"123"},
    )
    view._selected_provider = "nous"
    return view


def _collect_all(view, total_pages):
    """Walk every page and collect all rendered model ids."""
    rendered = []
    for page in range(total_pages):
        view._model_page = page
        view._build_model_select("nous")
        rendered.extend(v for _label, v in _model_buttons(view))
    return rendered


def test_nous_free_models_render_across_pages():
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

    view = _make_view(models)
    view._model_page = 0
    view._build_model_select("nous")

    # Page 1 shows 20 of 37; the tail lives on page 2.
    assert len(_model_buttons(view)) == 20
    assert view._model_pages == 2

    # Every model must be reachable across pages — the original regression.
    rendered = _collect_all(view, view._model_pages)
    assert len(rendered) == 37
    assert "tencent/hy3:free" in rendered
    assert "stepfun/step-3.7-flash:free" in rendered
    assert "poolside/laguna-s-2.1:free" in rendered
    assert "inclusionai/ling-3.0-flash:free" in rendered
    assert "poolside/laguna-xs-2.1:free" in rendered

    # No truncation, no dupes.
    assert sorted(rendered) == sorted(models)

    # Navigation buttons present on page 1 of a multi-page provider.
    view._model_page = 0
    view._build_model_select("nous")
    nav = _nav_labels(view)
    assert "◀ Prev" not in nav  # first page: nothing to go back to
    assert "Next ▶" in nav

    # ... and on the last page, Next disappears but Prev remains.
    view._model_page = view._model_pages - 1
    view._build_model_select("nous")
    nav = _nav_labels(view)
    assert "◀ Prev" in nav
    assert "Next ▶" not in nav

    # Each button label respects Discord's 80-char limit.
    for label, _value in _model_buttons(view):
        assert utf16_len(label) <= 80


def test_models_sorted_alphabetically_and_deduped():
    # Deliberately unsorted + containing a duplicate (NVIDIA NIM returns
    # duplicate entries; duplicate button values cause HTTP 400 on submit).
    models = [
        "z-ai/glm-5.2",
        "openai/gpt-5.6-sol",
        "deepseek/deepseek-v4-flash",
        "openai/gpt-5.6-sol",
        "anthropic/claude-fable-5",
        "openai/gpt-5.6-sol",
    ]
    view = _make_view(models)
    view._model_page = 0
    view._build_model_select("nous")

    rendered = [v for _label, v in _model_buttons(view)]
    assert rendered == sorted(
        ["anthropic/claude-fable-5", "deepseek/deepseek-v4-flash",
         "openai/gpt-5.6-sol", "z-ai/glm-5.2"]
    )
    assert len(rendered) == 4  # deduped from 6
    assert view._model_pages == 1


def test_small_provider_single_page_no_nav():
    """A <20-model provider renders one page with no pagination buttons."""
    models = [f"m/{i}" for i in range(10)]
    view = _make_view(models)
    view._build_model_select("nous")

    assert len(_model_buttons(view)) == 10
    assert view._model_pages == 1
    nav = _nav_labels(view)
    assert "◀ Prev" not in nav
    assert "Next ▶" not in nav
    assert "◀ Back" in nav
    assert "Cancel" in nav


def test_free_toggle_shows_free_subset_first():
    """Free-only mode (default) shows the free subset; toggling shows all."""
    models = [
        "anthropic/claude-fable-5",
        "openai/gpt-5.6-sol",
        "tencent/hy3:free",
        "stepfun/step-3.7-flash:free",
    ]
    free = ["tencent/hy3:free", "stepfun/step-3.7-flash:free"]
    view = _make_view(models, free_models=free)
    view._model_page = 0
    view._build_model_select("nous")

    # Default is free-only.
    rendered = [v for _label, v in _model_buttons(view)]
    assert rendered == sorted(free)
    nav = _nav_labels(view)
    assert "🆓 Free" in nav

    # Toggle to all models.
    view._free_only = False
    view._build_model_select("nous")
    rendered = [v for _label, v in _model_buttons(view)]
    assert rendered == sorted(models)
    nav = _nav_labels(view)
    assert "💳 All" in nav


def test_free_filter_note_in_embed():
    """The embed reports 'showing N free of M' so the filter is self-explanatory."""
    models = [
        "anthropic/claude-fable-5",
        "openai/gpt-5.6-sol",
        "tencent/hy3:free",
        "stepfun/step-3.7-flash:free",
    ]
    free = ["tencent/hy3:free", "stepfun/step-3.7-flash:free"]
    view = _make_view(models, free_models=free)

    captured = {}

    async def edit_message(**kwargs):
        captured["description"] = kwargs["embed"].description

    interaction = SimpleNamespace(
        user=SimpleNamespace(id=123),
        channel_id=456,
        data={"values": ["nous"]},
        response=SimpleNamespace(
            defer=AsyncMock(),
            send_message=AsyncMock(),
            edit_message=AsyncMock(side_effect=edit_message),
        ),
        edit_original_response=AsyncMock(),
    )

    # Free-only default: "showing 2 free of 4 models"
    view._free_only = True
    view._model_page = 0
    view._selected_provider = "nous"
    asyncio.get_event_loop().run_until_complete(
        view._render_model_page(interaction)
    )
    assert "Showing 2 free of 4 models" in captured["description"]

    # All-models toggle: "All 4 models"
    view._free_only = False
    asyncio.get_event_loop().run_until_complete(
        view._render_model_page(interaction)
    )
    assert "All 4 models" in captured["description"]


def test_139_model_provider_fully_reachable():
    """NVIDIA NIM returns 139 models — the case the 75-cap partition clips."""
    models = [f"nvidia/nim-model-{i:03d}" for i in range(139)]
    view = _make_view(models)
    view._model_page = 0
    view._build_model_select("nous")

    assert view._model_pages == 7  # ceil(139/20)
    rendered = _collect_all(view, view._model_pages)
    assert len(rendered) == 139
    assert rendered[0] == "nvidia/nim-model-000"
    assert rendered[-1] == "nvidia/nim-model-138"
