"""Tests for the per-slot reasoning_effort / max_tokens knobs in `hermes moa configure`.

Issues #102582 (reasoning_effort), #102584 (max_tokens), #102585 (edit an
existing slot's knobs without re-picking provider/model) — tracked by #102586.
"""

from __future__ import annotations

from hermes_cli import moa_cmd


def _provider(slug="openrouter", models=("model-a",), reasoning=True):
    return {
        "slug": slug,
        "name": slug,
        "models": list(models),
        "capabilities": {m: {"reasoning": reasoning} for m in models},
    }


def test_format_slot_shows_reasoning_and_max_tokens():
    slot = {"provider": "openrouter", "model": "model-a", "reasoning_effort": "high", "max_tokens": 512}
    assert moa_cmd._format_slot(slot) == "openrouter:model-a [reasoning=high, max_tokens=512]"


def test_format_slot_bare_when_no_tuning():
    slot = {"provider": "openrouter", "model": "model-a"}
    assert moa_cmd._format_slot(slot) == "openrouter:model-a"


def test_pick_new_slot_prompts_for_reasoning_effort_and_max_tokens(monkeypatch):
    """A fresh slot on a reasoning-capable model prompts for both knobs."""
    provider = _provider(reasoning=True)
    monkeypatch.setattr(moa_cmd, "_model_options", lambda: [provider])
    monkeypatch.setattr(moa_cmd, "_prompt_choice", lambda title, rows, default=0: 0)
    monkeypatch.setattr(moa_cmd, "line_input", lambda prompt_text: "512")
    monkeypatch.setattr(
        "hermes_cli.main._prompt_reasoning_effort_selection",
        lambda efforts, current_effort="": "high",
    )

    slot = moa_cmd._pick_slot(None)

    assert slot == {
        "provider": "openrouter",
        "model": "model-a",
        "reasoning_effort": "high",
        "max_tokens": "512",
    }


def test_pick_new_slot_skips_reasoning_prompt_when_unsupported(monkeypatch):
    """A model whose capabilities mark reasoning=False never sees the effort menu."""
    provider = _provider(reasoning=False)
    monkeypatch.setattr(moa_cmd, "_model_options", lambda: [provider])
    monkeypatch.setattr(moa_cmd, "_prompt_choice", lambda title, rows, default=0: 0)
    monkeypatch.setattr(moa_cmd, "line_input", lambda prompt_text: "")

    def _fail(*args, **kwargs):
        raise AssertionError("reasoning effort menu must not be shown for a non-reasoning model")

    monkeypatch.setattr("hermes_cli.main._prompt_reasoning_effort_selection", _fail)

    slot = moa_cmd._pick_slot(None)

    assert slot == {"provider": "openrouter", "model": "model-a"}


def test_pick_existing_slot_offers_edit_in_place(monkeypatch):
    """#102585: editing an already-picked slot must not re-walk provider/model."""
    provider = _provider(reasoning=True)
    monkeypatch.setattr(moa_cmd, "_model_options", lambda: [provider])

    seen_titles = []

    def fake_prompt_choice(title, rows, default=0):
        seen_titles.append(title)
        if title == "Edit slot":
            return 0  # keep provider/model
        raise AssertionError(f"unexpected prompt during edit-in-place: {title}")

    monkeypatch.setattr(moa_cmd, "_prompt_choice", fake_prompt_choice)
    monkeypatch.setattr(moa_cmd, "line_input", lambda prompt_text: "256")
    monkeypatch.setattr(
        "hermes_cli.main._prompt_reasoning_effort_selection",
        lambda efforts, current_effort="": "medium",
    )

    current = {"provider": "openrouter", "model": "model-a"}
    slot = moa_cmd._pick_slot(current)

    assert seen_titles == ["Edit slot"]
    assert slot == {
        "provider": "openrouter",
        "model": "model-a",
        "reasoning_effort": "medium",
        "max_tokens": "256",
    }


def test_pick_existing_slot_can_still_change_provider_model(monkeypatch):
    """Choosing 'Change provider/model' falls through to the full picker."""
    provider = _provider(reasoning=True)
    monkeypatch.setattr(moa_cmd, "_model_options", lambda: [provider])

    def fake_prompt_choice(title, rows, default=0):
        if title == "Edit slot":
            return 1  # change provider/model
        return 0

    monkeypatch.setattr(moa_cmd, "_prompt_choice", fake_prompt_choice)
    monkeypatch.setattr(moa_cmd, "line_input", lambda prompt_text: "")
    monkeypatch.setattr(
        "hermes_cli.main._prompt_reasoning_effort_selection",
        lambda efforts, current_effort="": None,
    )

    current = {"provider": "openrouter", "model": "model-a", "reasoning_effort": "low"}
    slot = moa_cmd._pick_slot(current)

    assert slot["provider"] == "openrouter"
    assert slot["model"] == "model-a"
    # No new selection made and no blank input given: prior effort is kept.
    assert slot["reasoning_effort"] == "low"
