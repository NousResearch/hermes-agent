"""The native task card's heading is operator-configurable (``platforms.slack.extra.task_card_title``).

The card heading is the only progress surface that names the bot in prose, so an operator running
the bot under its own name otherwise ships a card announcing the product name while the messages,
status line and avatar next to it use the new one. These assert the behaviour contract: an override
wins, an absent/blank one keeps the localized default, and the card and its text fallback never
disagree.
"""

import pytest

from gateway.config import PlatformConfig


def _adapter(extra):
    """A real SlackAdapter with only its config populated; no network, no token."""
    from plugins.platforms.slack.adapter import SlackAdapter

    adapter = object.__new__(SlackAdapter)
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    return adapter


@pytest.mark.parametrize("extra", [
    {"task_card_title": "Aria is working"},
    {"taskCardTitle": "Aria is working"},
    {"task_card_title": "  Aria is working  "},
])
def test_configured_title_is_used(extra):
    """Both spellings are accepted (as elsewhere in extra), and the value is trimmed."""
    assert _adapter(extra).native_task_card_title() == "Aria is working"


@pytest.mark.parametrize("extra", [
    {},
    {"task_card_title": ""},
    {"task_card_title": "   "},
    {"task_card_title": None},
    {"task_card_title": 42},
    {"native_task_cards": True},
])
def test_absent_or_unusable_title_defers_to_the_default(extra):
    """``None`` (not a literal) is what preserves the localized default and its translations."""
    assert _adapter(extra).native_task_card_title() is None


def test_non_dict_extra_does_not_raise():
    adapter = _adapter({})
    adapter.config = PlatformConfig(enabled=True, extra=None)
    assert adapter.native_task_card_title() is None


def test_absurdly_long_title_is_bounded():
    """Slack rejects an over-long heading; truncate rather than fail the whole card."""
    title = _adapter({"task_card_title": "A" * 5000}).native_task_card_title()
    assert title is not None and len(title) == 256


# ── the runner side: card heading and text fallback agree ────────────────────────────────────

def _state(adapter):
    from gateway.run_turn_runner import TurnRunner

    state = TurnRunner._TaskCardState(adapter=adapter)
    state.tasks = {"c1": {"id": "c1", "title": "terminal - ls", "status": "in_progress"}}
    state.task_order = ["c1"]
    return state


def test_runner_prefers_the_adapters_override():
    state = _state(_adapter({"task_card_title": "Aria is working"}))
    assert state.card_title() == "Aria is working"
    assert state.fallback_text().startswith("Aria is working\n")


def test_runner_falls_back_to_the_localized_default():
    from agent.i18n import t

    state = _state(_adapter({}))
    assert state.card_title() == t("gateway.progress.task_card_title")
    assert state.fallback_text().startswith(t("gateway.progress.task_card_title") + "\n")


def test_adapter_without_the_hook_is_unaffected():
    """The hook is additive: an adapter predating it must keep working (plugin compat contract)."""
    class _Old:
        pass

    from agent.i18n import t

    assert _state(_Old()).card_title() == t("gateway.progress.task_card_title")


def test_card_and_fallback_cannot_disagree():
    """One resolver feeds both lanes, so a native card and its text fallback always match."""
    state = _state(_adapter({"task_card_title": "Aria is working"}))
    assert state.fallback_text().split("\n")[0] == state.card_title()
