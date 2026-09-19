"""``model.context_length`` is a visible pin: one warning when it disagrees with the advertised
window, and a ``(pinned)`` label wherever the window is rendered (#66168)."""
import logging

from agent import context_pin
from agent.context_pin import context_pin_suffix, is_context_pinned, warn_once_on_pin_disagreement


def test_pin_disagreement_warns_once_and_keeps_pin(monkeypatch, caplog):
    monkeypatch.setattr(context_pin, "_warned_pins", set())
    monkeypatch.setattr(context_pin, "advertised_context_length", lambda model, base_url="": 200_000)
    with caplog.at_level(logging.WARNING, logger="agent.context_pin"):
        assert warn_once_on_pin_disagreement("claude-sonnet-4", "", 999_000) is True
        # Second session start in the same process: silent.
        assert warn_once_on_pin_disagreement("claude-sonnet-4", "", 999_000) is False
        # Control: a pin that matches the advertised window is not a disagreement.
        assert warn_once_on_pin_disagreement("claude-sonnet-4", "", 200_000) is False
    warnings = [r for r in caplog.records if "model.context_length pins" in r.getMessage()]
    assert len(warnings) == 1
    assert "999,000" in warnings[0].getMessage() and "200,000" in warnings[0].getMessage()


def test_pinned_label_only_when_shown_value_is_the_pin():
    assert is_context_pinned(999_000, 999_000) is True
    assert context_pin_suffix(999_000, 999_000) == " (pinned)"
    # Route changed / pin dropped, unpinned, or non-int pins never label.
    assert context_pin_suffix(200_000, 999_000) == ""
    assert context_pin_suffix(200_000, None) == ""
    assert context_pin_suffix(1, True) == ""
