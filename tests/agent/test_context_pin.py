"""``model.context_length`` is a visible pin: one warning when it disagrees with the advertised
window, and a ``(pinned)`` label wherever the window is rendered (#66168)."""
import logging

from agent import context_pin
from agent.context_pin import (
    advertised_context_length, context_pin_suffix, is_context_pinned, warn_once_on_pin_disagreement,
)


def test_pin_disagreement_warns_once_and_keeps_pin(monkeypatch, caplog):
    monkeypatch.setattr(context_pin, "_warned_pins", set())
    monkeypatch.setattr(context_pin, "advertised_context_length", lambda model, base_url="": 200_000)
    with caplog.at_level(logging.WARNING, logger="agent.context_pin"):
        assert warn_once_on_pin_disagreement("claude-sonnet-4", "", 999_000) is True
        # Second session start in the same process: silent.
        assert warn_once_on_pin_disagreement("claude-sonnet-4", "", 999_000) is False
        # Control: a pin that matches the advertised window is not a disagreement.
        assert warn_once_on_pin_disagreement("claude-sonnet-4", "", 200_000) is False
    warnings = [r for r in caplog.records if r.name == "agent.context_pin" and r.levelno >= logging.WARNING]
    assert len(warnings) == 1


def test_pinned_label_only_when_shown_value_is_the_pin():
    assert is_context_pinned(999_000, 999_000) is True
    assert context_pin_suffix(999_000, 999_000) == " (pinned)"
    # Route changed / pin dropped, unpinned, or non-int pins never label.
    assert context_pin_suffix(200_000, 999_000) == ""
    assert context_pin_suffix(200_000, None) == ""
    assert context_pin_suffix(1, True) == ""


def _catalog_only(monkeypatch):
    """Empty the learned and models.dev caches so resolution falls through to the catalog."""
    from agent import model_metadata
    monkeypatch.setattr(model_metadata, "get_cached_context_length", lambda *a, **k: None)
    monkeypatch.setattr(model_metadata, "_load_model_metadata_disk_cache", lambda: {})


def test_uncatalogued_deployment_suffix_is_not_an_advertised_value(monkeypatch):
    # #132271: an Ollama tag "glm-5.3:cloud" must not be compared against the hosted "glm-5.3"
    # entry — the tag names a local deployment the catalog knows nothing about.
    _catalog_only(monkeypatch)
    assert advertised_context_length("glm-5.3:cloud", "") is None


def test_catalogued_variant_and_bare_tag_still_advertise(monkeypatch):
    _catalog_only(monkeypatch)
    assert advertised_context_length("glm-5.3:batch", "") == 1_048_576
    assert advertised_context_length("glm-5.3", "") == 1_310_720
    # A dated snapshot suffix is a naming variant of the same model, not a deployment tag.
    assert advertised_context_length("gpt-5.4-mini-2026-01-01", "") == 400_000


def test_pin_matching_a_local_variant_window_stays_silent(monkeypatch, caplog):
    # End-to-end replay of #132271: the pin equals the endpoint's true window; the catalog only
    # knows the hosted variant, so the startup warning must not fire.
    _catalog_only(monkeypatch)
    monkeypatch.setattr(context_pin, "_warned_pins", set())
    with caplog.at_level(logging.WARNING, logger="agent.context_pin"):
        fired = warn_once_on_pin_disagreement("glm-5.3:cloud", "http://192.168.86.84:11434", 1_048_576)
    assert fired is False
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_hosted_catalog_disagreement_still_warns(monkeypatch, caplog):
    _catalog_only(monkeypatch)
    monkeypatch.setattr(context_pin, "_warned_pins", set())
    with caplog.at_level(logging.WARNING, logger="agent.context_pin"):
        fired = warn_once_on_pin_disagreement("glm-5.3", "", 999_000)
    assert fired is True
