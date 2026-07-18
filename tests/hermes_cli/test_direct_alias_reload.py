"""Tests for DIRECT_ALIASES hot-reload (model.aliases takes effect without restart).

Regression coverage for the load-once bug where `/model <alias>` resolved a stale
alias table because `_ensure_direct_aliases()` populated `DIRECT_ALIASES` only on
first use and never refreshed after a `model.aliases` config edit.
"""
import logging

import pytest

from hermes_cli import model_switch as ms


@pytest.fixture(autouse=True)
def _reset_direct_aliases():
    """Reset the module-global DIRECT_ALIASES (in place) and the degraded latch
    around each test so state doesn't leak between tests."""
    saved = dict(ms.DIRECT_ALIASES)
    saved_degraded = ms._DIRECT_ALIASES_DEGRADED
    ms.DIRECT_ALIASES.clear()
    ms._DIRECT_ALIASES_DEGRADED = False
    yield
    ms.DIRECT_ALIASES.clear()
    ms.DIRECT_ALIASES.update(saved)
    ms._DIRECT_ALIASES_DEGRADED = saved_degraded


def _fake_load(monkeypatch, aliases_map):
    """Point _load_direct_aliases at a synthetic config (builtins + given user aliases)."""
    def _loader():
        merged = dict(ms._BUILTIN_DIRECT_ALIASES)
        for k, v in aliases_map.items():
            prov, model = v.split("/", 1)
            merged[k] = ms.DirectAlias(model=model, provider=prov, base_url="")
        return merged, True
    monkeypatch.setattr(ms, "_load_direct_aliases", _loader)


def test_reload_on_config_change(monkeypatch):
    """The core bug: after first population, a changed config value is reflected
    on the NEXT call — with no manual clear. RED on the load-once implementation."""
    _fake_load(monkeypatch, {"zed": "claude-apr/claude-x"})
    ms._ensure_direct_aliases()
    assert ms.DIRECT_ALIASES["zed"] == ms.DirectAlias("claude-x", "claude-apr", "")

    # Config edited: zed now points somewhere else.
    _fake_load(monkeypatch, {"zed": "claude-apr/claude-y"})
    ms._ensure_direct_aliases()
    assert ms.DIRECT_ALIASES["zed"] == ms.DirectAlias("claude-y", "claude-apr", ""), (
        "alias table did not hot-reload after config change (load-once bug)"
    )


def test_resolve_alias_reflects_reload(monkeypatch):
    """End-to-end via resolve_alias(): a config edit changes what /model <alias> resolves to."""
    _fake_load(monkeypatch, {"opustest": "claude-apr/claude-opus-4-8"})
    ms._ensure_direct_aliases()
    assert ms.resolve_alias("opustest", "claude-apr") == (
        "claude-apr", "claude-opus-4-8", "opustest",
    )
    # Re-point the alias to a different provider; must follow.
    _fake_load(monkeypatch, {"opustest": "openai-codex/gpt-5.6-terra"})
    assert ms.resolve_alias("opustest", "claude-apr") == (
        "openai-codex", "gpt-5.6-terra", "opustest",
    )


def test_removed_alias_disappears(monkeypatch):
    """An alias deleted from config is pruned on the next refresh."""
    _fake_load(monkeypatch, {"gone": "claude-apr/claude-x"})
    ms._ensure_direct_aliases()
    assert "gone" in ms.DIRECT_ALIASES
    _fake_load(monkeypatch, {})  # user removed it
    ms._ensure_direct_aliases()
    assert "gone" not in ms.DIRECT_ALIASES
    # Builtins survive.
    assert "gpt-5.5" in ms.DIRECT_ALIASES


def test_degraded_read_retains_last_known_good(monkeypatch, caplog):
    """RC-A: a transient config-read failure must NOT prune user aliases back to
    builtins — the exact wrong-provider symptom this feature fixes."""
    _fake_load(monkeypatch, {"zed": "claude-apr/claude-x"})
    ms._ensure_direct_aliases()
    assert "zed" in ms.DIRECT_ALIASES

    # Next refresh: config read fails (ok=False, builtins only).
    monkeypatch.setattr(
        ms, "_load_direct_aliases", lambda: (dict(ms._BUILTIN_DIRECT_ALIASES), False)
    )
    with caplog.at_level(logging.WARNING, logger=ms.logger.name):
        ms._ensure_direct_aliases()
        # RC-B / Greptile P2: further degraded calls must NOT re-flood the log.
        ms._ensure_direct_aliases()
        ms._ensure_direct_aliases()

    assert ms.DIRECT_ALIASES.get("zed") == ms.DirectAlias("claude-x", "claude-apr", ""), (
        "user alias was pruned to builtins on a degraded config read (MB-2 regression)"
    )
    assert ms.resolve_alias("zed", "claude-apr") == ("claude-apr", "claude-x", "zed")
    # RC-B: the degraded path emits exactly ONE warning across repeated calls.
    warnings = [r for r in caplog.records if "retaining" in r.message]
    assert len(warnings) == 1, (
        f"degraded-retain path logged {len(warnings)} warnings, expected exactly 1 (RC-B/Greptile P2)"
    )


def test_degraded_then_healthy_rearms_warning(monkeypatch, caplog):
    """After recovery, a subsequent degraded read warns again (latch re-arms)."""
    _fake_load(monkeypatch, {"zed": "claude-apr/claude-x"})
    ms._ensure_direct_aliases()

    degraded = lambda: (dict(ms._BUILTIN_DIRECT_ALIASES), False)
    with caplog.at_level(logging.WARNING, logger=ms.logger.name):
        monkeypatch.setattr(ms, "_load_direct_aliases", degraded)
        ms._ensure_direct_aliases()          # warn #1
        _fake_load(monkeypatch, {"zed": "claude-apr/claude-x"})
        ms._ensure_direct_aliases()          # healthy: re-arm
        monkeypatch.setattr(ms, "_load_direct_aliases", degraded)
        ms._ensure_direct_aliases()          # warn #2
    warnings = [r for r in caplog.records if "retaining" in r.message]
    assert len(warnings) == 2


def test_in_place_mutation_and_never_empty(monkeypatch):
    """INV-2: reload mutates in place (stable id) and never observes an empty dict."""
    _fake_load(monkeypatch, {"a": "p/m1"})
    ms._ensure_direct_aliases()
    dict_id = id(ms.DIRECT_ALIASES)

    # Observe the dict during the reload: _load_direct_aliases is called first,
    # so snapshot the live dict's length at that moment — it must never be empty.
    seen_lengths = []
    real_loader = ms._load_direct_aliases

    def _observing_loader():
        seen_lengths.append(len(ms.DIRECT_ALIASES))
        merged = dict(ms._BUILTIN_DIRECT_ALIASES)
        merged["a"] = ms.DirectAlias("m2", "p", "")
        merged["b"] = ms.DirectAlias("m3", "p", "")
        return merged, True

    monkeypatch.setattr(ms, "_load_direct_aliases", _observing_loader)
    ms._ensure_direct_aliases()

    assert id(ms.DIRECT_ALIASES) == dict_id, "DIRECT_ALIASES was rebound (INV-2 violation)"
    # The dict held its prior contents the whole time (never cleared to empty).
    assert seen_lengths and all(n > 0 for n in seen_lengths), (
        "DIRECT_ALIASES was empty during reload (concurrency hazard)"
    )
    assert ms.DIRECT_ALIASES["a"] == ms.DirectAlias("m2", "p", "")
    assert "b" in ms.DIRECT_ALIASES
