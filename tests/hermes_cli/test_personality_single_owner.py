"""Tests for hermes_cli.personality — the single owner of personality state —
and the v34 one-time personality reset migration.

Regression coverage for the post-#81946 resurrection bug: personality state
used to be persisted differently per surface (TUI/desktop wrote the NAME to
display.personality, CLI/gateway wrote rendered TEXT to agent.system_prompt),
so making display.personality authoritative resurrected personalities users
had already turned off ("kawaii defaults on after updating").
"""

import os
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.personality import (
    BUILTIN_PERSONALITIES,
    available_personalities,
    active_personality_name,
    describe_personality,
    normalize_personality_name,
    persist_personality,
    prompt_text,
    render_personality_prompt,
    resolve_ephemeral_system_prompt,
    resolve_personality,
)

KAWAII = BUILTIN_PERSONALITIES["kawaii"]


# ── module semantics ──────────────────────────────────────────────────────────


def test_builtins_available_without_any_config():
    merged = available_personalities(None)
    assert len(merged) >= 1
    for name in merged:
        assert name == name.lower()
    # built-ins render to non-empty prompts
    for defn in merged.values():
        assert render_personality_prompt(defn)


def test_user_entries_overlay_builtins_by_name():
    cfg = {"agent": {"personalities": {"kawaii": "toned down", "custom": "hi"}}}
    merged = available_personalities(cfg)
    assert merged["kawaii"] == "toned down"
    assert merged["custom"] == "hi"


def test_neutral_names_normalize_to_empty():
    for raw in ("", "none", "None", " DEFAULT ", "neutral", None):
        assert normalize_personality_name(raw) == ""


def test_resolve_personality_neutral_and_case_insensitive():
    assert resolve_personality("none", {}) == ("", "")
    name, prompt = resolve_personality("  KAWAII ", {})
    assert name == "kawaii"
    assert prompt == KAWAII


def test_resolve_personality_unknown_raises_with_listing():
    with pytest.raises(ValueError) as exc:
        resolve_personality("doesnotexist", {})
    assert "Available" in str(exc.value)
    assert "`none`" in str(exc.value)


def test_resolve_overlay_personality_wins_over_manual_prompt():
    cfg = {
        "display": {"personality": "kawaii"},
        "agent": {"system_prompt": "manual forever"},
    }
    assert resolve_ephemeral_system_prompt(cfg) == KAWAII


def test_resolve_overlay_falls_back_to_manual_prompt():
    for neutral in ("", "none", "default", "neutral"):
        cfg = {
            "display": {"personality": neutral},
            "agent": {"system_prompt": "manual forever"},
        }
        assert resolve_ephemeral_system_prompt(cfg) == "manual forever"


def test_resolve_overlay_ignores_unknown_name():
    cfg = {
        "display": {"personality": "ghost"},
        "agent": {"system_prompt": "manual forever"},
    }
    assert resolve_ephemeral_system_prompt(cfg) == "manual forever"
    assert active_personality_name(cfg) == ""


def test_render_dict_personality():
    rendered = render_personality_prompt(
        {"system_prompt": "You are X.", "tone": "warm", "style": "brief"}
    )
    assert "You are X." in rendered
    assert "Tone: warm" in rendered
    assert "Style: brief" in rendered


def test_prompt_text_normalizes_none_str_list():
    assert prompt_text(None) == ""
    assert prompt_text("  hi  ") == "hi"
    assert prompt_text(["a", " b ", ""]) == "a\nb"


def test_describe_personality_truncates_and_flattens():
    assert describe_personality("x" * 80) == "x" * 50 + "..."
    assert "\n" not in describe_personality("a\nb")
    assert describe_personality({"description": "short desc"}) == "short desc"


# ── persistence (single write path) ──────────────────────────────────────────


def test_persist_personality_roundtrip(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
        assert persist_personality("KAWAII ") is True
        raw = yaml.safe_load((home / "config.yaml").read_text())
        assert raw["display"]["personality"] == "kawaii"

        assert persist_personality("none") is True
        raw = yaml.safe_load((home / "config.yaml").read_text())
        assert raw["display"]["personality"] == ""


def test_persist_personality_never_touches_system_prompt(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({"agent": {"system_prompt": "manual forever"}})
    )
    with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
        assert persist_personality("kawaii") is True
        raw = yaml.safe_load((home / "config.yaml").read_text())
        assert raw["agent"]["system_prompt"] == "manual forever"
        assert raw["display"]["personality"] == "kawaii"


# ── v34 migration: one-time reset of stale split-brain state ─────────────────


def _run_migration(home, cfg):
    (home / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
        from hermes_cli.config import migrate_config, read_raw_config

        results = migrate_config(interactive=False, quiet=True)
        return read_raw_config(), results


def test_migration_resets_stale_personality_name(tmp_path):
    # Shape 1: TUI/desktop wrote the name years ago; the old CLI/gateway
    # "/personality none" never cleared it. Post-#81946 it resurrected.
    home = tmp_path / ".hermes"
    home.mkdir()
    raw, results = _run_migration(
        home,
        {"_config_version": 33, "display": {"personality": "kawaii"}},
    )
    assert raw["display"]["personality"] == ""
    assert resolve_ephemeral_system_prompt(raw) == ""
    assert any("personality" in item for item in results["config_added"])


def test_migration_scrubs_personality_text_from_system_prompt(tmp_path):
    # Shape 2: old CLI/gateway wrote rendered personality TEXT into
    # agent.system_prompt. Verbatim match with a known personality render
    # proves machine-written — scrub it.
    home = tmp_path / ".hermes"
    home.mkdir()
    raw, _ = _run_migration(
        home,
        {"_config_version": 33, "agent": {"system_prompt": KAWAII}},
    )
    assert raw["agent"]["system_prompt"] == ""
    assert resolve_ephemeral_system_prompt(raw) == ""


def test_migration_preserves_manual_system_prompt(tmp_path):
    # Shape 3: a hand-written prompt never verbatim-matches a personality
    # render — it must survive untouched while the stale name is reset.
    home = tmp_path / ".hermes"
    home.mkdir()
    raw, _ = _run_migration(
        home,
        {
            "_config_version": 33,
            "display": {"personality": "pirate"},
            "agent": {"system_prompt": "my manual prompt"},
        },
    )
    assert raw["display"]["personality"] == ""
    assert raw["agent"]["system_prompt"] == "my manual prompt"
    assert resolve_ephemeral_system_prompt(raw) == "my manual prompt"


def test_migration_noop_when_nothing_stale(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    raw, results = _run_migration(home, {"_config_version": 33})
    assert not any("personality" in item for item in results["config_added"])


def test_post_v34_choice_is_never_reset(tmp_path):
    # The reset fires exactly once (33→34). A personality chosen AFTER the
    # migration is the user's real selection and must survive later runs.
    home = tmp_path / ".hermes"
    home.mkdir()
    raw, _ = _run_migration(
        home,
        {"_config_version": 34, "display": {"personality": "kawaii"}},
    )
    assert raw["display"]["personality"] == "kawaii"
    assert resolve_ephemeral_system_prompt(raw) == KAWAII
