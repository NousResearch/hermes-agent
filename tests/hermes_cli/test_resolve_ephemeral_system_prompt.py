"""Unit tests for resolve_ephemeral_system_prompt_from_config."""

from hermes_cli.config import (
    render_personality_prompt,
    resolve_ephemeral_system_prompt_from_config,
)


def test_resolve_uses_named_personality_when_set():
    cfg = {
        "display": {"personality": "helpful"},
        "agent": {
            "system_prompt": "manual forever",
            "personalities": {"helpful": "You are helpful."},
        },
    }
    assert resolve_ephemeral_system_prompt_from_config(cfg) == "You are helpful."


def test_resolve_falls_back_to_manual_system_prompt():
    cfg = {
        "display": {"personality": "none"},
        "agent": {
            "system_prompt": "manual forever",
            "personalities": {"helpful": "You are helpful."},
        },
    }
    assert resolve_ephemeral_system_prompt_from_config(cfg) == "manual forever"


def test_resolve_ignores_unknown_personality_name():
    cfg = {
        "display": {"personality": "missing"},
        "agent": {
            "system_prompt": "manual forever",
            "personalities": {"helpful": "You are helpful."},
        },
    }
    assert resolve_ephemeral_system_prompt_from_config(cfg) == "manual forever"


def test_resolve_renders_dict_personality():
    cfg = {
        "display": {"personality": "coder"},
        "agent": {
            "system_prompt": "manual forever",
            "personalities": {
                "coder": {
                    "system_prompt": "You are an expert programmer.",
                    "tone": "technical",
                    "style": "concise",
                }
            },
        },
    }
    resolved = resolve_ephemeral_system_prompt_from_config(cfg)
    assert "You are an expert programmer." in resolved
    assert "Tone: technical" in resolved
    assert "Style: concise" in resolved


def test_render_personality_prompt_string():
    assert render_personality_prompt("  hi  ") == "hi"
