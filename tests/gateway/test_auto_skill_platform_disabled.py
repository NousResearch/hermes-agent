"""Regression test: gateway auto-skill loading must respect the disabled-skill
gate (channel_skill_bindings / Telegram DM Topics).

`_resolve_auto_skill_content` (extracted from `_handle_message_with_agent`'s
auto-skill block) loads bound skills via `_load_skill_payload` with a raw
identifier — the same bypass class the stacked (#58888) and bundle (#59156)
invocation paths had: it skips `get_skill_commands()`'s scan-time disabled
filter, so an operator who disables a skill for this platform (or globally)
still had its full content injected into every new session bound to that
channel/topic.

This drives the real function with monkeypatched skill-loading/disabled-check
boundaries (each already covered by their own unit tests) rather than reading
gateway/run.py's source, per AGENTS.md's "never read source code in tests" —
the previous version of this test parsed the function via `inspect.getsource`
+ AST and only proved an import/call name existed, not that disabled content
is actually excluded.
"""
from __future__ import annotations

from gateway.run import _resolve_auto_skill_content


def _fake_load_skill_payload(disabled_skill_content):
    """Return a _load_skill_payload stand-in with two known skills:
    'writer' (enabled) and 'blocked' (its display name is disabled)."""

    def _load(skill_identifier, task_id=None):
        if skill_identifier == "writer":
            return ({"body": "Writer skill instructions."}, None, "writer")
        if skill_identifier == "blocked":
            return ({"body": disabled_skill_content}, None, "Blocked Skill")
        return None

    return _load


def _fake_build_skill_message(loaded_skill, skill_dir, note):
    return f"{note}\n{loaded_skill['body']}"


def test_disabled_skill_content_excluded_enabled_skill_included(monkeypatch):
    import agent.skill_commands as skill_commands_mod
    import agent.skill_utils as skill_utils_mod

    monkeypatch.setattr(
        skill_commands_mod, "_load_skill_payload",
        _fake_load_skill_payload("SECRET blocked-skill body"),
    )
    monkeypatch.setattr(
        skill_commands_mod, "_build_skill_message", _fake_build_skill_message,
    )
    monkeypatch.setattr(
        skill_utils_mod, "get_disabled_skill_names",
        lambda platform=None: {"Blocked Skill"},
    )

    parts, loaded_names = _resolve_auto_skill_content(
        ["writer", "blocked"], platform_name="telegram", quick_key="q1",
    )

    combined = "\n".join(parts)
    assert "Writer skill instructions." in combined
    assert "SECRET blocked-skill body" not in combined
    assert loaded_names == ["writer"]


def test_disabled_by_raw_identifier_also_excluded(monkeypatch):
    """get_disabled_skill_names may return the raw identifier instead of the
    display name (global disable via `hermes skills disable <id>`)."""
    import agent.skill_commands as skill_commands_mod
    import agent.skill_utils as skill_utils_mod

    monkeypatch.setattr(
        skill_commands_mod, "_load_skill_payload",
        _fake_load_skill_payload("SECRET blocked-skill body"),
    )
    monkeypatch.setattr(
        skill_commands_mod, "_build_skill_message", _fake_build_skill_message,
    )
    monkeypatch.setattr(
        skill_utils_mod, "get_disabled_skill_names",
        lambda platform=None: {"blocked"},
    )

    parts, loaded_names = _resolve_auto_skill_content(
        ["writer", "blocked"], platform_name="telegram", quick_key="q1",
    )

    assert loaded_names == ["writer"]
    assert not any("SECRET" in p for p in parts)


def test_no_skills_disabled_both_loaded(monkeypatch):
    import agent.skill_commands as skill_commands_mod
    import agent.skill_utils as skill_utils_mod

    monkeypatch.setattr(
        skill_commands_mod, "_load_skill_payload",
        _fake_load_skill_payload("Blocked skill body"),
    )
    monkeypatch.setattr(
        skill_commands_mod, "_build_skill_message", _fake_build_skill_message,
    )
    monkeypatch.setattr(
        skill_utils_mod, "get_disabled_skill_names", lambda platform=None: set(),
    )

    parts, loaded_names = _resolve_auto_skill_content(
        ["writer", "blocked"], platform_name="telegram", quick_key="q1",
    )

    assert loaded_names == ["writer", "blocked"]
    assert len(parts) == 2


def test_platform_is_forwarded_to_disabled_check(monkeypatch):
    """The gate must be platform-scoped: get_disabled_skill_names(platform=...)
    receives the actual channel's platform, not a hardcoded value."""
    import agent.skill_commands as skill_commands_mod
    import agent.skill_utils as skill_utils_mod

    seen = {}

    def _capture(platform=None):
        seen["platform"] = platform
        return set()

    monkeypatch.setattr(
        skill_commands_mod, "_load_skill_payload", _fake_load_skill_payload("x"),
    )
    monkeypatch.setattr(
        skill_commands_mod, "_build_skill_message", _fake_build_skill_message,
    )
    monkeypatch.setattr(skill_utils_mod, "get_disabled_skill_names", _capture)

    _resolve_auto_skill_content(
        ["writer"], platform_name="discord", quick_key="q1",
    )

    assert seen["platform"] == "discord"
