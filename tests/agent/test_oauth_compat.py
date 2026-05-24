"""Tests for agent/oauth_compat.py — Anthropic-OAuth Claude Code shim.

Covers:
- StealthMode.parse() lenient/strict cases
- ToolNameMap forward, reverse, idempotency, collision handling
- _default_pascal_case fallback for unmapped names
- is_third_party_classifier_rejection() narrow matching
- apply_to_kwargs() effect on tools, history, system across modes

See hermes-agent issue #15080 for the original report.
"""
from __future__ import annotations

import pytest

from agent import oauth_compat
from agent.oauth_compat import (
    CLAUDE_CODE_SYSTEM_PREFIX,
    CLAUDE_CODE_TOOLS,
    HERMES_TO_CLAUDE_CODE,
    StealthMode,
    ToolNameMap,
    _default_pascal_case,
    apply_to_kwargs,
    is_third_party_classifier_rejection,
)


# ---------------------------------------------------------------------------
# StealthMode
# ---------------------------------------------------------------------------

class TestStealthMode:
    def test_enum_values(self):
        # Pinned: external code reads these strings from config.yaml.
        assert StealthMode.OFF.value == "off"
        assert StealthMode.RENAME_ONLY.value == "rename_only"
        assert StealthMode.FULL_STEALTH.value == "full_stealth"

    def test_parse_passes_through_enum(self):
        assert StealthMode.parse(StealthMode.RENAME_ONLY) is StealthMode.RENAME_ONLY

    def test_parse_string(self):
        assert StealthMode.parse("off") is StealthMode.OFF
        assert StealthMode.parse("rename_only") is StealthMode.RENAME_ONLY
        assert StealthMode.parse("full_stealth") is StealthMode.FULL_STEALTH

    def test_parse_string_case_and_whitespace_insensitive(self):
        assert StealthMode.parse("  Full_Stealth  ") is StealthMode.FULL_STEALTH

    def test_parse_none_uses_default(self):
        assert StealthMode.parse(None) is StealthMode.OFF
        assert StealthMode.parse(None, default=StealthMode.RENAME_ONLY) is StealthMode.RENAME_ONLY

    def test_parse_empty_string_uses_default(self):
        assert StealthMode.parse("") is StealthMode.OFF

    def test_parse_unknown_falls_back_with_warning(self, caplog):
        with caplog.at_level("WARNING", logger="agent.oauth_compat"):
            result = StealthMode.parse("nope")
        assert result is StealthMode.OFF
        assert any("unrecognized StealthMode" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestDefaultPascalCase:
    def test_snake_case(self):
        assert _default_pascal_case("session_search") == "SessionSearch"
        assert _default_pascal_case("browser_back") == "BrowserBack"

    def test_already_pascal(self):
        # Idempotent — no underscores, capitalize first letter only.
        assert _default_pascal_case("Bash") == "Bash"
        assert _default_pascal_case("WebSearch") == "WebSearch"

    def test_single_word(self):
        assert _default_pascal_case("memory") == "Memory"

    def test_mixed_underscores(self):
        # Dropped empty parts (leading/trailing/double underscores).
        assert _default_pascal_case("_foo__bar_") == "FooBar"

    def test_empty_or_falsy(self):
        assert _default_pascal_case("") == ""
        assert _default_pascal_case(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Constants invariants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_hermes_targets_are_canonical(self):
        # Every mapped target should be a real Claude Code tool, otherwise
        # the rename gains us nothing on the classifier side.
        for orig, target in HERMES_TO_CLAUDE_CODE.items():
            assert target in CLAUDE_CODE_TOOLS, (
                f"{orig!r} → {target!r} but {target!r} is not in CLAUDE_CODE_TOOLS"
            )

    def test_canonical_set_is_pascal_case(self):
        for name in CLAUDE_CODE_TOOLS:
            assert name and name[0].isupper(), name
            assert "_" not in name, name


# ---------------------------------------------------------------------------
# ToolNameMap
# ---------------------------------------------------------------------------

class TestToolNameMap:
    def test_canonical_mapping(self):
        m = ToolNameMap()
        assert m.register("terminal") == "Bash"
        assert m.register("read_file") == "Read"
        assert m.register("write_file") == "Write"

    def test_pascal_fallback(self):
        m = ToolNameMap()
        assert m.register("session_search") == "SessionSearch"
        assert m.register("browser_back") == "BrowserBack"

    def test_reverse_lookup(self):
        m = ToolNameMap()
        m.register("terminal")
        m.register("session_search")
        assert m.unrename("Bash") == "terminal"
        assert m.unrename("SessionSearch") == "session_search"

    def test_unknown_reverse_returns_none(self):
        m = ToolNameMap()
        assert m.unrename("NeverSeen") is None
        assert m.unrename("") is None

    def test_idempotent_register(self):
        m = ToolNameMap()
        first = m.register("terminal")
        again = m.register("terminal")
        assert first == again
        assert len(m) == 1

    def test_collision_disambiguation(self, caplog):
        # If two hermes tools collapse to the same target (e.g. a
        # hypothetical future hermes 'bash' tool alongside 'terminal'),
        # the second registration gets a numeric suffix and a warning.
        m = ToolNameMap()
        first = m.register("terminal")          # → Bash
        with caplog.at_level("WARNING", logger="agent.oauth_compat"):
            second = m.register("bash")         # also → Bash, must disambiguate
        assert first == "Bash"
        assert second != "Bash"
        assert second.startswith("Bash") and second[4:].isdigit()
        assert m.unrename("Bash") == "terminal"
        assert m.unrename(second) == "bash"
        assert any("collision" in r.message for r in caplog.records)

    def test_empty_input_passes_through(self):
        m = ToolNameMap()
        assert m.register("") == ""


# ---------------------------------------------------------------------------
# is_third_party_classifier_rejection
# ---------------------------------------------------------------------------

class TestThirdPartyDetection:
    @pytest.mark.parametrize("msg", [
        "Third-party apps now draw from your extra usage, not your plan limits. "
        "Add more at claude.ai/settings/usage and keep going.",
        "You're out of extra usage. Add more at claude.ai/settings/usage and keep going.",
    ])
    def test_matches_known_phrasings(self, msg):
        assert is_third_party_classifier_rejection(status_code=400, error_message=msg)

    def test_wrong_status(self):
        msg = "Third-party apps … extra usage … claude.ai/settings/usage"
        assert not is_third_party_classifier_rejection(status_code=429, error_message=msg)
        assert not is_third_party_classifier_rejection(status_code=None, error_message=msg)

    def test_unrelated_400(self):
        assert not is_third_party_classifier_rejection(
            status_code=400,
            error_message="Invalid model: claude-bogus",
        )

    def test_partial_match_does_not_trigger(self):
        # The marker phrase alone (without the URL) is not enough — guards
        # against collision with the unrelated 429 long-context-tier rule.
        assert not is_third_party_classifier_rejection(
            status_code=400,
            error_message="The long context beta requires extra usage tier.",
        )

    def test_empty_message(self):
        assert not is_third_party_classifier_rejection(status_code=400, error_message="")
        assert not is_third_party_classifier_rejection(status_code=400, error_message=None)


# ---------------------------------------------------------------------------
# apply_to_kwargs
# ---------------------------------------------------------------------------

def _kwargs_with(tools=None, messages=None, system=None):
    return {
        "model": "claude-opus-4-7",
        "max_tokens": 16,
        "system": system if system is not None else [
            {"type": "text", "text": CLAUDE_CODE_SYSTEM_PREFIX},
            {"type": "text", "text": "# Persona\nLong content"},
        ],
        "messages": messages if messages is not None else [
            {"role": "user", "content": "hi"},
        ],
        "tools": tools if tools is not None else [
            {"name": "terminal", "description": "x", "input_schema": {"type": "object", "properties": {}}},
            {"name": "session_search", "description": "x", "input_schema": {"type": "object", "properties": {}}},
        ],
    }


class TestApplyToKwargs:
    def test_off_is_noop(self):
        m = ToolNameMap()
        kw = _kwargs_with()
        before = (
            [t["name"] for t in kw["tools"]],
            len(kw["system"]),
        )
        apply_to_kwargs(kw, mode=StealthMode.OFF, tool_map=m)
        after = (
            [t["name"] for t in kw["tools"]],
            len(kw["system"]),
        )
        assert before == after
        assert len(m) == 0

    def test_rename_only_renames_tools_and_history_but_keeps_system(self):
        m = ToolNameMap()
        kw = _kwargs_with(messages=[
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "id1", "name": "terminal", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "id1", "content": "ok"},
            ]},
        ])
        sys_blocks_before = list(kw["system"])
        apply_to_kwargs(kw, mode=StealthMode.RENAME_ONLY, tool_map=m)
        # Tools renamed
        assert [t["name"] for t in kw["tools"]] == ["Bash", "SessionSearch"]
        # History tool_use renamed
        assert kw["messages"][0]["content"][0]["name"] == "Bash"
        # tool_result untouched (it uses ID, not name)
        assert "name" not in kw["messages"][1]["content"][0]
        # System left alone
        assert kw["system"] == sys_blocks_before

    def test_full_stealth_collapses_system(self):
        m = ToolNameMap()
        kw = _kwargs_with()
        apply_to_kwargs(kw, mode=StealthMode.FULL_STEALTH, tool_map=m)
        assert kw["system"] == [
            {"type": "text", "text": CLAUDE_CODE_SYSTEM_PREFIX}
        ]
        assert [t["name"] for t in kw["tools"]] == ["Bash", "SessionSearch"]

    def test_full_stealth_idempotent(self):
        # Calling twice with the same tool_map should produce the same payload
        # (forward map is stable) and not double-rename.
        m = ToolNameMap()
        kw1 = _kwargs_with()
        apply_to_kwargs(kw1, mode=StealthMode.FULL_STEALTH, tool_map=m)
        snapshot = (
            [t["name"] for t in kw1["tools"]],
            kw1["system"],
        )
        # Re-running takes already-rewritten names through the map again. The
        # map registers them as new originals (Bash → Bash, SessionSearch →
        # SessionSearch in PascalCase fallback path), so the names stay stable.
        apply_to_kwargs(kw1, mode=StealthMode.FULL_STEALTH, tool_map=m)
        assert (
            [t["name"] for t in kw1["tools"]],
            kw1["system"],
        ) == snapshot

    def test_handles_missing_optional_fields(self):
        m = ToolNameMap()
        # No tools, no messages.
        kw = {"model": "claude-opus-4-7", "max_tokens": 16,
              "system": [{"type": "text", "text": CLAUDE_CODE_SYSTEM_PREFIX}]}
        apply_to_kwargs(kw, mode=StealthMode.FULL_STEALTH, tool_map=m)
        assert kw["system"] == [{"type": "text", "text": CLAUDE_CODE_SYSTEM_PREFIX}]

    def test_round_trip_via_unrename(self):
        # Dispatcher-side: after the model emits a tool_use with the renamed
        # name, unrename() restores the hermes original.
        m = ToolNameMap()
        kw = _kwargs_with()
        apply_to_kwargs(kw, mode=StealthMode.RENAME_ONLY, tool_map=m)
        for tool in kw["tools"]:
            assert m.unrename(tool["name"]) is not None
        assert m.unrename("Bash") == "terminal"
        assert m.unrename("SessionSearch") == "session_search"
