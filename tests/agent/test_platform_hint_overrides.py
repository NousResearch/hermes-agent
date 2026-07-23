"""Tests for per-platform prompt-hint overrides (config.yaml → platform_hints).

Covers agent/system_prompt.py::_resolve_platform_hint — the resolver that
applies append/replace overrides to a platform's default hint. Feature added
for enterprise managed profiles (per-platform behavior without affecting other
platforms). See HA Core ticket: configurable per-platform prompt hints.
"""

import types

from agent.system_prompt import _resolve_platform_hint


def _agent(overrides):
    """Minimal stand-in carrying just the override attribute the resolver reads."""
    a = types.SimpleNamespace()
    a._platform_hint_overrides = overrides
    return a


DEFAULT = "You are on WhatsApp. Do not use markdown."
EXTRA = "When tabular output would help, invoke the table_formatting skill."


class TestResolvePlatformHint:
    def test_no_overrides_returns_default(self):
        assert _resolve_platform_hint(_agent({}), "whatsapp", DEFAULT) == DEFAULT

    def test_missing_attr_returns_default(self):
        a = types.SimpleNamespace()  # no _platform_hint_overrides at all
        assert _resolve_platform_hint(a, "whatsapp", DEFAULT) == DEFAULT

    def test_platform_not_in_overrides_returns_default(self):
        a = _agent({"slack": {"append": "x"}})
        assert _resolve_platform_hint(a, "whatsapp", DEFAULT) == DEFAULT

    def test_append_dict(self):
        a = _agent({"whatsapp": {"append": EXTRA}})
        out = _resolve_platform_hint(a, "whatsapp", DEFAULT)
        assert out == f"{DEFAULT}\n\n{EXTRA}"
        assert DEFAULT in out and EXTRA in out

    def test_replace_dict(self):
        a = _agent({"whatsapp": {"replace": EXTRA}})
        out = _resolve_platform_hint(a, "whatsapp", DEFAULT)
        assert out == EXTRA
        assert DEFAULT not in out

    def test_replace_wins_over_append_but_both_applied(self):
        a = _agent({"whatsapp": {"replace": "BASE", "append": "TAIL"}})
        out = _resolve_platform_hint(a, "whatsapp", DEFAULT)
        # replace substitutes the base, append still tacks on
        assert out == "BASE\n\nTAIL"
        assert DEFAULT not in out

    def test_bare_string_is_append_shorthand(self):
        a = _agent({"whatsapp": EXTRA})
        out = _resolve_platform_hint(a, "whatsapp", DEFAULT)
        assert out == f"{DEFAULT}\n\n{EXTRA}"

    def test_other_platform_unaffected(self):
        """An override for whatsapp must not change telegram's hint."""
        a = _agent({"whatsapp": {"append": EXTRA}})
        tg_default = "You are on Telegram. Markdown works."
        assert _resolve_platform_hint(a, "telegram", tg_default) == tg_default

    def test_empty_platform_key_returns_default(self):
        a = _agent({"whatsapp": {"append": EXTRA}})
        assert _resolve_platform_hint(a, "", DEFAULT) == DEFAULT

    # --- defensive / malformed input: never break prompt assembly ---

    def test_malformed_spec_list_returns_default(self):
        a = _agent({"whatsapp": ["not", "valid"]})
        assert _resolve_platform_hint(a, "whatsapp", DEFAULT) == DEFAULT

    def test_overrides_not_a_dict_returns_default(self):
        a = _agent(["nope"])
        assert _resolve_platform_hint(a, "whatsapp", DEFAULT) == DEFAULT

    def test_empty_append_string_returns_default(self):
        a = _agent({"whatsapp": {"append": "   "}})
        assert _resolve_platform_hint(a, "whatsapp", DEFAULT) == DEFAULT

    def test_empty_replace_falls_back_to_default_base(self):
        a = _agent({"whatsapp": {"replace": "   "}})
        assert _resolve_platform_hint(a, "whatsapp", DEFAULT) == DEFAULT

    def test_non_string_append_ignored(self):
        a = _agent({"whatsapp": {"append": 123}})
        assert _resolve_platform_hint(a, "whatsapp", DEFAULT) == DEFAULT

    def test_replace_with_empty_default_hint(self):
        """replace works even when the platform had no built-in default."""
        a = _agent({"customplat": {"replace": "Custom hint."}})
        assert _resolve_platform_hint(a, "customplat", "") == "Custom hint."

    def test_append_with_empty_default_hint(self):
        """append on a platform with no default just yields the extra text."""
        a = _agent({"customplat": {"append": "Only this."}})
        assert _resolve_platform_hint(a, "customplat", "") == "Only this."
