"""Legacy ``api_mode`` spellings must keep selecting the transport they named.

Regression coverage for the silent api_mode vocabulary break: earlier
releases accepted ``api_mode: openai`` on custom provider entries. The
canonical set consumed by ``agent_init`` is now {chat_completions,
codex_responses, anthropic_messages, bedrock_converse, codex_app_server},
and an unrecognized value was silently ignored at BOTH consumption sites:

* ``hermes_cli.config._normalize_custom_provider_entry`` passed the raw
  string through, so ``agent_init``'s accepted-set check dropped it and
  fell through to hostname detection.
* ``hermes_cli.runtime_provider._parse_api_mode`` returned None, with the
  same fall-through.

For a host with a detection rule (e.g. api.actual.inc -> codex_responses)
the provider silently switched transports after an update and broke:
observed live as every reasoning-bearing request to a relay's untested
/v1/responses endpoint failing while chat_completions worked. See the
#66543 discussion.

The fix canonicalizes known legacy/alias spellings through one shared map
(``_canonical_api_mode``) at both sites. Unknown values still pass through
unchanged (normalizer) / return None (runtime gate) so existing invalid
config behavior is untouched.
"""

from __future__ import annotations

import pytest

from hermes_cli.config import _canonical_api_mode, _normalize_custom_provider_entry
from hermes_cli.runtime_provider import _parse_api_mode, _VALID_API_MODES


class TestCanonicalApiMode:
    """The shared alias map."""

    @pytest.mark.parametrize(
        "alias, canonical",
        [
            ("openai", "chat_completions"),
            ("OpenAI", "chat_completions"),
            (" openai ", "chat_completions"),
            ("openai_chat", "chat_completions"),
            ("chat-completions", "chat_completions"),
            ("responses", "codex_responses"),
            ("openai_responses", "codex_responses"),
            ("anthropic", "anthropic_messages"),
            ("messages", "anthropic_messages"),
            ("bedrock", "bedrock_converse"),
        ],
    )
    def test_alias_maps_to_canonical(self, alias, canonical):
        assert _canonical_api_mode(alias) == canonical

    @pytest.mark.parametrize(
        "canonical",
        sorted(_VALID_API_MODES),
    )
    def test_canonical_names_pass_through(self, canonical):
        assert _canonical_api_mode(canonical) == canonical

    def test_unknown_value_passes_through_unchanged(self):
        assert _canonical_api_mode("weird_thing") == "weird_thing"

    def test_every_alias_lands_in_the_valid_set(self):
        """Contract: aliasing must never produce a value the runtime rejects."""
        from hermes_cli.config import _API_MODE_ALIASES

        for target in _API_MODE_ALIASES.values():
            assert target in _VALID_API_MODES


class TestNormalizedEntryCanonicalizes:
    """Config-side consumption: _normalize_custom_provider_entry."""

    def _entry(self, api_mode):
        return {
            "name": "relay",
            "api": "https://relay.example.invalid/v1",
            "api_mode": api_mode,
        }

    def test_legacy_openai_becomes_chat_completions(self):
        normalized = _normalize_custom_provider_entry(
            self._entry("openai"), provider_key="relay"
        )
        assert normalized["api_mode"] == "chat_completions"

    def test_canonical_value_unchanged(self):
        normalized = _normalize_custom_provider_entry(
            self._entry("codex_responses"), provider_key="relay"
        )
        assert normalized["api_mode"] == "codex_responses"

    def test_transport_key_also_canonicalized(self):
        entry = {
            "name": "relay",
            "api": "https://relay.example.invalid/v1",
            "transport": "openai",
        }
        normalized = _normalize_custom_provider_entry(entry, provider_key="relay")
        assert normalized["api_mode"] == "chat_completions"


class TestRuntimeParseApiMode:
    """Runtime-side consumption: _parse_api_mode."""

    def test_legacy_openai_is_valid_chat_completions(self):
        assert _parse_api_mode("openai") == "chat_completions"

    def test_canonical_value_still_valid(self):
        assert _parse_api_mode("anthropic_messages") == "anthropic_messages"

    def test_unknown_value_still_rejected(self):
        assert _parse_api_mode("bogus") is None

    def test_non_string_still_rejected(self):
        assert _parse_api_mode(None) is None
        assert _parse_api_mode(42) is None
