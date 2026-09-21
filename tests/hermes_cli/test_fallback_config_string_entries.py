"""Fallback entry parsing: string shorthand accepted, malformed entries loud, parsers in parity.

Before the fix, ``_iter_fallback_entries`` silently dropped every non-dict entry. A chain
configured as string shorthand (``fallback_providers: [openrouter:qwen/qwen3.6-plus]``) resolved
to an EMPTY chain while looking configured: sessions died on the primary with a bare usage-limit
error and never activated a fallback (#51560, #117806). Contracts pinned here:

  1. ``'provider:model'`` string entries parse (first colon splits; model ids keep their colons).
  2. Any dropped entry warns — nothing about a configured chain is silent.
  3. An all-malformed chain warns that the effective chain is EMPTY.
  4. The agent-side parser (``agent.agent_init._fallback_entries``) accepts the same shapes.
"""

import logging

from hermes_cli.fallback_config import _iter_fallback_entries, get_fallback_chain


class TestStringShorthand:
    def test_provider_model_string_parses(self):
        entries = _iter_fallback_entries(["openrouter:qwen/qwen3.6-plus"])
        assert entries == [{"provider": "openrouter", "model": "qwen/qwen3.6-plus"}]

    def test_first_colon_splits_model_ids_with_colons(self):
        entries = _iter_fallback_entries(["nous:z-ai/glm-5.3-flash:free"])
        assert entries == [{"provider": "nous", "model": "z-ai/glm-5.3-flash:free"}]

    def test_whitespace_tolerated(self):
        entries = _iter_fallback_entries(["  nous : deepseek/deepseek-v4.1-flash  "])
        assert entries == [{"provider": "nous", "model": "deepseek/deepseek-v4.1-flash"}]

    def test_get_fallback_chain_merges_strings_and_dicts_without_dupes(self):
        cfg = {
            "fallback_providers": ["nous:z-ai/glm-5.3-flash"],
            "fallback_model": {"provider": "nous", "model": "z-ai/glm-5.3-flash"},
        }
        chain = get_fallback_chain(cfg)
        assert chain == [{"provider": "nous", "model": "z-ai/glm-5.3-flash"}]


class TestMalformedEntriesAreLoud:
    def test_unseparated_string_warns_and_drops(self, caplog):
        with caplog.at_level(logging.WARNING, logger="hermes_cli.fallback_config"):
            assert _iter_fallback_entries(["just-a-model-slug"]) == []
        assert "provider:model" in caplog.text

    def test_dict_missing_model_warns_and_drops(self, caplog):
        with caplog.at_level(logging.WARNING, logger="hermes_cli.fallback_config"):
            assert _iter_fallback_entries([{"provider": "nous"}]) == []
        assert "missing model" in caplog.text

    def test_non_string_non_dict_warns_and_drops(self, caplog):
        with caplog.at_level(logging.WARNING, logger="hermes_cli.fallback_config"):
            assert _iter_fallback_entries([42]) == []
        assert "malformed fallback entry" in caplog.text

    def test_all_entries_dropped_warns_chain_empty(self, caplog):
        with caplog.at_level(logging.WARNING, logger="hermes_cli.fallback_config"):
            assert _iter_fallback_entries([{"provider": "nous"}, "garbage"]) == []
        assert "effective fallback chain is EMPTY" in caplog.text

    def test_no_warning_when_entries_parse(self, caplog):
        with caplog.at_level(logging.WARNING, logger="hermes_cli.fallback_config"):
            assert _iter_fallback_entries([{"provider": "nous", "model": "deepseek/deepseek-v4.1-flash"}])
        assert "malformed" not in caplog.text
        assert "EMPTY" not in caplog.text


class TestAgentSideParserParity:
    def test_agent_parser_accepts_string_shorthand(self):
        from agent.agent_init import _fallback_entries
        assert _fallback_entries(["openrouter:qwen/qwen3.6-plus"]) == [
            {"provider": "openrouter", "model": "qwen/qwen3.6-plus"}]

    def test_agent_parser_accepts_legacy_single_dict(self):
        from agent.agent_init import _fallback_entries
        assert _fallback_entries({"provider": "nous", "model": "deepseek/deepseek-v4.1-flash"}) == [
            {"provider": "nous", "model": "deepseek/deepseek-v4.1-flash"}]

    def test_agent_parser_parity_with_shared_parser(self):
        from agent.agent_init import _fallback_entries
        raw = ["nous:z-ai/glm-5.3-flash", {"provider": "openrouter", "model": "qwen/qwen3.6-plus"}]
        assert _fallback_entries(raw) == _iter_fallback_entries(raw)
