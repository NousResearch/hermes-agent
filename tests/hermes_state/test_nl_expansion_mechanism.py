"""Tests for the NL query-expansion mechanism (language-agnostic contracts).

The mechanism must work with the default pack only: stopword stripping,
light stemming, additive fallback chain, graceful degradation for scripts
without a pack. Language packs are pure data and tested separately.
"""

import json
import re

import pytest

from hermes_state import SessionDB
from hermes_state_nl_expansion import (
    _NL_LANG_PACKS,
    NLSupport,
    detect_lang,
    morph_prefix,
)


class TestDetectLang:
    def test_unknown_script_gets_default(self):
        assert detect_lang("配置 服务器") == "default"

    def test_registry_packs_conform_to_schema(self):
        required = {
            "stopwords", "affinity_stopwords", "suffixes", "endings",
            "vowels", "min_stem", "trailing_vowel_drop", "fallback",
        }
        for lang, pack in _NL_LANG_PACKS.items():
            missing = required - set(pack)
            assert not missing, f"{lang} pack missing keys: {missing}"
            assert pack["fallback"] in {"keep", "drop1"}, lang
            assert pack["min_stem"] >= 3, lang


class TestExpandEn:
    @pytest.fixture()
    def host(self):
        return NLSupport()

    def test_english_question_strips_stopwords_and_prefixes(self, host):
        out = host.expand_nl_query("what did we decide about the configs?")
        assert out is not None
        assert out["and"] == "decide* AND about* AND config*"
        assert out["bare"] == "decide about configs"

    def test_two_meaningful_terms_minimum(self, host):
        assert host.expand_nl_query("what the?") is None
        assert host.expand_nl_query("config backup")["and"] == "config* AND backup*"

    def test_short_tokens_untouched(self, host):
        out = host.expand_nl_query("ssh api gateway")
        assert "api" in out["bare"].split()
        assert "ssh" in out["bare"].split()

    def test_separated_compound_splits_into_subtokens(self, host):
        out = host.expand_nl_query("check the ssh-config backup")
        assert "config*" in out["and"]
        assert "backup*" in out["and"]


class TestMorphPrefix:
    def test_english_suffix_strip(self):
        p = _NL_LANG_PACKS["default"]
        kw = dict(
            suffixes=p["suffixes"], endings=p["endings"], vowels=p["vowels"],
            min_stem=p["min_stem"], trailing_vowel_drop=p["trailing_vowel_drop"],
            fallback=p["fallback"],
        )
        assert morph_prefix("servers", **kw) == "server*"
        assert morph_prefix("walked", **kw) == "walk*"
        assert morph_prefix("config's", **kw) == "config*"

    def test_min_stem_floor(self):
        p = _NL_LANG_PACKS["default"]
        kw = dict(
            suffixes=p["suffixes"], endings=p["endings"], vowels=p["vowels"],
            min_stem=p["min_stem"], trailing_vowel_drop=p["trailing_vowel_drop"],
            fallback=p["fallback"],
        )
        assert morph_prefix("api", **kw) == "api"      # < min_stem → untouched
        assert morph_prefix("test", **kw) == "test*"   # == min_stem

    def test_latin_stem_kept_when_no_suffix_matches(self):
        p = _NL_LANG_PACKS["default"]
        kw = dict(
            suffixes=p["suffixes"], endings=p["endings"], vowels=p["vowels"],
            min_stem=p["min_stem"], trailing_vowel_drop=p["trailing_vowel_drop"],
            fallback=p["fallback"],
        )
        assert morph_prefix("config", **kw) == "config*"
        assert morph_prefix("testing", **kw) == "test*"

    def test_drop1_fallback_for_fusional_pack_data(self):
        """A hypothetical fusional pack: tail carries flexion → drop 1."""
        assert (
            morph_prefix(
                "router",
                suffixes=(), endings=frozenset(),
                vowels="aeiou", min_stem=4,
                trailing_vowel_drop=False, fallback="drop1",
            )
            == "route*"
        )


class TestAdditiveFallbackE2E:
    """The fallback must fire only on a zero-result miss and never reorder."""

    @pytest.fixture()
    def db(self, tmp_path):
        d = SessionDB(db_path=tmp_path / "state.db")
        d.create_session("sess-en", source="cli")
        d.append_message(
            session_id="sess-en", role="user",
            content="how do we deploy the k3s cluster backup",
        )
        d.append_message(
            session_id="sess-en", role="assistant",
            content="we decided about backups: deployment uses flux nightly",
        )
        yield d
        d.close()

    def test_exact_hit_not_replaced_by_expansion(self, db):
        rows = db.search_messages("nightly backups")
        assert rows
        first = json.dumps(rows[0], ensure_ascii=False).lower()
        assert "nightly" in first

    def test_natural_language_is_opt_in(self, db):
        query = "what did we decide about the backups?"
        assert db.search_messages(query) == []
        rows = db.search_messages(query, natural_language=True)
        assert rows, "NL mode must recover the session"
        assert "backup" in json.dumps(rows[:5], ensure_ascii=False).lower()

    @pytest.mark.parametrize("query", [
        '"exact phrase" backup',
        "python NOT java",
        "docker OR kubernetes",
        "deploy* backup",
        "NEAR(config backup, 5)",
    ])
    def test_fts_syntax_never_enters_nl_mode(self, db, query):
        assert db.search_messages(query, natural_language=True) == db.search_messages(query)

    def test_nl_path_preserves_source_and_role_filters(self, db):
        rows = db.search_messages(
            "what did we decide about the backups?",
            natural_language=True,
            source_filter=["cli"],
            role_filter=["assistant"],
        )
        assert rows and all(row["source"] == "cli" for row in rows)
        assert all(row["role"] == "assistant" for row in rows)

    def test_nl_route_is_visible_in_slow_search_log(self, db, monkeypatch, caplog):
        monkeypatch.setenv("HERMES_SEARCH_SLOW_MS", "0")
        with caplog.at_level("INFO", logger="hermes_state"):
            db.search_messages(
                "what did we decide about the backups?", natural_language=True
            )
        assert any("path=nl_fts_and" in record.message for record in caplog.records)

    def test_or_route_is_visible_when_terms_span_messages(self, db, monkeypatch, caplog):
        monkeypatch.setenv("HERMES_SEARCH_SLOW_MS", "0")
        with caplog.at_level("INFO", logger="hermes_state"):
            rows = db.search_messages(
                "flux cluster backups?", natural_language=True
            )
        assert rows
        assert any("path=nl_fts_or" in record.message for record in caplog.records)

    def test_nl_only_telemetry_is_opt_in_and_private(self, db, monkeypatch, caplog):
        monkeypatch.setenv("HERMES_SEARCH_NL_ROUTE_LOG", "1")
        with caplog.at_level("INFO", logger="hermes_state"):
            rows = db.search_messages(
                "what did we decide about the backups?", natural_language=True
            )
        assert rows
        events = [r.getMessage() for r in caplog.records if "NL fallback" in r.getMessage()]
        assert len(events) == 1
        assert "path=nl_fts_and" in events[0]
        assert "language=default" in events[0]
        assert "what did" not in events[0]

    def test_sanitizer_quotes_do_not_block_nl_for_hyphenated_words(self, db):
        db.append_message(
            session_id="sess-en", role="assistant", content="sauvegarde fonctionnement"
        )
        rows = db.search_messages(
            "comment les sauvegardes fonctionnent-elles", natural_language=True
        )
        assert rows
        assert "sauvegarde" in json.dumps(rows, ensure_ascii=False)

    def test_genuinely_absent_terms_stay_empty(self, db):
        assert db.search_messages("quantum unicorn recipes", natural_language=True) == []


class TestExpansionSafety:
    def test_cache_is_bounded(self):
        host = NLSupport()
        for number in range(300):
            host.expand_nl_query(f"question {number} about target")
        assert len(host._cache) <= host._CACHE_MAXSIZE

    def test_long_query_is_not_expanded_or_cached(self):
        host = NLSupport()
        query = "meaningful " * (host._MAX_QUERY_CHARS // 2)
        assert len(query) > host._MAX_QUERY_CHARS
        assert host.expand_nl_query(query) is None
        assert query not in host._cache

    def test_expansion_caps_meaningful_terms(self):
        host = NLSupport()
        out = host.expand_nl_query(" ".join(f"term{index}" for index in range(20)))
        assert out is not None
        assert out["and"].count(" AND ") + 1 == host._MAX_MEANINGFUL_TERMS
