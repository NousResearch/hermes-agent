"""Tests for the Slavic language packs (ru/be/uk + sr/bg/mk + cs/sk/hr).

Cyrillic and Latin Slavic packs are tested together since detection
uses script → affinity routing: Cyrillic queries score across all
Cyrillic packs, Latin Slavic packs score among Latin-script candidates.
"""

import json

import pytest

from hermes_state import SessionDB
from hermes_state_nl_expansion import _NL_LANG_PACKS, NLSupport, detect_lang


def _slavic_kwargs(lang: str) -> dict:
    p = _NL_LANG_PACKS[lang]
    return dict(
        suffixes=p["suffixes"], endings=p["endings"], vowels=p["vowels"],
        min_stem=p["min_stem"], trailing_vowel_drop=p["trailing_vowel_drop"],
        fallback=p["fallback"],
    )


class TestSlavicPackDetection:
    @pytest.mark.parametrize("query,expected_pack", [
        ("сколько алиасов в конфиге", "ru"),
        ("скільки аліасів у конфігу", "uk"),
        ("няма ды ці быць у каго блізу", "be"),  # unique Belarusian markers
        ("колькі сервераў", "be"),  # common Belarusian question, not Russian
        ("колико аласа у конфигу", "sr"),  # Serbian Cyrillic
        ("колко аликса в конфиг", "bg"),   # Bulgarian
        ("koliko alasa u konfiguraciji", "hr"),  # Croatian Latin
        ("kolik aliasů v konfiguraci", "cs"),    # Czech
        ("koľko aliasov v konfigurácii", "sk"),  # Slovak
    ])
    def test_cyrillic_affinity_routing(self, query, expected_pack):
        """Cyrillic queries should route through script detection + affinity scoring."""
        result = detect_lang(query)
        assert result == expected_pack, f"'{query}' → {result} (expected {expected_pack})"

    def test_english_question_gets_default(self):
        """English questions should never hit a Slavic pack."""
        assert detect_lang("what is the server config") == "default"


class TestSlavicMorphology:
    """Test that real inflections from each Slavic pack stem correctly."""

    @pytest.fixture()
    def host(self):
        return NLSupport()

    def test_ru_inflections(self, host):
        out = host.expand_nl_query("сколько серверов в конфиге прописано")
        assert out is not None
        # "серверов" → "сервер*" (2-char ending table)
        assert "сервер*" in out["and"] or "сервер" in out["bare"]
        # stopwords "сколько", "прописано" stripped
        sw = {"сколько", "прописано"}
        for word in out["bare"].split():
            assert word.lower() not in sw, f"{word} should be stripped"

    def test_uk_inflections(self, host):
        out = host.expand_nl_query("скільки серверів прописано")
        assert out is not None
        # Ukrainian endings differ from Russian
        assert any(s.startswith("серве") for s in out["and"].split())

    def test_sr_inflections(self, host):
        """Serbian Cyrillic inflection."""
        out = host.expand_nl_query("колико аласа у конфигу је прописан")
        assert out is not None
        # Should contain stemmed forms
        bare = out["bare"]
        assert "алас" in bare or "ала" in bare

    def test_bg_inflections(self, host):
        """Bulgarian loses cases but has rich prefixes."""
        out = host.expand_nl_query("колко сървъри са записани")
        assert out is not None
        bare = out["bare"]
        assert "сърв" in bare or "сервер" in bare

    def test_cs_inflections(self, host):
        """Czech with rich consonant clusters and long vowels."""
        out = host.expand_nl_query("kolik aliasů je v konfiguraci nastaveno")
        assert out is not None
        bare = out["bare"]
        assert "alias" in bare or "konfigur" in bare

    def test_sk_inflections(self, host):
        """Slovak close to Czech but different stopwords."""
        out = host.expand_nl_query("koľko aliasov je v konfigurácii zadaných")
        assert out is not None
        bare = out["bare"]
        assert "alias" in bare or "konfigur" in bare

    def test_be_inflections(self, host):
        out = host.expand_nl_query("колькі сервераў прапісана")
        assert out is not None
        # Belarusian -аў ending → stem
        bare = out["bare"]
        assert "сервер" in bare or "сервера" in bare

    def test_bg_inflections(self, host):
        out = host.expand_nl_query("колко сервери са записани")
        assert out is not None
        # Bulgarian -и plural suffix
        assert "серв" in out["and"] or "сервер" in out["bare"]

    def test_cs_inflections(self, host):
        out = host.expand_nl_query("kolik aliasů je v konfiguraci")
        assert out is not None
        # Czech -ů plural
        bare = out["bare"]
        assert "alias" in bare or "konfigur" in bare

    def test_slovak_question(self, host):
        out = host.expand_nl_query("koľko aliasov je v konfigurácii")
        assert out is not None
        # Slovak -ov suffix
        bare = out["bare"]
        assert "alias" in bare or "konfigur" in bare

    @pytest.mark.parametrize("query,lang,expected", [
        ("кога услугите стартираха", "bg", "услуг*"),
        ("каде се конфигурациите на серверот", "mk", "конфигураци*"),
        ("ako fungujú nočné zálohy", "sk", "záloh*"),
    ])
    def test_eval_regressions(self, host, query, lang, expected):
        assert detect_lang(query) == lang
        out = host.expand_nl_query(query)
        assert out is not None and expected in out["or"]


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("sess-cyrillic", source="cli")
    d.append_message(
        session_id="sess-cyrillic", role="user",
        content="сервер конфігурація запущений",
    )
    d.append_message(
        session_id="sess-cyrillic", role="assistant",
        content="сервер конфігурація запущений на traefik",
    )
    yield d
    d.close()


class TestSlavicE2E:
    def test_ukrainian_question_finds_session(self, db):
        query = "скільки серверів конфігурації запущено?"
        assert db.search_messages(query) == []  # strict low-level FTS5 API
        rows = db.search_messages(query, natural_language=True)
        assert rows, "conversational Ukrainian query must find Cyrillic session"
        blob = json.dumps(rows, ensure_ascii=False).lower()
        assert "сервер" in blob or "конф" in blob
