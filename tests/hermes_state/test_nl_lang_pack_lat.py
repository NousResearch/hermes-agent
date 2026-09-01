"""Tests for the Latin-script language packs (es/fr/de/pt/it + default).

Each pack must: be selected by stopword affinity, strip its stopwords,
stem real flexions into prefixes the unicode61 tokenizer can wildcard,
and never crash on accented input. No mechanism changes here.
"""

import json

import pytest

from hermes_state import SessionDB
from hermes_state_nl_expansion import _NL_LANG_PACKS, NLSupport, detect_lang


def _pack_kwargs(lang: str) -> dict:
    p = _NL_LANG_PACKS[lang]
    return dict(
        suffixes=p["suffixes"], endings=p["endings"], vowels=p["vowels"],
        min_stem=p["min_stem"], trailing_vowel_drop=p["trailing_vowel_drop"],
        fallback=p["fallback"],
    )


class TestLatinPackDetection:
    @pytest.mark.parametrize("query,lang", [
        ("dónde está la configuración del servidor", "es"),
        ("où est la configuration du serveur", "fr"),
        ("wo ist die Konfiguration des Servers", "de"),
        ("onde está a configuração do servidor", "pt"),
        ("dov'è la configurazione del server", "it"),
        ("quando girano i backup notturni", "it"),  # must beat Croatian's shared 'i'
        ("what about the server config", "default"),  # EN → default fallback
    ])
    def test_affinity_detection(self, query, lang):
        assert detect_lang(query) == lang


class TestLatinPackExpansion:
    """E2E through _expand_nl_query with each pack's own data."""

    @pytest.fixture()
    def host(self):
        return NLSupport()

    def test_spanish_question(self, host):
        out = host.expand_nl_query("¿dónde está la configuración del servidor?")
        assert out is not None
        assert "configuración" in out["bare"]
        # stopword 'del/la/está' stripped
        assert "está" not in f" {out['bare']} "
        assert "del" not in out["bare"].split()

    def test_french_question(self, host):
        out = host.expand_nl_query("où est la configuration du serveur")
        assert out is not None
        assert "serveu*" in out["and"] or "serveur*" in out["and"]
        assert "la" not in out["bare"].split()

    def test_german_question(self, host):
        out = host.expand_nl_query("wo ist die Konfiguration des Servers")
        assert out is not None
        assert "Server*" in out["and"]
        assert "die" not in out["bare"].split()
        assert "ist" not in out["bare"].split()

    def test_portuguese_question(self, host):
        out = host.expand_nl_query("onde está a configuração do servidor")
        assert out is not None
        # -ção flexion strips via the pack suffix table
        assert "configura*" in out["and"]
        assert "está" not in out["bare"].split()

    def test_italian_question(self, host):
        out = host.expand_nl_query("dov'è la configurazione del server")
        assert out is not None
        # -zione flexion strips via the pack suffix table
        assert "configura*" in out["and"]
        assert "della" not in out["bare"].split()


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("sess-es", source="cli")
    d.append_message(
        session_id="sess-es", role="user", content="configuración proxy documentada"
    )
    d.append_message(
        session_id="sess-es", role="assistant", content="proxy configurado con traefik"
    )
    yield d
    d.close()


class TestSpanishE2E:
    def test_inflected_question_finds_session(self, db):
        query = "¿dónde están las configuraciones del proxy?"
        assert db.search_messages(query) == []  # strict low-level FTS5 API
        rows = db.search_messages(query, natural_language=True)
        assert rows, "conversational ES query must reach 'configuré/configurado'"
        blob = json.dumps(rows, ensure_ascii=False).lower()
        assert "proxy" in blob or "configur" in blob
