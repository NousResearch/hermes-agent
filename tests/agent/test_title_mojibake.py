from unittest.mock import MagicMock, patch

import pytest

from agent.title_generator import _clean_title, _repair_mojibake, auto_title_session, generate_title
from hermes_state import SessionDB


def test_utf8_read_as_cp1250_is_restored():
    assert _clean_title("Salutation amicale \u0102\u00a0 Hugo") == "Salutation amicale à Hugo"


def test_utf8_read_as_cp1252_is_restored():
    assert _repair_mojibake("Pr\u00c3\u00a9parer la r\u00c3\u00a9union") == "Préparer la réunion"


def test_clean_text_is_untouched():
    for title in ("Préparer la réunion", "Mail à Paul", "Ça coûte 10 €", "Plain ASCII title", "日本語のタイトル"):
        assert _repair_mojibake(title) == title


TURKISH = "Merhaba arkadaşım, selamlaşma için yazıyorum: çiçek, dağ, ılık, İstanbul, gölge, şeker, ütü"


def _reply(content):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


@pytest.mark.parametrize(("source", "model_output", "expected"), [
    # Byte-spelled characters mixed with intact ones, so no whole-string round trip applies (#149025).
    (TURKISH, "ArkadaÅŸ selamlaşma", "Arkadaş selamlaşma"),
    (TURKISH, "Ã‡iÃ§ek ve dağ, Ä°stanbul gÃ¶lge ÅŸeker Ã¼tÃ¼ ılık", "Çiçek ve dağ, İstanbul gölge şeker ütü ılık"),
    ("Привет, расскажи про мир и şehir", "ÐŸÑ€Ð¸Ð²ÐµÑ‚ Ð¼Ð¸Ñ€ ve şehir", "Привет мир ve şehir"),
    ("日本語のタイトル in İzmir", "æ—¥æœ¬èªž İzmir", "日本語 İzmir"),
])
def test_byte_spelled_characters_the_user_typed_are_restored(source, model_output, expected):
    with patch("agent.title_generator.call_llm", return_value=_reply(model_output)):
        assert generate_title(source) == expected


def test_undecodable_byte_residue_is_rejected():
    # The reported title: ``ı``/``ş`` bytes as cp1253/cp1252 glyphs plus residue (``μƒ``) no repair
    # recovers. Storing it is the bug; reject so the derived title from the user's own words stays.
    with patch("agent.title_generator.call_llm", return_value=_reply("ArkadaμƒΔ±ΕŸ selamlaşma")):
        assert generate_title("Arkadaşım ile selamlaşma") is None


def test_rejected_damaged_title_falls_back_to_the_users_words(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="s1", source="telegram")
    with patch("agent.title_generator.call_llm", return_value=_reply("ArkadaμƒΔ±ΕŸ selamlaşma")):
        auto_title_session(db, "s1", "Arkadaşım ile selamlaşma")
    assert db.get_session_title("s1") == "Arkadaşım ile selamlaşma"


@pytest.mark.parametrize("title", [
    "Çiçek ğ ı İ ö ş ü", "İş planı", "Arkadaş selamlaşma", "Привет мир", "Ελληνικά θέματα", "日本語のタイトル",
    "Mail à Paul — notes", "Ăştia sunt pașii",
])
def test_correct_non_latin_titles_are_untouched(title):
    with patch("agent.title_generator.call_llm", return_value=_reply(title)):
        assert generate_title(TURKISH + " Привет мир Ελληνικά 日本語 à Paul Ăştia pașii") == title
