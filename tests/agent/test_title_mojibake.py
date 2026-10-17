from agent.title_generator import _clean_title, _repair_mojibake


def test_utf8_read_as_cp1250_is_restored():
    assert _clean_title("Salutation amicale \u0102\u00a0 Hugo") == "Salutation amicale à Hugo"


def test_utf8_read_as_cp1252_is_restored():
    assert _repair_mojibake("Pr\u00c3\u00a9parer la r\u00c3\u00a9union") == "Préparer la réunion"


def test_clean_text_is_untouched():
    for title in ("Préparer la réunion", "Mail à Paul", "Ça coûte 10 €", "Plain ASCII title", "日本語のタイトル"):
        assert _repair_mojibake(title) == title
