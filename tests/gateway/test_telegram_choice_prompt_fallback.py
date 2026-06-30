from gateway.run import _extract_choice_prompt_from_text
from gateway.platforms.telegram import _choice_reply_keyboard_rows


HEBREW_CHOICES = [
    "עצה מוקדמת מדי",
    "מרגיש שלא מכיר אותי מספיק",
    "לא הבין את הקושי האמיתי",
    "הכול נכון / שילוב של כמה מהם",
]


def test_extracts_hebrew_question_with_bullet_choices():
    text = """מה מפריע שם בעיקר?

• עצה מוקדמת מדי
• מרגיש שלא מכיר אותי מספיק
• לא הבין את הקושי האמיתי
• הכול נכון / שילוב של כמה מהם"""

    parsed = _extract_choice_prompt_from_text(text)

    assert parsed == ("מה מפריע שם בעיקר?", HEBREW_CHOICES)


def test_extracts_actual_option_text_not_hebrew_letter_labels():
    text = """מה מפריע שם בעיקר?

א. עצה מוקדמת מדי
ב. מרגיש שלא מכיר אותי מספיק
ג. לא הבין את הקושי האמיתי
ד. הכול נכון / שילוב של כמה מהם"""

    parsed = _extract_choice_prompt_from_text(text)

    assert parsed == ("מה מפריע שם בעיקר?", HEBREW_CHOICES)
    assert parsed is not None
    _, choices = parsed
    assert choices != ["בחירה א", "בחירה ב", "בחירה ג", "בחירה ד"]
    assert choices != ["א", "ב", "ג", "ד"]


def test_extracts_actual_option_text_not_number_labels():
    text = """מה מפריע שם בעיקר?

1. עצה מוקדמת מדי
2. מרגיש שלא מכיר אותי מספיק
3. לא הבין את הקושי האמיתי
4. הכול נכון / שילוב של כמה מהם"""

    parsed = _extract_choice_prompt_from_text(text)

    assert parsed == ("מה מפריע שם בעיקר?", HEBREW_CHOICES)
    assert parsed is not None
    _, choices = parsed
    assert choices != ["1", "2", "3", "4"]


def test_reply_keyboard_preserves_exact_choices_and_adds_all_and_several_when_missing():
    choices = [
        "עצה מוקדמת מדי",
        "מרגיש שלא מכיר אותי מספיק",
        "לא הבין את הקושי האמיתי",
    ]

    rows = _choice_reply_keyboard_rows(choices)

    assert rows == [
        ["עצה מוקדמת מדי"],
        ["מרגיש שלא מכיר אותי מספיק"],
        ["לא הבין את הקושי האמיתי"],
        ["כל האפשרויות"],
        ["כמה אפשרויות — אכתוב"],
    ]


def test_reply_keyboard_does_not_replace_real_all_or_combination_option():
    rows = _choice_reply_keyboard_rows(HEBREW_CHOICES)

    assert rows == [[choice] for choice in HEBREW_CHOICES]
    assert ["בחירה א"] not in rows
    assert ["1"] not in rows
    assert ["א"] not in rows


def test_does_not_extract_plain_bulleted_explanation():
    text = """Here are reasons:

• first fact
• second fact

This is not a prompt."""

    assert _extract_choice_prompt_from_text(text) is None
