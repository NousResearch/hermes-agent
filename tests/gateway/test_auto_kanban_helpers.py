from gateway.run import (
    _auto_kanban_extract_list_items,
    _auto_kanban_should_track,
    _auto_kanban_title,
)


def test_auto_kanban_ignores_commands_questions_and_short_text():
    assert _auto_kanban_should_track("/kanban create test") is False
    assert _auto_kanban_should_track("как сделать авто-канбан и почему он нужен") is False
    assert _auto_kanban_should_track("коротко сделай") is False


def test_auto_kanban_tracks_long_actionable_requests():
    text = (
        "Реализуй авто-канбан для входящих сообщений: нужно определить сложные запросы, "
        "создать задачу до обработки, закрыть ее после успешного ответа и заблокировать "
        "с причиной при ошибке. Это должно работать тихо и не создавать задачи для команд."
    )

    assert _auto_kanban_should_track(text) is True


def test_auto_kanban_extracts_explicit_lists_only():
    assert _auto_kanban_extract_list_items("1. Один пункт") is None
    assert _auto_kanban_extract_list_items("1. Первый\n2. Второй") == ["Первый", "Второй"]
    assert _auto_kanban_extract_list_items("- Первый\n- Второй\n- Третий") == [
        "Первый",
        "Второй",
        "Третий",
    ]


def test_auto_kanban_title_is_single_line_and_capped():
    title = _auto_kanban_title("  Реализуй\n\n" + "x" * 120)

    assert "\n" not in title
    assert len(title) <= 72
