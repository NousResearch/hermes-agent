from types import SimpleNamespace

import pytest

from gateway.kanban_status_card import render_kanban_status_card
from hermes_cli.kanban_db import CurrentRunProgress, Event
from hermes_cli.kanban_status_timing import (
    format_elapsed_age,
    format_relative_age,
    status_surface_refresh_period,
)


def _task(status="running", *, block_kind=None, title="Собрать статус-карточку", created_at=1, started_at=1, completed_at=None):
    return SimpleNamespace(
        id="t_card", status=status, assignee="heavy", block_kind=block_kind,
        title=title, created_at=created_at, started_at=started_at, completed_at=completed_at,
    )


def _event(kind, payload=None, created_at=1):
    return SimpleNamespace(kind=kind, payload=payload or {}, created_at=created_at)


def _parent(title, status="running"):
    return SimpleNamespace(title=title, status=status)


def _card(task, *events, now=1, latest_comment=None, parents=None):
    return render_kanban_status_card(
        sub={"task_id": "t_card"}, task=task, timeline=list(events), now=now,
        latest_comment=latest_comment, parents=parents,
    )


@pytest.mark.parametrize(
    ("age", "relative", "elapsed"),
    [
        (0, "только что", "меньше 15 сек"),
        (14, "только что", "меньше 15 сек"),
        (15, "15 сек назад", "15 сек"),
        (30, "30 сек назад", "30 сек"),
        (45, "45 сек назад", "45 сек"),
        (60, "1 мин назад", "1 мин"),
        (75, "1 мин 15 сек назад", "1 мин 15 сек"),
    ],
)
def test_status_clock_uses_fifteen_second_buckets(age, relative, elapsed):
    assert format_relative_age(age) == relative
    assert format_elapsed_age(age) == elapsed


def test_status_card_cadence_is_fifteen_seconds_at_all_ages():
    assert status_surface_refresh_period([0, 120]) == 15
    assert status_surface_refresh_period([59, 900]) == 15
    assert status_surface_refresh_period([60, 600]) == 15
    assert status_surface_refresh_period([900]) == 15
    assert status_surface_refresh_period([901]) == 15
    assert status_surface_refresh_period([]) == 0


def test_running_card_with_no_progress_uses_stale_progress_status_and_three_independent_metrics():
    text = _card(
        _task(title="Довести live-карточки до рабочего состояния"),
        _event("claimed"), _event("heartbeat", {"note": "Проверяю receipt и lease"}),
    )

    assert "🟢 меньше 15 сек · 🛠 нет данных · ⏱️ меньше 15 сек" in text
    assert "📍 Этап:" not in text
    assert "receipt" not in text


def test_running_card_uses_current_run_substantive_progress_clock():
    current_run = CurrentRunProgress(
        run_id=2,
        started_at=100,
        events=(Event(id=2, task_id="t_card", kind="heartbeat", payload=None, created_at=120, run_id=2),),
    )
    text = render_kanban_status_card(
        sub={"task_id": "t_card"}, task=_task(created_at=1, started_at=1),
        timeline=[_event("progress", {"checkpoint": "old run"}, created_at=5)],
        current_run=current_run, now=125,
    )
    assert "🛠 нет данных" in text
    assert "old run" not in text


def test_running_card_clamps_explicit_human_detail_in_own_block():
    detail = "Проверяю доставку в исходную тему и готовлю понятный итог для пользователя. " * 5
    text = _card(_task(title="Проверить длинную live-деталь"), _event("heartbeat", {"note": detail}))

    assert text.startswith("🗂 Проверить длинную live-деталь · t_card\n\n🤷‍♂️ @heavy на связи, но нет подтверждённого прогресса\n\n🧭 Сейчас:\n")
    detail_line = text.split("🧭 Сейчас:\n", 1)[1].split("\n\n🟢 ", 1)[0]
    assert len(detail_line) <= 220
    assert detail_line.endswith("…")
    assert "📍 Этап:" not in text


@pytest.mark.parametrize(
    ("block_kind", "heading", "next_action"),
    [
        ("needs_input", "📞 Нужен твой ответ, KD", "Нужен ответ"),
        ("transient", "⚠️ Перезапускается после временного сбоя", "ответ не нужен"),
    ],
)
def test_waiting_cards_keep_state_in_primary_line_without_internal_reason(block_kind, heading, next_action):
    text = _card(_task("blocked", block_kind=block_kind), _event("blocked", {"reason": "Нужен доступ"}))

    assert heading in text
    assert "🧭 Сейчас:" in text
    assert "Нужен доступ" not in text
    assert next_action in text
    assert "⏱ Ожидание: меньше 15 сек" in text
    assert "📍 Этап:" not in text


def test_superseded_and_error_cards_do_not_expose_internal_event_detail():
    capability = _card(_task("blocked", block_kind="capability"), _event("blocked", {"reason": "Нужен доступ к закрытому сервису"}))
    superseded = _card(_task("blocked"), _event("blocked", {"reason": "superseded by replacement t_b9435779"}))
    crashed = _card(_task(), _event("crashed", {"error": "traceback"}))

    assert "🔐 Нужна ручная помощь" in capability
    assert "Нужен доступ к закрытому сервису" not in capability
    assert "🔁 Работа продолжена в новой задаче" in superseded
    assert "traceback" not in crashed
    assert "⚠️ Worker перезапускается" in crashed
    assert "📍 Этап:" not in capability + superseded + crashed


def test_accepted_card_is_edited_to_auditor_acceptance_not_terminal_ping():
    text = _card(
        _task("done", title="Internal notifier architecture cleanup", completed_at=20),
        _event("claimed", created_at=10),
        _event("review_accepted", created_at=20),
        now=20,
    )

    assert text == (
        "🗂 Internal notifier architecture cleanup · t_card\n\n✅ Принято аудитором\n\n"
        "🧭 Сейчас:\nПроверка завершена\n\n⏱ Выполнено за: 0 мин"
    )
    assert "Готово — к твоему ревью" not in text
    assert "📍 Этап:" not in text


def test_internal_comments_are_never_rendered_in_current_block():
    text = _card(
        _task(), _event("heartbeat", {"note": "/tmp/status-card.json sha256=abcd1234"}),
        latest_comment=SimpleNamespace(body="OPS gate: sqlite receipt lease"),
    )

    assert "🧭 Сейчас:\nИсполнитель на связи; нового подтверждённого прогресса пока нет." in text
    assert "OPS gate" not in text
    assert "sqlite" not in text


def test_review_and_dependency_cards_show_only_applicable_clocks():
    review = _card(_task("review"), _event("review_requested", created_at=10), now=70)
    dependency = _card(
        _task("todo"), _event("created", {"parents": ["t_parent"]}), now=70,
        parents=[_parent("Подготовить данные")],
    )
    queued = _card(_task("ready"), now=70)

    assert "🔎 Аудитор проверяет результат" in review
    assert "⏱ На проверке: 1 мин" in review
    assert "Воркер на связи" not in review
    assert "⏳ Начнётся после «Подготовить данные»" in dependency
    assert "🧭 Сейчас:\nСвязанная задача ещё выполняется." in dependency
    assert "⏱️ 1 м" in dependency
    assert "Воркер на связи" not in dependency
    assert "⏳ @heavy ожидает запуска" in queued
    assert "🧭 Сейчас:\nОжидает свободного исполнителя" in queued
    assert "⏱️ 1 м" in queued


def test_dependency_wait_for_multiple_parents_shows_real_completion_count():
    text = _card(
        _task("todo"), _event("created", created_at=10), now=70,
        parents=[_parent("Первый", "done"), _parent("Второй")],
    )

    assert "⏳ Начнётся после 2 связанных задач" in text
    assert "🧭 Сейчас:\nГотово 1 из 2 связанных задач." in text
    assert "⏱️ 1 м" in text
    assert "зависимост" not in text.casefold()
    assert "todo" not in text.casefold()


def test_dependency_wait_for_parent_in_review_uses_user_safe_review_copy():
    text = _card(
        _task("todo"), _event("created", created_at=10), now=70,
        parents=[_parent("Проверить готовый отчёт", "review")],
    )

    assert "⏳ Начнётся после проверки «Проверить готовый отчёт»" in text
    assert "🧭 Сейчас:\nСвязанная задача ещё выполняется." in text
    assert "зависимост" not in text.casefold()


def test_dependency_wait_then_running_never_uses_human_needed_copy():
    waiting = _card(
        _task("todo", block_kind="dependency"), _event("dependency_wait"),
        parents=[_parent("Собрать данные")],
    )
    running = _card(
        _task("running", block_kind="dependency"), _event("claimed"), _event("heartbeat"),
        parents=[_parent("Собрать данные", "done")],
    )

    assert "Нужен" not in waiting
    assert "Начнётся после «Собрать данные»" in waiting
    assert "🤷‍♂️ @heavy на связи, но нет подтверждённого прогресса" in running
    assert "Нужен" not in running


def test_fake_clock_keeps_liveness_progress_and_elapsed_independent_across_retry():
    task = _task(created_at=1, started_at=10)
    text = _card(
        task,
        _event("claimed", created_at=10),
        _event("progress", {"checkpoint": "Нашёл причину ошибки"}, created_at=20),
        _event("heartbeat", created_at=100),
        _event("heartbeat", created_at=320),
        _event("reclaimed", created_at=400),
        _event("claimed", created_at=500),
        _event("heartbeat", created_at=600),
        now=620,
    )

    assert "🟢 15 сек · 🛠 10 м · ⏱️ 10 м" in text


def test_header_truncates_unicode_title_without_orphaning_task_id():
    text = _card(_task(title="Очень длинный заголовок задачи " * 8), _event("heartbeat", {"note": "Проверяю пользовательскую доставку"}))

    assert text.splitlines()[0].endswith(" · t_card")
    assert "… · t_card" in text.splitlines()[0]
    assert len(text.splitlines()[0]) <= 72


def test_short_header_is_not_changed():
    assert _card(_task(title="Коротко"), _event("heartbeat")).splitlines()[0] == "🗂 Коротко · t_card"


@pytest.mark.parametrize(
    ("task", "events", "parents", "expected"),
    [
        (_task("ready"), [], None, "⏳ @heavy ожидает запуска"),
        (_task("running"), [_event("heartbeat", created_at=100), _event("progress", created_at=100)], None, "🔫 @heavy выполняет задачу"),
        (_task("running"), [_event("heartbeat", created_at=100)], None, "🤷‍♂️ @heavy на связи, но нет подтверждённого прогресса"),
        (_task("ready"), [_event("review_rejected")], None, "😡 Аудитор вернул задачу на доработку"),
        (_task("running"), [_event("review_rejected")], None, "🤝 @heavy исправляет замечания аудитора"),
        (_task("review"), [_event("review_requested")], None, "🔎 Аудитор проверяет результат"),
        (_task("review"), [_event("review_requested"), _event("review_requested")], None, "🤔 Аудитор повторно проверяет"),
        (_task("review"), [_event("review_retry_scheduled")], None, "⚠️ Проверка аудитора перезапускается"),
        (_task("review"), [_event("needs_auditor")], None, "🔐 Нужна ручная проверка аудитора"),
        (_task("done"), [_event("review_accepted")], None, "✅ Принято аудитором"),
        (_task("todo"), [], None, "⏳ @heavy ожидает запуска"),
        (_task("todo"), [], [_parent("Родитель")], "⏳ Начнётся после «Родитель»"),
        (_task("todo"), [], [_parent("Первый"), _parent("Второй")], "⏳ Начнётся после 2 связанных задач"),
        (_task("blocked", block_kind="needs_input"), [], None, "📞 Нужен твой ответ, KD"),
        (_task("blocked", block_kind="capability"), [], None, "🔐 Нужна ручная помощь"),
        (_task("blocked", block_kind="transient"), [], None, "⚠️ Перезапускается после временного сбоя"),
        (_task("ready"), [_event("reclaimed")], None, "⚠️ Worker перезапускается"),
        (_task("ready"), [_event("gave_up")], None, "❌ Не удалось выполнить автоматически"),
        (_task("blocked"), [_event("superseded")], None, "🔁 Работа продолжена в новой задаче"),
        (_task("archived"), [], None, "📦 Перенесено в архив"),
    ],
)
def test_primary_status_vocabulary_covers_supported_lifecycle_states(task, events, parents, expected):
    text = _card(task, *events, parents=parents, now=100)

    assert expected in text
    assert "📍 Этап:" not in text
