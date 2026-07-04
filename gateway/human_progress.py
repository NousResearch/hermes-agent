"""Human-readable gateway progress labels.

This module deliberately avoids exposing raw tool names, command strings,
arguments, paths, JSON payloads, or other implementation details.  It is used
by chat platforms that want a concise activity feed for humans rather than a
developer-oriented tool trace.
"""

from __future__ import annotations

from typing import Any, Mapping


_GENERIC_PROGRESS = "Продолжаю работу над задачей."

_TOOL_MESSAGES: dict[str, str] = {
    "browser_back": "Возвращаюсь к предыдущей странице.",
    "browser_click": "Выбираю нужный элемент на странице.",
    "browser_console": "Проверяю состояние страницы.",
    "browser_get_images": "Смотрю изображения на странице.",
    "browser_navigate": "Открываю нужную страницу.",
    "browser_press": "Взаимодействую со страницей.",
    "browser_scroll": "Просматриваю страницу дальше.",
    "browser_snapshot": "Проверяю содержимое страницы.",
    "browser_type": "Заполняю поле на странице.",
    "browser_vision": "Разбираю, что видно на странице.",
    "clarify": "Уточняю недостающие детали.",
    "cronjob": "Обновляю расписание задачи.",
    "delegate_task": "Передаю часть работы отдельному агенту.",
    "execute_code": "Проверяю гипотезу в изолированном коде.",
    "image_generate": "Готовлю изображение по описанию.",
    "memory": "Сохраняю важную долгосрочную настройку.",
    "patch": "Вношу точечное изменение.",
    "process": "Проверяю фоновый процесс.",
    "read_file": "Изучаю найденный файл.",
    "search_files": "Ищу релевантные места в проекте.",
    "send_message": "Отправляю сообщение адресату.",
    "session_search": "Ищу подходящий контекст в прошлых сессиях.",
    "skill_manage": "Обновляю сохранённый рабочий навык.",
    "skill_view": "Подгружаю нужный рабочий навык.",
    "skills_list": "Проверяю доступные рабочие навыки.",
    "terminal": "Проверяю это в рабочем окружении.",
    "todo": "Обновляю план работы.",
    "vision_analyze": "Анализирую изображение.",
    "write_file": "Записываю подготовленные изменения.",
}

_TERMINAL_HINTS: tuple[tuple[str, str], ...] = (
    ("pytest", "Запускаю проверку тестами."),
    ("unittest", "Запускаю проверку тестами."),
    ("npm test", "Запускаю проверку тестами."),
    ("pnpm test", "Запускаю проверку тестами."),
    ("yarn test", "Запускаю проверку тестами."),
    ("ruff", "Проверяю качество Python-кода."),
    ("mypy", "Проверяю типы Python-кода."),
    ("eslint", "Проверяю качество JavaScript-кода."),
    ("tsc", "Проверяю типы TypeScript-кода."),
    ("git status", "Проверяю состояние изменений."),
    ("git diff", "Смотрю, что изменилось."),
    ("git log", "Смотрю историю изменений."),
    ("git branch", "Проверяю текущую ветку."),
    ("git fetch", "Обновляю сведения из репозитория."),
    ("git push", "Отправляю изменения в репозиторий."),
    ("git commit", "Фиксирую подготовленные изменения."),
    ("docker", "Проверяю контейнерное окружение."),
    ("systemctl", "Проверяю состояние системного сервиса."),
)

_SEARCH_TARGET_MESSAGES: dict[str, str] = {
    "files": "Ищу подходящие файлы в проекте.",
    "content": "Ищу совпадения внутри файлов.",
}


def human_tool_progress_message(tool_name: str | None, args: Mapping[str, Any] | None = None) -> str:
    """Return a concise human-readable progress message for a tool call.

    The returned text must be safe for end-user chat surfaces: no raw tool
    arguments, shell commands, file paths, URLs, JSON, or internal function
    names.  The message describes intent at a high level only.
    """
    name = (tool_name or "").strip()
    data: Mapping[str, Any] = args if isinstance(args, Mapping) else {}

    if name == "terminal":
        command = str(data.get("command") or "").lower()
        for needle, message in _TERMINAL_HINTS:
            if needle in command:
                return message
        return _TOOL_MESSAGES[name]

    if name == "search_files":
        target = str(data.get("target") or "content").lower()
        return _SEARCH_TARGET_MESSAGES.get(target, _TOOL_MESSAGES[name])

    return _TOOL_MESSAGES.get(name, _GENERIC_PROGRESS)
