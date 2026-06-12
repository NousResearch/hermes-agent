# Time (Т‑Банк) Safe Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Подключить Hermes к Slack‑совместимому мессенджеру Time (Т‑Банк) как plugin‑адаптер и гарантировать, что в Time‑сессиях агент физически не имеет опасных инструментов (shell, запись файлов, выполнение кода, browser, computer_use, home‑assistant, делегирование), сохранив ядро (чат/skills/memory/web/vision/session_search/todo/clarify/cron).

**Architecture:** Два слабосвязанных куска. (1) Адаптер Time = тонкий подкласс `SlackAdapter` с переопределяемыми «швами» для имён env‑токенов и `base_url`; транспорт — Socket Mode (websocket, фолбэк без webhook) или Events/webhook. (2) Профиль безопасности: новый toolset `corp_safe` + безусловное вычитание опасных toolset'ов для платформы `time` в единственном чокпоинте `_get_platform_tools`, плюс уже существующий defense‑in‑depth в `handle_function_call`.

**Tech Stack:** Python 3.11, `slack_bolt` (AsyncApp/AsyncWebClient/AsyncSocketModeHandler), pytest, существующая система toolset'ов Hermes.

---

## Файловая структура

- Modify: `gateway/platforms/slack.py` — добавить 4 переопределяемых seam‑метода (имена env, base_url, фабрики app/client). Без изменения поведения Slack.
- Create: `plugins/platforms/time/__init__.py`
- Create: `plugins/platforms/time/adapter.py` — `TimeAdapter(SlackAdapter)` + `register(ctx)`.
- Create: `plugins/platforms/time/plugin.yaml` — метаданные, requires_env/optional_env.
- Modify: `toolsets.py` — добавить toolset `corp_safe` и константу `CORP_DANGEROUS_TOOLSETS`.
- Modify: `hermes_cli/tools_config.py` — в `_get_platform_tools` безусловно вычитать опасные toolset'ы для restricted‑платформ (Time).
- Create: `tests/test_corp_safe_toolset.py`
- Create: `tests/test_time_platform_security.py`
- Create: `tests/test_time_adapter.py`

---

### Task 1: Seam‑методы в SlackAdapter (рефактор без смены поведения)

**Files:**
- Modify: `gateway/platforms/slack.py` (метод `connect`, ~строки 742–840; класс `SlackAdapter` ~строка 327)
- Test: `tests/test_time_adapter.py`

- [ ] **Step 1: Написать падающий тест на seam‑дефолты**

```python
# tests/test_time_adapter.py
from gateway.config import PlatformConfig, Platform
from gateway.platforms.slack import SlackAdapter

def _cfg():
    return PlatformConfig(platform=Platform.SLACK, enabled=True, token="xoxb-test")

def test_slack_seam_defaults():
    a = SlackAdapter(_cfg())
    assert a._app_token_env() == "SLACK_APP_TOKEN"
    assert a._api_base_url() is None  # Slack uses slack_bolt default endpoint
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `python -m pytest tests/test_time_adapter.py::test_slack_seam_defaults -v`
Expected: FAIL — `AttributeError: 'SlackAdapter' object has no attribute '_app_token_env'`

- [ ] **Step 3: Добавить seam‑методы в `SlackAdapter`** (сразу после `__init__`)

```python
    # ── Overridable seams for Slack-API-compatible platforms (e.g. Time) ──
    def _app_token_env(self) -> str:
        """Env var holding the Socket Mode app token."""
        return "SLACK_APP_TOKEN"

    def _api_base_url(self):
        """Web API base URL. None = slack_bolt default (https://slack.com/api/)."""
        return None

    def _make_async_app(self, token: str):
        """Build the AsyncApp, honoring a custom base_url when set."""
        base = self._api_base_url()
        if base:
            client = AsyncWebClient(token=token, base_url=base)
            return AsyncApp(token=token, client=client)
        return AsyncApp(token=token)

    def _make_web_client(self, token: str):
        """Build an AsyncWebClient, honoring a custom base_url when set."""
        base = self._api_base_url()
        if base:
            return AsyncWebClient(token=token, base_url=base)
        return AsyncWebClient(token=token)
```

- [ ] **Step 4: Перевести `connect()` на seam‑методы**

В `connect()` заменить:
```python
        app_token = os.getenv("SLACK_APP_TOKEN")
```
на:
```python
        app_token = os.getenv(self._app_token_env())
```
Заменить:
```python
            primary_token = bot_tokens[0]
            self._app = AsyncApp(token=primary_token)
```
на:
```python
            primary_token = bot_tokens[0]
            self._app = self._make_async_app(primary_token)
```
Заменить (в цикле по токенам):
```python
                client = AsyncWebClient(token=token)
```
на:
```python
                client = self._make_web_client(token)
```

- [ ] **Step 5: Запустить тест и существующие Slack‑тесты**

Run: `python -m pytest tests/test_time_adapter.py::test_slack_seam_defaults -v`
Expected: PASS
Run: `python -m pytest tests/ -k slack -q`
Expected: тот же результат, что до изменения (рефактор без смены поведения; новых падений нет)

- [ ] **Step 6: Commit**

```bash
git add gateway/platforms/slack.py tests/test_time_adapter.py
git commit -m "refactor(slack): add overridable seams for API base_url and token env"
```

---

### Task 2: TimeAdapter (подкласс SlackAdapter)

**Files:**
- Create: `plugins/platforms/time/adapter.py`
- Create: `plugins/platforms/time/__init__.py`
- Test: `tests/test_time_adapter.py`

- [ ] **Step 1: Написать падающий тест на переопределения Time**

```python
# tests/test_time_adapter.py  (добавить)
import os
from gateway.config import PlatformConfig, Platform

def test_time_adapter_overrides(monkeypatch):
    monkeypatch.setenv("TIME_API_BASE_URL", "https://time.tbank.ru/api/")
    from plugins.platforms.time.adapter import TimeAdapter
    cfg = PlatformConfig(platform=Platform.SLACK, enabled=True, token="t-bot")
    a = TimeAdapter(cfg)
    assert a._app_token_env() == "TIME_APP_TOKEN"
    assert a._api_base_url() == "https://time.tbank.ru/api/"

def test_time_make_web_client_uses_base_url(monkeypatch):
    monkeypatch.setenv("TIME_API_BASE_URL", "https://time.tbank.ru/api/")
    from plugins.platforms.time.adapter import TimeAdapter
    cfg = PlatformConfig(platform=Platform.SLACK, enabled=True, token="t-bot")
    a = TimeAdapter(cfg)
    client = a._make_web_client("t-bot")
    assert str(client.base_url).rstrip("/") == "https://time.tbank.ru/api"
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `python -m pytest tests/test_time_adapter.py::test_time_adapter_overrides -v`
Expected: FAIL — `ModuleNotFoundError: plugins.platforms.time.adapter`

- [ ] **Step 3: Создать `plugins/platforms/time/__init__.py`**

```python
```
(пустой файл — пакет‑маркер)

- [ ] **Step 4: Создать `plugins/platforms/time/adapter.py`**

```python
"""Time (Т‑Банк) platform adapter.

Time is Slack-API-compatible, so the adapter is a thin subclass of
``SlackAdapter`` that points the Slack Web API / Socket Mode transport at
Time's endpoints and reads Time-specific env vars.

Env vars:
    TIME_BOT_TOKEN       — bot token (Slack xoxb-equivalent)
    TIME_APP_TOKEN       — Socket Mode app token (Slack xapp-equivalent)
    TIME_API_BASE_URL    — Web API base URL (e.g. https://time.tbank.ru/api/)
    TIME_ALLOWED_USERS   — comma-separated allowed user IDs
    TIME_ALLOW_ALL_USERS — truthy = allow everyone (dev only)
    TIME_HOME_CHANNEL    — channel for cron/notification delivery
"""

import os
import logging
from typing import Optional

from gateway.platforms.slack import SlackAdapter
from gateway.config import Platform

logger = logging.getLogger(__name__)


class TimeAdapter(SlackAdapter):
    """Slack-compatible adapter for T-Bank's Time messenger."""

    def _app_token_env(self) -> str:
        return "TIME_APP_TOKEN"

    def _api_base_url(self) -> Optional[str]:
        return os.getenv("TIME_API_BASE_URL") or None


def _check_requirements() -> bool:
    try:
        import slack_bolt  # noqa: F401
        return True
    except Exception:
        return False


def _validate_config(cfg) -> bool:
    return bool(os.getenv("TIME_BOT_TOKEN")) and bool(os.getenv("TIME_API_BASE_URL"))


def _env_enablement():
    if not os.getenv("TIME_BOT_TOKEN"):
        return None
    extra = {"api_base_url": os.getenv("TIME_API_BASE_URL", "")}
    home = os.getenv("TIME_HOME_CHANNEL")
    if home:
        return {"extra": extra, "home_channel": {"chat_id": home}}
    return {"extra": extra}


def register(ctx):
    """Plugin entry point."""
    ctx.register_platform(
        name="time",
        label="Time",
        adapter_factory=lambda cfg: TimeAdapter(cfg),
        check_fn=_check_requirements,
        validate_config=_validate_config,
        required_env=["TIME_BOT_TOKEN", "TIME_APP_TOKEN", "TIME_API_BASE_URL"],
        install_hint="pip install slack-bolt",
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="TIME_HOME_CHANNEL",
        allowed_users_env="TIME_ALLOWED_USERS",
        allow_all_env="TIME_ALLOW_ALL_USERS",
        max_message_length=39000,
        emoji="🕐",
        platform_hint=(
            "You are chatting via Time (Т‑Банк corporate messenger). "
            "Time renders Slack-style mrkdwn. Keep replies professional."
        ),
    )
```

Примечание: `TimeAdapter` вызывает `super().__init__(config, Platform.SLACK)` через
наследование от `SlackAdapter`, поэтому отдельный enum `Platform.TIME` не требуется —
платформа различается по plugin‑ключу `"time"` в реестре. Если в ходе реализации
окажется, что внутренняя логика Slack завязана на `Platform.SLACK` так, что Time‑сессии
конфликтуют со Slack‑сессиями, добавить `Platform.TIME` через механизм plugin‑enum
реестра (см. `platform_registry.py`) — но НЕ редактировать ядровой enum напрямую.

- [ ] **Step 5: Запустить тесты Time‑адаптера**

Run: `python -m pytest tests/test_time_adapter.py -v`
Expected: PASS (все тесты файла)

- [ ] **Step 6: Создать `plugins/platforms/time/plugin.yaml`**

```yaml
name: time-platform
label: Time
kind: platform
version: 1.0.0
description: >
  Time (Т‑Банк) corporate messenger adapter for Hermes Agent.
  Time is Slack-API-compatible; this adapter subclasses the Slack
  adapter and points it at Time's endpoints.
author: densin
requires_env:
  - name: TIME_BOT_TOKEN
    description: "Bot token for Time (Slack xoxb-equivalent)"
    prompt: "Time bot token"
    password: true
  - name: TIME_APP_TOKEN
    description: "Socket Mode app token (Slack xapp-equivalent)"
    prompt: "Time app token"
    password: true
  - name: TIME_API_BASE_URL
    description: "Time Web API base URL (e.g. https://time.tbank.ru/api/)"
    prompt: "Time API base URL"
    password: false
optional_env:
  - name: TIME_ALLOWED_USERS
    description: "Comma-separated Time user IDs allowed to talk to the bot"
    prompt: "Allowed users (comma-separated)"
    password: false
  - name: TIME_ALLOW_ALL_USERS
    description: "Allow anyone to talk to the bot (dev only)"
    prompt: "Allow all users? (true/false)"
    password: false
  - name: TIME_HOME_CHANNEL
    description: "Channel for cron / notification delivery"
    prompt: "Home channel (or empty)"
    password: false
```

- [ ] **Step 7: Commit**

```bash
git add plugins/platforms/time/
git commit -m "feat(gateway): add Time (T-Bank) Slack-compatible platform adapter"
```

---

### Task 3: Toolset `corp_safe` + список опасных toolset'ов

**Files:**
- Modify: `toolsets.py` (словарь `TOOLSETS`, рядом с `"safe"` ~строка 337)
- Test: `tests/test_corp_safe_toolset.py`

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_corp_safe_toolset.py
from toolsets import resolve_toolset, CORP_DANGEROUS_TOOLSETS, TOOLSETS

DANGEROUS_TOOL_NAMES = {
    "terminal", "process", "read_terminal",
    "write_file", "patch",
    "execute_code", "computer_use",
    "browser_navigate", "browser_click", "browser_type",
    "ha_call_service", "delegate_task",
}

def test_corp_safe_has_no_dangerous_tools():
    tools = resolve_toolset("corp_safe")
    leaked = tools & DANGEROUS_TOOL_NAMES
    assert not leaked, f"corp_safe leaks dangerous tools: {leaked}"

def test_corp_safe_keeps_core_tools():
    tools = resolve_toolset("corp_safe")
    for core in ("web_search", "vision_analyze", "memory",
                 "session_search", "skills_list", "clarify", "todo"):
        assert core in tools, f"corp_safe missing core tool {core}"

def test_dangerous_toolsets_listed():
    for ts in ("terminal", "file", "code_execution", "computer_use",
               "browser", "homeassistant", "delegation"):
        assert ts in CORP_DANGEROUS_TOOLSETS
        assert ts in TOOLSETS  # name is real
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `python -m pytest tests/test_corp_safe_toolset.py -v`
Expected: FAIL — `ImportError: cannot import name 'CORP_DANGEROUS_TOOLSETS'` / unknown toolset `corp_safe`

- [ ] **Step 3: Добавить `corp_safe` в `TOOLSETS`** (после блока `"safe"`)

```python
    "corp_safe": {
        "description": (
            "Corporate-safe toolkit: chat, skills, memory, recall, web/vision, "
            "planning — NO terminal, file-write, code execution, browser, "
            "computer-use, home-assistant, or delegation."
        ),
        "tools": [],
        "includes": [
            "web", "vision", "image_gen",
            "skills", "memory", "session_search",
            "messaging", "todo", "clarify", "cronjob",
        ],
    },
```

- [ ] **Step 4: Добавить константу `CORP_DANGEROUS_TOOLSETS`** (после словаря `TOOLSETS`)

```python
# Toolsets a corporate-restricted platform (e.g. Time) must NEVER receive,
# because they can cause real-world / physical side effects:
#   terminal       — arbitrary shell command execution
#   file           — file write/patch (overwrite/delete)
#   code_execution — arbitrary Python (bypasses every other restriction)
#   computer_use   — desktop control
#   browser        — browser automation (acts as the user)
#   homeassistant  — controls physical smart-home devices
#   delegation     — a subagent could regain dangerous tools
CORP_DANGEROUS_TOOLSETS = frozenset({
    "terminal", "file", "code_execution",
    "computer_use", "browser", "homeassistant", "delegation",
})
```

- [ ] **Step 5: Запустить тесты**

Run: `python -m pytest tests/test_corp_safe_toolset.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add toolsets.py tests/test_corp_safe_toolset.py
git commit -m "feat(toolsets): add corp_safe toolset and CORP_DANGEROUS_TOOLSETS"
```

---

### Task 4: Безусловное вычитание опасных toolset'ов для Time (hard guarantee)

**Files:**
- Modify: `hermes_cli/tools_config.py` (`_get_platform_tools`, конец функции перед `return`)
- Test: `tests/test_time_platform_security.py`

- [ ] **Step 1: Написать падающий тест на чокпоинт**

```python
# tests/test_time_platform_security.py
from hermes_cli.tools_config import _get_platform_tools
from toolsets import resolve_toolset

DANGEROUS = {"terminal", "process", "write_file", "patch",
             "execute_code", "computer_use", "browser_navigate",
             "ha_call_service", "delegate_task"}

def test_time_platform_strips_dangerous_even_if_misconfigured():
    # Even if an admin wrongly grants the coding toolset to Time:
    config = {"platform_toolsets": {"time": ["coding"]}}
    tools = _get_platform_tools(config, "time")
    leaked = tools & DANGEROUS
    assert not leaked, f"Time leaked dangerous tools: {leaked}"

def test_non_time_platform_unaffected():
    config = {"platform_toolsets": {"slack": ["coding"]}}
    tools = _get_platform_tools(config, "slack")
    # Slack keeps terminal — the hard-deny is Time-only.
    assert "terminal" in tools
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `python -m pytest tests/test_time_platform_security.py -v`
Expected: FAIL — `test_time_platform_strips_dangerous_even_if_misconfigured` (terminal leaks)

- [ ] **Step 3: Добавить hard‑deny в конец `_get_platform_tools`**

Найти финальный `return` функции `_get_platform_tools` (возвращает множество `enabled_toolsets`/имён). Непосредственно перед ним вставить:

```python
    # ── Corporate hard-deny: restricted platforms can NEVER receive
    # dangerous toolsets, regardless of config/plugins. This is the single
    # chokepoint both gateway call sites flow through. ──
    from toolsets import CORP_DANGEROUS_TOOLSETS, resolve_toolset
    CORP_RESTRICTED_PLATFORMS = {"time"}
    if platform in CORP_RESTRICTED_PLATFORMS:
        dangerous_tool_names = set()
        for ts in CORP_DANGEROUS_TOOLSETS:
            dangerous_tool_names |= set(resolve_toolset(ts))
        # Remove both the toolset names and their expanded tool names.
        enabled_toolsets = {
            t for t in enabled_toolsets
            if t not in CORP_DANGEROUS_TOOLSETS and t not in dangerous_tool_names
        }
```

Примечание для исполнителя: переменная, которую возвращает функция, может
называться иначе (`enabled_toolsets`, `result`, `tools`). Подставить её
фактическое имя из `return <name>`. Если функция в нескольких ветках
возвращает разные множества — вынести фильтр в локальную функцию
`_apply_corp_hard_deny(names, platform)` и обернуть каждый `return`.

- [ ] **Step 4: Запустить тесты безопасности + corp_safe + slack**

Run: `python -m pytest tests/test_time_platform_security.py tests/test_corp_safe_toolset.py -v`
Expected: PASS
Run: `python -m pytest tests/ -k "slack or tools_config" -q`
Expected: без новых падений

- [ ] **Step 5: Commit**

```bash
git add hermes_cli/tools_config.py tests/test_time_platform_security.py
git commit -m "feat(security): hard-deny dangerous toolsets for Time platform"
```

---

### Task 5: Defense-in-depth — bridge не может вызвать опасный тулз в Time‑scope

**Files:**
- Test: `tests/test_time_platform_security.py` (добавить)

- [ ] **Step 1: Написать тест, что get_tool_definitions для corp_safe не отдаёт опасные схемы**

```python
# tests/test_time_platform_security.py  (добавить)
from model_tools import get_tool_definitions

def test_corp_safe_tool_defs_exclude_dangerous():
    defs = get_tool_definitions(enabled_toolsets=["corp_safe"]) or []
    names = set()
    for d in defs:
        fn = d.get("function") if isinstance(d, dict) else None
        if isinstance(fn, dict) and "name" in fn:
            names.add(fn["name"])
        elif isinstance(d, dict) and "name" in d:
            names.add(d["name"])
    for bad in ("terminal", "process", "execute_code",
                "write_file", "browser_navigate", "delegate_task"):
        assert bad not in names, f"corp_safe tool schema leaked {bad}"
```

- [ ] **Step 2: Запустить**

Run: `python -m pytest tests/test_time_platform_security.py::test_corp_safe_tool_defs_exclude_dangerous -v`
Expected: PASS (схемы опасных тулзов отсутствуют — подтверждает первый слой защиты)

Примечание: defense‑in‑depth моста `tool_search`→`handle_function_call`
уже реализован в ядре (`model_tools.handle_function_call`, проверка
`scoped_deferrable_names`). Отдельная реализация не требуется — этот тест
фиксирует, что `corp_safe` корректно сужает scope, на который опирается мост.

- [ ] **Step 3: Commit**

```bash
git add tests/test_time_platform_security.py
git commit -m "test(security): assert corp_safe scope excludes dangerous tool schemas"
```

---

### Task 6: Конфиг по умолчанию + документация запуска

**Files:**
- Modify: `cli-config.yaml.example` (блок `platform_toolsets`, ~строка 694)
- Create: `plugins/platforms/time/README.md`

- [ ] **Step 1: Добавить пример в `cli-config.yaml.example`** (в блок `platform_toolsets:`)

```yaml
  # Time (Т‑Банк) — corporate-safe profile. The platform additionally
  # hard-denies dangerous toolsets in code, so even if you widen this list
  # the agent never gets shell/file-write/code-exec/browser/etc.
  time: [corp_safe]
```

- [ ] **Step 2: Создать `plugins/platforms/time/README.md`**

```markdown
# Time (Т‑Банк) adapter

Time is Slack-API-compatible. This adapter subclasses the Slack adapter.

## Setup (local prototype)

```bash
export TIME_BOT_TOKEN="..."        # bot token
export TIME_APP_TOKEN="..."        # Socket Mode app token (no public webhook needed)
export TIME_API_BASE_URL="https://<time-host>/api/"
export TIME_ALLOW_ALL_USERS=true   # dev only
hermes gateway
```

## Security

Time sessions run the `corp_safe` toolset and a code-level hard-deny of
dangerous toolsets (`terminal`, `file`, `code_execution`, `computer_use`,
`browser`, `homeassistant`, `delegation`). The agent physically cannot
execute shell commands, write files, run code, drive a browser, or control
devices from Time. See `docs/superpowers/specs/2026-06-12-...-design.md`.

## Transport

- **Socket Mode (default):** websocket, no inbound webhook URL required —
  use this when corporate policy forbids public webhooks.
- **Events/webhook:** supported via the inherited Slack Events path when a
  signing secret + public endpoint are configured.
```

- [ ] **Step 3: Smoke‑проверка импорта плагина и регистрации**

Run:
```bash
python -c "import plugins.platforms.time.adapter as m; print('register' in dir(m) and 'TimeAdapter' in dir(m))"
```
Expected: `True`

- [ ] **Step 4: Commit**

```bash
git add cli-config.yaml.example plugins/platforms/time/README.md
git commit -m "docs(time): default corp_safe profile + adapter README"
```

---

### Task 7: Полный прогон и финальная проверка

- [ ] **Step 1: Прогнать все новые тесты**

Run: `python -m pytest tests/test_time_adapter.py tests/test_corp_safe_toolset.py tests/test_time_platform_security.py -v`
Expected: все PASS

- [ ] **Step 2: Проверить отсутствие регрессий в смежных областях**

Run: `python -m pytest tests/ -k "slack or toolset or tools_config or platform" -q`
Expected: без новых падений относительно базовой ветки

- [ ] **Step 3: Финальный commit (если остались несохранённые правки)**

```bash
git add -A && git commit -m "chore(time): finalize safe Time integration" || echo "nothing to commit"
```

---

## Self-Review (выполнено автором плана)

- **Покрытие спеки:** адаптер Time (Task 1–2, §4.1) ✓; транспорт Socket
  Mode/webhook (Task 2, plugin.yaml/README) ✓; `corp_safe` (Task 3, §3/§4.2)
  ✓; hard‑deny чокпоинт (Task 4, §4.2) ✓; defense‑in‑depth (Task 5, §6) ✓;
  тестовый план §7 → Tasks 3–7 ✓; конфиг по умолчанию (Task 6) ✓.
- **O‑1 (read‑only файлы):** решено НЕ включать `file` в `corp_safe`
  (консервативно). `read_file`/`search_files` отсутствуют в составе corp_safe.
- **Плейсхолдеры:** нет TBD/TODO; все шаги с кодом содержат код.
- **Согласованность имён:** `_app_token_env`, `_api_base_url`,
  `_make_async_app`, `_make_web_client`, `corp_safe`,
  `CORP_DANGEROUS_TOOLSETS`, `CORP_RESTRICTED_PLATFORMS` — используются
  единообразно во всех задачах.
