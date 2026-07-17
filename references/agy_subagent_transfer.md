# Перенос Hermes-интеграции Antigravity `agy` на другую машину

## 1. Коммиты

Применять по порядку:

1. `4030c8d8a` — `feat: support agy delegated subagents`
   - базовый внешний CLI shim
   - one-shot fallback через `agy -p`
2. `194c2143a` — `feat: add agy agentapi transport`
   - полноценный stateful `agy agentapi`
   - create/send/poll lifecycle
   - message delta, locking, timeout reset и тесты

## 2. Что дает перенос

После обоих коммитов доступны два режима:

```python
# Persistent conversation transport
delegate_task(
    goal="Return exactly AGY_OK",
    acp_command="agy",
    acp_args=["agentapi"],
)

# One-shot fallback
delegate_task(
    goal="Return exactly AGY_OK",
    acp_command="agy",
    acp_args=[],
)
```

## 3. Файлы реализации

- `agent/antigravity_agentapi_client.py`
- `agent/antigravity_runtime.py`
- `agent/copilot_acp_client.py`
- `tools/delegate_tool.py`
- `tests/agent/test_antigravity_runtime.py`
- `tests/agent/test_antigravity_agentapi_transport.py`
- `tests/agent/test_copilot_acp_client.py`

Отдельно, вне git checkout Hermes, могут переноситься skills:

- `skills/productivity/antigravity-cli/`
- `skills/autonomous-ai-agents/generic-acp-subagents/`
- `skills/autonomous-ai-agents/generic-acp-cli-subagents/`

Копирование skills без кодовых коммитов не добавит сам transport.

## 4. Git-перенос

```bash
cd /c/Path/To/hermes-agent
git fetch <remote-with-commits>
git cherry-pick 4030c8d8a
git cherry-pick 194c2143a
```

Если коммиты еще не опубликованы в remote, перенести их через bundle/patch либо скопировать перечисленные файлы с сохранением структуры.

## 5. Требования целевой машины

Минимум:

- рабочий Hermes checkout и его Python 3.11 venv;
- установленный и авторизованный `agy`;
- `agy` доступен в PATH процесса Hermes;
- пакет `winpty`, доступный в Hermes venv на Windows.

`ANTIGRAVITY_PROJECT_ID` можно передать через env. При его отсутствии используется `~/.gemini/antigravity-cli/cache/default_project_id.txt`.

Не переносить и не закреплять `ANTIGRAVITY_LS_ADDRESS` как постоянное значение: HTTP-порт выбирается каждым `agy` runtime случайно. Durable runtime manager валидирует configured address, ищет живой порт в managed state/logs и при необходимости автоматически запускает detached singleton daemon через ConPTY. После gateway restart или Windows reboot первый agentapi вызов восстанавливает runtime без ручного обновления env.

Managed state находится в:

```text
~/.gemini/antigravity-cli/log/hermes-antigravity-runtime.json
```

## 6. Targeted tests

```bash
cd /c/Path/To/hermes-agent
PYTHONPATH=. pytest \
  tests/agent/test_antigravity_runtime.py \
  tests/agent/test_antigravity_agentapi_transport.py \
  tests/agent/test_copilot_acp_client.py -q
```

Подтверждённый результат для durable runtime:

```text
39 passed
```

## 7. Direct live smoke

```bash
PYTHONPATH=. python - <<'PY'
from agent.copilot_acp_client import CopilotACPClient

client = CopilotACPClient(
    acp_command="agy",
    acp_args=["agentapi", "--model=flash", "--title=Hermes transport smoke"],
)
try:
    first = client.chat.completions.create(
        model="flash",
        messages=[{"role": "user", "content": "Reply exactly LIVE_ONE"}],
        timeout=180,
    )
    first_text = first.choices[0].message.content
    second = client.chat.completions.create(
        model="flash",
        messages=[
            {"role": "user", "content": "Reply exactly LIVE_ONE"},
            {"role": "assistant", "content": first_text},
            {"role": "user", "content": "Reply exactly LIVE_TWO"},
        ],
        timeout=180,
    )
    print(first_text)
    print(second.choices[0].message.content)
finally:
    client.close()
PY
```

## 8. Fresh Hermes delegation smoke

```bash
venv/Scripts/hermes.exe --provider openai-codex -m gpt-5.6-sol -t delegation -z \
  'Call delegate_task exactly once with goal="Reply with exactly AGY_DURABLE_FRESH_OK", acp_command="agy", acp_args=["agentapi"]. Return the delegated child result verbatim.'
```

Ожидаемый stdout:

```text
AGY_DURABLE_FRESH_OK
```

## 9. Если agentapi недоступен

Временно использовать one-shot fallback:

```python
delegate_task(
    goal="...",
    acp_command="agy",
    acp_args=[],
)
```

Это запускает `agy -p`, не требует `ANTIGRAVITY_LS_ADDRESS`/`ANTIGRAVITY_PROJECT_ID`, но не сохраняет stateful conversation между turns.

## 10. Что не копировать автоматически

Не переносить без отдельной необходимости:

- `~/.gemini/antigravity-cli/` целиком
- auth tokens и session credentials
- transcript/conversation databases
- machine-specific service URLs

На целевой машине лучше выполнить локальную авторизацию и получить собственные runtime-параметры.

## 11. Быстрый чек-лист переноса

1. Перенести два базовых коммита и durable-runtime изменения.
2. Проверить `agy --version`, локальную авторизацию и project-id cache/env.
3. Не сохранять случайный `ANTIGRAVITY_LS_ADDRESS` как постоянную конфигурацию.
4. Прогнать targeted tests.
5. Перезапустить Hermes-gateway.
6. Проверить fresh-runtime `delegate_task(..., acp_command="agy", acp_args=["agentapi"])`.
7. Только после этого подключать transport к постоянным Telegram workflows.

## 12. Контрольный список

1. Перенести базовые коммиты и durable-runtime изменения.
2. Убедиться, что `agy --version` работает из того же service account/environment.
3. Проверить `ANTIGRAVITY_PROJECT_ID` или `default_project_id.txt`.
4. Выполнить targeted tests.
5. Перезапустить Hermes/gateway.
6. Выполнить fresh-runtime `delegate_task` smoke без ручного экспорта LS address.
7. Только после этого подключать transport к постоянным Telegram workflows.
