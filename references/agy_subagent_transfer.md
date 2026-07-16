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
- `agent/copilot_acp_client.py`
- `tools/delegate_tool.py`
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

- рабочий Hermes checkout и его Python 3.11 venv
- установленный и авторизованный `agy`
- запущенный локальный Antigravity service/language server
- `agy` доступен в PATH процесса Hermes

Для stateful agentapi обязательны:

```bash
export ANTIGRAVITY_LS_ADDRESS='<local Antigravity service URL>'
export ANTIGRAVITY_PROJECT_ID='<local project id>'
```

На исходной машине подтверждены:

```text
ANTIGRAVITY_LS_ADDRESS=http://localhost:61727
ANTIGRAVITY_PROJECT_ID=default-cli-project
```

Эти значения machine-specific. Не переносить их вслепую на другой компьютер.

Для Telegram/gateway запуска переменные должны присутствовать именно в окружении gateway-процесса до его старта. Экспорт в отдельном интерактивном shell не изменит уже запущенный gateway.

## 6. Targeted tests

```bash
cd /c/Path/To/hermes-agent
PYTHONPATH=. pytest \
  tests/agent/test_antigravity_agentapi_transport.py \
  tests/agent/test_copilot_acp_client.py -q
```

Ожидаемый результат для коммита `194c2143a`:

```text
24 passed
```

## 7. Direct live smoke

```bash
export ANTIGRAVITY_LS_ADDRESS='<local URL>'
export ANTIGRAVITY_PROJECT_ID='<local project id>'
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
export ANTIGRAVITY_LS_ADDRESS='<local URL>'
export ANTIGRAVITY_PROJECT_ID='<local project id>'
venv/Scripts/hermes.exe --provider openai-codex -m gpt-5.4 -t delegation -z \
  'Call delegate_task exactly once with goal="Reply with exactly AGY_AGENTAPI_DELEGATE_OK", acp_command="agy", acp_args=["agentapi"]. Return the delegated child result verbatim.'
```

Ожидаемый stdout:

```text
AGY_AGENTAPI_DELEGATE_OK
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

## 11. Контрольный список

1. Cherry-pick двух коммитов.
2. Убедиться, что `agy --version` работает из того же service account/environment.
3. Настроить два env-параметра agentapi.
4. Перезапустить Hermes/gateway.
5. Выполнить targeted tests.
6. Выполнить fresh-runtime `delegate_task` smoke.
7. Только после этого подключать transport к постоянным Telegram workflows.
