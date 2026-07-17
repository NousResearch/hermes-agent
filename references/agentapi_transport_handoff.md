# Реализовано: Antigravity `agentapi` transport для Hermes

Дата проверки: 2026-07-17
Базовый коммит print-mode интеграции: `4030c8d8a` (`feat: support agy delegated subagents`)
Коммит agentapi реализации: `194c2143a` (`feat: add agy agentapi transport`)

## Статус

Полноценный stateful transport реализован и проверен.

Доступны два режима:

```python
# Stateful agentapi transport
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

Если `agentapi` отсутствует в `acp_args`, сохраняется прежний путь `agy -p <prompt>`.

## Требования окружения

Для `agy agentapi` Hermes-процесс должен получить:

```bash
export ANTIGRAVITY_LS_ADDRESS='http://localhost:61727'
export ANTIGRAVITY_PROJECT_ID='default-cli-project'
```

На Windows можно задать их постоянно для текущего пользователя так:

```powershell
[Environment]::SetEnvironmentVariable('ANTIGRAVITY_LS_ADDRESS','http://localhost:61727','User')
[Environment]::SetEnvironmentVariable('ANTIGRAVITY_PROJECT_ID','default-cli-project','User')
```

После этого нужно открыть новый PowerShell / новый Hermes process. Это machine-specific значения. На другой машине их нужно получить из локального Antigravity runtime, а не копировать вслепую.

## Реализованная архитектура

### `agent/antigravity_agentapi_client.py`

Отдельный transport-класс отвечает за:

1. Первый turn:
   - `agy agentapi new-conversation ... <prompt>`
   - разбор `response.newConversation.conversationId`
2. Следующие turns:
   - baseline по максимальному `step_index`
   - `agy agentapi send-message ... <conversation_id> <prompt>`
3. Polling результата:
   - файл `~/.gemini/antigravity-cli/brain/<conversation_id>/.system_generated/logs/transcript.jsonl`
   - выбирается новый `source=MODEL`, `type=PLANNER_RESPONSE`, `status=DONE`
   - записи с `tool_calls`, незавершенные и шумовые записи игнорируются
4. Безопасность состояния:
   - полный lifecycle turn сериализован через `RLock`
   - после timeout/ошибки conversation ID сбрасывается
   - следующий запрос начинает новую conversation и не может принять запоздавший старый ответ
5. Надежность:
   - общий timeout budget для CLI-вызова и polling
   - sleep ограничивается оставшимся budget
   - torn/partial UTF-8 JSONL tail не ломает разбор уже завершенных строк

### `agent/copilot_acp_client.py`

Compatibility shim теперь:

- выбирает `antigravity-agentapi`, если `acp_command="agy"` и в `acp_args` есть `agentapi`
- сохраняет `antigravity-print` для старого режима
- держит один agentapi transport на логический client lifecycle
- сравнивает chat messages по стабильным fingerprints
- в следующую Antigravity turn отправляет только новый message delta, не дублируя уже существующую conversation history
- при divergence/compression нового chat history сбрасывает conversation и отправляет полный prompt
- очищает state при `close()` и при transport failure

### `tools/delegate_tool.py`

Описание `acp_args` документирует оба режима Antigravity:

- `['agentapi']` — persistent conversation
- `[]` — one-shot print fallback

## Реально подтвержденный CLI-контракт

Создание:

```bash
agy agentapi new-conversation --model=flash --title='Hermes task' '<prompt>'
```

Ответ содержит:

```json
{"response":{"newConversation":{"conversationId":"..."}}}
```

Продолжение:

```bash
agy agentapi send-message --title='Hermes task' '<conversation_id>' '<prompt>'
```

Финальный ответ появляется асинхронно в transcript JSONL.

## Проверки

### Targeted unit tests

```bash
PYTHONPATH=. pytest \
  tests/agent/test_antigravity_agentapi_transport.py \
  tests/agent/test_copilot_acp_client.py -q
```

Результат:

```text
24 passed in 1.98s
```

Покрыты:

- create/send/poll lifecycle
- malformed JSON и missing env
- timeout reset
- stale-response isolation
- concurrent turns и lazy-init race
- message delta
- torn UTF-8 tail
- bounded polling sleep
- `close()` reset
- print-mode regression

### Direct live two-turn transport

Conversation:

```text
0ed60f9a-ff0f-48bf-ade9-ee4446071c8a
```

Фактический результат:

```text
first=HERMES_AGENTAPI_LIVE_ONE
second=HERMES_AGENTAPI_LIVE_TWO
```

Transcript подтвердил, что второй turn содержал только новый user delta, без повторной отправки первой history.

### Fresh Hermes `delegate_task` smoke

Команда:

```bash
export ANTIGRAVITY_LS_ADDRESS='http://localhost:61727'
export ANTIGRAVITY_PROJECT_ID='default-cli-project'
venv/Scripts/hermes.exe --provider openai-codex -m gpt-5.4 -t delegation -z \
  'Call delegate_task exactly once with goal="Reply with exactly AGY_AGENTAPI_DELEGATE_OK", acp_command="agy", acp_args=["agentapi"]. Return the delegated child result verbatim.'
```

Фактический stdout:

```text
AGY_AGENTAPI_DELEGATE_OK
```

### Broad agent test suite

```text
4210 passed, 2 skipped, 37 failed
```

37 failures относятся к существующим Windows/path/temp/plugin/vision baseline-проблемам в других модулях; ни один failure не относится к agentapi transport или измененным targeted tests.

## Файлы реализации

- `agent/antigravity_agentapi_client.py`
- `agent/copilot_acp_client.py`
- `tools/delegate_tool.py`
- `tests/agent/test_antigravity_agentapi_transport.py`
- `tests/agent/test_copilot_acp_client.py`

## Короткий чек-лист для другого ПК

1. Применить оба коммита: `4030c8d8a` и `194c2143a`.
2. Задать user env на целевой машине:
   - `ANTIGRAVITY_LS_ADDRESS`
   - `ANTIGRAVITY_PROJECT_ID`
3. Открыть новый PowerShell / перезапустить Hermes-gateway.
4. Проверить `agy agentapi new-conversation`.
5. Проверить `delegate_task(..., acp_command="agy", acp_args=["agentapi"])`.
6. Если `agentapi` недоступен, временно использовать `acp_args=[]`.

## Что осталось опционально

- вынести machine-specific Antigravity service discovery в отдельную конфигурацию/auto-discovery
- подготовить upstream PR с более общим названием compatibility shim вместо исторического `CopilotACPClient`
- добавить live integration test под feature flag, если CI получит доступ к запущенному Antigravity service
