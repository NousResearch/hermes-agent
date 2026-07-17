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

## Требования окружения и runtime

`ANTIGRAVITY_PROJECT_ID` можно передать через env. Если он отсутствует, Hermes читает `~/.gemini/antigravity-cli/cache/default_project_id.txt`.

`ANTIGRAVITY_LS_ADDRESS` нельзя считать постоянной настройкой: каждый интерактивный `agy` runtime выбирает новый **случайный** HTTP-порт. Сохранённый User env вроде `http://localhost:61727` становится stale сразу после завершения того процесса, который выбрал этот порт.

Текущая реализация перед каждым agentapi turn:

1. проверяет configured address коротким TCP probe;
2. при stale address читает managed runtime state;
3. затем проверяет свежие Antigravity logs;
4. если живого runtime нет на Windows, запускает detached singleton daemon через текущий Hermes Python и `winpty.PtyProcess`;
5. ждёт новый случайный HTTP-порт в пределах общего caller timeout.

Поэтому новый Hermes process и перезапущенный gateway не требуют ручного обновления `ANTIGRAVITY_LS_ADDRESS`. После Windows reboot runtime поднимается автоматически при первом agentapi вызове.

Runtime-файлы:

- state: `~/.gemini/antigravity-cli/log/hermes-antigravity-runtime.json`
- log: `~/.gemini/antigravity-cli/log/hermes-antigravity-managed.log`
- singleton lock: `~/.gemini/antigravity-cli/log/hermes-antigravity-runtime.lock`

## Реализованная архитектура

### `agent/antigravity_runtime.py`

Отдельный runtime manager отвечает за bounded TCP probe, project-id fallback, random-port discovery, atomic state, detached daemon startup, Windows singleton lock, ConPTY output draining и автоматический restart дочернего `agy`.

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
  tests/agent/test_antigravity_runtime.py \
  tests/agent/test_antigravity_agentapi_transport.py \
  tests/agent/test_copilot_acp_client.py -q
```

Результат:

```text
39 passed in 2.10s
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
- stale static address replacement
- project-id cache fallback
- managed daemon auto-start and concurrent launch protection
- random-port parser and non-Windows actionable failure

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
venv/Scripts/hermes.exe --provider openai-codex -m gpt-5.6-sol -t delegation -z \
  'Call delegate_task exactly once with goal="Reply with exactly AGY_DURABLE_FRESH_OK", acp_command="agy", acp_args=["agentapi"]. Return the delegated child result verbatim.'
```

Фактический stdout:

```text
AGY_DURABLE_FRESH_OK
```

### Broad agent test suite

```text
4210 passed, 2 skipped, 37 failed
```

37 failures относятся к существующим Windows/path/temp/plugin/vision baseline-проблемам в других модулях; ни один failure не относится к agentapi transport или измененным targeted tests.

## Файлы реализации

- `agent/antigravity_agentapi_client.py`
- `agent/antigravity_runtime.py`
- `agent/copilot_acp_client.py`
- `tools/delegate_tool.py`
- `tests/agent/test_antigravity_runtime.py`
- `tests/agent/test_antigravity_agentapi_transport.py`
- `tests/agent/test_copilot_acp_client.py`

## Короткий чек-лист для другого ПК

1. Перенести оба базовых коммита и durable-runtime изменения.
2. Проверить `agy --version` и локальную авторизацию Antigravity.
3. Убедиться, что `default_project_id.txt` существует, либо задать `ANTIGRAVITY_PROJECT_ID`.
4. Не закреплять случайный `ANTIGRAVITY_LS_ADDRESS` как постоянную настройку.
5. Перезапустить Hermes-gateway и проверить `delegate_task(..., acp_command="agy", acp_args=["agentapi"])`.
6. Если `agentapi` недоступен, временно использовать `acp_args=[]`.

## Что осталось опционально

- подготовить upstream PR с более общим названием compatibility shim вместо исторического `CopilotACPClient`
- добавить live integration test под feature flag, если CI получит доступ к запущенному Antigravity service
