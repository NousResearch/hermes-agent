# Symphony: установка и эксплуатация

## Назначение

Symphony добавляет безопасный контур задач разработки поверх Hermes Kanban:

```text
Telegram → DevelopmentTask → Kanban worktree → worker → review → PR
```

Kanban остаётся единственным диспетчером worktree и worker-процессов. Symphony не запускает merge, production deploy, миграции или PR approval.

## Предварительные условия

1. Отдельный профиль для development-бота:
   ```text
   hermes profile create symphony
   ```
2. В profile `.env` должны быть только секреты:
   ```text
   TELEGRAM_BOT_TOKEN=...
   ```
   Не добавляйте токены в `config.yaml`, карточки задач или документацию.
3. Зарегистрируйте Hermes Project для каждого репозитория, который разрешено обрабатывать.
4. Для GitHub PR настройте `gh auth login` в account/environment, в котором будет работать worker. GitLab MR сейчас не поддержан adapter-ом.

### Hermes-native worker

Symphony запускает не Codex и не сторонний coding-agent, а обычный Hermes CLI в отдельном subprocess:

```text
hermes -p <assignee> --cli chat -q "work kanban task <task-id>"
```

Worker получает только task-scoped Kanban environment: `HERMES_HOME`, `HERMES_KANBAN_TASK`, `HERMES_KANBAN_WORKSPACE`, board/database paths и `TERMINAL_CWD`. Родительские API keys и messaging tokens scrubbed перед spawn; секреты worker читает из назначенного Hermes profile. Не используйте `C:\Users\Home\\.codex\\config.toml` для Symphony.

## Операторский путь

1. В Telegram: `🛠 Разработка → ➕ Новая задача`.
2. Введите:
   ```text
   <Hermes Project> | <точный текст задачи>
   ```
3. Нажмите `Подготовить`, затем `В очередь`.
4. Проверьте состояние:
   ```text
   hermes symphony status
   hermes symphony show <development-task-id>
   ```
5. Для controlled локального tick используйте:
   ```text
   hermes symphony run-once --max-concurrency 1
   ```
   В production обычно работает уже существующий Gateway-embedded Kanban dispatcher.
6. Worker работает только в task-specific git worktree. При первом запуске Symphony создаёт `WORKFLOW.md`, если оператор не создал свой.
7. Если задача получила `needs_clarification`, оператор может ответить без Telegram-кнопки:
   ```text
   hermes symphony answer-clarification <task-id> "точный ответ на уточнение"
   ```
   Ответ redacted и сохраняется durable; после этого задача возвращается в `ready_for_development`.
8. После реальных тестов и evidence подготовьте GitHub PR:
   ```text
   hermes symphony publish-review <task-id> \
     --commit-message "feat: description" \
     --test-summary "pytest: 12 passed" \
     --result-summary "Что изменено и ограничения"
   ```
   Эта команда не merge'ит и не deploy'ит.

## Evidence contract

Worker обязан оставить machine-readable artifact внутри своего worktree:

```text
.symphony/<task-id>-result.json
```

Для стандартного Hermes-native worker artifact создаётся непосредственно через `kanban_complete(evidence={...})`; отдельная ручная запись JSON не требуется. `evidence.tests` обязан содержать реальные команды и нулевые exit codes. Worker должен вызвать `kanban_complete` только после успешного создания и validation artifact.

Artifact содержит безопасные поля: task ID, branch, PR URL, изменённые файлы, тестовые команды/result, summary и limitations. Значения redacted before write. Не указывайте secrets, full logs или environment dumps.

`kanban_complete` сам по себе не заменяет Symphony evidence artifact. Если Kanban уже стал `done`, но `.symphony/<task-id>-result.json` отсутствует или не проходит schema validation, `hermes symphony reconcile` оставляет Symphony task в текущем non-terminal статусе и записывает `task.completion_blocked`; это предотвращает ложное подтверждение результата.

## Recovery и rollback

- Worker без heartbeat переводится в `failed`; recovery сначала завершает только durable `worker_pid` этого run, затем используйте `▶️ Продолжить` после проверки причины.
- Для зависшего validation timeout сохраняется failure artifact; `validation.check` содержит только redacted command, `cwd=task workspace`, exit code и `timed_out`, без env или сырых логов.
- `validating → paused → queued` допустим: pause сначала проверяет durable transition, затем останавливает Kanban task; активный AgentRun помечается `paused`, а следующий dispatch создаёт новый run.
- Старый worker callback с устаревшим `run_id` не может обновить heartbeat нового retry.
- Если процесс падает после claim, но до создания `AgentRun`, устаревший `provisioning_workspace` переводится в `failed` с `agent.recovered(reason=workspace_claim_expired)`; повторный запуск возможен только через явный resume.
- `hermes symphony cancel <task-id>` останавливает и архивирует связанный Kanban task.
- Для rollback перестаньте ставить задачи в очередь. Не удаляйте worktree автоматически; сначала сохраните evidence и проверьте branch.
- Symphony state: `$HERMES_HOME/symphony/symphony.db`. Он additive и не заменяет Kanban/state DB.

## Troubleshooting

| Симптом | Проверка | Безопасное действие |
|---|---|---|
| Задача не запускается | `hermes symphony status`; Kanban board/task | убедиться, что project существует и worker profile назначен |
| Нет worktree | `hermes symphony show <task-id>` | проверить, что project root — git repository |
| PR не создан | `gh auth status` в worktree | настроить GitHub auth; не вставлять token в task/Telegram |
| Worker завис | AgentRun last progress/status | отменить или дождаться stale recovery, затем resume |
| Telegram не отвечает | `hermes --profile symphony gateway status` | проверить Gateway/logs; не перезапускать при активных задачах без оценки состояния |

## Установочная проверка

```text
scripts/run_tests.sh tests/hermes_cli/test_symphony.py tests/hermes_cli/test_symphony_runner.py tests/gateway/test_telegram_development_menu.py tests/gateway/test_telegram_task_menu.py -q
python -m hermes_cli.main symphony status
hermes --profile symphony gateway status
```
