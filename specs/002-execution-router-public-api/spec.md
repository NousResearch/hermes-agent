# Product Specification — Hermes Public `execution_router` Plugin API

Stage: PRODUCT
Status: ACCEPTED
Accepted-Draft-SHA256: `b55b5a1fc9eb4e9c39b67a4a6df22b6a5f7dd82d22d3641105793912f7e3abc7`
Accepted-Amendment-PA-001-SHA256: `5cc39122232ca68ff406f73e76a55d2ed76191175bdcae69a9998d990cc04b9f`
Owner-Acceptance: Telegram current session — «Принять canonical product spec по указанному SHA-256»
Amendment-Acceptance: Telegram current session — «Принять PA-001 по указанному SHA-256»
Accepted-Correction-C9-SHA256: `8c46a9e0083edd9dd6b8591664e62d530f1c80d6e2692c49b75cde1080db2223`
Correction-C9-Acceptance: Telegram current session — «Хорошо. Делаем по твоей рекомендации, а это полезное дополнение оставим для следующей версии»
Accepted-Correction-C10-SHA256: `f714322813b498a778059d691b2ebb2d953aa297d4a10631d8ff39c3a926ba0a`
Correction-C10-Acceptance: Telegram current session — «Минимальный T005: existing claim + defer в той же lane»
Project-Version: 0.1.1
Change-Level: patch
Traceability-Schema: 1
Spec-Version: 0.1.1
Accepted-Source-Draft-SHA256: `b6be7be534636d84d09e93da1893a978c09c2ef7560c13e532b81374552a6434`
Canonicalization: accepted source behavior preserved; requirement namespace mechanically normalized from `PR-xxx` to `FR-xxx`, and FR section headings mechanically normalized to Project Standard requirement records
Implementation-Authorization: DISABLED

## Purpose

Добавить в Hermes универсальный policy-neutral публичный plugin API, позволяющий одному явно разрешённому provider предложить immutable per-attempt execution route до credential binding и создания исполнителя. Product-specific model selection policy остаётся во внешнем устанавливаемом плагине.

## External project and runtime boundaries

- EXT-001: hermes-multimodel-routing

Все существующие упоминания Feature 002 в этой спецификации относятся к Feature 002-omh-derived-enhancements зарегистрированного project EXT-001. EXT-001 является project-потребителем execution_router и единственным владельцем product-specific routing policy, model capability profiles, execution chains, plugin-provider/adapter, локализованных объяснений, compatibility qualification, pilot и LIVE-решений. Проект hermes-execution-router-api владеет только policy-neutral публичным host API и не принимает authority этих consumer-механизмов.

## Current upstream evidence

- Inspected latest upstream `main`: `110736c0bc9fd249f1ce7f7ca5d353040f640be6` (2026-09-10T15:26:17Z); complete `execution_router` contract absent.
- Installed/runtime-authoritative checkout remains `b2aa855b626ff8688eb34b95c60ee8b6a4af3679`, Hermes v0.21.1 (2026.9.7), inspected read-only.
- Upstream issue [#41190](https://github.com/NousResearch/hermes-agent/issues/41190) proposed a broader all-call-site router and was closed `not_planned`.
- Open [PR #96051](https://github.com/NousResearch/hermes-agent/pull/96051) covers only a main-turn `pre_llm_call` routing override and does not satisfy native-child/Kanban coverage or this spec's full lifecycle contract.
- This proposal is limited to the concrete consumer required by Feature 002 and adds no routing classifier, model tiers, heuristics, policy, config defaults, dispatcher, gateway or store to Hermes core.

## Accepted block provenance

The owner accepted the following exact product blocks without changes:
- Block 1: SHA-256 `2f66bb86f996110fea9984c1c18ed22ba122d2ac14cb417d068291f99b452b7c`.
- Block 2: SHA-256 `9f6d54e896a4bd24c9e5da3e3f87811fdb5702ca550b089200128882e82b19d4`.
- Block 3: SHA-256 `1bd6204fd69e8b7601fd830de047b3ed6c6fae137ab1bcce56e05f88460c6e90`.
- Block 4: SHA-256 `57b414bce3409f7ede0695410b784fa7d3ba7adf466fbfde473015fa29408087`.
- Block 5: SHA-256 `a076beb5f0c3f5faf1c6178a9ffe1798a11a7509c95888e428317ca90455337d`.

- FR-001: **Назначение**

Hermes предоставляет универсальную публичную точку расширения `execution_router`, через которую ровно один явно разрешённый plugin-provider может предложить маршрут для одного нового execution attempt до создания исполнителя и до привязки provider credentials.

Host Hermes остаётся единственным владельцем проверки решения, разрешений, credentials, создания исполнителя, lifecycle, fallback/escalation запуска и событий фактического маршрута. Плагин только предлагает route decision и не выполняет dispatch самостоятельно.

- FR-002: **Минимальное покрытие**

Один и тот же продуктовый контракт должен применяться к трём execution kinds, необходимым Feature 002:

1. `main_turn` — новый attempt основной пользовательской задачи;
2. `native_child` — новый Hermes child/subagent attempt;
3. `kanban_worker` — новый attempt штатного Kanban worker.

Пока любой из этих путей не поддерживает контракт, интеграция Feature 002 остаётся `DEFER` для этого пути и не изображает частичное покрытие как полную T005.

- FR-003: **Поведение без router-provider**

Если provider не зарегистрирован, не разрешён владельцем или возвращает `None`, Hermes выполняет существующий штатный маршрут byte-for-byte/семантически без изменений. API не включает автоматическую маршрутизацию по умолчанию и не добавляет model tiers, classifier или routing policy в core.

- FR-004: **Жёсткие границы**

`execution_router` не может:

- выдавать или читать credentials;
- расширять tools, permissions, approvals, workspace или write scope;
- менять общий `config.yaml` или профиль;
- выбирать задачу, менять её цель или объявлять completion;
- создавать собственный dispatcher, gateway, scheduler, task store или retry loop;
- применять product-specific complexity/model policy внутри Hermes core.

- FR-005: **Стадийность**

Разработка API ведётся отдельно от Feature 002 и отдельно от установленного Hermes. Локальный implementation branch/project не означает merge, publication, installation, pilot или LIVE. Feature 002 подключается к API только после его отдельной проверки и подтверждения exact host version/contract.

- FR-006: **Момент решения**

Hermes вызывает `execution_router` после определения и авторизации session/task/execution context, но до выбора target credentials и до создания нового execution attempt. Router получает immutable credential-free request и не получает live agent, writable task, config object или dispatcher handle.

- FR-007: **Явные результаты**

Router возвращает ровно один из трёх результатов:

1. `route` — предлагает конкретный model/provider/reasoning route для нового attempt;
2. `pass_through` — явно отказывается от выбора, после чего Hermes использует существующий штатный маршрут;
3. `stop` — запрещает начинать attempt и возвращает пользователю видимую причину.

Неявный `None` может быть совместимым представлением `pass_through`, но не должен означать успешное применение route.

- FR-008: **Приоритет explicit pin**

Явный pin пользователя или канонической Kanban-задачи является жёстким ограничением. Router видит credential-free pin, но не может заменить или ослабить его. Hermes отклоняет несовместимый `route`; router может только подтвердить совместимый маршрут, выбрать `pass_through` либо `stop`.

Pin включает каждое явно заданное измерение отдельно: model, provider и reasoning. Неуказанное измерение не считается pin и может быть предложено router, если это не создаёт конфликт с указанными измерениями.

- FR-009: **Host validation**

До запуска Hermes проверяет предложенный route против:

- explicit pins;
- доступных и разрешённых provider/model targets;
- разрешений, tool/workspace ceilings и execution kind;
- формата и версии контракта;
- attempt identity и отсутствия повторного применения решения к другому attempt.

Router не может расширить eligibility. Невалидный или конфликтующий route не применяется.

- FR-010: **Ошибка зарегистрированного router**

Если зарегистрированный и разрешённый router выбросил исключение, превысил host-owned timeout либо вернул malformed/conflicting decision, новый attempt не запускается. Hermes возвращает видимое состояние `router_error` с безопасной причиной. Он не делает скрытый `pass_through`, потому что это могло бы обойти обязательную routing policy.

Отсутствие зарегистрированного/разрешённого router по-прежнему сохраняет штатное поведение согласно принятому блоку 1.

- FR-011: **Изоляция решения**

Решение связано с одним immutable execution/attempt ID. Оно не записывается как новый profile default, не меняет параллельные sessions и не может быть повторно использовано для другого запуска. Для `main_turn` и `kanban_worker` новый fallback/escalation attempt требует нового host-mediated решения. В v1 router-selected `native_child`, которому для fallback нужна смена route/model, завершается bounded видимой ошибкой без автоматического replacement/resubmit внутри того же `delegate_task` unit; последующий обычный parent `delegate_task` является новой execution и маршрутизируется заново.

- FR-012: **Единственный активный владелец**

Для одного execution scope Hermes допускает ровно один активный `execution_router` provider. Второй provider не получает неявный приоритет и не образует цепочку router-плагинов: регистрация или активация завершается видимым конфликтом до первого routed attempt.

Отключение или удаление активного provider возвращает Hermes к штатному поведению для будущих attempts и не переписывает историю уже начатых или завершённых attempts.

- FR-013: **Явное разрешение владельца**

Plugin не получает routing authority только из факта установки. Активация `execution_router` требует отдельного host-owned consent для точного plugin/provider identity и версии capability contract. Обновление, добавляющее или расширяющее routing capability, не наследует разрешение молча.

Consent разрешает только предложение route decision. Он не разрешает credentials, новые tools, permissions, workspace, delivery, completion или LIVE rollout конкретной продуктовой policy.

- FR-014: **Requested / accepted / actual**

Hermes создаёт связанные структурированные события минимум для следующих состояний:

1. `route_requested` — host передал credential-free request активному router;
2. `route_accepted` — host проверил decision и зафиксировал маршрут нового attempt;
3. `route_started` — исполнитель действительно создан с фактическими model/provider/reasoning;
4. `route_not_started` — запуск остановлен до создания исполнителя;
5. `route_finished` — attempt завершился с фактическим route identity и terminal state.

События различают requested, accepted и actual route; один этап не считается доказательством другого.

- FR-015: **Корреляция**

Каждое событие содержит opaque root/task/execution/attempt IDs, execution kind, router identity, decision/result state и временную привязку. Эти идентификаторы выдаёт и связывает Hermes. Плагин не создаёт параллельный execution graph или task store.

- FR-016: **Причина и пользовательское уведомление**

Router может вернуть bounded безопасную reason-code/reason-text пару. Hermes сохраняет её рядом с decision и предоставляет штатным поверхностям до старта attempt либо при `stop/router_error`.

Generic API не содержит русскую или иную product-specific формулировку. Локализованный текст остаётся ответственностью feature plugin/surface adapter; host гарантирует структурированные поля и безопасный fallback-текст, если plugin text отсутствует или отклонён.

- FR-017: **Минимизация данных**

Route events не содержат credentials, approval tokens, полный prompt, скрытое reasoning или произвольный plugin payload. Task text передаётся router только отдельным явно ограниченным полем контракта, если это будет принято в следующем блоке; по умолчанию используются структурированные task attributes и opaque IDs.

- FR-018: **Read-only наблюдение**

Router и другие разрешённые observers могут читать связанные route events через публичный read-only контракт. Чтение события не даёт права изменить attempt, повторить dispatch, объявить completion или изменить canonical task state.

- FR-019: **Credential-free task envelope**

Router получает immutable bounded envelope только для нового attempt:

- opaque root/task/execution/attempt IDs;
- execution kind и platform/surface class;
- bounded task instruction text, необходимый plugin-owned classifier;
- признак truncation, исходный размер и host-computed digest текста;
- explicit model/provider/reasoning pins;
- текущий штатный route candidate;
- credential-free eligibility projection разрешённых model/provider targets;
- bounded outcome/reason предыдущего attempt только для fallback/escalation.

Router не получает system prompt, conversation history, credentials, approval tokens, writable task/config objects, live agent или dispatcher handles.

- FR-020: **Text disclosure**

Доступ к bounded task instruction text является частью отдельного consent на `execution_router`. Host передаёт только текст, уже авторизованный для данного execution kind, и применяет существующие redaction/size rules до вызова router.

Если текст усечён или недостаточен, router может вернуть `pass_through` либо `stop`; он не должен изображать полную классификацию. Digest служит привязкой решения к exact bounded input, а не способом восстановить скрытый текст.

- FR-021: **Eligibility не расширяется**

Eligibility projection создаёт Hermes после проверки профиля, provider availability, operator policy и explicit constraints. Она не содержит API keys, endpoints с secrets или credential metadata. Router может выбрать только из projection либо вернуть `pass_through/stop`.

Host повторно проверяет projection revision/freshness непосредственно перед созданием attempt. Устаревшее решение не применяется.

- FR-022: **Нерекурсивность**

Вызов router не может сам создавать routed execution. Контекст `execution_router` не предоставляет `ctx.llm`, subagent launch, tools или иной host dispatch handle для принятия решения. Попытка инициировать вложенный routing через публичный host API отклоняется как recursion.

Обычная full-trust природа Python plugins не изображается sandbox-гарантией: capability contract ограничивает предоставленные host surfaces, но не утверждает, что произвольный вредоносный plugin-код технически изолирован от ОС.

- FR-023: **Deadline и late result**

Host задаёт короткий bounded deadline ожидания route decision. Если decision не получен вовремя:

- attempt не запускается;
- создаётся `router_error/timeout` event;
- поздний decision окончательно отбрасывается и не может быть применён к текущему или следующему attempt;
- пользователь получает безопасную видимую причину.

Отмена in-process/async callback является cooperative. Версия 1 не обещает принудительное завершение произвольного Python-кода, descendant cleanup или sandbox/process isolation. Provider обязан возвращать решение локально и быстро; network/model/tool work внутри router callback не входит в поддерживаемый контракт.

- FR-024: **Cost boundary**

`execution_router` API не предоставляет оплачиваемый model call или tools во время решения. Host учитывает только собственное время ожидания; self-reported plugin cost не используется как authority. Если product plugin хочет предварительную LLM-классификацию, она должна происходить через отдельный будущий host-owned контракт и authoritative receipt, а не внутри router callback.

- FR-025: **Determinism and idempotency**

Повторная доставка одного immutable request ID до запуска возвращает то же принятое решение либо тот же terminal router error. Host применяет решение не более одного раза. Изменение attempt ID, pins, bounded-input digest или eligibility revision требует нового решения.

- FR-026: **Immutable route внутри attempt**

После `route_accepted` Hermes привязывает validated model/provider/reasoning и route identity к одному новому attempt. Этот маршрут не меняется внутри attempt. Provider error, rate limit, timeout или verification conflict не превращают тот же attempt в исполнение другой моделью.

- FR-027: **Failure, fallback и escalation**

Если attempt с router-selected route не может продолжаться, Hermes завершает его наблюдаемым terminal state. Для `main_turn` и `kanban_worker` новый attempt создаётся только когда соответствующая штатная authority разрешает retry/fallback/escalation; новый request получает bounded failure context и новый attempt ID, а active router заново возвращает `route | pass_through | stop`.

Для `native_child` v1 не создаёт и не resubmit-ит replacement внутри того же `delegate_task` unit: fallback, требующий смены route/model, завершает child attempt bounded видимым failure. Parent может позднее выполнить обычный новый delegate call; это новая execution с новой начальной route resolution.

Старое решение не переносится автоматически. Router не решает сам, разрешён ли повторный запуск, не увеличивает retry/depth/cost budget и не объявляет verification outcome. No-router и explicit `pass_through` сохраняют native in-agent fallback для каждого execution kind.

- FR-028: **Native pass-through semantics**

Если router явно выбрал `pass_through`, Hermes сохраняет существующее штатное поведение этого execution path, включая его native failure handling. Такое исполнение помечается как `native/pass_through`, а не как router-selected route.

Feature plugin, которому требуется гарантия нового attempt на fallback/escalation, не использует `pass_through` для управляемого запуска и возвращает explicit `route` либо `stop`.

- FR-029: **Host-owned credential binding**

После принятия decision Hermes заново разрешает provider identity, credentials, endpoint/API mode и совместимость reasoning для выбранной цели. Router никогда не возвращает API key или credential handle.

Если credential binding или capability validation не удались, исполнитель не создаётся; host фиксирует `route_not_started`. Для `main_turn` и `kanban_worker` возможный retry является отдельным решением их штатной authority. Для `native_child` v1 не создаёт same-unit replacement; позднейший обычный parent delegate call является новой execution.

- FR-030: **Уведомления**

До запуска router-selected attempt штатная поверхность получает структурированное route notice с execution kind, requested/accepted route и bounded reason. Для `main_turn` и `kanban_worker` при разрешённом fallback/escalation новое уведомление показывает предыдущий terminal state и новый accepted route до старта нового attempt. Для router-selected `native_child` route-change fallback в v1 вместо replacement выдаёт bounded видимый terminal failure.

Отсутствие surface-specific renderer не блокирует host event: используется безопасный текстовый fallback. Generic core не содержит product-specific русских формулировок; feature plugin предоставляет локализованный reason, а surface локализует стандартные host states.

- FR-031: **Совместимость и отключение**

Без активного router-provider все существующие execution paths и fallback semantics остаются неизменными. Отключение provider влияет только на будущие attempts; уже запущенный attempt сохраняет зафиксированный route и lifecycle.

API version/capability discovery позволяет feature plugin проверить поддержку `main_turn`, `native_child` и `kanban_worker`. Неполная поддержка не маскируется: unsupported kind возвращается как unsupported/DEFER до попытки dispatch.

- FR-032: **Первая версия и исключения**

Первая версия считается функционально полной только при одинаковом host contract для:

- `main_turn`;
- `native_child`;
- `kanban_worker`.

Auxiliary LLM calls, cron scheduler jobs, plugin-owned `ctx.llm` calls, model aggregation и provider-internal retries не входят в v1. Их добавление требует отдельного продуктового amendment, compatibility review и versioned contract extension.

- FR-033: **Delivery boundary**

Результатом upstream-проекта является generic, документированный и протестированный Hermes plugin API плюс reference fixtures. В него не входит routing policy Feature 002.

Локальная реализация и commit не означают merge в upstream, публикацию, установку в рабочий Hermes, pilot или LIVE. Подключение Feature 002 начинается только после отдельной qualification exact API/version и отдельного разрешения на возврат к T005.

- FR-034: **Минимальная Kanban reservation в исходной lane**

Для `kanban_worker` каждый фактически предлагаемый worker attempt маршрутизируется до credentials, `task_run`, process или executor. Reservation использует существующие `claim_lock`/`claim_expires`, не меняет исходный статус `ready` или `review` и не требует нового task status, task column/index/dashboard mapping, reservation store, queue, scheduler или dispatcher. Истёкшая pre-run claim освобождается без phantom run; `stop/router_error` оставляет задачу в исходной lane под коротким bounded defer, записывает bounded/redacted not-started evidence в существующие `task_events` и не затрагивает worker failure accounting/circuit breaker.

Validated route/open выполняется одной существующей write transaction: повторно проверяет claim token/expiry/current lane, создаёт canonical `task_run` с route metadata/events и только затем переводит задачу в running. Корреляция использует существующую run primary identity и metadata/event payload; невозможность обеспечить identity/idempotency этими полями является STOP для отдельного решения, а не разрешением импровизировать schema. После реально начатого worker/model failure действуют существующие close/failure-budget/retry/requeue/circuit-breaker rules; следующий обычный eligible attempt получает fresh request/attempt IDs и заново вызывает router. No-router/pass-through сохраняет точный текущий claim/open/spawn/retry path. Crash contract различает только reserved/no run и opened run, если реализация не докажет ещё одно внешне различимое состояние. C10 полностью supersedes более широкое прежнее T005 wording.

## Whole-spec acceptance criteria

The product specification is accepted only if the owner approves this exact whole-file DRAFT and confirms all of the following:

1. `execution_router` is a generic host capability; Feature 002 policy remains in its installable plugin.
2. v1 covers exactly `main_turn`, `native_child`, and `kanban_worker` through one semantic contract.
3. Explicit pins, eligibility, credentials, permissions, dispatch, lifecycle, retry budgets and completion authority remain host-owned.
4. Route decisions are `route | pass_through | stop`, one-attempt-bound, validated, observable and non-recursive.
5. Active-router error/timeout stops dispatch visibly; late results are discarded; no hard-kill/sandbox guarantee is claimed for arbitrary trusted plugin code.
6. Router-selected routes are immutable within an attempt; `main_turn` and `kanban_worker` fallback/escalation create a new attempt and decision when their existing authority permits, while v1 `native_child` route-change fallback terminates visibly without automatic same-unit replacement.
7. No-router and explicit pass-through preserve native behavior.
8. One active consented provider owns the capability; route events distinguish requested, accepted and actual without a plugin-owned store.
9. Auxiliary LLM, cron, `ctx.llm`, aggregation and provider-internal retry are outside v1 and require amendment.
10. Planning and local implementation do not authorize merge, publication, installation, pilot or LIVE.
11. `kanban_worker` reuses the existing claim in the unchanged `ready`/`review` lane, defers `stop/router_error` without worker-failure accounting, opens the canonical run atomically after routing, and adds no routing status or mandatory task schema.

## Planning boundary

Acceptance of this product specification authorizes architecture drafting only. It does not authorize implementation. Architecture must be reviewed in separate blocks, then accepted as one exact DRAFT; only after that may an integrated task map be drafted and separately accepted. Implementation requires a further explicit authorization and exact project/task entry.
