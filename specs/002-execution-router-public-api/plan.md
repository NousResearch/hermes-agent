# Architecture — Hermes Public `execution_router` Plugin API

Stage: ARCHITECTURE
Status: ACCEPTED
Project-Version: 0.1.1
Architecture-Version: 1
Product-Spec: spec.md
Traceability-Schema: 1
Accepted-Integrated-Plan-Source-SHA256: `b6f800775e5835c99cd2bcb12f13f553e00b9553b0af9ce4b8345565d9022ce3`
Integrated-Plan-Acceptance: Telegram current session — owner selected «Принимаю интегрированную архитектуру по SHA-256»
Accepted-Architecture-Corrections: `C1` SHA-256 `ca048c1245ab4f2efa0f966615d4fe8e97a121bdb345810356aa8ccf815043d7`; `C2` SHA-256 `6ca74b01325ffa7a91d5f749a7f6521b6020498722ea25852ac62579ef8e431c`; `C3` SHA-256 `c6237c72bd027246427cede1652443139d0b53e9696bb80a94b044ccf88993c6`
Accepted-Architecture-Blocks: `ARC-001` from exact owner-approved DRAFT SHA-256 `a88f46d1def25e94f9f2e98429a7200f45ba3aff00988a73e97ebefddd0b0466`; `ARC-002` from exact owner-approved DRAFT SHA-256 `96458abe33bb37e933ba56e82a19e3707c376925e15a46e4494022b5fb6a78a7`; `ARC-003` from exact owner-approved DRAFT SHA-256 `09dbc5c7e039ba6547d0b7baf2b4f189929aea772a8c155cbd2522ef741f2f69`; `ARC-004` from exact owner-approved DRAFT SHA-256 `f8ec787cc9e58c3427a6fd7b960c8d2936b7e2ecd208ad8d6634315f9825b8c0`; `ARC-005` from exact owner-approved DRAFT SHA-256 `e2cda4001160c15451fdda696ea7ca51e8d17787774256e60720afd44b2137bf`; `ARC-006` from exact owner-approved DRAFT SHA-256 `56d47b56798f49c16637ba048cddf08fa60e642c400b79754089d5176a11d886`
Correction-Acceptance: Telegram current session — C1 «Утверждаю correction set C1 по SHA-256»; C2 «Утверждаю С2»; C3 «Утверждаю СЗ»
Owner-Acceptance: Telegram current session — ARC-001 «Утверждаю блок по правильному SHA-256»; ARC-002 «Утверждаю» following exact displayed SHA-256; ARC-003 «Утверждаю» following exact displayed SHA-256; ARC-004 «Утверждаю» following exact displayed SHA-256; ARC-005 «Хорошо. Утверждаю» following detailed explanation and exact displayed SHA-256; ARC-006 «Утверждаю» following exact displayed SHA-256
Canonicalization: accepted block behavior preserved; `ARC-001` heading normalized to canonical plan syntax
Implementation-Authorization: DISABLED

## Planning state

`ARC-001` through `ARC-006`, correction sets C1/C2/C3 and the whole integrated architecture are accepted. C1, C2 and C3 are integrated into the normative clauses below; no superseded contradictory clause remains authoritative. Acceptance authorizes the dependent task-map stage only; source implementation, commit, publication, installation, pilot and LIVE remain unauthorized.

## Простое объяснение

В Hermes появляется одна штатная «дверь» для запроса маршрута перед запуском модели. Через неё может отвечать только один отдельно разрешённый routing-provider. Provider сообщает предложение, но не запускает модель и не получает управление Hermes. Один и тот же host-owned механизм используется основной задачей, native child и Kanban worker.

### ARC-001: Host-owned execution-router registry and resolution seam
Source-Requirements: FR-001, FR-002, FR-003, FR-004, FR-005, FR-006, FR-012, FR-013
External-Boundaries: EXT-001

1. **Владелец механизма — Hermes host.** Публичный plugin contract, регистрация capability, consent, проверка единственности provider и вызов router принадлежат Hermes. `EXT-001` поставляет только product-specific provider и не владеет dispatch path.

2. **Одна capability и один активный provider.** Plugin объявляет capability `execution_router` с собственной plugin identity и поддерживаемой contract version. Установка сама по себе не активирует capability. Host-owned consent связывается с точной identity/version. Попытка активировать второй provider завершается видимым конфликтом до routed attempt; цепочки router-плагинов не создаются.

3. **Одна общая resolution seam.** Hermes предоставляет один внутренний host-owned orchestration entry, условно `resolve_execution_route(request)`, который реализует регистрацию/consent lookup и вызывает публичный provider callback. Это название описывает архитектурную роль, а не фиксирует модуль или сигнатуру исходного кода.

4. **Три тонких host adapter.** `main_turn`, `native_child` и `kanban_worker` не реализуют собственную routing policy. Каждый путь после определения и авторизации execution context, но до credential binding и создания исполнителя, формирует request и вызывает одну resolution seam. Различия путей выражаются полем execution kind, а не тремя несовместимыми API.

5. **Provider не является dispatcher.** Callback не получает credentials, live agent, writable task/config, executor factory или dispatcher handle. Он не создаёт process/task, не делает retry/fallback и не изменяет permissions, tools, workspace либо profile defaults.

6. **Нулевое влияние без provider.** Если capability не зарегистрирована, не разрешена либо provider явно выбирает pass-through, adapter продолжает существующий штатный путь без нового model policy, default route или side effect. Наличие API само по себе не включает автоматическую маршрутизацию.

7. **Attempt boundary.** Resolution выполняется ровно один раз для нового attempt через host seam. Fallback/escalation не переиспользует старое решение и при создании нового attempt снова проходит эту seam. Детальная identity/correlation model будет определена отдельным архитектурным блоком.

8. **Repository and release boundary.** Изменение generic API разрабатывается и проверяется в отдельном project/repository `hermes-execution-router-api` с upstream Hermes ancestry. Оно не vendored в `EXT-001`, не устанавливается в runtime и не становится LIVE только из-за принятия этой архитектуры.

## Явно не входит в этот блок

- поля `ExecutionRouteRequest` и `ExecutionRouteDecision`;
- validation матрица route/pins/eligibility;
- timeout, exception и malformed-decision behavior;
- requested/accepted/actual events и redaction;
- точные модули, классы и сигнатуры исходного кода;
- implementation, commit, upstream publication, installation, pilot или LIVE.

## Acceptance effect

Принятие блока фиксирует только ownership и место единого вызова. Оно не принимает остальные архитектурные блоки и не разрешает реализацию.


---

## Простое объяснение

Hermes передаёт router небольшой неизменяемый «конверт» с описанием запуска и уже разрешённым списком вариантов. Router не пишет произвольные provider/model и не видит credentials: он выбирает один `candidate_id` из этого списка, пропускает выбор либо останавливает запуск.

### ARC-002: Versioned immutable route contract
Source-Requirements: FR-007, FR-008, FR-009, FR-016, FR-017, FR-019, FR-020, FR-021, FR-025, FR-029, FR-031, FR-032
External-Boundaries: EXT-001

#### 1. Capability descriptor

При регистрации provider объявляет публичный descriptor:

- `capability = execution_router`;
- одну exact major/minor contract version;
- поддерживаемые execution kinds из закрытого v1-набора `main_turn | native_child | kanban_worker`;
- plugin/provider identity, которую host связывает с consent.

Hermes отклоняет несовместимую contract version или неполную capability до routed attempt. Auxiliary, cron, `ctx.llm`, aggregation и provider-internal retry отсутствуют в enum v1.

#### 2. `ExecutionRouteRequestV1`

Request является immutable value object и содержит только:

- `contract_version`;
- host-issued opaque `request_id`, `root_id`, `task_id`, `execution_id`, `attempt_id`;
- `execution_kind` и bounded `surface_class`;
- `instruction`: уже разрешённый и redacted bounded text либо явное отсутствие текста; `truncated`, `original_utf8_bytes`, host-computed `digest_algorithm` и `digest`;
- `pins`: отдельные optional model/provider/reasoning constraints;
- `native_candidate_id`, если штатный route уже представлен в projection;
- immutable `eligibility_revision` и непустой либо явно пустой tuple `eligible_candidates`;
- optional bounded `previous_attempt` только для host-authorized fallback/escalation: предыдущий attempt ID, terminal-state code, route identity и безопасный bounded reason.

Request не содержит system prompt, историю conversation, credentials, endpoints с secrets, approval tokens, writable objects, live agent, config, tools или dispatch handles.

#### 3. Eligibility candidate

Каждый `eligible_candidate` создаётся Hermes и содержит:

- opaque host-issued `candidate_id`;
- конкретные public `provider`, `model` и `reasoning` values, которые вместе образуют один допустимый route tuple;
- только credential-free capability attributes, необходимые для выбора;
- revision binding к request.

Projection создаётся только после host checks profile/provider availability, operator policy, explicit pins/constraints, permissions, tool ceiling, workspace ceiling и authorization данного execution kind. `eligibility_revision` связывает canonical digest всех этих authority inputs, candidate tuples и host contract generation. Изменение любого измерения инвалидирует revision.

Credentials, credential labels, secret endpoints, permission objects и policy internals не включаются. Explicit pins уже сужают projection, но также передаются отдельно для объяснимости и immediate host revalidation.
#### 4. `ExecutionRouteDecisionV1`

Decision — закрытый discriminated union из трёх вариантов:

1. `route(candidate_id)` — выбрать ровно один candidate из request projection;
2. `pass_through` — использовать native behavior;
3. `stop` — не создавать attempt.

Каждый вариант возвращает exact `request_id` и `attempt_id`. Дополнительно разрешена только bounded пара `reason_code` / `reason_text`. Произвольный payload, credentials, новый candidate, retry instruction, task mutation или completion claim запрещены schema.

Для Python compatibility возвращённый `None` нормализуется host в explicit `pass_through`; в событиях он не изображается как `route`.

#### 5. Candidate-reference rule

`route` принимает только `candidate_id`, присутствующий в exact request и связанный с той же `eligibility_revision`. Provider не может вернуть собственные provider/model/reasoning strings.

Непосредственно перед credential binding и executor/attempt creation Hermes повторно проверяет candidate membership и freshness, explicit pins, profile/provider availability, operator policy, permissions, tool/workspace ceilings и execution-kind authorization. Stale/changed authority input даёт `router_error/stale_eligibility`; executor не создаётся.

Это делает «eligibility не расширяется» свойством контракта и immediate host validation, а не доверием к plugin.
#### 6. Immutability and serialization

Все public contract objects являются frozen/immutable после создания, не содержат callbacks или mutable mappings. Candidate order не означает приоритет; v1 не вводит policy-neutral rank.

Canonical v1 serialization для request/decision/fixture digests — UTF-8 JSON: object keys lexicographically sorted; separators exactly `,` and `:` without insignificant whitespace; enums use exact lowercase wire strings; arrays preserve contract order; every field is present and absent optionals are `null`; integers are bounded base-10 JSON integers; floats/NaN/Infinity forbidden; strings are not Unicode-normalized; `ensure_ascii=false`; standard JSON escaping; duplicate keys rejected.

`serialized_request_bytes` — длина canonical UTF-8 bytes. Request digest — SHA-256 canonical request с digest field=`null` на время вычисления. Instruction digest — SHA-256 exact bounded instruction UTF-8 bytes delivered to router. Eligibility revision uses the same canonical rules over candidates plus all authority revisions.

Все fields имеют finite host-owned limits from contract v1. Boundary tests cover limit-1/limit/limit+1, multibyte UTF-8, escaping, null optionals and oversized integers.
#### 7. Version evolution

Неизвестный required field, unknown decision variant, unsupported execution kind или incompatible contract version отклоняются fail-closed до dispatch. Новые optional observational fields могут добавляться только по правилам совместимости; новый execution kind или расширение routing authority требует product amendment и новой contract version.

## Что этот блок намеренно не решает

- timeout, exception и late-result handling;
- idempotency cache/lifecycle и recursion guard implementation;
- requested/accepted/actual event sequence и read API;
- конкретные Python module/class names;
- точные числовые byte/count limits;
- implementation, commit, publication, installation, pilot или LIVE.

## Решение, требующее явного согласия

Главный выбор блока: `route` ссылается на host-issued `candidate_id`, а не передаёт произвольные model/provider/reasoning strings. Concrete tuple остаётся видимым router в credential-free projection, но authority списка принадлежит Hermes.


---

## Простое объяснение

Перед запуском Hermes ещё раз проверяет решение router. Ошибка активного router не превращается в скрытый запуск по старым настройкам. Один attempt получает один неизменный маршрут. Если нужен fallback, Hermes завершает старый attempt и создаёт новый, для которого router вызывается заново.

### ARC-003: Host validation and immutable attempt lifecycle
Source-Requirements: FR-008, FR-009, FR-010, FR-011, FR-022, FR-023, FR-024, FR-025, FR-026, FR-027, FR-028, FR-029, FR-031
External-Boundaries: EXT-001

#### 1. Единая последовательность до запуска

Для каждого нового routed attempt Hermes выполняет один host-owned pipeline:

1. формирует credential-free projection после всех authority checks и фиксирует immutable request/`route_requested`;
2. вызывает ровно один consented provider под bounded deadline;
3. нормализует ответ в `route | pass_through | stop | router_error`;
4. проверяет identity, contract version, decision shape и canonical correlation;
5. для `route` непосредственно revalidate-ит candidate, eligibility revision, pins, profile/provider availability, operator policy, permissions, tool/workspace ceilings и execution-kind authorization;
6. только после успешной revalidation разрешает provider identity и credentials;
7. создаёт executor либо фиксирует non-start outcome.

Ни provider, ни consumer plugin не могут пропустить или переставить стадии.
#### 2. Матрица результатов

- **Нет зарегистрированного/разрешённого provider:** routing pipeline не вызывается; Hermes сохраняет native behavior будущего attempt.
- **`pass_through` или `None`:** Hermes использует native route и native failure handling; состояние помечается как `native/pass_through`.
- **Валидный `route(candidate_id)`:** Hermes фиксирует accepted concrete tuple и только затем выполняет credential binding.
- **`stop`:** executor не создаётся; bounded причина возвращается штатной поверхности.
- **Exception, timeout, malformed decision, identity/version mismatch, конфликт с pins, отсутствующий/устаревший candidate:** результат `router_error`; executor не создаётся; скрытого pass-through нет.
- **Credential/capability failure после валидного route:** executor не создаётся; это host start failure, а не новое решение того же attempt.

#### 3. Deadline и late result

Deadline задаётся и измеряется Hermes monotonic clock. Exact duration является versioned host constant и не управляется plugin. При deadline expiry host атомарно закрывает resolution как `router_error/timeout`, cooperative-cancel callback task и окончательно отвергает поздний ответ по закрытому `request_id`.

Контракт не обещает hard kill, descendant cleanup или process sandbox для произвольного Python plugin-кода. Поддерживаемый provider должен быть локальным, быстрым и не выполнять network/model/tool calls. API не предоставляет `ctx.llm`, tools или dispatch handles.

#### 4. Idempotency без второго store

Resolution/idempotency ownership использует существующие surface authorities, а не абстрактный новый router store:

- main_turn/native_child — existing profile `hermes_state.py::SessionDB`/`state.db` route lifecycle records;
- native-child live process metadata — existing `_active_subagents`, но не durable event authority;
- kanban_worker — existing board DB `task_events`/`task_runs` plus route metadata.

Unique `(request_id, attempt_id)` and transition constraints ensure one decision application. Повтор того же request до executor creation возвращает тот же accepted decision/error. Изменение attempt ID, pins, input digest или eligibility revision создаёт новый request.

Public read API selects the already-authoritative store by authorized scope and returns one immutable schema; aggregate DB не создаётся.
#### 5. Route immutability

После `route_accepted` concrete provider/model/reasoning tuple и route identity неизменны внутри attempt. Rate limit, provider failure, verification conflict или другая ошибка не меняют модель «на ходу» и не переиспользуют attempt ID.

#### 6. Fallback, retry и escalation

Право на новый запуск остаётся у существующей Hermes authority и её budget/depth/permission checks. Если она разрешает retry/fallback/escalation:

- текущий attempt получает terminal state;
- создаётся новый attempt ID и новый request ID;
- в request передаётся только bounded previous-attempt context;
- active router вызывается заново и возвращает новое `route | pass_through | stop`.

Router не создаёт retry loop, не увеличивает budget/depth и не переносит старое решение автоматически.

#### 7. Recursion guard

Во время router callback host помечает execution context как `route_resolution`. Публичные routed launch surfaces отклоняют запуск из этого context как recursion. Основная защита — отсутствие dispatch-capable handles в callback contract; context guard закрывает случайный повторный вход через другие публичные host surfaces.

Это contract isolation, а не утверждение об OS sandbox для full-trust plugin.

#### 8. Безопасные ошибки

Публичный error result содержит стабильный host reason code и bounded безопасный текст без exception traceback, credentials, raw plugin payload или скрытого config. Полная диагностическая информация может идти только в существующий защищённый Hermes diagnostic channel согласно его правилам; `execution_router` не создаёт отдельный лог или evidence authority.

## Что блок пока не решает

- точную event schema и read-only observation API;
- surface notification rendering;
- конкретные числовые deadline/size limits;
- размещение компонентов по исходным модулям;
- тестовую/qualification матрицу;
- implementation, commit, publication, installation, pilot или LIVE.

## Решения, требующие подтверждения

1. Ошибка активного router является fail-closed для нового attempt, а не скрытым pass-through.
2. Credential binding failure не меняет route внутри attempt; возможный retry — отдельный attempt.
3. Idempotency record используется внутри существующего attempt lifecycle, без нового router store.
4. Deadline cooperative; hard-kill/sandbox гарантия не заявляется.


---

## Простое объяснение

Hermes отдельно фиксирует: router был вызван, решение прошло проверку, executor действительно стартовал, старт не состоялся либо attempt завершился. Поэтому «router предложил модель» нельзя перепутать с «эта модель реально работала». События хранятся и читаются через существующую систему Hermes, без отдельной базы плагина.

### ARC-004: Host route-event projection and read-only observation
Source-Requirements: FR-014, FR-015, FR-016, FR-017, FR-018, FR-030, FR-031
External-Boundaries: EXT-001

#### 1. Единственный владелец событий

Route events создаёт Hermes в существующем execution/event lifecycle. `execution_router` не создаёт собственный event store, task graph, журнал истины или delivery channel. Plugin может предоставить bounded reason и локализованный renderer, но event identity, timestamps, correlation, accepted/actual route и terminal state вычисляет host.

#### 2. Общий event envelope

Каждое route event содержит только структурированные credential-free поля:

- host-issued `event_id` и monotonic sequence внутри attempt;
- host timestamp;
- opaque `root_id`, `task_id`, `execution_id`, `attempt_id`, `request_id`;
- optional `previous_attempt_id` для разрешённого fallback/escalation;
- `execution_kind` и bounded `surface_class`;
- active router plugin/provider identity и contract version;
- event type и decision/result state;
- route identity: requested candidate reference, accepted concrete public tuple и actual concrete public tuple — каждое в своём nullable поле;
- bounded host reason code и безопасный reason text;
- terminal state только там, где он фактически известен.

Envelope не содержит credentials, credential labels, secret endpoints, prompt/history, hidden reasoning, approval tokens, traceback или произвольный plugin payload.

#### 3. События и переходы

1. **`route_requested`** — перед вызовом active router; фиксирует request/input/eligibility digests/revision, не копируя candidate list целиком.
2. **`route_accepted`** — только после host validation `route(candidate_id)` и фиксации concrete accepted route.
3. **`route_started`** — только после фактического executor creation; actual tuple берётся из host start receipt. Для `pass_through`: `route_requested` → `route_started` с `decision_state=pass_through` и actual native route, без `route_accepted`.
4. **`route_not_started`** — после `route_requested`, если executor не создан: `stop`, `router_error`, stale eligibility, validation or credential/start failure. Для `stop` decision state явный; `route_accepted` отсутствует.
5. **`route_finished`** — только для started attempt при terminal lifecycle state; не объявляет product/task completion.

`route_started` и `route_not_started` взаимоисключающи. `route_finished` невозможен без `route_started`. Host transition constraints reject impossible order.
#### 4. No-router и pass-through

Если active router отсутствует, Hermes не создаёт фиктивный `route_requested`; существующий native event flow остаётся неизменным. Если active router явно вернул `pass_through`, создаются route events с decision state `pass_through`, а `route_started` получает actual native route из start receipt.

Таким образом отсутствие router не меняет legacy telemetry, а явное участие router остаётся наблюдаемым.

#### 5. Read-only observation API

Hermes предоставляет bounded read-only projection route events через публичный наблюдательный контракт. Чтение доступно только в пределах уже разрешённых session/task scopes и существующих permission checks. Минимальные фильтры: attempt/request IDs, execution kind, bounded time/sequence range и event type.

API возвращает immutable projections, имеет host-owned pagination/record limits и не предоставляет update/delete/replay/retry/dispatch/completion methods. Читатель не может изменить canonical task state или объявить execution outcome.

#### 6. Notices для пользовательских поверхностей

Каждый pre-start route notice содержит execution kind, requested candidate/public tuple (если запрошен route), accepted candidate/public tuple (если принят), decision state, bounded reason code/text и attempt/request IDs. Для fallback/escalation добавляются previous attempt ID и bounded terminal-state code.

`stop/router_error` notice содержит execution kind, decision/error state и bounded reason до возврата управления пользователю. Generic host предоставляет нейтральные state/reason codes и fallback text. `EXT-001` может локализовать plugin reason, но не меняет host fields, actual route или history. Отсутствие native renderer не блокирует event/execution: используется текстовый fallback.
#### 7. Delivery semantics

Event publication следует transaction/durability semantics существующей surface authority; этот API не обещает отдельную stronger delivery guarantee. Main/native construction owner записывает resolution в SessionDB до executor construction; Kanban route/open transaction atomically связывает accepted route и run. Event IDs/attempt sequence support read deduplication.

API не вводит новое правило, что failure существующего persistence backend само по себе блокирует dispatch: действует уже существующая host policy. Однако qualification не может получить PASS, если required transitions не воспроизводятся и не читаются. Optional UI renderer failure не меняет decision и не удаляет host event.
## Что блок пока не решает

- конкретную существующую event backend/таблицу/класс Hermes;
- retention policy сверх действующих правил Hermes;
- surface-specific UI layouts;
- числовые pagination/size limits;
- source module placement и migration strategy;
- test/qualification matrix;
- implementation, commit, publication, installation, pilot или LIVE.

## Решения, требующие подтверждения

1. Отсутствующий router не создаёт новые route events; явный `pass_through` создаёт.
2. `route_started` строится только из фактического host start receipt.
3. События используют существующую Hermes event authority, без plugin-owned store.
4. Ошибка необязательного renderer не блокирует attempt. Ошибка или недоступность существующей host lifecycle persistence обрабатывается ровно действующей policy соответствующей surface authority; `execution_router` не добавляет отдельного fail-closed основания. Полнота mandatory lifecycle projection остаётся qualification criterion: неполная или невоспроизводимая фиксация не получает qualification PASS, но сам API не переопределяет dispatch semantics.


---

## Простое объяснение

В текущем Hermes нет одной готовой точки, которая одновременно находится достаточно рано для основной задачи, native child и Kanban worker. Поэтому общий router-service остаётся один, но к нему ведут три маленьких host adapter. Каждый adapter вызывается до credentials и до создания executor/attempt. Существующие dispatcher, gateway, Kanban и delegation остаются владельцами запуска.

### ARC-005: Surface adapters before credential and executor boundaries
Source-Requirements: FR-002, FR-006, FR-008, FR-011, FR-026, FR-027, FR-028, FR-029, FR-031, FR-032
External-Boundaries: EXT-001

#### 1. Не использовать `AIAgent` как routing seam

`run_agent.py::AIAgent.__init__` и `agent/agent_init.py::init_agent` являются слишком поздней общей точкой:

- explicit pin provenance уже частично сведена к runtime kwargs;
- parent credential native child может быть уже выбран;
- Kanban subprocess и persistent run могут быть уже созданы;
- существующий fallback способен заменить client/model внутри того же agent.

Router resolution SHALL NOT добавляться только в `AIAgent` construction. Общий host service из ARC-001 вызывается surface adapters до этих границ.

#### 2. `main_turn` adapter family

`main_turn` является одним execution kind, но routing выполняется на каждом новом turn/attempt, не только при session agent construction.

- Classic CLI: `hermes_cli/cli_chat_turn_mixin.py::CLIChatTurnMixin.chat()` выполняет credential-free normalization/projection нового turn и вызывает `prepare_main_turn_attempt` **до** `_ensure_runtime_credentials()`, `_resolve_turn_agent_config(...)` и `_init_agent(...)`. Для `route/pass_through` validated route передаётся в существующие `_resolve_turn_agent_config(...)` / `_init_agent(...)` paths; agent reuse допускается только при exact accepted route signature, иначе existing construction path rebuild-ит agent. `stop/router_error` возвращают управление до credential binding, agent construction и `self.agent.run_conversation(...)`.
- TUI/backend: `tui_gateway/prompt_turn.py::_prepare_turn_input`, до binding `session["agent"]` to turn state и `_invoke_agent`; route change rebuild-ит session agent through existing path.
- Gateway: `gateway/run_turn_runner.py::TurnRunner.run_sync` splits current `_resolve_session_agent_runtime` into credential-free native candidate/pin projection before routing and post-decision credential resolution before `_resolve_turn_agent`.
- One-shot: route before runtime-provider resolution/AIAgent construction.

Shared `prepare_main_turn_attempt` only prepares validated route and invokes existing construct/reuse callbacks; it does not run conversation or become dispatcher. History/session authority remains unchanged.
#### 3. `native_child` adapter

В `tools/delegate_tool.py::delegate_task` router вызывается отдельно для каждого нового child после `_normalize_task_list()`/schema coercion, но **до** `_resolve_delegation_credentials(...)`, `_build_children(...)` и `_run_batch(...)`.

Это сохраняет raw explicit model/provider/reasoning pins и не создаёт child `AIAgent`, credential lease или executor work до решения. `_build_children()` и `_run_batch()` остаются единственными штатными construction/dispatch paths; новый dispatcher не добавляется.

Параллельный batch сначала получает независимые immutable decisions для разрешённых child attempts, затем строит только разрешённые children. `stop/router_error` одного child не превращает его в другой route и не создаёт скрытого child executor.

#### 4. `kanban_worker` two-phase boundary

Current `_claim_and_open_run()` creates `task_runs` too early. Existing Kanban DB is extended without a new store/scheduler:

1. CAS `ready|review -> routing` sets existing `claim_lock/claim_expires`, `routing_source_status` and host UUID `planned_attempt_id`; no `task_runs` or process.
2. `release_stale_claims` also scans `routing`; expiry without PID restores exact source lane, clears routing fields and appends recovery event, for ready/review.
3. After validated `route|pass_through`, one transaction verifies lock/expiry/source/planned ID; inserts `task_runs` with unique `attempt_id=planned_attempt_id` and accepted route metadata; sets running/current_run_id; clears routing-only fields; existing spawn runs only after commit.
4. Crash before open uses routing recovery; crash after open uses existing running/run reclaim; spawn failure closes run through existing failure authority.
5. `stop/router_error` creates no run, appends route_not_started to existing task_events, restores source lane and feeds existing failure accounting/circuit breaker. Same dispatch cycle cannot immediately reselect it.
6. Migration adds only required fields/indexes/constraints to existing tasks/task_runs; no new queue/table/store.

Provider-only pin remains unsupported where canonical Kanban already rejects it; router cannot weaken that rule.
#### 5. Credential binding

После accepted decision каждый surface использует существующий provider/runtime resolver для selected candidate. API передаёт только candidate identity; raw credentials, base URL secrets и pool metadata не переходят через router.

Если credential resolution не удалась, соответствующий attempt не стартует. Возможный retry требует нового attempt через тот же surface adapter.

#### 6. Fallback semantics

For router-selected attempts, `try_activate_fallback` and its call sites do not mutate client/model in place. Existing fallback authority first checks and atomically consumes its permitted slot; agent terminal result carries host-only `routed_restart_required` with reason/budget receipt and no target.

Re-entry owners:

- main_turn per-turn adapter creates new IDs, reruns router and constructs/reuses only matching fresh agent;
- native child existing delegate/single-child lifecycle receives child spec/factory and builds a new child only after a new decision;
- Kanban worker returns terminal restart receipt; `kanban_db_dispatch` closes current run and re-enters ordinary reserve/route/open.

If budget/permission denies restart, attempt remains terminal. `native/pass_through` preserves current in-place/native fallback. No global retry dispatcher or plugin retry loop is added.
#### 7. Attempt identity and cached agents

Host-issued attempt/request IDs создаются до router invocation, но executor/session/run object создаётся только после accepted result. Они не переиспользуются при retry.

Main-turn cache key/signature включает accepted route identity и contract generation. Смена route, provider generation или capability disablement принудительно создаёт fresh agent; cached credentials/client прежнего route не используются.

#### 8. Scope boundary

Adapter code не классифицирует задачи, не выбирает модель самостоятельно, не локализует policy и не владеет retry budget. Оно только:

1. собирает bounded host context;
2. вызывает общий resolver;
3. передаёт validated route существующему construction path;
4. публикует host lifecycle state.

Auxiliary LLM calls, cron jobs, plugin-owned `ctx.llm`, aggregation и provider-internal retries не получают adapters в v1.

## Verified blockers, которые этот блок закрывает

- единой достаточно ранней seam в upstream нет;
- routing только в Gateway не покроет CLI/TUI/one-shot main turns;
- routing внутри `_build_child_agent` будет позже credential selection;
- routing только внутри Kanban worker произойдёт после process/run creation;
- существующий in-place fallback несовместим с immutable per-attempt route;
- cached main agent может скрыто сохранить прежний client без route-aware signature.

## Что блок не решает

- точные новые module/class names;
- capability ID и consent wiring;
- exact constants и supported-Hermes version floor;
- полный набор файлов/tests и upstream release procedure;
- implementation, commit, publication, installation, pilot или LIVE.

## Решения, требующие подтверждения

1. Один resolver плюс три surface adapter, а не routing внутри `AIAgent`.
2. `main_turn` покрывает Gateway, classic CLI, TUI/backend и one-shot.
3. Kanban claim/open-run делится на reservation и route/open, чтобы attempt/process не создавался до routing.
4. In-place fallback запрещён только для router-selected attempt; pass-through сохраняет native semantics.


---

## Простое объяснение

Публичные типы router будут маленьким самостоятельным contract module. Plugin регистрирует provider через штатный `PluginContext`, но только после отдельного consent `execution.routing`. Hermes хранит один active provider, вызывает его через host runtime service и снимает регистрацию при unload. Совместимость доказывается не названием API, а точной Hermes-версией и тестами всех трёх execution kinds.

### ARC-006: Public host module, capability consent and qualification boundary
Source-Requirements: FR-001, FR-002, FR-003, FR-004, FR-005, FR-012, FR-013, FR-017, FR-019, FR-020, FR-022, FR-023, FR-024, FR-031, FR-032, FR-033
External-Boundaries: EXT-001

#### 1. Public contract module

Новый policy-neutral module `agent/execution_router.py` владеет только публичным v1 contract:

- closed enums execution kind и decision kind;
- immutable `ExecutionRouteRequestV1`, eligibility candidate, pins, previous-attempt projection и `ExecutionRouteDecisionV1`;
- immutable `ExecutionRouterHostCapabilitiesV1` и public read-only discovery function;
- provider Protocol/ABC;
- contract constants;
- pure shape/correlation/membership/serialization validation helpers без credentials, config mutation, plugin discovery или dispatch.

Host capabilities descriptor содержит supported contract versions, execution kinds, active limits, availability event/read contract и host release/build identity. Discovery не активирует provider и не раскрывает credentials/config internals. `unsupported` является host preflight result, не четвёртым router decision: отсутствующий API или unsupported kind сохраняет consumer `DEFER`, а partial development build возвращает `unsupported_kind` до callback/executor creation. Provider decision union остаётся ровно `route | pass_through | stop`.

Module не содержит model catalog, complexity classifier, chains, русскую policy, provider credentials, retry loop или consumer implementation. Используются tuples/primitives/frozen value objects; mutable dict/list payload не входит в public contract.
#### 2. Host runtime owner

Новый `hermes_cli/execution_router_runtime.py` owns provider lookup, version-bound consent/generation check, canonical request invocation/deadline/normalization/validation, recursion marker, host notices and read projection. It returns validated result to existing surface adapter and never creates executor.

Durable route lifecycle is surface-owned: main/native use existing SessionDB/state.db route records; Kanban uses existing task_events/task_runs. Native live registry remains live metadata only. The public reader selects authority by authorized scope and does not create aggregate storage. No second persistent store or dispatcher is introduced.
#### 3. Registration and lifecycle

`hermes_cli/plugins.py::PluginContext` получает public method `register_execution_router(provider)`. `PluginManager` получает один profile-scoped slot с exact owner identity, contract version и generation.

Регистрация:

1. требует объявленной и выданной capability `execution.routing`;
2. проверяет provider protocol и совместимую contract version;
3. отклоняет второй active provider видимым conflict до routed attempt;
4. записывается в существующий plugin ownership ledger;
5. снимается при targeted unload/disable без влияния на уже started attempts.

Автоматического last-registration-wins и цепочки providers нет. Обновление provider требует unload старой generation, повторной проверки capability set/consent и новой регистрации. `registration_lifecycle.py` может использоваться только для безопасного generation teardown; оно не разрешает неявную replacement policy.

#### 4. Dedicated two-phase consent

`hermes_cli/plugin_capabilities.py` gains `execution.routing`, but generic capability-set grant alone is insufficient. Persisted consent subject is exact `(plugin_id, plugin_version, provider_id, execution_router_contract_version, capability_set_hash)`.

Host extends the existing `plugins.entries.<plugin_id>` record in normal Hermes config; no new database or consent store is created. Under `execution_router`, host stores a validated `pending` descriptor and the last explicitly confirmed `consent` tuple/timestamp. Host derives plugin identity/version and capability hash from the discovered manifest; provider identity and contract version come from the validated public registrar descriptor. Plugin never writes config directly.

Phase 1 remains the existing generic capability flow: install/enable may grant `execution.routing` through `_run_capability_consent(...)` / `record_consent(...)`, but this alone never activates a router.

Phase 2 begins when `PluginContext.register_execution_router(provider)` forms the exact tuple. Without matching consent, host `record_pending_execution_router_consent(...)` stores only the pending descriptor, returns typed inactive result `consent_required`, emits a bounded operator notice, and does not claim the active slot, expose the callable, or create an active ownership-ledger entry. Runtime/non-interactive load never prompts and remains fail closed. Pending state is not resolver authority.

Existing `hermes plugins enable <plugin_id>` is the explicit interactive acquisition path even when the plugin is already enabled. It displays all five tuple elements, the capability description, and the host-issued-candidates boundary. Affirmation calls `record_execution_router_consent(...)`; decline leaves the router inactive and unrelated grants unchanged; non-interactive invocation never grants consent. The command states that activation occurs only on the next normal load/reload and does not inject a callable into a running process. `hermes plugins capabilities <plugin_id>` stays read-only and reports `pending / consented / stale`.

On subsequent registration, `execution_router_consent_status(...)` recomputes the tuple. Exact match plus live generic grant may claim the single slot. Absent, declined, stale, forged, superseded, or host/manifest-mismatched state cannot activate. Any tuple element change requires re-consent. Unload removes the active slot but preserves consent history; disable/revocation makes the slot unusable and uses normal unload cleanup.

Consent grants only candidate selection and bounded text disclosure. It does not imply `llm.provider_override`, ctx.llm, tools, credentials, workspace, retry, completion or LIVE. Separate broad override capability is unnecessary for host-issued candidate selection.
#### 5. V1 constants

Следующие ceilings входят в exact public contract v1 и применяются host до callback:

- contract version: `1.0`;
- execution kinds: ровно 3 (`main_turn`, `native_child`, `kanban_worker`);
- router deadline: `250 ms` monotonic host time;
- bounded instruction: максимум `16,384` UTF-8 bytes после host redaction; larger input truncates with original size and SHA-256 digest;
- полный serialized request projection: максимум `65,536` bytes;
- eligible candidates: максимум `128` concrete tuples;
- opaque ID и `candidate_id`: максимум `128` ASCII characters each;
- `surface_class` и host/plugin reason code: максимум `64` ASCII characters each;
- reason text и previous-attempt safe reason: максимум `512` UTF-8 bytes each;
- event observation page: максимум `100` records;
- previous-attempt projection: максимум одна непосредственная predecessor link в v1.

Превышение request/candidate limits до callback даёт host `router_error/request_too_large`, кроме instruction text, для которого contract явно разрешает deterministic truncation. Plugin не может повышать ceilings через config. Изменение authority, execution kinds или required fields требует contract review/version; увеличение безопасного ceiling не применяется молча без compatibility evidence.

#### 6. Manifest and compatibility

Current upstream facts:

- runtime parser поддерживает plugin manifest schema 2, но installer path всё ещё содержит schema-1 ceiling;
- `api_version` читается, но не является enforced admission gate;
- `requires_hermes` является реально применяемой version boundary;
- `execution_router` и `execution.routing` в inspected upstream отсутствуют.

Поэтому v1 не зависит от manifest-v2-only behavior и не изображает `api_version` защитой. Consumer plugin использует installer-supported manifest form, объявляет `execution.routing` и после первого upstream release указывает exact `requires_hermes` floor. Capability descriptor дополнительно проверяет contract `1.0` runtime-side.

Несогласованность installer/runtime manifest ceilings фиксируется как отдельный upstream defect, но не расширяет `execution_router`: если выбранная поддерживаемая manifest form уже несёт capabilities и `requires_hermes`, исправление schema ceiling не является обязательным deliverable этого API. Если без него install/consent невозможны, defect становится явной blocking task, а не скрытым обходом.

#### 7. Source integration set

Минимальный ожидаемый source set:

- `agent/execution_router.py` — public contract and pure validation;
- `hermes_cli/execution_router_runtime.py` — host resolver/lifecycle adapter;
- `hermes_cli/plugins.py` — `PluginContext` registrar, typed pending/active result and single slot;
- `hermes_cli/plugin_capabilities.py` — `execution.routing`, existing-config pending/consent schema, exact tuple/hash validation, `record_pending_execution_router_consent(...)`, `execution_router_consent_status(...)` and `record_execution_router_consent(...)`;
- `hermes_cli/plugins_cmd.py` — `cmd_enable(...)` interactive pending-tuple consent and `cmd_capabilities(...)` read-only status rendering;
- `hermes_cli/plugins_loader.py` — bounded `consent_required` status without treating pending registration as active;
- ARC-005 existing surface modules — только thin adapters;
- existing host execution/event authority — route-event projection, без нового store;
- public docs and one policy-neutral reference fixture provider.

Generic middleware/event fan-out не используется как authoritative decision transport: hooks многополучательны, а plugin event bus asynchronous/lossy. Existing `subagent_start/stop`, stream и approval hooks сохраняются как observation around the host-owned decision.

#### 8. Required qualification

Новая focused suite размещается рядом с existing ownership:

- `tests/agent/test_execution_router.py` — frozen types, canonical JSON/constants, decision validation, correlation, complete eligibility invalidation, deadline/late result, recursion/idempotency and host capability discovery;
- `tests/hermes_cli/test_plugin_execution_router.py` — registrar, dedicated consent, single-provider conflict, unload/generation, no-router behavior;
- `tests/hermes_cli/test_plugin_capabilities.py` — declaration, grant/decline/re-consent and non-implication of broader capabilities;
- gateway/CLI/TUI/one-shot integration tests — все main-turn entry surfaces and route-aware cache rebuild; Classic CLI additionally proves one router call per new user turn before `_ensure_runtime_credentials()`, no credential/config/init/conversation calls for `stop/router_error`, accepted-route propagation into existing config/init paths, exact-signature reuse, changed-signature rebuild and unchanged no-router kwargs/control flow;
- `tests/tools/test_delegate.py` — one decision per native child before credentials/build, pins, batch partial stop/error and routed-restart new-attempt fallback;
- Kanban dispatch/DB tests — routing reservation CAS/expiry for ready+review, route/open transaction, crash at every boundary, stop/error restore, restart receipt and no phantom `task_runs`;
- event/read tests — valid ordering, requested/accepted/actual distinction, redaction, bounded reads and renderer isolation;
- packaging/docs tests — public module in wheel/sdist and documented contract/capability;
- first-load consent tests — generic grant creates pending descriptor but no active slot/callable/ledger entry;
- CLI consent tests — non-interactive denial, exact five-field TTY display/persistence, decline isolation and subsequent-load activation;
- stale-state tests — tuple changes, forged/superseded pending and host/manifest mismatch fail closed;
- lifecycle tests — read-only pending/consented/stale display and unload/disable cleanup with preserved consent history.

Обязательный regression gate запускает неизменённые existing suites для plugin loading/ledger/capabilities, strict provider selection, approvals, stream hooks, delegation, Kanban, gateway/CLI/TUI/one-shot construction, fallback and packaging. No-router cases должны подтверждать прежние kwargs/control flow, а не только проходить новые mocks.

#### 9. Compatibility evidence and delivery gates

Qualification artifact связывает:

- exact execution-router contract `1.0`;
- exact upstream Hermes commit/release;
- отдельный PASS/FAIL для `main_turn`, `native_child`, `kanban_worker`;
- exact focused/regression test commands and outputs;
- no-router compatibility evidence;
- reference fixture version/hash;
- known unsupported kinds;
- security/scope review, включая отсутствие product policy и второго dispatcher/store.

Только полный PASS всех трёх kinds позволяет consumer project снять T005 `DEFER`. Local branch/commit, частичный adapter или passing unit tests одного kind не являются совместимым release.

#### 10. Repository and release boundary

Implementation, если будет отдельно разрешена, выполняется в sibling Git project с сохраняемой upstream ancestry. Local commit не означает upstream merge/release. Upstream publication, установка exact Hermes build, API qualification, consumer compatibility qualification, pilot и LIVE являются разными последующими gates.

Проект `hermes-execution-router-api` не устанавливает consumer plugin и не принимает решения его model policy/rollout. `EXT-001` не vendor-ит API source и не использует local monkey patch для снятия `DEFER`.

## Что ARC-006 не разрешает

- source implementation или tests;
- создание branch/commit/tag/remote/push;
- изменение установленного Hermes или профилей;
- установку reference/consumer plugin;
- pilot или LIVE;
- routing auxiliary/cron/`ctx.llm`/aggregation/provider retries;
- новую security capability сверх `execution.routing`, уже необходимой принятому consent requirement.

## Решения, требующие подтверждения

1. Public types — `agent/execution_router.py`; host resolver — `hermes_cli/execution_router_runtime.py`.
2. Единственная новая capability — узкая `execution.routing`; широкая `llm.provider_override` из неё не следует и для candidate selection не требуется.
3. V1 использует указанные конечные constants, включая deadline 250 ms и instruction 16 KiB.
4. Authoritative decision не проводится через middleware/plugin event bus.
5. T005 снимается только после exact compatibility PASS всех трёх execution kinds.
