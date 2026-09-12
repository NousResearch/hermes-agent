# Architecture — Hermes Public `execution_router` Plugin API

Stage: ARCHITECTURE
Status: ACCEPTED
Project-Version: 0.1.1
Architecture-Version: 1
Product-Spec: spec.md
Traceability-Schema: 1
Accepted-Integrated-Plan-Source-SHA256: `b6f800775e5835c99cd2bcb12f13f553e00b9553b0af9ce4b8345565d9022ce3`
Integrated-Plan-Acceptance: Telegram current session — owner selected «Принимаю интегрированную архитектуру по SHA-256»
Accepted-Architecture-Corrections: `C1` SHA-256 `ca048c1245ab4f2efa0f966615d4fe8e97a121bdb345810356aa8ccf815043d7`; `C2` SHA-256 `6ca74b01325ffa7a91d5f749a7f6521b6020498722ea25852ac62579ef8e431c`; `C3` SHA-256 `c6237c72bd027246427cede1652443139d0b53e9696bb80a94b044ccf88993c6`; `C4` SHA-256 `8530bc9e37b3c95e64c8c88b17310fdc95e3a77a9518d24199a78685091c4509`; `C5` accepted-source SHA-256 `63fb0951747260c95ce33ee81f968ef8a63aaf7fa43ae97de6db0a197de2905b`, canonical-record SHA-256 `3a2754b0206225a2541623337c4717f2e7dc56a0921cc5345c5843bd2406e603`; `C6` accepted-source SHA-256 `b1d306fd1f6252d9034d0a84c5ba733f218cb101772ada0842d077bc4f7796f9`, canonical-record SHA-256 `cfc6dc76b0882df627abd60ec65ed8bf18d8aa9c6097fe79af2ac9432dc2377d`; `C7` accepted-source SHA-256 `60a2d38d69c4d1f9d1651b07d5b1eec6ac47033776db38626f89e667df8f7f40`, canonical-record SHA-256 `065b71217af1f6cfdc594f56d6d43b7889a5bdc6d902dffb50f1449c832f4157`; `C9` SHA-256 `8c46a9e0083edd9dd6b8591664e62d530f1c80d6e2692c49b75cde1080db2223` (supersedes C8); `C10` SHA-256 `f714322813b498a778059d691b2ebb2d953aa297d4a10631d8ff39c3a926ba0a`
Accepted-Architecture-Blocks: `ARC-001` from exact owner-approved DRAFT SHA-256 `a88f46d1def25e94f9f2e98429a7200f45ba3aff00988a73e97ebefddd0b0466`; `ARC-002` from exact owner-approved DRAFT SHA-256 `96458abe33bb37e933ba56e82a19e3707c376925e15a46e4494022b5fb6a78a7`; `ARC-003` from exact owner-approved DRAFT SHA-256 `09dbc5c7e039ba6547d0b7baf2b4f189929aea772a8c155cbd2522ef741f2f69`; `ARC-004` from exact owner-approved DRAFT SHA-256 `f8ec787cc9e58c3427a6fd7b960c8d2936b7e2ecd208ad8d6634315f9825b8c0`; `ARC-005` from exact owner-approved DRAFT SHA-256 `e2cda4001160c15451fdda696ea7ca51e8d17787774256e60720afd44b2137bf`; `ARC-006` from exact owner-approved DRAFT SHA-256 `56d47b56798f49c16637ba048cddf08fa60e642c400b79754089d5176a11d886`
Correction-Acceptance: Telegram current session — C1 «Утверждаю correction set C1 по SHA-256»; C2 «Утверждаю С2»; C3 «Утверждаю СЗ»; C4 «Принять C4 по указанному SHA-256»; C5 `Принять весь C5 по указанному SHA-256`; C6 `Owner accepted all C6-01 through C6-07 blocks and then explicitly accepted the unchanged whole DRAFT specs/002-execution-router-public-api/drafts/architecture-correction-c6.md by exact SHA-256 b1d306fd1f6252d9034d0a84c5ba733f218cb101772ada0842d077bc4f7796f9.`; C7 `Owner explicitly accepted unchanged C7 whole artifact specs/002-execution-router-public-api/drafts/architecture-correction-c7.md by exact SHA-256 60a2d38d69c4d1f9d1651b07d5b1eec6ac47033776db38626f89e667df8f7f40.`
Correction-C9-Acceptance: Telegram current session — «Хорошо. Делаем по твоей рекомендации, а это полезное дополнение оставим для следующей версии»
Correction-C10-Acceptance: Telegram current session — «Минимальный T005: existing claim + defer в той же lane»
Owner-Acceptance: Telegram current session — ARC-001 «Утверждаю блок по правильному SHA-256»; ARC-002 «Утверждаю» following exact displayed SHA-256; ARC-003 «Утверждаю» following exact displayed SHA-256; ARC-004 «Утверждаю» following exact displayed SHA-256; ARC-005 «Хорошо. Утверждаю» following detailed explanation and exact displayed SHA-256; ARC-006 «Утверждаю» following exact displayed SHA-256
Canonicalization: accepted block behavior preserved; `ARC-001` heading normalized to canonical plan syntax
Implementation-Authorization: DISABLED

## Planning state

`ARC-001` through `ARC-006`, correction sets C1/C2/C3/C4/C5/C6/C7/C9/C10 and the whole integrated architecture are accepted. C9 supersedes non-executable C8 and limits v1 `native_child` to initial pre-credential routing plus bounded terminal failure when a router-selected child requires route/model-change fallback; it adds no same-unit replacement/re-entry. C10 supersedes the disproportionate T005 routing-state/schema design with existing claim plus bounded defer in the unchanged source lane. Completed C6/C7 `main_turn` and C9 `native_child` semantics are unchanged. No correction changes the public contract `1.0`. Source implementation remains governed by separately authorized exact tasks; commit, publication, installation, pilot and LIVE remain unauthorized.

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

7. **Attempt boundary.** Resolution выполняется ровно один раз для каждого нового attempt, который execution-kind owner действительно создаёт, через host seam. `main_turn` и `kanban_worker` при разрешённом создании fallback/escalation attempt снова проходят seam. V1 `native_child` не создаёт replacement внутри того же delegate unit; позднейший обычный parent delegate call является новой execution и проходит начальную seam нормально.

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

По принятому C4 все integer fields являются JSON-safe неотрицательными integers `0..9007199254740991`; `bool`, negative integers, floats, NaN и infinities не принимаются. Более узкие contract ceilings сохраняют приоритет. `ExecutionRouteRequestV1` содержит явный `request_digest`: SHA-256 canonical request с временным `request_digest=null`, но с сохранённым nested instruction digest. Для отсутствующего instruction используются `text=null`, `truncated=false`, `original_utf8_bytes=0`, `digest_algorithm=sha256`, `digest=null`; отсутствие не эквивалентно пустой строке.

Credential-free candidate attributes представлены только immutable tuple элементов `ExecutionRouteCapabilityAttributeV1(name, value)`: максимум 32, уникальные lexicographically sorted ASCII names длиной `1..64`, values только `str | int | bool`, string максимум 256 UTF-8 bytes, integer по общему ceiling. Mappings, lists, nested payloads, floats, nulls, duplicate names и arbitrary objects запрещены; tuple может быть пустым и никогда не расширяет eligibility.
#### 7. Version evolution

Неизвестный required field, unknown decision variant, unsupported execution kind или incompatible contract version отклоняются fail-closed до dispatch. Новые optional observational fields могут добавляться только по правилам совместимости; новый execution kind или расширение routing authority требует product amendment и новой contract version.

Для exact type/parser `1.0` принимаются только frozen public types и закрытые enums. Mapping, неизвестное поле или unknown variant отклоняются; будущее additive observational field требует нового явно поддержанного contract type/version и не принимается молча validator `1.0`.

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

Перед запуском Hermes ещё раз проверяет решение router. Ошибка активного router не превращается в скрытый запуск по старым настройкам. Один attempt получает один неизменный маршрут. Новый routed fallback attempt создаётся только там, где это принято для execution kind; v1 `native_child` вместо same-unit replacement завершается bounded видимым failure.

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

По принятому C5 provider реализует ровно один synchronous callback `resolve_execution_route(request: ExecutionRouteRequestV1, cancellation: ExecutionRouterCancellationSignalV1) -> ExecutionRouteDecisionV1 | None`. `ExecutionRouterCancellationSignalV1` — host-owned immutable read-only capability с единственным public method `is_cancelled() -> bool`; он не предоставляет setter, `cancel()`, mutable attribute, wait/join primitive, callback registration, host object reference или state-mutation method. Host сохраняет private controller и устанавливает cancellation при первом из: истечение 250 ms monotonic deadline, targeted unload/disable owning provider, revoke/supersede active generation.

Host запускает callback в bounded daemon worker, фиксирует monotonic deadline до старта и принимает публикацию результата только через атомарное закрытие open arbitration. При 250 ms host закрывает resolution как `router_error/timeout`, persists closed outcome через existing lifecycle authority, не ждёт и не join-ит callback; любой late return/exception отвергается. Unload или generation revocation аналогично закрывает open resolution fail-closed, сигнализирует cancellation и отвергает late output. Awaitable, mapping или другой return type является malformed decision. Cancellation остаётся cooperative и не даёт dispatch, credentials, tools, retries, config/lifecycle mutation или иной host authority. Одноаргументный callback несовместим с unreleased contract `1.0`; fallback overload отсутствует.

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

Право на новый запуск остаётся у существующей Hermes authority и её budget/depth/permission checks. Для routed `main_turn` точный history-safe порядок, immutable transcript/cursor carry и private continuation seam определены интегрированным C6 ниже. Для `kanban_worker`, если существующая authority разрешает retry/fallback/escalation, текущий attempt получает terminal state, создаются новые attempt/request IDs, request получает только bounded previous-attempt context, и active router вызывается заново.

Для v1 `native_child` router-selected failure, требующий смены route/model, завершает child attempt bounded видимым failure без replacement/resubmit внутри того же `delegate_task` unit. Позднейший обычный parent delegate call является новой execution и маршрутизируется как новый initial child launch. No-router и explicit `pass_through` сохраняют native in-agent fallback. Router не создаёт retry loop, не увеличивает budget/depth и не переносит старое решение автоматически.

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

По принятым C4/C5 exact frozen event fields включают `contract_version`, host IDs (`event_id`, `root_id`, nullable `task_id`, `execution_id`, `attempt_id`, `request_id`, nullable `previous_attempt_id`), `sequence`, `timestamp_utc_ms`, execution kind/surface, router plugin/provider identity и contract version, closed event type, nullable closed decision state (`route | pass_through | stop | router_error`), отдельные nullable requested candidate reference / accepted route identity / actual route identity, bounded host reason, nullable bounded ASCII terminal state и ровно три nullable binding fields: `request_digest`, `instruction_digest`, `eligibility_revision`. Integers подчиняются C4 integer domain. Requested reference использует exact candidate ID request, accepted identity копируется из validated candidate, actual identity — из host start receipt; новые arbitrary route strings событие не принимает.

Для `route_requested` exact `request_digest` и exact `eligibility_revision` обязательны и копируются из validated immutable request. `instruction_digest` равен SHA-256 post-redaction/post-truncation UTF-8 instruction bytes, фактически переданных provider; он non-null при наличии instruction text и null тогда и только тогда, когда request содержит accepted explicit no-instruction representation. `request_digest` и non-null `instruction_digest` — lowercase 64-character hexadecimal SHA-256. Для `route_accepted`, `route_started`, `route_not_started`, `route_finished` все три binding fields обязательны null; correlation идёт через тот же `(request_id, attempt_id)` и authoritative preceding `route_requested`.

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

`sequence` monotonic только внутри exact `(request_id, attempt_id)` и никогда не является session-global. Attempt-local `after_sequence`/`before_sequence` — exclusive bounds; любой из них требует одновременно exact `request_id` и exact `attempt_id`, а missing/partial/non-authoritative identity pair отклоняется до query. Результаты exact attempt упорядочены по `(sequence, event_id)`.

Session-wide pagination начинается без token и детерминированно упорядочивает уже авторизованные и отфильтрованные rows по `(timestamp_utc_ms, event_id)` ascending. Если rows остаются, host возвращает opaque next-page token, связанный с authorized session/scope, normalized filters, contract/token version и последним возвращённым cursor. Продолжение использует только этот token; host до query проверяет structure, integrity, version, scope/session и filter binding. Malformed, altered, expired/unsupported-version или mismatched token возвращает no records fail-closed. Page-token mode несовместим с sequence bounds, caller-supplied timestamp cursors и иным filter set; caller не задаёт внутренний timestamp/event cursor. Page ceiling остаётся 100; global sequence, cursor row/store, aggregate store и mutation/replay/dispatch authority не создаются.

#### 6. Notices для пользовательских поверхностей

Каждый pre-start route notice содержит execution kind, requested candidate/public tuple (если запрошен route), accepted candidate/public tuple (если принят), decision state, bounded reason code/text и attempt/request IDs. Для разрешённых `main_turn`/`kanban_worker` fallback/escalation добавляются previous attempt ID и bounded terminal-state code; v1 `native_child` вместо нового pre-start fallback notice возвращает bounded terminal failure.

`stop/router_error` notice содержит execution kind, decision/error state и bounded reason до возврата управления пользователю. Generic host предоставляет нейтральные state/reason codes и fallback text. `EXT-001` может локализовать plugin reason, но не меняет host fields, actual route или history. Отсутствие native renderer не блокирует event/execution: используется текстовый fallback.
#### 7. Delivery semantics

Event publication следует transaction/durability semantics существующей surface authority; этот API не обещает отдельную stronger delivery guarantee. Main/native construction owner записывает resolution в SessionDB до executor construction; Kanban route/open transaction atomically связывает accepted route и run. Event IDs/attempt sequence support read deduplication.

API не вводит новое правило, что failure существующего persistence backend само по себе блокирует dispatch: действует уже существующая host policy. Однако qualification не может получить PASS, если required transitions не воспроизводятся и не читаются. Optional UI renderer failure не меняет decision и не удаляет host event.

По принятому C4 main/native route attempt и immutable event rows живут только в exact profile-owned existing `SessionDB`/`state.db` и содержат owning `session_id`; pre-start native-child rows принадлежат авторизованной parent session. Эти таблицы включаются в штатное session recovery, а удаление/pruning owning session удаляет связанные route rows в той же host-owned operation. Cross-profile orphan, отдельный store и event-bus authority запрещены. Duplicate transition возвращает существующую immutable запись без увеличения sequence.

По принятому C5 authoritative `execution_route_attempts` row хранит exact `request_digest`, exact `eligibility_revision` и новый nullable `instruction_digest`. Creation/replay requested lifecycle валидирует все три значения против exact immutable request. Duplicate `(request_id, attempt_id)` с отличием любого binding value — conflicting replay, который fail-closed не создаёт event и не увеличивает sequence. Event reconstruction копирует requested-event binding values только из authoritative attempt row; raw instruction, candidate list, system prompt, history, credentials и arbitrary plugin payload не сохраняются.
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
Source-Requirements: FR-002, FR-006, FR-008, FR-011, FR-026, FR-027, FR-028, FR-029, FR-031, FR-032, FR-034
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

В `tools/delegate_tool.py::delegate_task` router вызывается отдельно для каждого initial normalized child после `_normalize_task_list()`/schema coercion, но **до** `_resolve_delegation_credentials(...)`, `_build_children(...)` и `_run_batch(...)`.

Это сохраняет raw explicit model/provider/reasoning pins и не создаёт child `AIAgent`, credential lease или executor work до решения. `_build_children()` и `_run_batch()` остаются единственными штатными construction/dispatch paths; новый dispatcher не добавляется.

Параллельный batch сначала получает независимые immutable decisions для разрешённых child attempts, затем строит только разрешённые children. `stop/router_error` одного child не превращает его в другой route и не создаёт скрытого child executor.

#### 4. `kanban_worker` existing-claim boundary (C10)

Every actually proposed worker attempt routes before credentials, `task_run`, process or executor. Selection already excludes claimed rows, so reservation reuses existing `claim_lock` and `claim_expires` while status remains exactly its original `ready` or `review` lane. The claim carries a bounded routing-reservation token; no `routing` status, `routing_source_status`, task-level planned-attempt field, routing-specific dashboard mapping/index, reservation table/store, queue, scheduler or dispatcher is added.

Router requested/decision lifecycle is projected into existing `task_events`. After validated `route|pass_through`, one existing write transaction revalidates claim token, expiry and unchanged lane, inserts the canonical `task_run` with accepted route metadata/events, then moves the task to running; spawn follows commit. Existing run primary identity and metadata/event payload correlate request and attempt. If exact implementation proves they cannot satisfy identity/idempotency, T005 stops for a later exact owner decision rather than adding a schema column.

`stop/router_error` creates no credentials, run, process or executor. It appends bounded/redacted not-started evidence and retains the exact source lane under the current claim until a short bounded defer expires; existing or minimally extended stale-claim release clears the claim and records recovery evidence. It does not increment worker failure accounting or circuit breaker, and repeated stop/error may defer again without a blocked status/policy.

Crash handling has only two externally distinct classes: reserved/no run is released without a phantom run; opened run uses existing running/run recovery. A real worker/model failure after run start remains a real attempt under existing close/failure-budget/retry/requeue/circuit-breaker authority. Generic lifecycle, notices and redaction use only existing `task_events` and run metadata; dashboard/status consumers remain unchanged.

Provider-only pin remains unsupported where canonical Kanban already rejects it; router cannot weaken that rule.
#### 5. Credential binding

После accepted decision каждый surface использует существующий provider/runtime resolver для selected candidate. API передаёт только candidate identity; raw credentials, base URL secrets и pool metadata не переходят через router.

Если credential resolution не удалась, attempt не стартует. Для `main_turn` и `kanban_worker` возможный retry требует нового attempt через тот же surface adapter. V1 `native_child` не создаёт replacement внутри того же delegate unit; позднейший обычный parent delegate call является новой execution. Active `pass_through` children разделяют один штатный native credential bundle, а ошибки его credential/build path сохраняют существующую batch-wide семантику; T004 не преобразует их в per-child recovery results.

#### 6. Fallback semantics

For router-selected attempts, `try_activate_fallback` and its call sites do not mutate client/model in place. Existing fallback authority first checks its permitted behavior. `main_turn` uses its accepted host-owned restart flow; `kanban_worker` closes a real started run and relies on the next ordinary eligible retry/requeue attempt; v1 `native_child` route/model-change fallback terminates with a bounded visible failure and produces no same-unit replacement.

Execution-kind outcomes:

- `main_turn` surface owner uses the integrated C6 private history-safe continuation seam: it seals the terminal old transcript/current-turn boundary, closes the old route, validates the carried spent-slot cursor, then creates fresh IDs, reruns router before credentials/construction, rebinds callbacks and adopts one final result without replaying the user or completed tools;
- v1 `native_child` closes the router-selected child attempt and returns one bounded visible terminal failure to the existing delegate result path; it does not construct, resubmit or hand off a replacement inside the same `delegate_task` unit;
- Kanban worker closes the current run through existing authority; the next ordinary eligible retry/requeue reserves with existing claim fields and performs a fresh route/open with fresh request/attempt IDs. A minimal routed marker may prevent in-place fallback only if worker initialization requires it, reusing the existing host receipt/fence rather than adding a restart orchestrator.

If budget/permission denies restart, attempt remains terminal. A parent may later issue a normal new delegate call, which is a new execution and routes normally. No-router and explicit `pass_through` preserve current in-agent/native fallback. No global retry dispatcher or plugin retry loop is added.
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
- существующий in-place fallback несовместим с immutable router-selected route; C6 handles `main_turn`, future Kanban keeps its accepted re-entry, and v1 `native_child` terminates visibly instead of replacing;
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
4. In-place fallback запрещён только для router-selected attempt; pass-through сохраняет native semantics. V1 `native_child` не получает automatic same-unit replacement/re-entry.


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

По принятому C5 provider form фиксируется как runtime-checkable Python Protocol с frozen descriptor и одним synchronous callback `resolve_execution_route(request: ExecutionRouteRequestV1, cancellation: ExecutionRouterCancellationSignalV1) -> ExecutionRouteDecisionV1 | None`; одноаргументная форма, awaitable или mapping не являются поддерживаемым provider contract/result `1.0`. Public cancellation signal — frozen immutable read-only capability только с `is_cancelled() -> bool`; private cancellation controller остаётся у host.

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
- все public integers: `0..9007199254740991`, если более узкий ceiling не указан;
- candidate capability attributes: максимум `32`, unique sorted ASCII name `1..64`, string value максимум `256` UTF-8 bytes, value type только `str | int | bool`.

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
- `tests/tools/test_delegate.py` — one decision per initial normalized native child before credentials/build, pins, independent mixed outcomes, SessionDB lifecycle/renderer isolation, native compatibility and bounded terminal routed-fallback behavior without same-unit replacement;
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

---

## Accepted C6 — History-Safe Main-Turn Continuation

Accepted-source SHA-256: `b1d306fd1f6252d9034d0a84c5ba733f218cb101772ada0842d077bc4f7796f9`
Canonical-record SHA-256: `cfc6dc76b0882df627abd60ec65ed8bf18d8aa9c6097fe79af2ac9432dc2377d`
Affected-Task: T003

#### C6 clause 01 — Minimum host-owned same-semantic-turn continuation seam
Source-Requirements: FR-011, FR-025, FR-026, FR-027, FR-029, FR-031

Hermes SHALL add one private, host-owned continuation seam for an already-started `main_turn`:

`continue_main_turn_attempt(agent, continuation_record, *, existing_surface_callbacks) -> turn_result`

The name describes the exact internal role; it is not a public plugin capability or compatibility API. The seam is owned by the existing agent/turn host and is callable only by the four T003 surface owners after a routed attempt returns the accepted host-only restart receipt.

The seam SHALL continue from the exact transcript produced by the terminal old attempt. It SHALL NOT accept or synthesize a new user message. It SHALL initialize the new agent turn loop from the sealed transcript and the existing current-turn boundary, without executing the normal new-user-turn append path, firing new-turn admission hooks for the same logical turn, or converting continuation into `/steer`, interrupt, queued prompt, auto-continue prompt or crash-recovery prompt.

The continuation transcript contains every assistant tool-call row and tool-result row already produced. Continuation begins after that tail. Completed tool calls and side effects are historical facts and SHALL NOT be dispatched again. A new model may inspect existing results and issue a genuinely new tool call, but the host never recreates a previously completed call from the continuation boundary.

Strict role alternation remains authoritative. No synthetic user replay is inserted. If the sealed transcript cannot form a provider-valid continuation under existing role/tool-call repair rules without replaying, rewinding or deleting history, continuation fails closed and the surface returns the bounded terminal failure.

The host SHALL seal one immutable internal `MainTurnContinuationRecord` containing only:

- owning profile/session identity and execution kind `main_turn`;
- old `request_id` and `attempt_id`;
- exact current `turn_id` and `current_turn_user_idx` exported by the old result;
- a recursively immutable snapshot of exact returned `messages`, including assistant tool-call and tool-result metadata required by existing provider projection;
- existing persistence watermark/row-binding metadata maintained by the session persistence path, so previously durable rows are recognized rather than appended again;
- old accepted route signature and router generation/contract generation;
- bounded old terminal reason and accepted host-only fallback receipt;
- `consumed_fallback_slot` and `next_fallback_cursor = consumed_fallback_slot + 1`.

It SHALL NOT contain credentials, writable SessionDB/config objects, executor/dispatcher handles, a retry count invented by execution_router, a target route, synthetic prompt or plugin payload. It is not passed to the router; the router receives only the existing bounded `previous_attempt` projection in the fresh `ExecutionRouteRequestV1`.

The immutable snapshot is the cross-agent ownership boundary. A continuation agent receives a fresh mutable working copy; it cannot mutate the sealed record or old attempt result. The existing surface remains owner of display state, callbacks and final transcript adoption.

#### C6 clause 02 — Mandatory close, budget and routing order
Source-Requirements: FR-011, FR-025, FR-026, FR-027, FR-029, FR-030

For router-selected `main_turn` fallback, every surface SHALL perform this order exactly:

1. The active agent's existing fallback authority decides whether fallback is permitted and atomically consumes exactly one existing slot. It returns a host-only receipt with `consumed_fallback_slot`; it does not choose the next target.
2. The old attempt stops all model execution and returns its exact current transcript and current-turn boundary. No completed tool call is re-entered.
3. The surface seals the immutable continuation record under existing transcript/persistence synchronization. It invokes existing persistence only for rows not already durable; the original user row is never appended again.
4. The old route lifecycle is terminally closed as `route_finished` with bounded terminal state/reason identifying routed restart, before any new `request_id` or `attempt_id` is issued.
5. The surface validates `next_fallback_cursor` against the same existing fallback budget/chain authority. Agent construction, primary-runtime restoration and route-signature rebuild cannot reset, decrement or replace this cursor. A stale, missing, contradictory or exhausted cursor fails closed without router invocation and without a new attempt.
6. Only after validation does the host issue a fresh `request_id` and `attempt_id`, build the remaining credential-free candidate projection at `next_fallback_cursor`, include bounded previous-attempt projection and invoke the active router exactly once.
7. `stop/router_error` records the fresh non-start lifecycle and returns a bounded terminal result. `route/pass_through` is revalidated under accepted rules.
8. Only after the fresh decision is resolved and accepted may the surface bind credentials and construct/select an agent/executor. Changed route signature or router/contract generation forces existing reconstruction. Exact-signature reuse is allowed only where already supported and only while the immutable continuation record and cursor remain attached; reuse cannot resurrect the terminal old attempt.
9. The surface rebinds existing callbacks to the selected continuation agent, calls the private seam with the sealed transcript and adopts the final result once.

`consumed_fallback_slot` is evidence that existing authority spent a slot, not a new budget. Existing per-surface fallback chain, permission/depth and exhaustion rules remain sole authority. C6 creates no extra retry, refund or reset on rebuild or `restore_primary_runtime`. Consumed and earlier slots are excluded from the next candidate projection and cannot be selected again. Exhaustion is terminal and fail-closed.

No-router and explicit `pass_through` attempts retain accepted native in-agent fallback behavior. They create no C6 continuation record and do not enter the seam.

#### C6 clause 03 — Transcript ownership and persistence ordering by surface
Source-Requirements: FR-003, FR-027, FR-029, FR-031

Classic CLI: `CLIChatTurnMixin.chat()` remains sole UI/turn owner; `_chat_stage_user_message()` runs exactly once. `_chat_run_agent()` owns an internal routed-continuation loop, never recursively calls `chat()` and never stages the original message again. It seals `turn.result["messages"]`, current-turn boundary and persistence watermark under the existing persist lock, flushes only missing rows, closes the old route, resolves before `_ensure_runtime_credentials()`, `_resolve_turn_agent_config(...)` and `_init_agent(...)`, rebinds stream/approval/secret/sudo/interim/TTS/status callbacks, and `_chat_settle_turn()` adopts only the final result.

TUI/backend: existing `_run_prompt_submit` admission, inflight marker and `message.start` run once. `_prepare_turn_input` snapshots history once. `_invoke_agent` owns continuation before `_absorb_turn_result` and does not call `_run_prompt_submit`, `_dispatch_followup_turn`, `_enqueue_prompt` or queued drain. It persists only missing rows, closes old route, resolves fresh route, reconstructs `session["agent"]` when signature changes, rebinds stream/interim/title/usage/approval/session-context callbacks, and `_commit_turn_history` runs once under existing `history_version` rules.

Gateway: `TurnRunner.run_sync()` remains sole owner of one admitted inbound turn. Continuation lives inside `run_sync` after original history selection and before stream finalization, `_sync_session_after_run` and outer delivery/persistence. It never recursively calls `run_sync`, re-enters inbound handling, enqueues `ctx.message`, invokes queued-followup, or uses crash auto-resume. Original user persistence/timestamp/display/platform metadata apply once; continuation omits user-persistence arguments, resolves before post-decision runtime/agent construction, rebinds approval/stream/interim/tool/status/session-context/notification callbacks, and finalization/delivery occur only after the final result.

One-shot: `hermes_cli.oneshot._run_agent()` owns the logical turn and one SessionDB handle. It keeps the same store and sealed transcript across routed restart, closes old attempt, validates cursor, obtains fresh IDs/router decision before credentials and `AIAgent`, rebinds noninteractive callbacks and calls the private seam. It never recursively calls `_run_agent()`, reloads the original prompt as a user message or closes SessionDB between attempts. Existing finalizer closes final agent/store once.

#### C6 clause 04 — Route signatures, prompt cache and callbacks
Source-Requirements: FR-011, FR-026, FR-029, FR-031

The accepted route signature remains the construction/reuse key and includes fresh accepted route identity plus router contract/generation identity. Changed signature discards the old agent for execution and uses existing reconstruction. Exact-signature reuse preserves the same sealed continuation and cursor. Credential/executor construction occurs only after fresh decision. Only the positively activated routed-continuation branch may rebind normal callbacks, and every reconstruction cleanup change is guarded at the routed mutation point. No-router/native turns retain the same callback objects, cleanup behavior and exact pre-T003 control flow. Callbacks enter neither public request nor continuation record.

The transcript prefix is not rewritten, reordered or truncated by C6. Existing sanctioned compression/provider-projection repair remain the only transformations. Existing system-prompt restoration, route identity validation, tool-schema freeze and prompt-cache policy remain authoritative; no second system prompt, synthetic user cache break or rewind is added. Rebuild uses existing persisted system-prompt authority and the exact sealed transcript.

#### C6 clause 05 — Notices, lifecycle and failure behavior
Source-Requirements: FR-011, FR-025, FR-027, FR-029, FR-030

- old routed attempt: `route_finished` with terminal state `routed_restart_required`, bounded reason and original IDs;
- fresh attempt, only after budget/cursor validation: `route_requested`, then accepted `route_accepted`/`route_started` or `route_not_started` under fresh IDs;
- final started continuation attempt: `route_finished` with actual terminal state.

A pre-start fallback notice is emitted through the existing renderer after fresh decision and before executor start. It includes only execution kind, old attempt ID/terminal code, fresh request/attempt IDs, accepted route fields and bounded reason. Renderer failure remains non-authoritative.

Failure to seal a valid transcript/current-turn boundary or contradictory persistence watermark fails closed without new attempt or replay. Budget exhaustion/invalid cursor returns existing terminal fallback-exhausted result after closing old attempt, with no fresh IDs/router call. Router stop/error, credential failure or construction failure closes fresh attempt via existing `route_not_started` and never returns to old attempt. Continuation exception closes fresh started attempt via existing `route_finished`; another slot may be consumed only if existing authority permits. C6 adds no cross-process continuation journal or automatic replay. On the positively activated routed-continuation path only, each concrete transcript- or delivery-mutating callback proven able to produce late output after attempt closure receives the minimum surface-owned attempt/request token guard; late old results rejected by that guard cannot mutate fresh state. No universal callback inventory or generic fencing subsystem is authorized.

#### C6 clause 06 — Required RED→GREEN tests
Source-Requirements: FR-003, FR-011, FR-025, FR-026, FR-027, FR-029, FR-030, FR-031

The required test set is exactly 16 red-capable tests: four shared private-seam tests and three each for Classic CLI, TUI/backend, Gateway and one-shot. They prove exact transcript continuation without user replay, completed side effect exactly once, spent cursor across rebuild/primary restore, fresh identity/lifecycle/pre-credential order, per-surface same-turn continuation and callback rebinding, exhaustion without recursion/queue/replay/router call, and unchanged no-router/native fallback. Every positive per-surface test directly asserts SessionDB has exactly one original user row and one copy of the completed assistant tool-call/tool-result pair.

#### C6 clause 07 — Traceability and exact supersession
Source-Requirements: FR-003, FR-011, FR-025, FR-026, FR-027, FR-029, FR-030, FR-031

C6 refines only FR-003/011/025/026/027/029/030/031, ARC-001 sections 3/4/7, ARC-003 sections 1/4/5/6/8, ARC-004 sections 3/6/7, ARC-005 sections 2/6/7, ARC-006 sections 2/7 and T003 S002-S008. It supersedes exactly the old ARC-005 section 6 `main_turn` re-entry bullet, the phrase `existing surface re-entry` for routed `main_turn` only where it claimed Classic CLI/Gateway already provided the seam, and ARC-003 section 6 only for routed `main_turn` ordering. All other ARC-001–ARC-006, C1–C5 and T003 clauses remain unchanged.

C6 explicitly rejects user replay, partial transcript rewrite/removal, completed tool re-execution, steer/interrupt/queued/crash-resume continuation, second dispatcher/gateway/store/journal/retry engine/budget/owner, plugin transcript exposure, public continuation API/capability/version change, queue continuation, T004/T005 expansion, product routing policy, commit, publication, installation, profile/runtime mutation, pilot and LIVE.

---

## Accepted C7 — Optional-Path Scope Containment

Accepted-source SHA-256: `60a2d38d69c4d1f9d1651b07d5b1eec6ac47033776db38626f89e667df8f7f40`
Canonical-record SHA-256: `065b71217af1f6cfdc594f56d6d43b7889a5bdc6d902dffb50f1449c832f4157`
Accepted-C6-Source-SHA256: `b1d306fd1f6252d9034d0a84c5ba733f218cb101772ada0842d077bc4f7796f9`
Affected-Task: T003

The source audit disproved C6-05's claim that the relevant callbacks already have attempt/request identity fencing. C7 authorizes only a minimal, surface-owned token/guard on the concrete transcript- or delivery-mutating callbacks proven able to produce late output after a routed attempt closes. The guard exists only on the positively activated routed-continuation path. No-router/native turns retain the same callback objects and exact pre-T003 branch, with no new wrapper, catch/remap block, lifecycle metadata, cleanup or callback replacement.

Route-signature reconstruction cleanup is likewise conditional at the routed mutation point. It must not alter shared native `/model`, cache-mismatch, credential-error or existing rebuild semantics. Continuation symbols are private underscored internal names and are absent from `__all__`, facades, compatibility exports and public/plugin surfaces. C7 adds no universal callback inventory, generic fencing subsystem, speculative hardening or ordinary-turn change.

The C6-06 ceiling remains exactly 16 tests, with these exact retained nodes:

1. `test_main_turn_continuation_uses_exact_transcript_without_user_replay`
2. `test_main_turn_continuation_does_not_repeat_completed_tool_side_effect`
3. `test_main_turn_continuation_carries_spent_cursor_across_rebuild_and_primary_restore`
4. `test_main_turn_continuation_fresh_identity_and_lifecycle_order`
5. `test_cli_routed_fallback_continues_same_turn_once`
6. `test_cli_routed_fallback_budget_exhaustion_is_not_reset`
7. `test_cli_no_router_main_turn_and_native_fallback_are_unchanged`
8. `test_tui_routed_fallback_continues_inside_one_prompt_submit`
9. `test_tui_routed_fallback_budget_exhaustion_is_not_a_queued_followup`
10. `test_tui_no_router_main_turn_and_native_fallback_are_unchanged`
11. `test_gateway_routed_fallback_continues_one_inbound_turn`
12. `test_gateway_routed_fallback_budget_exhaustion_never_queues_or_replays`
13. `test_gateway_no_router_main_turn_and_native_fallback_are_unchanged`
14. `test_oneshot_routed_fallback_continues_before_single_close`
15. `test_oneshot_routed_fallback_budget_exhaustion_does_not_recurse`
16. `test_oneshot_no_router_main_turn_and_native_fallback_are_unchanged`

C7 supersedes only the C6-05 phrase `ignored by the existing attempt/request identity fencing`, replacing that false premise with the routed-only surface-owned guard above, and C6-04 phrases requiring callback rebinding and existing reconstruction/cleanup only to require a positive routed activation guard at each mutation point and preserve exact no-router/native object identity and control flow. All other C6 and T003 scope remains unchanged. Public contract version remains `1.0`. C7 authorizes no T004, T005, commit, installation, runtime/profile/config change or LIVE action.
