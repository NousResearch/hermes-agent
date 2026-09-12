# Architecture Correction Set C4 — T002 Contract Closure

Stage: ARCHITECTURE_CORRECTION
Status: DRAFT
Project-Version: 0.1.1
Architecture-Version: 1
Reviewed-Plan-Record-SHA256: `3d43b28dc773fe8efd1cd0d74246c76b87abdb5ddcbcfc0ec8cbad56beae2464`
Affected-Task: T002
Implementation-Authorization: DISABLED_UNTIL_ACCEPTED

## Причина

При входе в T002 выяснилось, что принятая архитектура требует bounded integers, canonical request digest, immutable candidate capability attributes и immutable event projection, но не фиксирует несколько wire-level решений. Реализация без их принятия создала бы новый публичный контракт по усмотрению исполнителя. C4 закрывает только эти пробелы и не меняет продукт, execution kinds, authority, consent, routing policy или границы T002.

## C4-01 — Integer domain

Все integer-поля публичного contract `1.0` являются JSON-safe неотрицательными целыми числами в диапазоне `0..9007199254740991` (`2^53-1`). `bool` не считается integer. Отрицательные значения, floats, `NaN`, infinities и значения выше ceiling отклоняются до provider callback. Это правило применяется к byte/count/sequence/timestamp-полям; более узкие принятые ceilings сохраняют приоритет.

## C4-02 — Instruction and request digests

`ExecutionRouteRequestV1` содержит явное поле `request_digest`.

- Instruction присутствует: `digest_algorithm="sha256"`, а `digest` равен SHA-256 exact post-redaction/post-truncation UTF-8 bytes, переданных provider.
- Instruction отсутствует: `text=null`, `truncated=false`, `original_utf8_bytes=0`, `digest_algorithm="sha256"`, `digest=null`; отсутствие текста не приравнивается к пустой строке.
- Request digest равен SHA-256 canonical UTF-8 JSON всего request при временно установленном `request_digest=null`; nested instruction digest не обнуляется.
- Host принимает request только при совпадении указанного `request_digest` с пересчитанным значением.

## C4-03 — Candidate capability attributes

Credential-free capability attributes представлены только как immutable tuple элементов `ExecutionRouteCapabilityAttributeV1(name, value)`:

- максимум 32 элемента на candidate;
- `name` — уникальная ASCII-строка длиной `1..64`, элементы отсортированы лексикографически по `name`;
- `value` — только `str | int | bool`; integer подчиняется C4-01, string ограничена 256 UTF-8 bytes;
- mappings, lists, nested payloads, floats, `null`, duplicate names и произвольные объекты запрещены.

Tuple может быть пустым. Эти attributes наблюдательные и не расширяют host eligibility; provider всё равно может выбрать только host-issued `candidate_id`.

## C4-04 — Exact provider callback and v1 shape rule

Public provider contract `1.0` — runtime-checkable Python Protocol с frozen descriptor и одним синхронным callback:

`resolve_execution_route(request: ExecutionRouteRequestV1) -> ExecutionRouteDecisionV1 | None`

Callback выполняется host в bounded daemon worker под deadline 250 ms. Awaitable, mapping или иной return type является malformed decision и даёт видимый fail-closed `router_error`; `None` нормализуется host в explicit `pass_through`.

V1 принимает только exact frozen public types и закрытые enums. Неизвестные поля, mappings и неизвестные variants отклоняются. Будущее additive observational field требует нового явно поддержанного contract type/version; оно не принимается молча текущим parser/validator `1.0`.

## C4-05 — Event wire fields and transitions

`ExecutionRouteEventV1` использует exact frozen fields:

- `contract_version`, host-issued `event_id`, `root_id`, nullable `task_id`, `execution_id`, `attempt_id`, `request_id`, nullable `previous_attempt_id`;
- `sequence` и `timestamp_utc_ms`, подчинённые C4-01;
- `execution_kind`, `surface_class`, router plugin/provider identity и router contract version;
- closed `event_type`: `route_requested | route_accepted | route_started | route_not_started | route_finished`;
- closed `decision_state`: `route | pass_through | stop | router_error | null`;
- nullable requested candidate reference, accepted route identity и actual route identity как три разные поля;
- nullable bounded host `reason_code`, `reason_text` и nullable bounded ASCII `terminal_state`.

Route identity не принимает новые строки: requested reference использует candidate ID exact request, accepted identity копируется из validated candidate, actual identity — из host start receipt. Event не содержит instruction text, candidate list, credentials, traceback или plugin payload.

Переходы остаются принятыми ARC-004: `route_started` XOR `route_not_started`; `route_finished` требует `route_started`; `route_accepted` существует только для validated `route`; `pass_through` не создаёт `route_accepted`. Duplicate transition возвращает существующую immutable запись и не увеличивает sequence.

## C4-06 — SessionDB ownership, recovery and deletion

Main/native lifecycle rows принадлежат exact profile-owned `SessionDB` и содержат owning `session_id`; pre-start native-child row принадлежит авторизованной parent session. Route attempt и immutable event tables добавляются в существующий `state.db`, включаются в штатное session recovery, а удаление/pruning owning session удаляет связанные route rows в той же host-owned operation. Отдельный store, event bus authority или orphaned cross-profile row не допускается.

## Scope proof

C4 не добавляет:

- новые execution kinds;
- model catalog, classifier, chains или product policy;
- credentials, tools, workspace, dispatcher, scheduler, queue, retry engine или второй store;
- async/network/model/tool work в provider callback;
- новую capability, кроме уже принятой `execution.routing`;
- T003–T005 surface adapters, commit, publication, installation, pilot или LIVE.

## Acceptance effect

Принятие exact C4 разрешает интегрировать эти шесть правил в нормативные clauses ARC-002/003/004/006 и продолжить уже авторизованную T002. Оно не изменяет product spec или task map и не разрешает commit либо любую следующую задачу.
