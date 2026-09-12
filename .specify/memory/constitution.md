# Hermes Project Constitution Core

Core-Standard-Version: 2.0.0
Core-Standard-SHA256: 04b24bf2fd15fd92a19edd3f66f7227c178212b3d364feb1c2cef3b4802ef26d
Ratified: 2026-08-20
Last-Amended: 2026-08-31

## Core Principles

### I. One Authority per State

Каждый класс intent, состояния, side effect, результата и evidence MUST иметь одного владельца. Дублирующие активные источники истины, dual writers и неограниченные compatibility layers запрещены.

### II. Mandatory Product-to-Production Stages

Каждая новая версия продукта MUST пройти три принятых этапа до implementation: (1) `spec.md` полностью и понятным языком описывает назначение, пользователей, функциональность, бизнес-логику, данные, границы, ошибки, non-goals и `DONE_MEANS`; (2) `plan.md` описывает минимальную архитектуру, модули, ответственность, взаимодействия, данные, runtime и deployment; (3) `tasks.md` содержит полную dependency-ordered карту от первой реализации до запуска и production readback. Документы минимальны по форме, но полны по содержанию и обязательны к точному исполнению.

### III. Scope and Target-Level Discipline

До действия MUST быть известны target, requested outcome, `DONE_MEANS`, разрешённый уровень (`SPEC_ONLY`, `LOCAL_ARTIFACT`, `TEST_ENV`, `LIVE_ENV`) и forbidden side effects. Нельзя выполнять ниже или выше авторизованного уровня.

### IV. Product Outcome Over Supporting Artifacts

Статусы, спецификации, планы, тесты, scanners, reviews, evidence bundles, активный процесс, HTTP 2xx, исторический успех и выполненная карточка не являются продуктовым результатом. Положительный verdict MUST опираться на реальный artifact или business readback из acceptance criteria. Поддерживающие артефакты создаются только в объёме, необходимом для достижения и проверки результата.

### V. Ponytail Architecture Gate

Architecture MUST implement the complete accepted product with the fewest necessary components and transitions. During architecture design every proposed component is challenged for necessity, merge/reuse, platform support, existing dependencies and removable stores, queues, adapters or services. Ponytail may simplify implementation structure but MUST NOT remove accepted product behavior. Architecture/security review remains proportional to concrete risk.

### VI. Binding Ponytail Task Map

After architecture acceptance, Ponytail is applied again while generating `tasks.md`: reuse existing capabilities, standard library, native platform and installed dependencies before creating minimum new code, tests or documentation. Every product capability and architecture component MUST be covered; orphan work is forbidden. Once accepted, the map is the sole execution route, mirrored by Hermes `todo`, and is executed in dependency order without silent redesign. Tests and review remain the minimum required by acceptance and material risk.

### VII. Safe Mutation and Recovery

Перед внешней или LIVE-мутацией MUST быть снят baseline. После мутации обязателен readback. Неопределённая identity side effect требует reconciliation до retry. Recovery выполняется missing-only. Replacement включает rollback и удаление legacy authority.

### VIII. Security and Least Exposure

Секреты MUST храниться только в утверждённых защищённых хранилищах и не попадать в chat, specs, logs, evidence или git. Профиль получает минимальные toolsets, credentials и Knowledge scope, необходимые его задаче.

### IX. Knowledge Boundaries

Skills, Knowledge, Memory и execution state являются разными слоями. Public, Customers и Internal Knowledge MUST быть разделены. Метрики, цены, live state, competitor facts и mutable specifications MUST иметь дату и provenance и не храниться только в memory.

### X. Specialized Execution

Задачи SHOULD назначаться узкоспециализированным профилям с явными role boundaries, required skills, Knowledge scope, output contract, evidence minimums и escalation rules. Профиль не должен расширять профессиональную или операционную область только потому, что технически имеет доступ.

### XI. Simplicity and Delivery

Используется минимальная архитектура и минимальный процесс, полностью закрывающие утверждённый результат. Дополнительные сервисы, поля, fallback, cron, storage, adapters, документы, тесты и review loops запрещены без доказанной необходимости. Если supporting work становится больше product change без конкретного risk justification, дальнейшее усложнение MUST остановиться, а реализация продолжиться кратчайшим путём к `DONE_MEANS`. После cutover в активном контуре остаётся одна архитектура.

### XII. Honest Project State

Проектные отчёты MUST различать `ДОКАЗАНО`, `ЧАСТИЧНО`, `НЕ ДОКАЗАНО`, `НЕ ВНЕСЕНО В LIVE`, `ОТКЛОНЕНО` и `HISTORICAL`. `STATUS.md` описывает текущее доказанное состояние; `HANDOFF.md` — точку продолжения; ни один из них не заменяет feature spec или canonical brief.

### XIII. Living Project Versions

Одна версия проекта связывает согласованные `spec.md`, `plan.md`, `tasks.md`, implementation и release/readback. Состояния версии: `DRAFT`, `ACCEPTED`, `IN_PROGRESS`, `RELEASED`, `SUPERSEDED`. Если новое авторизованное решение не совпадает с текущими документами, открывается новая version/amendment и документы обновляются; решение не отклоняется ради устаревшего текста. Released history не переписывается. Gate MUST иметь штатные version-bootstrap и document-recovery пути, которые не дают implementation или LIVE authority.

## Development Workflow

1. Создать `DRAFT` версии и полностью описать продукт в `spec.md`.
2. Принять продукт; спроектировать `plan.md` с архитектурным Ponytail-проходом.
3. Принять архитектуру; построить полную `tasks.md` с task-map Ponytail-проходом.
4. Принять карту и выполнить её в dependency order, синхронизируя активную работу с Hermes `todo`.
5. Проверять каждую задачу минимально достаточным способом; supporting artifacts не заменяют delivery.
6. Запустить продукт, получить production/product readback и перевести версию в `RELEASED`.
7. Для изменений открыть следующую patch/minor/major версию; прежнюю после cutover отметить `SUPERSEDED`.

## Governance

- Core constitution имеет semantic version.
- Проектный overlay может усиливать, но MUST NOT ослаблять core.
- Изменение core требует impact report и migration plan.
- Каждый проект хранит `Core-Standard-Version` и `Core-Standard-SHA256` в локальной constitution.
- Validator проверяет структуру, обязательные поля и отсутствие unresolved placeholders; semantic review добавляется только при материальном риске и не заменяет product acceptance.
- Отклонение от принципа допускается только как явное, датированное, ограниченное exception с владельцем, риском и сроком удаления.

**Version**: 2.0.0 | **Ratified**: 2026-08-20 | **Last Amended**: 2026-08-31
