# Бэклог и документация проекта: Telegram News Aggregator (v2)

Этот документ служит центральным реестром состояния проекта, его архитектуры и отслеживания выполненных, текущих и запланированных задач.

---

## 1. Общее описание и архитектура

Проект представляет собой автономный новостной конвейер на базе **Hermes Agent**. Система собирает новости из международных и российских источников, выполняет интеллектуальную дедупликацию, анализирует статьи с помощью LLM-агентов (Scout, Fact-check, Summarizer, Formatter), отправляет черновики на модерацию в закрытый чат и публикует одобренные посты в Telegram-канал.

### Схема взаимодействия компонентов

```
                  ┌────────────────────────────────────────┐
                  │ 1. INGESTION (Сбор)                    │
                  │ - RSS-ленты (rss.py + ETag/Modified)   │
                  │ - Telegram-каналы (telegram_ingest.py)  │
                  └──────────────────┬─────────────────────┘
                                     │ (сохранение в articles)
                                     ▼
                  ┌────────────────────────────────────────┐
                  │ 2. CLUSTERING (Кластеризация)          │
                  │ - Группировка дубликатов               │
                  │ - Перевод заголовков на русский язык   │
                  │ - Привязка к duplicate_clusters        │
                  └──────────────────┬─────────────────────┘
                                     │ (выбор лучшего источника)
                                     ▼
                  ┌────────────────────────────────────────┐
                  │ 3. LLM-PIPELINE (Обработка Hermes)     │
                  │ - Scout: скоринг и рубрики             │
                  │ - Fact-check: проверка противоречий    │
                  │ - Summarizer: Fair Use рерайтинг       │
                  │ - Formatter: генерация текста поста    │
                  └──────────────────┬─────────────────────┘
                                     │ (запись в drafts)
                                     ▼
                  ┌────────────────────────────────────────┐
                  │ 4. MODERATION (Модерация)              │
                  │ - bot.py: отправка карточки в чат      │
                  │ - Кнопки Approve/Reject/Edit           │
                  └──────────────────┬─────────────────────┘
                                     │ (запись в queue)
                                     ▼
                  ┌────────────────────────────────────────┐
                  │ 5. DELIVERY (Публикация)               │
                  │ - sender.py: отправка в канал          │
                  │ - Соблюдение лимитов и Retry 429       │
                  └────────────────────────────────────────┘
```

---

## 2. Структура директорий и БД

### Файловая структура
* **[src/ingest/](file:///D:/hermes/TG_news/src/ingest)**: Сборщики данных ([rss.py](file:///D:/hermes/TG_news/src/ingest/rss.py), [telegram_ingest.py](file:///D:/hermes/TG_news/src/ingest/telegram_ingest.py)).
* **[src/dedup/](file:///D:/hermes/TG_news/src/dedup)**: Алгоритмы дедупликации ([simhash.py](file:///D:/hermes/TG_news/src/dedup/simhash.py), [cluster.py](file:///D:/hermes/TG_news/src/dedup/cluster.py)).
* **[src/pipeline/](file:///D:/hermes/TG_news/src/pipeline)**: LLM-клиент и координатор конвейера ([hermes_client.py](file:///D:/hermes/TG_news/src/pipeline/hermes_client.py), [run_pipeline.py](file:///D:/hermes/TG_news/src/pipeline/run_pipeline.py)).
* **[src/moderation/](file:///D:/hermes/TG_news/src/moderation)**: Telegram-бот модерации ([bot.py](file:///D:/hermes/TG_news/src/moderation/bot.py)).
* **[src/delivery/](file:///D:/hermes/TG_news/src/delivery)**: Демон отправки постов ([sender.py](file:///D:/hermes/TG_news/src/delivery/sender.py)).
* **[storage/](file:///D:/hermes/TG_news/storage)**: SQLite БД `news_aggregator.db` и файлы сессий Telethon.

---

## 3. Таблица бэклога задач (Backlog)

| Статус | Фаза / Задача | Описание / Технические детали |
| :--- | :--- | :--- |
| **Выполнено** | **Фаза 1: Сбор данных** | Настройка схемы БД SQLite, сбор RSS (с поддержкой ETag/Last-Modified и Browserless fallbacks), интеграция Telethon для Telegram-каналов. |
| **Выполнено** | **Фаза 2: Кластеризация** | Написание SimHash и алгоритмов расстояния Хэмминга. Создание скрипта [cluster.py](file:///D:/hermes/TG_news/src/dedup/cluster.py) для перевода заголовков на русский язык и группировки новостей через LLM. |
| **Выполнено** | **Фаза 3: Интеграция с Hermes** | Настройка промптов Scout, Fact-check, Summarizer, Formatter и вызовов LLM API Proxy. Авто-аппрув неполитических новостей от Tier-1/2 СМИ. |
| **Выполнено** | **Фаза 4: Модерация** | Написание интерактивного бота модерации [bot.py](file:///D:/hermes/TG_news/src/moderation/bot.py) с кнопками Approve/Reject. |
| **Выполнено** | **Фаза 5: Публикатор** | Логика Token Bucket очереди публикаций и обработка лимитов 429 в [sender.py](file:///D:/hermes/TG_news/src/delivery/sender.py). |
| **Выполнено** | **Развертывание (WSL/Docker)** | Добавление контейнеров `news-moderation-bot` и `news-delivery-sender` в общий `docker-compose.yml`. |
| **Выполнено** | **Добавление Telegram-источников** | Интеграция новых каналов Ньюсач (`@ru2ch`) и NEXTA Live (`@nexta_live`) в БД через [integrate_resources.py](file:///D:/hermes/TG_news/src/integrate_resources.py). |
| **Выполнено** | **Кроссплатформенная совместимость** | Исправление путей (`/tmp` -> `tempfile.gettempdir()`) и обеспечение параллельной отказоустойчивости SQLite при одновременном запуске вручную и по cron. |
| **Выполнено** | **Гибридная кластеризация (Векторы + LLM)** | Оптимизация [cluster.py](file:///D:/hermes/TG_news/src/dedup/cluster.py) с использованием локальных эмбеддингов BGE-M3 и фильтрации по косинусному сходству. |
| **Выполнено** | **Агентный режим RANK** | Реализация режима `RANK` (агентский скоринг приоритетов) согласно [russia_news_editorial_pipeline_skill_en.md](file:///D:/hermes/TG_news/russia_news_editorial_pipeline_skill_en.md). |
| **Выполнено** | **Миграция на parse_mode='HTML'** | Обновление Formatter, [sender.py](file:///D:/hermes/TG_news/src/delivery/sender.py) и [bot.py](file:///D:/hermes/TG_news/src/moderation/bot.py) для перевода верстки постов на HTML. |
| **Выполнено** | **Интеграция графа связей (Knowledge Graph)** | Добавление хронологического отслеживания сюжетов (`related_to_id` в таблице `duplicate_clusters`). |
| **Выполнено** | **Диагностика блокировок канала** | Установлена точная причина ошибки `chat not found` (неверный ID канала, полученный прибавлением `-100` к приватному ID пользователя). Прописана инструкция по созданию канала и получению реального ID. |
| **Выполнено** | **Устранение голодания очереди** | Перевод SQL-запроса в [run_pipeline.py](file:///D:/hermes/TG_news/src/pipeline/run_pipeline.py) на сортировку по свежести (`published_at DESC`) и фильтрацию статей старше 24 часов на уровне БД. |
| **Выполнено** | **Временное заземление фактчека** | Передача `current_time` в [run_factcheck](file:///D:/hermes/TG_news/src/pipeline/hermes_client.py#L126-L150) для предотвращения анахронистических галлюцинаций модели (например, ложного срабатывания на события ЧМ-2026). |
| **Выполнено** | **Карточки модерации и редактирование** | Реализация подробных карточек с аналитикой (Ranker, Factcheck) и кнопки `✏️ Редактировать` с инлайн-редактированием через reply в [bot.py](file:///D:/hermes/TG_news/src/moderation/bot.py). |
| **Выполнено** | **Приоритизация юмора и позитива** | Интеграция категорий «Юмор» и «Позитив» в Scout (с разделением на чистый юмор и ироничные дополнения в серьезных новостях), сохранение шуток/гифок в Summarizer/Formatter, и бонус +15 баллов в Ranker только для чисто развлекательного контента. |
| **Выполнено (2026-07-23)** | **NEWS-P1-001 — evolving stories** | `src/intelligence.py`: стабильные story IDs и хронология обновлений; synthetic E2E в `tests/communication/test_news_intelligence_xdom.py`. |
| **Выполнено (2026-07-23)** | **NEWS-P1-002 — claims/contradictions** | Утверждения, stance/evidence и объяснимая матрица противоречий. |
| **Выполнено (2026-07-23)** | **NEWS-P1-003 — source reliability** | Исторические outcomes и explanation/evidence для оценки источника. |
| **Выполнено (2026-07-23)** | **NEWS-P2-001 — watchlists** | Фильтры по topics/entities/geography с preview. |
| **Выполнено (2026-07-23)** | **NEWS-P2-002 — breaking confirmation** | Alert подтверждается только несколькими независимыми источниками. |
| **Выполнено (2026-07-23)** | **NEWS-P2-003 — normalization/translation** | Сохраняются язык, нормализованный и переведённый текст с provenance. |
| **Выполнено (2026-07-23)** | **NEWS-P3-001 — editorial explanation** | Selection/dedup/reject решения имеют reason/evidence. |
| **Выполнено (2026-07-23)** | **NEWS-P3-002 — source health** | Dashboard данных, quarantine и recovery transitions. |
| **Выполнено (2026-07-23)** | **NEWS-P3-003 — digests/archive** | Daily/weekly digest и архив с детерминированным составом. |
| **Выполнено, публикация отключена** | **Communication Core / XDOM boundary** | Принимаются только публичные topic/entity/story/source refs; private message/article bodies отклоняются; результат — приватная подсказка или draft. Production Telegram publication не выполнялась. |

---

## 4. Журнал текущих инцидентов и решений

### Инцидент 1: Блокировка доставки (`chat not found`)
* **Симптом:** Сообщения в `publication_queue` переходят в статус `failed` с ошибкой `Bad Request: chat not found`.
* **Причина:** Переменная `TELEGRAM_CHANNEL_ID` в [.env](file:///D:/hermes/TG_news/.env) была установлена в `-100283186658` путем добавления `-100` к личному ID пользователя `283186658`. Бот пытается отправить сообщение в несуществующий чат.
* **Решение:** Создать реальный публичный/приватный канал, добавить бота `@Alex_oca_bot` администратором с правами на отправку сообщений, получить настоящий ID канала и обновить переменную `TELEGRAM_CHANNEL_ID` в [.env](file:///D:/hermes/TG_news/.env). После этого перезапустить или запустить публикатор [sender.py](file:///D:/hermes/TG_news/src/delivery/sender.py).

### Инцидент 2: Голодание источников Tier-2 и Telegram-источников
* **Симптом:** Новости от Ньюсач и Маркеттвитс копились в базе данных, но не переходили в черновики.
* **Причина:** Очередь была забита более чем 6600+ статьями Tier-1 из RSS-лент. Из-за сортировки по `s.tier ASC` и лимита в 3–5 статей за раз, пайплайн обрабатывал только Tier-1 и не доходил до Tier-2.
* **Решение:** SQL-запрос изменен на сортировку по времени (`a.published_at DESC`) и дополнен фильтрацией статей старше 24 часов на уровне БД, что разблокировало обработку свежих новостей из Telegram.

### Инцидент 3: Ложные флаги фактчекинга (Анахронизм)
* **Симптом:** Фактчекер пометил реальную новость о Трампе на финале ЧМ-2026 как фейковую, решив, что ЧМ-2026 еще не состоялся.
* **Причина:** У модели не было временного контекста (знала только свой cutoff — 2024/2025 год).
* **Решение:** В промпт [run_factcheck](file:///D:/hermes/TG_news/src/pipeline/hermes_client.py#L126-L150) теперь передается `current_time` системы, что позволяет модели сопоставлять события с текущим 2026 годом.
