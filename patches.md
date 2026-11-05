# patches.md — что изменено в этом форке относительно upstream

Форк: `git@github.com:dappvibe/hermes-agent.git` (origin), апстрим:
`https://github.com/NousResearch/hermes-agent.git`.

Точка расхождения (merge-base на момент написания): `f0888d77bf6`
(31 окт 2026). Всё ниже — 40 коммитов форка сверху, от старых к новым.
Актуальный список: `git log upstream/main..main --oneline --reverse`.

Это сводка ПО СУТИ изменений. Технические детали восстановления 4 из них
после `hermes update` (конкретные grep-команды, что проверять живьём) —
в `~/.hermes/hermes-agent-local-patches.md` (вне этого репозитория,
не трогается git-историей форка).

---

## 1. Новые MCP-коннекторы (optional-mcps/)

Три собственных коннектора под магазин M-Style на Ozon + личный телеграм-аккаунт,
плюс чистка каталога до нужного набора.

### Avito (`optional-mcps/avito/`)
- `07074a1e6f` — первая версия: обёртка над `avito-api` (PyPI) к `api.avito.ru`
  (auth, мессенджер, объявления, профиль).
- `5ca2deada0` — автоматический OAuth без копипаста кода: Avito требует реальный
  domain-callback (webhook.site не подходит — «domain must belong to you»).
  Поднят `optional-mcps/oauth-callback/` — локальный HTTP-сервер, Cloudflare
  Tunnel `cb.mstylehome.ru` → `localhost:8743`. Новый тул
  `avito_login_wait_callback` ждёт код автоматически.

### Ozon (`optional-mcps/ozon/`)
- `ca6a0123f3` — первая версия: 3 generic-тула (`ozon_seller_call`,
  `ozon_performance_call`, `ozon_list_endpoints`) на 222 документированных
  эндпоинта (Seller API + Performance API, 18 категорий).
- `99ad42f0fe` — заменено на 222 именованных типизированных тула (по одному на
  эндпоинт, конвенция как у Cloudflare MCP): 31 реальная реализация
  (подтверждённые схемы запроса) + 191 явная `not_implemented`-заглушка
  (путь/метод известны, схема тела — нет; вызов не бьёт по живому API).
- `dc13c3b487` — свёрнуто в 10 тулов с параметром `action` (`ozon_fbs`,
  `ozon_fbp`, `ozon_returns`, `ozon_delivery`, `ozon_carriage`,
  `ozon_products`, `ozon_reviews`, `ozon_ads`, `ozon_fbo`, `ozon_finance`) —
  та же логика на 222 действия, меньше шума в тул-списке модели. Текущее
  рабочее состояние.

### Telegram user-client (`optional-mcps/telegram-client/`)
- `76e722991a` — обёртка над Telethon (настоящий MTProto-клиент
  пользовательского аккаунта, не Bot API): 19 тулов (диалоги, сообщения,
  участники групп, медиа, профиль). Логин интерактивно через чат
  (`telegram_login_start/_code/_password`), сессия в
  `~/.hermes/mcp-tokens/telegram-client.session`.
- `eca59264b3` — по умолчанию публичные TD `api_id/hash`, у пользователя
  спрашивается только номер телефона (не нужно регистрировать своё приложение
  в Telegram).

### Инфраструктура коннекторов
- `b0f7d98d29` — каталог `optional-mcps/` обрезан с ~68 до 5 записей:
  avito, ozon, cloudflare, supabase, vercel. Остальное апстримовское не
  удалено из git-истории/апстрима, просто не в каталоге этого форка.
- `76d0e80280` — venv-окружения коннекторов (avito/ozon/telegram-client)
  перенесены внутрь форка (`optional-mcps/<name>/server/venv/`), раньше жили
  во внешней `~/.hermes/mcp-servers/<name>/venv/`, привязанной к одному хосту.

## 2. MAX messenger — новая платформа (plugins/platforms/max/)
- `d2e9212a9c` — адаптер мессенджера MAX (Минцифры РФ): long-poll `/updates`,
  `POST /messages`, 1:1 текстовые диалоги, карточка на странице Channels
  (`MAX_BOT_TOKEN` / `MAX_ALLOWED_USERS`), как у Slack/Matrix/Telegram.
- `5f276e516a` — доверие российскому gov root CA для `platform-api2.max.ru`
  (сертификат не входит в стандартные trust stores вне РФ).
- `1e1cd42a07` — фикс роутинга ответов в диалогах: по `sender.user_id`, а не
  `recipient.chat_id` (chat_id — внутренний id диалога MAX, 404 на отправке).
- `8eaf53a8ab` / `f2cb884d84` — синхронизация меню команд с Telegram (те же
  команды, тот же лимит из конфига `command_menu`, а не хардкод 32).
- `e42787f68a` — транскрипция входящих голосовых (MAX не различает voice/audio,
  переиспользован STT-пайплайн Telegram).
- `145e2c9cae` — индикатор «печатает» (`typing_on`), как в Telegram.

## 3. Bot Mode / мультиагентная оркестрация (tools/bot_mode_dm.py и др.)
- `918ae61d82` — `message_agent` доступен оркестратору (`default`-профиль) из
  ЛЮБОЙ его сессии, не только из канонической «Bot Chat»; плюс запрет
  открывать «Bot Chat» вторым CLI-писателем, когда гейтвей профиля жив
  (`--force-writer` как форточка для восстановления). Раньше координатор был
  вынужден шеллаутить `hermes chat -c "Bot Chat"`, что создавало конфликт
  писателей и доставки зависали.
- `cca41522bc` — ответы `message_agent` в ГРУППОВЫХ Telegram-сессиях реально
  доходили адресату (раньше терялись: `notify_on_complete: True` — только
  обещание push, а страховочная запись в транскрипт вооружалась лишь для DM).
- `ca11f96813` — фикс обрезания хвоста уведомления о завершении: однострочный
  JSON-ответ (как у `message_agent`) мог полностью пропасть при обрезке по
  границе строки.

## 4. Поиск: нативный провайдер Yandex (plugins/web/yandex/)
- `1ea0d3ea4a` — замена самодельного SearXNG-совместимого моста на нормальный
  плагин через `yandex-ai-studio-sdk` (нужны `YC_API_KEY` + `YC_FOLDER_ID`).
  Только поиск (без extract — Yandex Search API отдаёт сниппеты, не полный
  текст страницы).

## 5. Прочие фиксы и мелкие фичи
- `b2a15ca0a1` — фикс гейтвея: `profile_mismatch` на приёме сообщений от
  Telegram-пользователей — `user_id` не передавался при повторном матчинге
  профильного маршрута при допуске сессии (расходилось со стампингом на входе).
- `bdb66fb963` — новая команда `hermes profile update-clients`: батч-обновление
  всех профилей с именем `client*`/`device*` (пропускает без
  `distribution.yaml`, ошибка одного профиля не блокирует остальные,
  `--force-config`, `-y`).
- `088504f93b` + серия `i18n(ru)` — ручная коррекция русских описаний команд
  в `locales/ru.yaml` (`/status`, `/usage`, `/sessions`, `/retry`, `/stop`,
  `/undo`, `/resume`, `/title`, `/new`).

## 6. LOCAL PATCH — внутренние, не для апстрима
Помечены в истории как `LOCAL PATCH:` — персональные патчи одной инсталляции,
не задуманы как contribution. Подробные инструкции восстановления после
`hermes update`/merge апстрима — в `~/.hermes/hermes-agent-local-patches.md`.

- `4e229ff3cb` — необязательное поле `change_summary` (≤72 симв.) в схемах
  `write_file` / `patch` / `skill_manage`. Только схема, handler'ы не менялись
  (они и так читают поля через `.get()`, игнорируя неизвестные ключи). Поле
  долетает до `post_tool_call`-хука и используется плагином
  `memory-git-autocommit` как заголовок git-коммита в `~/.hermes/memory`.
- `c467d4b71e` + `e06952c5f7` — плагин `memory-git-autocommit` перенесён
  из `~/.hermes/plugins/` (родительский репозиторий) внутрь форка
  (`plugins/memory-git-autocommit/`), чтобы схемный патч и сам плагин жили в
  одной истории коммитов. Принятый риск: `plugins/AGENTS.md` резервирует
  `plugins/` под официальные типы — это отступление ради одной инсталляции.

## 7. Слияния с апстримом
- `8219774d24` — merge upstream/main в форк (после первой партии коннекторов).
- `097b80fa4f` — merge upstream/main (конфликты разрешены в
  `gateway/session_ingress_context.py`, `locales/ru.yaml`).
- `81c105c6cd` — merge ветки `patch/feature/profile-update-clients`.

---

### Как актуализировать этот файл
```
cd ~/.hermes/hermes-agent
git fetch upstream        # сеть к GitHub нужна; через SSH туннель/VPN при необходимости
git log upstream/main..main --oneline --reverse
```
Новые коммиты сверху списка — дописать сюда разделом, в хронологии.
