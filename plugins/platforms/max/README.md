<div align="center">

<br/>

<img src="assets/github-logo.jpg" alt="MAX Hermes Plugin" width="720">

### MAX — российский мессенджер как канал связи с Hermes Agent.

**<span style="color:#f59e0b">Пиши боту в MAX — отвечает твой Hermes. С памятью, инструментами и без белого IP.</span>**

<br/>

[![Version](https://img.shields.io/badge/version-0.20.0-blue.svg)](https://github.com/FraN-arti/max-hermes-plugin)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Hermes](https://img.shields.io/badge/Hermes-0.20+-7C3AED.svg)](https://hermes-agent.nousresearch.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![MAX](https://img.shields.io/badge/MAX-API-FF6B00.svg)](https://dev.max.ru)

[Русский](README.md) · [English](README.en.md)

</div>

---

<br/>

## Что это

**MAX** (max.ru) — российский мессенджер от VK. Этот плагин подключает его к
[Hermes Agent](https://hermes-agent.nousresearch.com) как полноценный канал связи —
ровно как Telegram, WhatsApp или Slack.

Ты пишешь боту в MAX → отвечает **твой** Hermes: с общей памятью, инструментами,
cron-задачами и всем, что он умеет. Это не «болванка-эхо», а тот же агент,
который живёт у тебя на компе.

<br/>

## Почему MAX

- 🇷🇺 **Российский мессенджер** — работает без VPN, не блокируется, данные в РФ
- 📡 **Long Polling** — не нужен белый IP, домен, проброс портов или HTTPS-сертификат.
  Работает из-за NAT (Ростелеком и другие провайдеры — не проблема)
- 🔐 **Allowlist** — только допущенные тобой люди могут писать боту
- ⏰ **Cron-доставка** — уведомления и напоминания от Hermes приходят в MAX
- 🪄 **Zero-config TLS** — сертификат Минцифры скачивается сам при первом запуске

<br/>

## Быстрый старт (5 минут)

### 1. Создай бота в MAX

Нужен верифицированный профиль на платформе **MAX для партнёров**
(юрлицо / ИП / самозанятый): [подключение](https://dev.max.ru/docs/maxbusiness/connection)

1. Создай бота: [business.max.ru](https://business.max.ru/self) → **Чат-боты** → создать
2. Дождись модерации
3. Забери токен: **Чат-боты** → **Расширенные настройки** → **Настроить**

### 2. Установи плагин

```bash
# Скопируй папку плагина в Hermes
# Linux/macOS:
cp -r plugins/platforms/max ~/.hermes/plugins/platforms/
# Windows:
#   положи plugins/platforms/max в %LOCALAPPDATA%\hermes\plugins\platforms\

# Включи плагин
hermes plugins enable max
```

### 3. Настрой

```bash
# Через мастер (рекомендуется):
hermes setup gateway        # → выбери MAX → вставь токен

# Или вручную — добавь в .env:
MAX_BOT_TOKEN=токен_бота
MAX_ALLOWED_USERS=твой_user_id
```

> **Как узнать свой user_id?** Напиши боту любое сообщение и посмотри в логе
> gateway: `Message from <имя>` — рядом будет user_id. Или временно поставь
> `MAX_ALLOW_ALL_USERS=true`, узнай ID, потом убери.

### 4. Запусти

```bash
hermes gateway restart
```

Готово! Напиши боту в MAX — он ответит. 🎉

<br/>

## Как это работает

```
 Ты (MAX)  →  MAX API (platform-api2.max.ru)  ←─ Long Polling ←─  Hermes Gateway
                                                                      ↓
 Ты (MAX)  ←  POST /messages  ←──────────────  Hermes Gateway (ответ)
```

- **Приём:** Long Polling `GET /updates` с marker-курсором. Комп сам опрашивает
  MAX-серверы — внешний доступ к нему не нужен.
- **Маркер персистится** в `max/marker.json` — после перезапуска gateway старые
  сообщения не обрабатываются повторно.
- **Отправка:** `POST /messages` c `user_id` (личка) или `chat_id` (группы/каналы).
  Rate limit MAX (2 сообщ./сек на чат) соблюдается автоматически.
- **«Печатает…»:** индикатор набора через `POST /chats/{id}/actions` (`typing_on`).
- **TLS:** MAX использует Russian Trusted Root CA (Минцифры). Плагин сам скачивает
  его с официального источника [gu-st.ru](https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt)
  при первом запуске (с таймаутом 10с и проверкой PEM), либо берёт путь из `MAX_CA_CERT_PATH`.

<br/>

## Переменные окружения

| Переменная | Обязательна | Описание |
|---|---|---|
| `MAX_BOT_TOKEN` | ✅ | Токен бота из business.max.ru |
| `MAX_ALLOWED_USERS` | ❌ | Разрешённые user_id (через запятую) |
| `MAX_ALLOW_ALL_USERS` | ❌ | `true` = пускать всех (только для dev!) |
| `MAX_HOME_CHANNEL` | ❌ | chat_id для cron-доставки |
| `MAX_HOME_USER_ID` | ❌ | user_id для cron-доставки в личку |
| `MAX_CA_CERT_PATH` | ❌ | Свой путь к сертификату Минцифры |

<br/>

## Структура

```
plugins/platforms/max/
├── __init__.py      # регистрация плагина
├── adapter.py       # Long Polling + отправка + авто-сертификат + typing
├── plugin.yaml      # метаданные для hermes setup gateway
├── README.md        # этот файл (RU)
└── README.en.md     # English version
```

<br/>

## Ограничения

- MAX рекомендует **webhook** для production, но он требует HTTPS + публичный URL.
  Long Polling работает везде (даже за NAT) — для личного использования идеально.
- **Вложения** (картинки, файлы) пока не отправляются — только текст. В планах.
- Сообщения > 4000 символов **разбиваются на несколько** (умный сплит по границам строк/слов).

<br/>

## Лицензия

[MIT](LICENSE)

---

<div align="center">

Сделано для русского комьюнити Hermes 🇷🇺 · [MAX для разработчиков](https://dev.max.ru)

</div>
