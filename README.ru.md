<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/">Hermes Agent</a> | <a href="https://hermes-agent.nousresearch.com/">Hermes Desktop</a>
</p>
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Документация"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-blue?style=for-the-badge" alt="English"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Español-orange?style=for-the-badge" alt="Español"></a>
</p>

**Самообучающийся AI-агент от [Nous Research](https://nousresearch.com).** Это единственный агент со встроенным циклом обучения: он создаёт навыки из опыта, улучшает их в процессе работы, напоминает себе сохранять знания, ищет по своим прошлым разговорам и постепенно строит более глубокую модель того, кто вы — между сессиями. Запускайте на VPS за $5, на GPU-кластере или на serverless-инфраструктуре, которая почти ничего не стоит в простое. Он не привязан к ноутбуку — пишите ему в Telegram, пока он работает на облачной VM.

Используйте любую модель — [Nous Portal](https://portal.nousresearch.com), OpenRouter, OpenAI, свой endpoint и [многие другие](https://hermes-agent.nousresearch.com/docs/integrations/providers). Переключайтесь через `hermes model` — без правок кода и без vendor lock-in.

<table>
<tr><td><b>Настоящий терминальный интерфейс</b></td><td>Полноценный TUI: многострочное редактирование, автодополнение slash-команд, история диалогов, прерывание и перенаправление, потоковый вывод инструментов.</td></tr>
<tr><td><b>Живёт там, где вы</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal и CLI — всё из одного gateway-процесса. Транскрипция голосовых, непрерывность разговоров между платформами.</td></tr>
<tr><td><b>Замкнутый цикл обучения</b></td><td>Память, которую курирует сам агент, с периодическими «подталкиваниями». Автономное создание skills после сложных задач. Skills самоулучшаются в работе. Поиск сессий FTS5 + LLM-суммаризация для recall между сессиями. Диалектическое моделирование пользователя <a href="https://github.com/plastic-labs/honcho">Honcho</a>. Совместимость с открытым стандартом <a href="https://agentskills.io">agentskills.io</a>.</td></tr>
<tr><td><b>Плановые автоматизации</b></td><td>Встроенный cron с доставкой на любую платформу. Ежедневные отчёты, ночные бэкапы, еженедельные аудиты — на естественном языке, без присмотра.</td></tr>
<tr><td><b>Делегирует и параллелит</b></td><td>Изолированные субагенты для параллельных потоков работы. Python-скрипты, вызывающие tools через RPC, сжимают многошаговые пайплайны в ходы с нулевой стоимостью контекста.</td></tr>
<tr><td><b>Работает где угодно, не только на ноутбуке</b></td><td>Шесть terminal backends — local, Docker, SSH, Singularity, Modal и Daytona. Daytona и Modal дают serverless-персистентность: окружение агента «засыпает» в простое и просыпается по запросу, почти без стоимости между сессиями. VPS за $5 или GPU-кластер.</td></tr>
<tr><td><b>Готов к research</b></td><td>Пакетная генерация trajectories, сжатие trajectories для обучения следующего поколения tool-calling моделей.</td></tr>
</table>

---

## Быстрая установка

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows (нативно, PowerShell)

> **Важно:** На native Windows Hermes работает **без WSL** — CLI, gateway, TUI и tools. Если удобнее WSL2, используйте one-liner Linux/macOS выше. Нашли баг? Пожалуйста, [откройте issue](https://github.com/NousResearch/hermes-agent/issues).

В PowerShell:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

Установщик делает всё сам: uv, Python 3.11, Node.js, ripgrep, ffmpeg и **портативный Git Bash** (MinGit в `%LOCALAPPDATA%\hermes\git` — без admin, изолирован от системного Git). Hermes использует этот Git Bash для shell-команд.

Если Git уже установлен, installer его подхватит. Иначе скачается ~45MB MinGit — без вмешательства в системный Git.

> **Android / Termux:** Проверенный ручной путь — в [гайде Termux](https://hermes-agent.nousresearch.com/docs/getting-started/termux). На Termux ставится curated-extra `.[termux]`, потому что полный `.[all]` тянет voice-зависимости, несовместимые с Android.
>
> **Windows:** Native Windows полностью поддерживается — PowerShell one-liner выше ставит всё. WSL2 — тот же Linux-путь. Native install: `%LOCALAPPDATA%\hermes`; WSL2/Linux: `~/.hermes`.

После установки:

```bash
source ~/.bashrc    # перезагрузить shell (или: source ~/.zshrc)
hermes              # начать чат!
```

### Устранение неполадок

#### Windows Defender / антивирус помечает `uv.exe` как malware

Если антивирус (Bitdefender, Windows Defender и т.п.) карантинит `uv.exe` из папки Hermes `bin` (`%LOCALAPPDATA%\hermes\bin\uv.exe`) — это **ложное срабатывание**. Файл — `uv` от Astral: Rust-менеджер Python-пакетов, который Hermes бандлит. ML-движки часто флагуют unsigned Rust-бинари, качающие пакеты.

**Проверить подлинность:**

```powershell
# При необходимости: winget install --id GitHub.cli
# gh auth login

$uv = "$env:LOCALAPPDATA\hermes\bin\uv.exe"
$ver = (& $uv --version).Split(' ')[1]
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$zip = "$env:TEMP\uv.zip"
Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$ver/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
gh attestation verify $zip --repo astral-sh/uv
Expand-Archive $zip "$env:TEMP\uv_x" -Force
(Get-FileHash "$env:TEMP\uv_x\uv.exe").Hash -eq (Get-FileHash $uv).Hash
```

Если attestation: «Verification succeeded», а последняя строка — `True`, всё в порядке.

**Whitelist Hermes:**
- **Windows Defender:** PowerShell as Admin → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
- **Bitdefender:** исключение в консоли (Protection > Antivirus > Settings > Manage Exceptions)
- Whitelist **папки**, не hash файла — Hermes обновляет `uv`, hash меняется

Контекст upstream: [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553), [astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011), [astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079).

---

## С чего начать

```bash
hermes              # Интерактивный CLI — начать разговор
hermes model        # Выбрать LLM-провайдера и модель
hermes tools        # Настроить, какие tools включены
hermes config set   # Задать отдельные значения config
hermes config get   # Прочитать отдельные значения config
hermes gateway      # Запустить messaging gateway (Telegram, Discord, …)
hermes setup        # Полный setup-wizard (всё сразу)
hermes claw migrate # Миграция с OpenClaw
hermes update       # Обновить до последней версии
hermes doctor       # Диагностика проблем
```

📖 **[Полная документация →](https://hermes-agent.nousresearch.com/docs/)**

---

## Без коллекции API-ключей — Nous Portal

Hermes работает с любым провайдером — это не меняется. Но если не хочется собирать пять отдельных ключей (модель, web search, image gen, TTS, cloud browser), **[Nous Portal](https://portal.nousresearch.com)** закрывает всё одной подпиской:

- **300+ моделей** — `/model <name>`
- **Tool Gateway** — web search (Firecrawl), image generation (FAL), TTS (OpenAI), cloud browser (Browser Use) — всё через вашу подписку, без лишних аккаунтов

Одна команда после чистой установки:

```bash
hermes setup --portal
```

OAuth-логин, Nous как провайдер, Tool Gateway включён. Состояние: `hermes portal info`. Подробности: [Tool Gateway docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway).

Свои ключи per-tool по-прежнему можно — gateway per-backend, не all-or-nothing.

---

## CLI vs Messaging — краткая шпаргалка

Две точки входа: TUI через `hermes`, или gateway + Telegram / Discord / Slack / WhatsApp / Signal / Email. Многие slash-команды общие.

| Действие | CLI | Messaging |
| -------- | --- | --------- |
| Начать чат | `hermes` | `hermes gateway setup` + `hermes gateway start`, затем сообщение боту |
| Новый разговор | `/new` или `/reset` | `/new` или `/reset` |
| Сменить модель | `/model [provider:model]` | `/model [provider:model]` |
| Личность | `/personality [name]` | `/personality [name]` |
| Retry / undo | `/retry`, `/undo` | `/retry`, `/undo` |
| Сжать контекст / usage | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]` |
| Skills | `/skills` или `/<skill-name>` | `/<skill-name>` |
| Прервать работу | `Ctrl+C` или новое сообщение | `/stop` или новое сообщение |
| Статус платформы | `/platforms` | `/status`, `/sethome` |

Полные списки: [CLI guide](https://hermes-agent.nousresearch.com/docs/user-guide/cli) и [Messaging Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## Документация

Вся документация: **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**:

| Раздел | О чём |
| ------ | ----- |
| [Quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | Install → setup → первый разговор за 2 минуты |
| [CLI Usage](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | Команды, keybindings, personalities, sessions |
| [Configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | Config, провайдеры, модели, все опции |
| [Messaging Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant |
| [Security](https://hermes-agent.nousresearch.com/docs/user-guide/security) | Approval команд, DM pairing, container isolation |
| [Tools & Toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | 40+ tools, toolsets, terminal backends |
| [Skills System](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | Procedural memory, Skills Hub, создание skills |
| [Memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | Persistent memory, user profiles, best practices |
| [MCP Integration](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | Любой MCP server для расширенных возможностей |
| [Cron Scheduling](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | Плановые задачи с доставкой на платформы |
| [Context Files](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | Контекст проекта для каждого разговора |
| [Architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | Структура, agent loop, ключевые классы |
| [Contributing](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) | Dev setup, PR process, code style |
| [CLI Reference](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) | Все команды и флаги |
| [Environment Variables](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | Полный справочник env vars |

---

## Миграция с OpenClaw

Если вы пришли из OpenClaw, Hermes может автоматически импортировать настройки, memories, skills и API keys.

**При первой настройке:** wizard (`hermes setup`) находит `~/.openclaw` и предлагает миграцию до конфигурации.

**В любой момент после install:**

```bash
hermes claw migrate              # Интерактивная миграция (full preset)
hermes claw migrate --dry-run    # Превью без изменений
hermes claw migrate --preset user-data   # Без секретов
hermes claw migrate --overwrite  # Перезаписать конфликты
```

Что импортируется:

- **SOUL.md** — persona
- **Memories** — MEMORY.md и USER.md
- **Skills** — user skills → `~/.hermes/skills/openclaw-imports/`
- **Command allowlist** — паттерны approval
- **Messaging settings** — платформы, allowed users, working directory
- **API keys** — allowlisted secrets (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **TTS assets** — audio workspace
- **Workspace instructions** — AGENTS.md (с `--workspace-target`)

См. `hermes claw migrate --help` или skill `openclaw-migration` для guided-миграции с dry-run.

---

## Участие (Contributing)

Приветствуем вклад! См. [Contributing Guide](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing): dev setup, code style, PR process.

Быстрый старт для контрибьюторов — стандартный installer, затем работа в полном git checkout `$HERMES_HOME/hermes-agent` (обычно `~/.hermes/hermes-agent`). Так совпадает layout с `hermes update`, managed venv, gateway и docs tooling.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

Fallback: ручной clone (throwaway / CI без managed layout).

venv **вне** дерева исходников — venv внутри workspace агент может снести relative-path командой:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.hermes/venvs/hermes-dev --python 3.11
source ~/.hermes/venvs/hermes-dev/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

---

## Сообщество

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — Linux desktop-control MCP server для Hermes и других MCP hosts: AT-SPI, Wayland/X11 input, screenshots, compositor window targeting.
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — community WeChat bridge: Hermes Agent и OpenClaw на одном WeChat-аккаунте.

---

## Лицензия

MIT — см. [LICENSE](LICENSE).

Сделано [Nous Research](https://nousresearch.com).
