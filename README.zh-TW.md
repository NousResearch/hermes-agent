<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="說明文件"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-lightgrey?style=for-the-badge" alt="English"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
</p>

**由 [Nous Research](https://nousresearch.com) 打造的自進化 AI Agent。** 它是唯一內建學習閉環的智慧 Agent——從經驗中建立技能，在使用中改進技能，主動持久化知識，搜尋過往對話，並在跨會話中逐步建立對你的深度理解。可以在 $5 的 VPS 上執行，也可以在 GPU 叢集上執行，或者使用幾乎零成本的 Serverless 基礎設施。它不綁定你的筆電——你可以透過 Telegram 與它對話，而它在雲端 VM 上運作。

支援任意模型——[Nous Portal](https://portal.nousresearch.com)、[OpenRouter](https://openrouter.ai)（200+ 款模型）、[NVIDIA NIM](https://build.nvidia.com)（Nemotron）、[小米 MiMo](https://platform.xiaomimimo.com)、[z.ai/GLM](https://z.ai)、[Kimi/Moonshot](https://platform.moonshot.ai)、[MiniMax](https://www.minimax.io)、[Hugging Face](https://huggingface.co)、OpenAI，或自訂端點。使用 `hermes model` 即可切換——無需修改程式碼，無綁定鎖定。

<table>
<tr><td><b>真正的終端機介面</b></td><td>完整的 TUI，支援多行編輯、斜線指令自動補全、對話歷史記錄、中斷重定向和串流工具輸出。</td></tr>
<tr><td><b>隨處使用</b></td><td>Telegram、Discord、Slack、WhatsApp、Signal 和 CLI——全部從單一網關程序執行。語音備忘錄轉錄、跨平台對話連續性。</td></tr>
<tr><td><b>閉環學習</b></td><td>Agent 管理記憶並定期自我提醒。完成複雜任務後自動建立技能。技能在使用中自我改進。FTS5 會話搜尋配合 LLM 摘要實現跨會話回溯。<a href="https://github.com/plastic-labs/honcho">Honcho</a> 辯證式使用者建模。相容 <a href="https://agentskills.io">agentskills.io</a> 開放標準。</td></tr>
<tr><td><b>排程自動化</b></td><td>內建 cron 排程器，支援傳送至任何平台。每日報告、夜間備份、每週稽核——全部用自然語言描述，無人值守執行。</td></tr>
<tr><td><b>委派與平行處理</b></td><td>建立隔離的子 Agent 處理平行工作流程。撰寫 Python 腳本透過 RPC 呼叫工具，將多步驟管道壓縮為零上下文開銷的對話輪次。</td></tr>
<tr><td><b>隨處執行</b></td><td>六種終端機後端——本地、Docker、SSH、Daytona、Singularity 和 Modal。Daytona 和 Modal 提供 Serverless 持久化——Agent 環境閒置時休眠、按需喚醒，閒置期間幾乎零成本。$5 VPS 或 GPU 叢集都能執行。</td></tr>
<tr><td><b>研究就緒</b></td><td>批次軌跡產生、軌跡壓縮——用於訓練下一代工具呼叫模型。</td></tr>
</table>

---

## 快速安裝

```bash
curl -fsSL [https://hermes-agent.nousresearch.com/install.sh](https://hermes-agent.nousresearch.com/install.sh) | bash
```

支援 Linux、macOS、WSL2 和 Android (Termux)。安裝程式會自動處理平台特定的設定。

> **Android / Termux：** 已測試的手動安裝流程請參考 [Termux 指南](https://hermes-agent.nousresearch.com/docs/getting-started/termux)。在 Termux 上，Hermes 會安裝精選的 `.[termux]` 擴充套件，因為完整的 `.[all]` 擴充套件會拉取與 Android 不相容的語音相依套件。
>
> **Windows：** 在 PowerShell 中執行：
> ```powershell
> iex (irm [https://hermes-agent.nousresearch.com/install.ps1](https://hermes-agent.nousresearch.com/install.ps1))
> ```
> 安裝完成後，可能需要重新啟動終端機，然後執行 `hermes` 開始對話。

安裝後：

```bash
source ~/.bashrc    # 重新載入 shell（或: source ~/.zshrc）
hermes              # 開始對話！
```

---

## 快速入門

```bash
hermes              # 互動式 CLI — 開始對話
hermes model        # 選擇 LLM 提供業者與模型
hermes tools        # 設定已啟用的工具
hermes config set   # 設定單一設定項
hermes gateway      # 啟動訊息網關（Telegram、Discord 等）
hermes setup        # 執行完整設定精靈（一次性設定所有內容）
hermes claw migrate # 從 OpenClaw 遷移（若來自 OpenClaw）
hermes update       # 更新至最新版本
hermes doctor       # 診斷問題
```

📖 **[完整說明文件 →](https://hermes-agent.nousresearch.com/docs/)**

---

## 免去到處收集 API 金鑰 — Nous Portal

Hermes 始終允許你使用任意服務商，這點不會改變。但如果你不想為了模型、網頁搜尋、圖片生成、TTS、雲端瀏覽器而分別去申請五個不同的 API 金鑰，**[Nous Portal](https://portal.nousresearch.com)** 只要一個訂閱就能涵蓋全部：

- **300+ 款模型** — 用 `/model <name>` 隨時切換
- **Tool Gateway** — 網頁搜尋（Firecrawl）、圖片生成（FAL）、文字轉語音（OpenAI）、雲端瀏覽器（Browser Use），全部透過訂閱託管。無需額外註冊任何帳號。

全新安裝時只需一條指令：

```bash
hermes setup --portal
```

它會透過 OAuth 登入、把 Nous 設定為推論服務商，並啟用 Tool Gateway。隨時用 `hermes portal info` 查看路由狀態。完整說明請見 [Tool Gateway 說明文件](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway)。

你隨時可以依工具單獨切回自己的 API 金鑰 — Gateway 是按工具粒度生效的，不是一刀切。

---

## CLI 與訊息平台 快速對照

Hermes 有兩種入口：用 `hermes` 啟動終端機 UI，或執行網關透過 Telegram、Discord、Slack、WhatsApp、Signal 或 Email 與之對話。進入對話後，許多斜線指令在兩種介面中皆通用。

| 操作 | CLI | 訊息平台 |
|------|-----|----------|
| 開始對話 | `hermes` | 執行 `hermes gateway setup` + `hermes gateway start`，然後傳送訊息給機器人 |
| 開始新對話 | `/new` 或 `/reset` | `/new` 或 `/reset` |
| 更換模型 | `/model [provider:model]` | `/model [provider:model]` |
| 設定人格 | `/personality [name]` | `/personality [name]` |
| 重試或復原上一輪 | `/retry`、`/undo` | `/retry`、`/undo` |
| 壓縮上下文 / 查看用量 | `/compress`、`/usage`、`/insights [--days N]` | `/compress`、`/usage`、`/insights [days]` |
| 瀏覽技能 | `/skills` 或 `/<skill-name>` | `/skills` 或 `/<skill-name>` |
| 中斷當前工作 | `Ctrl+C` 或傳送新訊息 | `/stop` 或傳送新訊息 |
| 平台特定狀態 | `/platforms` | `/status`、`/sethome` |

完整指令列表請參閱 [CLI 指南](https://hermes-agent.nousresearch.com/docs/user-guide/cli) 和 [訊息網關指南](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)。

---

## 說明文件

所有說明文件位於 **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**：

| 章節 | 內容 |
|------|------|
| [快速開始](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | 安裝 → 設定 → 2 分鐘內開始首次對話 |
| [CLI 使用指南](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | 指令、快捷鍵、人格、會話 |
| [設定選項](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | 設定檔、提供商、模型、所有選項 |
| [訊息網關](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram、Discord、Slack、WhatsApp、Signal、Home Assistant |
| [安全性](https://hermes-agent.nousresearch.com/docs/user-guide/security) | 指令審核、DM 配對、容器隔離 |
| [工具與工具集](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | 40+ 種工具、工具集系統、終端機後端 |
| [技能系統](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | 程序記憶、技能中心、建立技能 |
| [記憶系統](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | 持久記憶、使用者輪廓、最佳實踐 |
| [MCP 整合](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | 連接任意 MCP 伺服器擴充能力 |
| [排程自動化](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | 排程任務與平台推送 |
| [上下文檔案](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | 影響每次對話的專案上下文 |
| [架構說明](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | 專案結構、Agent 迴圈、核心類別 |
| [貢獻指南](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) | 開發環境設定、PR 流程、程式碼風格 |
| [CLI 參考手冊](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) | 所有指令與旗標 |
| [環境變數](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | 完整環境變數參考 |

---

## 從 OpenClaw 遷移

如果你來自 OpenClaw，Hermes 可以自動匯入你的設定、記憶、技能與 API 金鑰。

**首次安裝時：** 安裝精靈（`hermes setup`）會自動偵測 `~/.openclaw` 并在設定開始前提供遷移選項。

**安裝後任意時間：**

```bash
hermes claw migrate              # 互動式遷移（完整預設）
hermes claw migrate --dry-run    # 預演將要遷移的內容
hermes claw migrate --preset user-data   # 僅遷移使用者資料，不含金鑰
hermes claw migrate --overwrite  # 覆蓋已有衝突
```

匯入內容：
- **SOUL.md** — 人格檔案
- **記憶** — MEMORY.md 和 USER.md 條目
- **技能** — 使用者建立的技能 → `~/.hermes/skills/openclaw-imports/`
- **指令白名單** — 審核模式
- **訊息設定** — 平台設定、允許的使用者、工作目錄
- **API 金鑰** — 白名單中的金鑰（Telegram、OpenRouter、OpenAI、Anthropic、ElevenLabs）
- **TTS 資產** — 工作區音訊檔案
- **工作區指令** — AGENTS.md（使用 `--workspace-target`）

使用 `hermes claw migrate --help` 查看所有選項，或使用 `openclaw-migration` 技能進行互動式 Agent 引導遷移（包含預演預覽）。

---

## 貢獻

歡迎貢獻！請參閱 [貢獻指南](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) 了解開發設定、程式碼風格與 PR 流程。

貢獻者快速入門——使用標準安裝程式，然後在其建立的完整 git checkout 中開發：
`$HERMES_HOME/hermes-agent`（通常是 `~/.hermes/hermes-agent`）。這會符合
`hermes update`、託管 venv、lazy dependencies、gateway 與 docs tooling 所使用的配置架構。

```bash
curl -fsSL [https://hermes-agent.nousresearch.com/install.sh](https://hermes-agent.nousresearch.com/install.sh) | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

手動複製備用路徑（用於一次性 clone / CI，或你明確不想使用 managed install layout 時）：

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
python -m pytest tests/ -q
```

---

## 社群

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [技能中心](https://agentskills.io)
- 🐛 [問題回報](https://github.com/NousResearch/hermes-agent/issues)
- 💡 [討論區](https://github.com/NousResearch/hermes-agent/discussions)
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — 社群微信橋接：在同一微信帳號上執行 Hermes Agent 和 OpenClaw。

---

## 授權條款

MIT — 詳見 [LICENSE](LICENSE)。

由 [Nous Research](https://nousresearch.com) 打造。 