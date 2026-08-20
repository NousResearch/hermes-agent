<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/">Hermes Agent</a> | <a href="https://hermes-agent.nousresearch.com/">Hermes Desktop</a>
</p>
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-lightgrey?style=for-the-badge" alt="English"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-简体中文-red?style=for-the-badge" alt="简体中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Español-orange?style=for-the-badge" alt="Español"></a>
</p>

**由 [Nous Research](https://nousresearch.com) 打造的自我改進 AI 代理。** 它是唯一內建學習迴圈的代理——從經驗中建立技能、在使用過程中改進技能、主動提醒自己保存知識、搜尋自己過去的對話，並跨工作階段逐步加深對你的了解。可以跑在 $5 的 VPS、GPU 叢集，或閒置時幾乎零成本的無伺服器（serverless）基礎設施上。它不綁定你的筆電——它在雲端 VM 上工作時，你可以從 Telegram 跟它對話。

想用什麼模型都可以——[Nous Portal](https://portal.nousresearch.com)、OpenRouter、OpenAI、你自己的端點，以及[其他多種選擇](https://hermes-agent.nousresearch.com/docs/integrations/providers)。用 `hermes model` 即可切換——不必改程式碼、沒有鎖定。

<table>
<tr><td><b>真正的終端機介面</b></td><td>完整的 TUI，支援多行編輯、斜線指令自動補全、對話歷史、中斷並重新導向，以及串流工具輸出。</td></tr>
<tr><td><b>在你常用的地方待命</b></td><td>Telegram、Discord、Slack、WhatsApp、Signal 與 CLI——全部由單一閘道行程提供。支援語音訊息轉逐字稿、跨平台對話接續。</td></tr>
<tr><td><b>閉環學習迴圈</b></td><td>由代理自行管理的記憶，並定期自我提醒。完成複雜任務後自主建立技能，技能在使用過程中自我改進。FTS5 工作階段搜尋搭配 LLM 摘要，實現跨工作階段回想。<a href="https://github.com/plastic-labs/honcho">Honcho</a> 辯證式使用者建模。相容 <a href="https://agentskills.io">agentskills.io</a> 開放標準。</td></tr>
<tr><td><b>排程自動化</b></td><td>內建 cron 排程器，可將結果投遞到任何平台。每日報告、夜間備份、每週稽核——全部用自然語言描述，無人值守執行。</td></tr>
<tr><td><b>委派與平行處理</b></td><td>產生隔離的子代理處理平行工作流。撰寫 Python 指令碼透過 RPC 呼叫工具，把多步驟管線壓縮成零上下文成本的回合。</td></tr>
<tr><td><b>隨處可跑，不只在你的筆電</b></td><td>六種終端機後端——本機、Docker、SSH、Singularity、Modal 與 Daytona。Daytona 與 Modal 提供無伺服器持久化——代理的環境閒置時休眠、需要時喚醒，工作階段之間幾乎零成本。$5 VPS 或 GPU 叢集都能跑。</td></tr>
<tr><td><b>研究就緒</b></td><td>批次軌跡（trajectory）生成、軌跡壓縮——用於訓練下一代工具呼叫模型。</td></tr>
</table>

---

## 快速安裝

### Linux、macOS、WSL2、Termux

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows（原生，PowerShell）

> **注意：** 原生 Windows 不需要 WSL 即可執行 Hermes——CLI、閘道、TUI 和工具全部原生運作。若你偏好使用 WSL2，上面的 Linux/macOS 一行指令在 WSL2 裡同樣適用。發現 bug 了嗎？請[回報 issue](https://github.com/NousResearch/hermes-agent/issues)。

在 PowerShell 中執行：

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

安裝程式會處理好一切：uv、Python 3.11、Node.js、ripgrep、ffmpeg，**以及一份可攜式 Git Bash**（MinGit，解壓至 `%LOCALAPPDATA%\hermes\git`——不需要系統管理員權限，與系統上任何既有的 Git 安裝完全隔離）。Hermes 會用這份隨附的 Git Bash 執行 shell 指令。

如果你已經安裝了 Git，安裝程式會偵測到並直接使用它。否則只需下載約 45MB 的 MinGit——它不會碰到或干擾系統的 Git。

> **Android / Termux：** 經過測試的手動安裝步驟請見 [Termux 指南](https://hermes-agent.nousresearch.com/docs/getting-started/termux)。在 Termux 上，Hermes 會安裝精選的 `.[termux]` extra，因為完整的 `.[all]` extra 目前會拉入與 Android 不相容的語音相依套件。
>
> **Windows：** 原生 Windows 已完整支援——上面的 PowerShell 一行指令會安裝所有東西。若你偏好 WSL2，Linux 指令在那裡也適用。原生 Windows 安裝位於 `%LOCALAPPDATA%\hermes`；WSL2 則和 Linux 一樣安裝在 `~/.hermes`。

安裝完成後：

```bash
source ~/.bashrc    # 重新載入 shell（或：source ~/.zshrc）
hermes              # 開始聊天！
```

### 疑難排解

#### Windows Defender 或防毒軟體將 `uv.exe` 標記為惡意程式

如果你的防毒軟體（Bitdefender、Windows Defender 等）隔離了 Hermes `bin` 資料夾中的 `uv.exe`（`%LOCALAPPDATA%\hermes\bin\uv.exe`），這是**誤判**。該檔案是 Astral 的 `uv`——Hermes 隨附用來管理 Python 環境的 Rust 版 Python 套件管理器。以機器學習為基礎的防毒引擎經常誤判會下載並安裝套件的未簽章 Rust 執行檔。

**驗證你的檔案是否為正版：**

```powershell
# 如有需要，先安裝 GitHub CLI
winget install --id GitHub.cli

# 登入 GitHub
gh auth login

# 執行驗證
$uv = "$env:LOCALAPPDATA\hermes\bin\uv.exe"
$ver = (& $uv --version).Split(' ')[1]
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$zip = "$env:TEMP\uv.zip"
Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$ver/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
gh attestation verify $zip --repo astral-sh/uv
Expand-Archive $zip "$env:TEMP\uv_x" -Force
(Get-FileHash "$env:TEMP\uv_x\uv.exe").Hash -eq (Get-FileHash $uv).Hash
```

如果 attestation 顯示「Verification succeeded」且最後一行印出 `True`，就沒問題。

**將 Hermes 加入白名單：**
- **Windows Defender：** 以系統管理員身分執行 PowerShell → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
- **Bitdefender：** 在 Bitdefender 主控台新增例外（防護 > 防毒 > 設定 > 管理例外）
- 白名單請加**資料夾**，不要加檔案雜湊——Hermes 會更新 `uv`，每個版本的雜湊值都不同

更多背景請見 Astral 上游的回報：[astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553)、[astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011)、[astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079)。

---

## 開始使用

```bash
hermes              # 互動式 CLI——開始對話
hermes model        # 選擇你的 LLM 供應商與模型
hermes tools        # 設定要啟用哪些工具
hermes config set   # 設定個別設定值
hermes gateway      # 啟動訊息閘道（Telegram、Discord 等）
hermes setup        # 執行完整設定精靈（一次設定所有項目）
hermes claw migrate # 從 OpenClaw 遷移（如果你來自 OpenClaw）
hermes update       # 更新到最新版本
hermes doctor       # 診斷問題
```

📖 **[完整文件 →](https://hermes-agent.nousresearch.com/docs/)**

---

## 免去湊齊 API 金鑰的麻煩——Nous Portal

Hermes 可以搭配任何你想用的供應商——這點不會改變。但如果你不想為模型、網頁搜尋、圖片生成、TTS 和雲端瀏覽器分別申請五把 API 金鑰，**[Nous Portal](https://portal.nousresearch.com)** 用一份訂閱涵蓋全部：

- **300+ 模型**——用 `/model <名稱>` 任選
- **Tool Gateway**——網頁搜尋（Firecrawl）、圖片生成（FAL）、文字轉語音（OpenAI）、雲端瀏覽器（Browser Use），全部透過你的訂閱路由。不需要額外帳號。

全新安裝只要一個指令：

```bash
hermes setup --portal
```

它會透過 OAuth 登入、將 Nous 設為你的供應商，並開啟 Tool Gateway。隨時可以用 `hermes portal info` 檢查目前接了哪些服務。完整細節請見 [Tool Gateway 文件頁面](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway)。

你隨時可以針對個別工具改用自己的金鑰——閘道是逐後端設定的，不是全有或全無。

---

## CLI 與訊息平台快速對照

Hermes 有兩個入口：用 `hermes` 啟動終端機 UI，或執行閘道後從 Telegram、Discord、Slack、WhatsApp、Signal 或 Email 與它對話。進入對話後，許多斜線指令在兩種介面間通用。

| 動作                     | CLI                                           | 訊息平台                                                                     |
| ------------------------ | --------------------------------------------- | ---------------------------------------------------------------------------- |
| 開始聊天                 | `hermes`                                      | 執行 `hermes gateway setup` + `hermes gateway start`，然後傳訊息給機器人     |
| 開始新對話               | `/new` 或 `/reset`                            | `/new` 或 `/reset`                                                           |
| 更換模型                 | `/model [provider:model]`                     | `/model [provider:model]`                                                    |
| 設定 personality         | `/personality [名稱]`                         | `/personality [名稱]`                                                        |
| 重試或復原上一回合       | `/retry`、`/undo`                             | `/retry`、`/undo`                                                            |
| 壓縮上下文／檢查用量     | `/compress`、`/usage`、`/insights [--days N]` | `/compress`、`/usage`、`/insights [天數]`                                    |
| 瀏覽技能                 | `/skills` 或 `/<技能名稱>`                    | `/<技能名稱>`                                                                |
| 中斷目前工作             | `Ctrl+C` 或直接傳新訊息                       | `/stop` 或直接傳新訊息                                                       |
| 平台專屬狀態             | `/platforms`                                  | `/status`、`/sethome`                                                        |

完整指令清單請見 [CLI 指南](https://hermes-agent.nousresearch.com/docs/user-guide/cli)與[訊息閘道指南](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)。

---

## 文件

所有文件都在 **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**：

| 章節                                                                                                 | 涵蓋內容                                                   |
| --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [快速上手](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)                    | 安裝 → 設定 → 2 分鐘內開始第一次對話                       |
| [CLI 使用](https://hermes-agent.nousresearch.com/docs/user-guide/cli)                                | 指令、快捷鍵、personality、工作階段                        |
| [組態設定](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)                      | 設定檔、供應商、模型、所有選項                             |
| [訊息閘道](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)                          | Telegram、Discord、Slack、WhatsApp、Signal、Home Assistant |
| [安全性](https://hermes-agent.nousresearch.com/docs/user-guide/security)                             | 指令核准、DM 配對、容器隔離                                |
| [工具與工具集](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools)                 | 40+ 工具、工具集系統、終端機後端                           |
| [技能系統](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)                    | 程序性記憶、Skills Hub、建立技能                           |
| [記憶](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)                        | 持久記憶、使用者檔案、最佳實務                             |
| [MCP 整合](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)                       | 連接任何 MCP 伺服器以擴充能力                              |
| [Cron 排程](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)                     | 排程任務並投遞到各平台                                     |
| [情境檔案](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files)             | 影響每一次對話的專案情境                                   |
| [架構](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)                      | 專案結構、代理迴圈、關鍵類別                               |
| [貢獻指南](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)                  | 開發環境設定、PR 流程、程式碼風格                          |
| [CLI 參考](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)                        | 所有指令與旗標                                             |
| [環境變數](https://hermes-agent.nousresearch.com/docs/reference/environment-variables)               | 完整環境變數參考                                           |

---

## 從 OpenClaw 遷移

如果你來自 OpenClaw，Hermes 可以自動匯入你的設定、記憶、技能與 API 金鑰。

**首次設定時：** 設定精靈（`hermes setup`）會自動偵測 `~/.openclaw`，並在開始設定前詢問是否要遷移。

**安裝後隨時可以執行：**

```bash
hermes claw migrate              # 互動式遷移（完整預設集）
hermes claw migrate --dry-run    # 預覽將被遷移的內容
hermes claw migrate --preset user-data   # 遷移但不含密鑰
hermes claw migrate --overwrite  # 覆寫既有的衝突項目
```

會匯入的內容：

- **SOUL.md**——角色（persona）檔案
- **記憶**——MEMORY.md 與 USER.md 條目
- **技能**——使用者自建技能 → `~/.hermes/skills/openclaw-imports/`
- **指令允許清單**——核准模式
- **訊息設定**——平台組態、允許的使用者、工作目錄
- **API 金鑰**——允許清單內的密鑰（Telegram、OpenRouter、OpenAI、Anthropic、ElevenLabs）
- **TTS 資產**——工作區音訊檔
- **工作區指示**——AGENTS.md（搭配 `--workspace-target`）

所有選項請見 `hermes claw migrate --help`，或使用 `openclaw-migration` 技能進行由代理引導、含 dry-run 預覽的互動式遷移。

---

## 貢獻

歡迎貢獻！開發環境設定、程式碼風格與 PR 流程請見[貢獻指南](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)。

貢獻者快速上手——使用標準安裝程式，然後在它建立的完整 git checkout 中工作，位置在 `$HERMES_HOME/hermes-agent`（通常是 `~/.hermes/hermes-agent`）。這與 `hermes update`、受管 venv、延遲載入相依套件、閘道及文件工具所使用的目錄配置一致。

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

手動 clone 的備援做法（適用於刻意不想採用受管安裝配置的一次性 clone／CI）：

請把 venv 建在 clone 下來的原始碼樹之外——如果 venv 位於代理工作的目錄裡，代理對自己 checkout 執行的相對路徑指令可能會把它清掉，導致執行中的 runtime 在工作階段進行到一半時被摧毀。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.hermes/venvs/hermes-dev --python 3.11
source ~/.hermes/venvs/hermes-dev/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

---

## 社群

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux)——供 Hermes 及其他 MCP 主機使用的 Linux 桌面控制 MCP 伺服器，支援 AT-SPI 無障礙樹、Wayland/X11 輸入、螢幕截圖與合成器視窗鎖定。
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw)——社群 WeChat 橋接：在同一個 WeChat 帳號上同時執行 Hermes Agent 與 OpenClaw。

---

## 授權

MIT——請見 [LICENSE](LICENSE)。

由 [Nous Research](https://nousresearch.com) 打造。
