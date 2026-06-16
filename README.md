<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent — zapabob Windows / Operations Fork

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://github.com/NousResearch/hermes-agent"><img src="https://img.shields.io/badge/Upstream-NousResearch-blueviolet?style=for-the-badge" alt="Upstream"></a>
  <a href="https://github.com/zapabob/hermes-agent"><img src="https://img.shields.io/badge/Fork-zapabob-black?style=for-the-badge" alt="This fork"></a>
  <a href="https://github.com/zapabob/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
</p>

[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) を土台に、**常時稼働する Windows ワークステーション**向けの運用レイヤーを載せた fork です。公式の CLI / TUI / Gateway / Desktop / Dashboard / Memory / Skills / Cron は追従しつつ、**無料クラウド推論・ローカル秘書・X 自動投稿・NotebookLM 連携・VRChat 運用**など、このマシンで実際に使う機能をリポジトリ内にまとめています。

> 公式の汎用マーケティング README ではなく、**この checkout に入っている独自機能の索引**として読んでください。

### 日本語で一言

クラウドは OpenCode Zen の `auto-free` を主軸に、詰まったら llama.cpp ローカルへロールバック。Telegram / Discord などの Gateway、Electron Desktop、ローカル秘書（読み取り自動・書き込みは確認ゲート）、`lm-twitterer` による X 投稿、`notebooklm` プラグインによる実装ログ集約、Quest 2 + Virtual Desktop 向け VRChat ドクターまで、**1 台の Windows で回すための部品**がここにあります。

---

## 独自機能マップ

| 領域 | この fork の追加価値 | 入口 |
|------|----------------------|------|
| **無料クラウド推論** | `opencode-zen` + 仮想モデル `auto-free`、カタログ自動ローテーション、OpenClaw 鍵ブリッジ | [OpenCode Free](#opencode-zen-auto-free) |
| **ローカル推論 / 秘書** | RTX 3060 向け llama.cpp 秘書ランタイム、65536 ctx、`--jinja` 契約、書き込み確認ゲート | [Local Secretary](#local-secretary-rtx-3060) |
| **llama ロールバック** | TurboQuant `llama-server` @ `:8080`、自動起動・OOM 段階降下 | [Llama fallback](#llama-fallback) |
| **X / LM-twitterer** | Hermes 生成 + X cookie 投稿、はくあ署名、whitelist 返信、cron ラッパー | [LM-twitterer](#lm-twitterer) |
| **NotebookLM** | 実装ログ・X 活動の収集、投稿案ドラフト、Enterprise 同期 | [NotebookLM](#notebooklm) |
| **Desktop / Dashboard** | `apps/desktop/` Electron チャット、ダッシュボード PTY 埋め込み TUI | [Desktop](#desktop--dashboard) |
| **WebUI 連携** | 兄弟リポ `hermes-webui` の起動スクリプト | [WebUI](#hermes-webui-companion) |
| **Gateway 運用** | Windows UTF-8 / サブプロセス修復、Discord 100 コマンド整理、Telegram 接続予算 | [Gateway](#gateway--windows-hardening) |
| **VRChat / Quest 2** | Neuro ブリッジ、OSC ツール、OpenXR ActiveRuntime 修復 | [VRChat](#vrchat--quest-2) |
| **Windows 常駐** | Task Scheduler ゲートウェイ自動起動、ホスト移行スクリプト | [Autostart](#windows-autostart) |
| **Hypura / OpenClaw** | `hermes harness`、`hermes claw migrate`、チャネル readiness | [Harness](#hypura-harness--openclaw) |
| **その他プラグイン** | `irodori_tts`, `teams_pipeline`, `google_colab`, `questframe_fh6vr`, `unsloth_studio` など | `plugins/` |

---

## Quick Start (Windows)

```powershell
git clone https://github.com/zapabob/hermes-agent.git
cd hermes-agent
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[all,dev]"
python -m hermes_cli.main setup
```

`~/.hermes/.env`（秘密情報のみ）:

```env
OPENCODE_ZEN_API_KEY=...       # https://opencode.ai/auth
# または OpenClaw 由来の OPENCODE_API_KEY を共用
GATEWAY_ALLOW_ALL_USERS=true   # 個人用 gateway
```

`docs/migration/opencode_free_webui_config.example.yaml` から `model` / `fallback_providers` を `~/.hermes/config.yaml` にマージ:

```powershell
hermes fallback list
hermes doctor
hermes gateway run
hermes --tui
```

**注意:** `~/.hermes/.env` に UTF-8 BOM があるとキーが読めないことがあります。BOM を除去してください。

---

## OpenCode Zen `auto-free`

クラウド推論の既定ルート。`auto-free` は実行時に [OpenCode Zen の無料モデル一覧](https://opencode.ai/zen/v1/models) から解決し、レート制限時はカタログ内の次候補へ進みます。

```yaml
model:
  provider: opencode-zen
  default: auto-free

fallback_providers:
  - provider: opencode-zen
    model: auto-free
  - provider: llama-cpp
    model: your-fallback.gguf
    base_url: http://127.0.0.1:8080/v1
```

| 操作 | コマンド |
|------|----------|
| カタログ更新 | `py -3 scripts/refresh_opencode_free_catalog.py --force` |
| 解決確認 | `hermes fallback list` |
| 手順スキル | `skills/autonomous-ai-agents/opencode-free-rotation/SKILL.md` |
| OpenClaw 鍵 | `OPENCODE_API_KEY` だけで Zen 変数を満たす（`hermes_cli/auth.py`） |

---

## Local Secretary (RTX 3060)

コーディング専用エージェントではなく、**ローカル秘書**として動かすための構成。主推論は llama.cpp の OpenAI 互換 API（`:8080`）、Ollama は試験用。

| 項目 | 内容 |
|------|------|
| 主モデル例 | `qwen35-9b-secretary` @ `:8080`（GGUF は `~/.hermes/.env` の `HERMES_LLAMA_GGUF_PATH` 等で指定） |
| フォールバック | Hermes-3 8B `:8081`、Phi-4 mini `:8082` |
| 必須 | llama.cpp 起動に `--jinja`（tool calling 契約） |
| Context | 目標 65536（64000 未満は起動前警告） |
| 書き込み | X 投稿・Gmail 送信・Calendar 変更・シェル等は **ユーザー確認必須**（読み取り系は自動 OK） |

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\start-llama-secretary.ps1
powershell -ExecutionPolicy Bypass -File scripts\windows\check-local-llm.ps1
```

詳細: [`docs/local-secretary-runtime.md`](docs/local-secretary-runtime.md) · 設定例: `config/local-secretary.example.yaml` · ゲート実装: `agent/local_secretary/write_action_gate.py`

---

## Llama fallback

クラウドが使えないときのロールバック。`hermes_cli/llama_fallback_runtime.py` が `:8080` をプローブし、必要なら TurboQuant `llama-server` を起動します。

| スクリプト | 用途 |
|------------|------|
| `scripts/windows/start-hermes-llama-fallback-rtx3060.ps1` | **日常運用**（RTX 3060 / 秘書向け） |
| `scripts/windows/start-hermes-llama-fallback-rtx3080.ps1` | レガシー / 手動（RTX 3080 プロファイル） |
| `scripts/windows/start-hermes-llama-fallback.ps1` | 汎用ラッパー |

環境変数: `HERMES_LLAMA_MODEL_PATH`, `HERMES_LLAMA_GPU_PROFILE`, `HERMES_LLAMA_FALLBACK_AUTOSTART=auto`

---

## LM-twitterer

[LM-twitterer](https://github.com/soichi11208/LM-twitterer) 由来の Hermes プラグイン。**本文生成は Hermes（`ctx.llm`）**、**投稿・返信は X session cookie**（`auth_token` / `ct0`）。既定署名は `はくあ #hermesagent`。

```powershell
hermes plugins enable lm-twitterer
hermes lm-twitterer install-deps --yes
hermes lm-twitterer auth-browser --screen-name YOUR_NAME --wait-seconds 600
hermes lm-twitterer post "公開向けメモ"              # dry-run
hermes lm-twitterer post "公開向けメモ" --live      # 投稿
hermes lm-twitterer cron install --post-topic "..." --provider opencode-zen --model auto-free
```

| 要点 | 説明 |
|------|------|
| トピック検証 | `validate_public_topic` — `environment` / `secretary` 等の自然文は OK、`API_KEY=` や `.env` パスは拒否 |
| Cron | `~/.hermes/scripts/lm-twitterer-*.py` が `auth-check` → `post/replies --live`；`cron.script_timeout_seconds` は 900 以上推奨 |
| Gateway | `/lm-twitterer post [topic...] [--live]` |

フル手順: [`plugins/lm-twitterer/README.md`](plugins/lm-twitterer/README.md)

---

## NotebookLM

実装ログや（秘匿済み）X 活動を NotebookLM 向けソースにまとめ、投稿案のブレインストームや Enterprise API 同期を行うプラグイン。

```powershell
hermes plugins enable notebooklm
hermes notebooklm status
hermes notebooklm collect
hermes notebooklm brainstorm
```

ツール: `notebooklm_status`, `notebooklm_collect`, `notebooklm_brainstorm`, `notebooklm_sync`, `notebooklm_run`  
Enterprise 利用時は `NOTEBOOKLM_ENTERPRISE_PROJECT_NUMBER` 等を `~/.hermes/.env` に設定。

---

## Desktop / Dashboard

| 面 | パス / 起動 |
|----|-------------|
| **Electron Desktop** | `apps/desktop/` — `scripts/windows/start-hermes-desktop.ps1` |
| **Dashboard（埋め込み TUI）** | `hermes dashboard` / `scripts/windows/start-hermes-dashboard.ps1` |
| **Classic CLI / Ink TUI** | `hermes` / `hermes --tui` |

Desktop は `tui_gateway` 上の独自チャット面（PTY 埋め込み TUI とは別）。バグ切り分け時は Desktop 用スキル `hermes-desktop-app-work` を参照。

---

## hermes-webui Companion

兄弟リポ [zapabob/hermes-webui](https://github.com/zapabob/hermes-webui) を同じマシンで動かすラッパー:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\start-hermes-webui.ps1
```

`HERMES_WEBUI_ROOT` で checkout パスを上書き（例: `...\hermes-WebUI`）。WebUI は raw `config.yaml` を読み、`auto-free` は表示上そのまま・実行時に Hermes 側で解決されます。

---

## Gateway / Windows hardening

- **メッセージ:** Telegram, Discord, Slack, LINE, … — `hermes gateway run`
- **Discord:** スラッシュコマンド 100 件上限前の stale コマンド掃除
- **Telegram:** 90s 接続予算、必要時フォールバック IP 無効化
- **ターミナル:** Git Bash 優先、UTF-8 出力、`search_files` が Windows 上で `rg` / grep を解決
- **テスト / I/O:** `WinError 10106` 回避の env バックフィル、CRLF 正規化ハッシュ、Windows `atomic_json_write` 再試行

---

## VRChat / Quest 2

| リソース | 内容 |
|----------|------|
| スキル | `skills/gaming/vrchat/`, `skills/gaming/neuro-vrchat/` |
| ツール | `vrchat_osc`, `vrchat_neuro_*`, `vrchat_preflight`, … |
| ハーネス | `scripts/vrchat_neuro_bridge.py`, `scripts/vrchat_runtime_doctor.py` |
| Quest 2 診断 | `scripts/windows/vrchat_quest2_controller_doctor.ps1` |
| OpenXR 修復 | `scripts/windows/run-vrchat-openxr-fix-admin.ps1`（Virtual Desktop 向け） |

移行ガイド: [`docs/migration/vrchat_neurosama_autonomy.md`](docs/migration/vrchat_neurosama_autonomy.md)

---

## Windows autostart

```powershell
# ゲートウェイのみ（既定）
powershell -ExecutionPolicy Bypass -File scripts\windows\register-hermes-autostart.ps1

# llama ロールバックも登録
powershell -ExecutionPolicy Bypass -File scripts\windows\register-hermes-autostart.ps1 -IncludeLlama
```

ゲートウェイラッパー `scripts/windows/start-hermes-gateway.ps1` は既定で llama を起動しません（`-StartLlama` または `HERMES_GATEWAY_RECOVERY_START_LLAMA=1` で回復時のみ）。

ホスト移行: `docs/migration/windows_surface_to_mothership.md` · `export/import/verify-hermes-host-migration.ps1`

---

## Hypura Harness / OpenClaw

```powershell
hermes harness status
hermes harness start
hermes claw migrate
```

OpenClaw 由来の VRChat / VOICEVOX / チャネル設定は `scripts/openclaw_ports/` と `tools/openclaw/` 経由で取り込み可能。Hypura Harness 連携は `hermes_cli/harness.py`。

---

## 上流 Hermes との関係

この fork は **公式機能を置き換えない**運用レイヤーです。バグ修正や汎用改善は [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) への PR を優先し、Windows 専用スクリプト・個人 gateway 設定・VRChat / NotebookLM / lm-twitterer など **マシン固有の導線**だけを fork に残します。

```powershell
git fetch upstream main
py -3 scripts\sync_all.py --dry-run
py -3 scripts\sync_all.py --merge --target main --allow-preflight-blockers
```

マージ時に価値を守るパス例: `scripts/windows/`, `hermes_cli/llama_fallback_runtime.py`, `plugins/lm-twitterer/`, `plugins/notebooklm/`, `gateway/platforms/`, `agent/local_secretary/`

公式ドキュメント: [Quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) · [Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) · [Skills](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)

---

## Development

```powershell
pip install -e ".[all,dev]"
scripts\run_tests.sh tests\plugins\test_lm_twitterer_plugin.py
```

fork 固有のスモーク例:

```powershell
py -3 -m pytest -o addopts="" -p no:randomly tests\hermes_cli\test_opencode_openclaw_bridge.py -q
py -3 -m pytest -o addopts="" -p no:randomly tests\hermes_cli\test_opencode_free_rotation.py -q
```

フルスイート（CI 同等）: `scripts/run_tests.sh`  
ローカル生成物 `career_docs_output/` は git 無視（個人用途）。

---

## Links

| | |
|---|---|
| この fork | https://github.com/zapabob/hermes-agent |
| 上流 | https://github.com/NousResearch/hermes-agent |
| 公式 Docs | https://hermes-agent.nousresearch.com/docs/ |
| WebUI | https://github.com/zapabob/hermes-webui |
| OpenCode Zen | https://opencode.ai/auth |

---

## License

MIT — see [LICENSE](LICENSE).  
Core by [Nous Research](https://nousresearch.com). Windows / operations layer by [zapabob](https://github.com/zapabob).
