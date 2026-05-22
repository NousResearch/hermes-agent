<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤ - zapabob Windows / Operations Fork

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/zapabob/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://github.com/NousResearch/hermes-agent"><img src="https://img.shields.io/badge/Upstream-NousResearch-blueviolet?style=for-the-badge" alt="Upstream: NousResearch"></a>
</p>

このリポジトリは [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) を基盤にした、Windows ネイティブ運用と常駐ゲートウェイ運用を厚くした fork です。

公式版の強みであるモデル非依存、TUI、メッセージングゲートウェイ、スキル、メモリ、cron、自律的な学習ループを基盤にし、公式の最新変更を継続的に取り込む方針です。その上で、この fork は実運用で詰まりやすい Windows shell、Discord / Telegram gateway、OpenClaw / Hypura / VRChat 周辺を先に補修し、ローカル PC でそのまま動かすための道具を追加しています。

比較基準: 2026-05-22 に `git fetch upstream main` で確認した公式 `upstream/main` と、この fork の `main`。この README は fork 独自価値の説明であり、公式最新が先行している変更は sync policy に沿って取り込みます。

---

## 公式版に対する利点

| 領域 | 公式 `NousResearch/hermes-agent` | この fork の利点 |
|---|---|---|
| Windows terminal | Native Windows は early beta。`bash` 探索は PATH 上の `bash.exe` を拾うため、環境によって WSL 起動スタブを誤認する余地があります。 | Git for Windows / portable Git Bash を優先し、`System32\bash.exe` と `WindowsApps\bash.exe` の WSL スタブを除外します。`terminal` の文字化けと `search_files` の `rg/grep/find` 不検出を避けます。 |
| Windows xurl | `skills/social-media/xurl` は `platforms: [linux, macos]`。 | `windows` を有効化し、Hermes 用 Windows shim を追加。公開 OAuth2 / PKCE client の client secret なし登録にも対応します。 |
| Discord gateway | 公式の gateway 機能を基盤にします。 | stale な Discord global slash command を先に削除してから再作成します。100 command limit 到達時の起動失敗を避け、`DISCORD_ALLOWED_USERS=*` を明示的な allow-all として扱います。 |
| Telegram startup | 公式の Telegram adapter を基盤にします。 | fallback IP retry を見込んだ 90 秒接続予算を戻し、`HERMES_TELEGRAM_DISABLE_FALLBACK_IPS=1` のときは設定済み fallback IP と DoH discovery を本当に読まないようにしています。 |
| Harness / Hypura | 公式 README には Hypura Harness CLI の運用面は前面に出ていません。 | `hermes harness status/start/stop/restart` を復元し、daemon 不在時も argparse ではなく明確な診断を返します。 |
| Skills Hub safety | 公式の Skills Hub を基盤にします。 | uninstall lock の `install_path` を検証し、絶対パス、traversal、skills root 削除を拒否します。CRLF / LF 差による bundle hash ぶれも抑えます。 |
| OpenClaw / VRChat / Voice | 公式にも OpenClaw migration はあります。 | OpenClaw 移行、VRChat OSC、VoiceVox、channel readiness、Hypura native helper を Windows PC 向けに拡張しています。 |
| Windows desktop operation | 公式 installer は Windows native を提供します。 | desktop shortcut、autostart、gateway 起動 wrapper、xurl shim など、実機常駐用の補助スクリプトを追加しています。 |

私の見方では、この fork は公式版の代替品というより、Windows 常駐運用を前提にした実務向けの運用レイヤーです。Linux / macOS / WSL2 で素直に使うなら公式版が自然です。Windows 11 上で gateway、ローカル shell、X 投稿、VRChat / Hypura 連携までまとめて動かしたい場合は、この fork の方が手戻りが少ない構成です。

---

## Fork 独自機能

### Windows native terminal hardening

- Git Bash を優先して検出し、WSL の `bash.exe` 起動スタブを terminal backend から除外。
- `terminal` tool の stdout / stderr を UTF-8 前提で扱い、Windows の文字化けを抑制。
- `search_files` が `rg`、Git Bash 同梱 `grep`、`find` を見つけられる状態を維持。
- Windows で誤って POSIX 専用 PATH を混ぜないようにし、ローカル shell の実行環境を安定化。

検証例:

```powershell
py -3.12 -m pytest -o addopts="" -p no:randomly tests\tools\test_local_env_windows_msys.py tests\tools\test_terminal_tool.py -q
```

### Gateway reliability hardening

- Discord global command sync は stale command の削除を先に行い、100 command limit に近いアプリでも起動しやすい順序に調整。
- Discord の `DISCORD_ALLOWED_USERS=*` は username 解決ではなく wildcard allow-all として処理。
- Telegram の platform connect timeout を fallback retry に合わせて 90 秒へ調整。
- Telegram fallback IP を無効化した場合、設定値と DoH discovery の両方をスキップ。
- `gateway --replace` は Windows でも安全な PID 生存判定を使い、古い gateway の置換を安定化。

検証例:

```powershell
py -3.12 -m pytest -o addopts="" -p no:randomly `
  tests\gateway\test_discord_connect.py `
  tests\gateway\test_discord_component_auth.py `
  tests\gateway\test_discord_slash_auth.py `
  tests\gateway\test_platform_reconnect.py `
  tests\gateway\test_runner_startup_failures.py `
  tests\gateway\test_telegram_conflict.py -q
```

### Hypura Harness CLI

`hermes harness` を復元し、Harness daemon の状態確認と起動停止を CLI から扱えるようにしています。

```bash
hermes harness status
hermes harness start
hermes harness stop
hermes harness restart
```

daemon script が未配置でも、`invalid choice: 'harness'` ではなく、どの runtime dependency が足りないかを返します。

### Windows xurl skill

公式 xurl skill は Linux / macOS 向けです。この fork では Windows を有効化し、Hermes から扱いやすい shim を追加しています。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_xurl_windows_shim.ps1
xurl auth status
```

対応済みの shim コマンド:

- `auth status`
- `auth apps add/remove/default/list`
- `auth oauth2`
- `whoami`
- `post`
- `reply`
- `quote`

token 値を標準出力に出さないことを重視しています。

### Skills Hub path safety

Skills Hub の uninstall 処理で lock file の `install_path` をそのまま信頼しないようにしています。

- `skill` または `category/skill` 形式だけを許可。
- final segment が対象 skill 名と一致しない場合は拒否。
- `SKILLS_DIR` の外側、絶対パス、traversal、skills root 自体の削除を拒否。
- text asset は改行差を正規化して hash し、Windows checkout と Unix checkout の差を抑制。

---

## 公式版から引き継ぐ主な機能

| 機能 | 内容 |
|---|---|
| Real terminal interface | TUI、複数行編集、slash command 補完、会話履歴、interrupt、streaming tool output。 |
| Messaging gateway | Telegram、Discord、Slack、WhatsApp、Signal、Email などを単一 gateway から扱います。 |
| Closed learning loop | メモリ、session search、skill creation、skill self-improvement、Honcho user modeling。 |
| Cron automations | 自然言語で定義する scheduled task と、各 messaging platform への配信。 |
| Delegation | subagent と RPC tool 呼び出しによる並列作業。 |
| Terminal backends | local、Docker、SSH、Singularity、Modal、Daytona、Vercel Sandbox。 |
| Research workflows | trajectory generation と compression。 |

---

## Quick Install

### この fork を Windows で使う

PowerShell:

```powershell
git clone https://github.com/zapabob/hermes-agent.git
cd hermes-agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[all,dev]"
python -m hermes_cli.main --help
```

Git Bash:

```bash
git clone https://github.com/zapabob/hermes-agent.git
cd hermes-agent
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install -U pip
pip install -e ".[all,dev]"
hermes
```

### 公式版を使う

Linux / macOS / WSL2 / Termux:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc
hermes
```

Native Windows 公式 installer:

```powershell
iex (irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1)
```

---

## Getting Started

```bash
hermes              # interactive CLI
hermes model        # model / provider selection
hermes tools        # toolset configuration
hermes gateway run  # foreground messaging gateway
hermes setup        # setup wizard
hermes doctor       # environment diagnostics
hermes logs         # inspect runtime logs
hermes harness      # fork-specific Hypura Harness management
```

CLI と messaging platform の詳細は公式ドキュメントを参照してください。

- [Quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)
- [CLI Usage](https://hermes-agent.nousresearch.com/docs/user-guide/cli)
- [Messaging Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)
- [Tools & Toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools)
- [Skills System](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)
- [Cron Scheduling](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)

---

## OpenClaw / VRChat / Hypura

この fork には、OpenClaw 由来の運用資産を Hermes へ移すための補助と、Windows PC 上での VRChat / Hypura 周辺連携が含まれています。

```bash
hermes claw migrate
hermes claw migrate --dry-run
hermes claw migrate --preset user-data
hermes harness status
```

関連ファイル:

- `scripts/openclaw_ports/`
- `tools/openclaw/`
- `skills/gaming/vrchat/`
- `hermes_cli/hypura_native.py`
- `tools/harness_tools.py`

---

## Upstream sync policy

この fork は公式 `NousResearch/hermes-agent` を追従します。同期時は公式差分を取り込みつつ、Windows / gateway / OpenClaw-Hypura 層の fork 独自価値を残します。

```powershell
git fetch upstream main
py -3 scripts\sync_upstream.py --dry-run
py -3 scripts\sync_upstream.py --merge
py -3 scripts\sync_upstream.py --pytest-only
```

watch list:

- `tools/environments/local.py`
- `tools/environments/persistent_shell.py`
- `tools/environments/platform_shell_compat.py`
- `gateway/platforms/discord.py`
- `gateway/platforms/telegram.py`
- `hermes_cli/harness.py`
- `tools/skills_hub.py`
- `README.md`

---

## Development

```bash
git clone https://github.com/zapabob/hermes-agent.git
cd hermes-agent
py -3.12 -m venv .venv
source .venv/Scripts/activate
pip install -e ".[all,dev]"
py -3.12 -m pytest -o addopts="" -p no:randomly tests\tools\test_local_env_windows_msys.py -q
```

広い confidence が必要な場合は、公式 Hermes test environment を持つ CI または developer host で full test suite を実行してください。

---

## Community

- [Official docs](https://hermes-agent.nousresearch.com/docs/)
- [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
- [zapabob/hermes-agent issues](https://github.com/zapabob/hermes-agent/issues)
- [Nous Research Discord](https://discord.gg/NousResearch)
- [Skills Hub](https://agentskills.io)

---

## License

MIT - see [LICENSE](LICENSE).

Built by [Nous Research](https://nousresearch.com). Windows and operations fork maintained by [zapabob](https://github.com/zapabob).
