<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤ — Windows Native Fork

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/zapabob/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://github.com/NousResearch/hermes-agent"><img src="https://img.shields.io/badge/Upstream-NousResearch-blueviolet?style=for-the-badge" alt="Upstream: NousResearch"></a>
</p>

**このリポジトリは [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) の Windows ネイティブ対応フォークです。**
WSL2 なしで Windows 11 上でそのまま動作します。

> **Original:** The self-improving AI agent built by [Nous Research](https://nousresearch.com). Use any model — Nous Portal, OpenRouter (200+ models), OpenAI, Anthropic, or your own endpoint. Switch with `hermes model` — no lock-in.

---

## このフォークの変更点

アップストリームの `NousResearch/hermes-agent` に対して以下の Windows ネイティブ対応を追加しています。

| 修正箇所 | 内容 |
|---|---|
| `persistent_shell.py` `_temp_prefix` | `/tmp` ハードコードを `tempfile.gettempdir()` に変更。Python の `open()` と Git Bash 両方からアクセス可能に |
| `local.py` `_temp_prefix` | 同上 (`/tmp/hermes-local-*` → `C:/Users/.../AppData/Local/Temp/hermes-local-*`) |
| `local.py` `_make_run_env` | Windows では Unix 専用の `_SANE_PATH` (`/opt/homebrew/bin` 等) をサブプロセス PATH に混入しないよう修正 |
| `local.py` `_kill_shell_children` | `pkill -P` (Unix 専用) を `taskkill /F /FI "PPID eq <pid>"` (Windows 標準) に置き換え |

実装は集約モジュール [`tools/environments/platform_shell_compat.py`](tools/environments/platform_shell_compat.py) に寄せてあり、公式側の `local.py` / `persistent_shell.py` 更新時のマージ衝突を減らしています。

---

## アップストリーム（公式）との同期

`NousResearch/hermes-agent` の `main` を取り込みつつ上記の Windows 差分を維持する手順です。

1. **リモート**（初回のみ）: `git remote add upstream https://github.com/NousResearch/hermes-agent.git`
2. **差分確認**: `py -3 scripts/sync_upstream.py --dry-run`（`fetch` + 分岐サマリ + ウォッチリスト表示）
3. **作業ツリーをクリーンにしてから** マージ用ブランチ作成＋マージ: `py -3 scripts/sync_upstream.py --merge`  
   オプションでマージ直後にツール系テスト: `--merge --pytest`
4. **コンフリクト時**はスクリプトが案内する **ウォッチリスト** を優先確認:
   - `tools/environments/local.py`
   - `tools/environments/persistent_shell.py`
   - `tools/environments/platform_shell_compat.py`
   - `README.md`
5. 解決後: `py -3 scripts/sync_upstream.py --pytest-only`（`tests/tools/test_local_persistent.py` 等）

自動で `ours`/`theirs` を機械適用しない設計です（セキュリティ修正の取りこぼし防止）。

---

## クイックインストール

### Windows ネイティブ（Git Bash 必須）

1. **[Git for Windows](https://git-scm.com/download/win)** をインストール（Git Bash が含まれます）
2. **[uv](https://docs.astral.sh/uv/getting-started/installation/)** をインストール

```powershell
# PowerShell
irm https://astral.sh/uv/install.ps1 | iex
```

3. このリポジトリをクローンしてインストール：

```bash
# Git Bash または PowerShell
git clone https://github.com/zapabob/hermes-agent.git
cd hermes-agent
uv venv venv --python 3.11
source venv/Scripts/activate   # Git Bash の場合
# または: .\venv\Scripts\activate  # PowerShell の場合
uv pip install -e ".[all]"
hermes
```

### Linux / macOS / WSL2

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc
hermes
```

---

## はじめに

```bash
hermes              # 会話を開始
hermes model        # LLM プロバイダーとモデルを選択
hermes tools        # 有効にするツールを設定
hermes config set   # 個別の設定値を変更
hermes gateway      # メッセージングゲートウェイを起動 (Telegram, Discord など)
hermes setup        # セットアップウィザードを実行
hermes update       # 最新バージョンに更新
hermes doctor       # 問題を診断
```

📖 **[公式ドキュメント →](https://hermes-agent.nousresearch.com/docs/)**

---

## 主な機能

<table>
<tr><td><b>リアルターミナル UI</b></td><td>マルチライン編集、スラッシュコマンド補完、会話履歴、割り込み、ストリーミング出力付きの TUI</td></tr>
<tr><td><b>どこでも使える</b></td><td>Telegram、Discord、Slack、WhatsApp、Signal、CLI — 単一のゲートウェイプロセスから</td></tr>
<tr><td><b>学習ループ</b></td><td>エージェント主導のメモリ管理。複雑なタスク後の自律スキル生成。スキルの自己改善。セッション横断の FTS5 検索と LLM 要約</td></tr>
<tr><td><b>スケジュール自動化</b></td><td>自然言語によるクロンスケジューラー。日次レポート、夜間バックアップなど</td></tr>
<tr><td><b>並列サブエージェント</b></td><td>独立したサブエージェントを並列スポーン。Python スクリプトから RPC 経由でツール呼び出し可能</td></tr>
<tr><td><b>複数実行環境</b></td><td>ローカル、Docker、SSH、Daytona、Singularity、Modal の 6 バックエンド対応</td></tr>
</table>

---

## CLI とメッセージングのクイックリファレンス

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### Windows (native, PowerShell) — Early Beta

> **Heads up:** Native Windows support is **early beta**. It installs and runs, but hasn't been road-tested as broadly as our Linux/macOS/WSL2 paths. Please [file issues](https://github.com/NousResearch/hermes-agent/issues) when you hit rough edges. For the most battle-tested Windows setup today, run the Linux/macOS one-liner above inside **WSL2**.

Run this in PowerShell:

```powershell
iex (irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1)
```

The installer handles everything: uv, Python 3.11, Node.js, ripgrep, ffmpeg, **and a portable Git Bash** (MinGit, unpacked to `%LOCALAPPDATA%\hermes\git` — no admin required, completely isolated from any system Git install).  Hermes uses this bundled Git Bash to run shell commands.

If you already have Git installed, the installer detects it and uses that instead.  Otherwise a ~45MB MinGit download is all you need — it won't touch or interfere with any system Git.

> **Android / Termux:** The tested manual path is documented in the [Termux guide](https://hermes-agent.nousresearch.com/docs/getting-started/termux). On Termux, Hermes installs a curated `.[termux]` extra because the full `.[all]` extra currently pulls Android-incompatible voice dependencies.
>
> **Windows:** Native Windows is supported as an **early beta** — the PowerShell one-liner above installs everything, but expect rough edges and please file issues when you hit them. If you'd rather use WSL2 (our most battle-tested Windows path), the Linux command works there too. Native Windows install lives under `%LOCALAPPDATA%\hermes`; WSL2 installs under `~/.hermes` as on Linux.  The only Hermes feature that currently needs WSL2 specifically is the browser-based dashboard chat pane (it uses a POSIX PTY — classic CLI and gateway both run natively).

After installation:

```bash
source ~/.bashrc    # reload shell (or: source ~/.zshrc)
hermes              # start chatting!
```

---

## ドキュメント

| セクション | 内容 |
|---|---|
| [クイックスタート](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | インストール → セットアップ → 最初の会話 |
| [CLI 使い方](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | コマンド、キーバインド、ペルソナ、セッション |
| [設定](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | 設定ファイル、プロバイダー、モデル |
| [メッセージングゲートウェイ](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram、Discord、Slack、WhatsApp、Signal |
| [セキュリティ](https://hermes-agent.nousresearch.com/docs/user-guide/security) | コマンド承認、DM ペアリング、コンテナ分離 |
| [ツール & ツールセット](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | 40+ ツール、ツールセットシステム |
| [スキルシステム](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | 手続きメモリ、Skills Hub、スキル作成 |
| [メモリ](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | 永続メモリ、ユーザープロファイル |
| [MCP 連携](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | 任意の MCP サーバーを接続 |
| [クロンスケジューリング](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | スケジュールタスクとプラットフォーム配信 |

---

## OpenClaw からの移行

```bash
hermes claw migrate              # インタラクティブ移行
hermes claw migrate --dry-run    # プレビュー
hermes claw migrate --preset user-data  # シークレットなしで移行
```

移行される内容: SOUL.md、メモリ、スキル、コマンド許可リスト、メッセージング設定、API キー

---

## 開発者向けセットアップ

```bash
git clone https://github.com/zapabob/hermes-agent.git
cd hermes-agent
uv venv venv --python 3.11
source venv/Scripts/activate  # Windows Git Bash
uv pip install -e ".[all,dev]"
python -m pytest tests/ -q
```

コントリビューションは [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) へのプルリクエストもご検討ください。

---

## コミュニティ

- 💬 [Discord (NousResearch)](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues (upstream)](https://github.com/NousResearch/hermes-agent/issues)
- 🐛 [Issues (this fork)](https://github.com/zapabob/hermes-agent/issues)

---

## ライセンス

MIT — [LICENSE](LICENSE) 参照。

Built by [Nous Research](https://nousresearch.com). Windows native patches by [zapabob](https://github.com/zapabob).
