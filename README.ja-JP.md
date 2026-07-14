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
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Español-orange?style=for-the-badge" alt="Español"></a>
  <a href="README.ja-JP.md"><img src="https://img.shields.io/badge/Lang-日本語-green?style=for-the-badge" alt="日本語"></a>
</p>

**[Nous Research](https://nousresearch.com)が構築した、自己改善型のAIエージェントです。** 学習ループを組み込んだ唯一のエージェントであり、経験からスキルを生成し、使用中にそれらを改善し、自身に知識の永続化を促し、過去の会話を自ら検索し、セッションをまたいであなたという人物への理解を深めていきます。月5ドルのVPSでも、GPUクラスタでも、アイドル時にはほぼゼロコストのサーバーレス基盤でも動作します。ノートPCに縛られることはなく、Telegramから話しかけている間にクラウドVM上で作業を進めさせることもできます。

どんなモデルでも使用できます — [Nous Portal](https://portal.nousresearch.com)、OpenRouter、OpenAI、独自のエンドポイント、その他[多数](https://hermes-agent.nousresearch.com/docs/integrations/providers)。`hermes model`で切り替え可能 — コード変更もロックインも不要です。

<table>
<tr><td><b>本格的なターミナルインターフェース</b></td><td>マルチライン編集、スラッシュコマンドの自動補完、会話履歴、割り込みとリダイレクト、ストリーミングツール出力に対応したフル機能のTUI。</td></tr>
<tr><td><b>あなたのいる場所で動く</b></td><td>Telegram、Discord、Slack、WhatsApp、Signal、CLI — すべて単一のゲートウェイプロセスから。ボイスメモの文字起こし、プラットフォームをまたいだ会話の継続性。</td></tr>
<tr><td><b>閉じた学習ループ</b></td><td>エージェント自身が管理し、定期的にナッジを行うメモリ。複雑なタスク完了後の自律的なスキル生成。使用中に自己改善するスキル。LLM要約を活用したクロスセッション想起のためのFTS5セッション検索。<a href="https://github.com/plastic-labs/honcho">Honcho</a>による弁証法的ユーザーモデリング。<a href="https://agentskills.io">agentskills.io</a>のオープン標準と互換。</td></tr>
<tr><td><b>スケジュール自動化</b></td><td>任意のプラットフォームへの配信に対応した、組み込みcronスケジューラ。日次レポート、深夜バックアップ、週次監査 — すべて自然言語で指示し、無人で実行できます。</td></tr>
<tr><td><b>委譲と並列化</b></td><td>独立したサブエージェントを生成し、並列ワークストリームを実行。RPC経由でツールを呼び出すPythonスクリプトを書くことで、複数ステップのパイプラインをコンテキストコストゼロのターンに圧縮します。</td></tr>
<tr><td><b>ノートPCだけでなく、どこでも動作</b></td><td>6つのターミナルバックエンド — local、Docker、SSH、Singularity、Modal、Daytona。DaytonaとModalはサーバーレス永続性を提供し、エージェントの環境はアイドル時に休止し、必要に応じて目覚めるため、セッション間のコストはほぼゼロです。月5ドルのVPSでもGPUクラスタでも動かせます。</td></tr>
<tr><td><b>研究にすぐ使える</b></td><td>バッチでの軌跡生成、次世代のツール呼び出しモデルを訓練するための軌跡圧縮。</td></tr>
</table>

---

## クイックインストール

### Linux、macOS、WSL2、Termux

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows（ネイティブ、PowerShell）

> **ご注意:** WindowsネイティブはWSLなしでHermesを実行します — CLI、ゲートウェイ、TUI、ツールはすべてネイティブで動作します。WSL2を使いたい場合は、上記のLinux/macOS用ワンライナーもそのまま使えます。バグを見つけたら[Issueを起票](https://github.com/NousResearch/hermes-agent/issues)してください。

PowerShellで以下を実行します:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

インストーラはすべてを処理します: uv、Python 3.11、Node.js、ripgrep、ffmpeg、**ポータブル版Git Bash**（MinGit、`%LOCALAPPDATA%\hermes\git`に展開 — 管理者権限不要、システムのGitインストールから完全に隔離されます）。Hermesはこのバンドル版Git Bashを使ってシェルコマンドを実行します。

既にGitがインストールされている場合、インストーラはそれを検出して代わりに使用します。なければ約45MBのMinGitをダウンロードするだけで済み — システムのGitに干渉することはありません。

> **Android / Termux:** テスト済みの手動インストール手順は[Termuxガイド](https://hermes-agent.nousresearch.com/docs/getting-started/termux)に記載されています。Termux上では、`.[all]`エクストラがAndroid非対応の音声関連依存を引き込むため、Hermesはキュレートされた`.[termux]`エクストラをインストールします。
>
> **Windows:** Windowsネイティブは完全にサポートされています — 上記のPowerShellワンライナーがすべてをインストールします。WSL2を使いたい場合は、Linuxコマンドがそのまま使えます。Windowsネイティブのインストール先は`%LOCALAPPDATA%\hermes`、WSL2のインストール先はLinuxと同じく`~/.hermes`です。

インストール後:

```bash
source ~/.bashrc    # シェルを再読み込み（または: source ~/.zshrc）
hermes              # 会話開始！
```

### トラブルシューティング

#### Windows Defenderやアンチウイルスが`uv.exe`をマルウェアとして検出する

アンチウイルス（Bitdefender、Windows Defenderなど）がHermesの`bin`フォルダ（`%LOCALAPPDATA%\hermes\bin\uv.exe`）から`uv.exe`を隔離する場合、これは**誤検知**です。このファイルはAstralの`uv` — HermesがPython環境を管理するために同梱しているRust製Pythonパッケージマネージャです。MLベースのアンチウイルスエンジンは、パッケージをダウンロード・インストールする署名なしのRustバイナリを頻繁に誤検知します。

**手元のコピーが正規のものか検証するには:**

```powershell
# 必要ならGitHub CLIをインストール
winget install --id GitHub.cli

# GitHubにログイン
gh auth login

# 検証を実行
$uv = "$env:LOCALAPPDATA\hermes\bin\uv.exe"
$ver = (& $uv --version).Split(' ')[1]
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$zip = "$env:TEMP\uv.zip"
Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$ver/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
gh attestation verify $zip --repo astral-sh/uv
Expand-Archive $zip "$env:TEMP\uv_x" -Force
(Get-FileHash "$env:TEMP\uv_x\uv.exe").Hash -eq (Get-FileHash $uv).Hash
```

attestation（証明）が「Verification succeeded」と表示され、最後の行が`True`を出力すれば問題ありません。

**Hermesをホワイトリストに登録するには:**
- **Windows Defender:** PowerShellを管理者として実行 → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
- **Bitdefender:** Bitdefenderコンソールで例外を追加（Protection > Antivirus > Settings > Manage Exceptions）
- ファイルのハッシュではなく**フォルダ**をホワイトリストに登録してください — Hermesは`uv`を更新し、バージョンごとにハッシュが変わります

詳しい背景は、Astralのアップストリームレポートを参照してください: [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553)、[astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011)、[astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079)。

---

## はじめに

```bash
hermes              # インタラクティブCLI — 会話を開始
hermes model        # LLMプロバイダとモデルを選択
hermes tools        # 有効にするツールを設定
hermes config set   # 個別の設定値をセット
hermes gateway      # メッセージングゲートウェイを起動（Telegram、Discord等）
hermes setup        # フルセットアップウィザードを実行（一括設定）
hermes claw migrate # OpenClawから移行（OpenClaw利用者の場合）
hermes update       # 最新バージョンへ更新
hermes doctor       # 問題を診断
```

📖 **[完全なドキュメント →](https://hermes-agent.nousresearch.com/docs/)**

---

## APIキー集めをスキップ — Nous Portal

Hermesは好きなプロバイダで動作します — それは変わりません。ただ、モデル、ウェブ検索、画像生成、TTS、クラウドブラウザのために5つもの個別APIキーを集めたくないのであれば、**[Nous Portal](https://portal.nousresearch.com)**が1つのサブスクリプションでそのすべてをカバーします:

- **300以上のモデル** — `/model <name>`でいずれも選択可能
- **Tool Gateway** — ウェブ検索（Firecrawl）、画像生成（FAL）、テキスト読み上げ（OpenAI）、クラウドブラウザ（Browser Use）をすべてあなたのサブスクリプション経由でルーティング。追加アカウントは不要です。

新規インストールから1コマンド:

```bash
hermes setup --portal
```

これでOAuthによるログイン、Nousのプロバイダ設定、Tool Gatewayの有効化が行われます。何が構成されているかは`hermes portal info`でいつでも確認できます。詳細は[Tool Gatewayドキュメントページ](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway)を参照してください。

ツールごとに独自のキーを使うこともいつでも可能です — ゲートウェイはバックエンド単位であり、全か無かではありません。

---

## CLIとメッセージングのクイックリファレンス

Hermesには2つのエントリポイントがあります: `hermes`でターミナルUIを起動するか、ゲートウェイを起動してTelegram、Discord、Slack、WhatsApp、Signal、Emailから対話します。会話が始まれば、多くのスラッシュコマンドは両インターフェースで共通です。

| アクション | CLI | メッセージングプラットフォーム |
|---------|-----|---------------------|
| 会話を開始 | `hermes` | `hermes gateway setup` + `hermes gateway start`を実行し、ボットにメッセージを送信 |
| 新しい会話を開始 | `/new` または `/reset` | `/new` または `/reset` |
| モデルを変更 | `/model [provider:model]` | `/model [provider:model]` |
| パーソナリティを設定 | `/personality [name]` | `/personality [name]` |
| 直前のターンを再試行・取り消し | `/retry`、`/undo` | `/retry`、`/undo` |
| コンテキスト圧縮 / 使用量確認 | `/compress`、`/usage`、`/insights [--days N]` | `/compress`、`/usage`、`/insights [days]` |
| スキルを参照 | `/skills` または `/<skill-name>` | `/<skill-name>` |
| 現在の作業を中断 | `Ctrl+C` または新規メッセージ送信 | `/stop` または新規メッセージ送信 |
| プラットフォーム固有のステータス | `/platforms` | `/status`、`/sethome` |

完全なコマンド一覧は[CLIガイド](https://hermes-agent.nousresearch.com/docs/user-guide/cli)と[メッセージングゲートウェイガイド](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)を参照してください。

---

## ドキュメント

すべてのドキュメントは**[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**にあります:

| セクション | 内容 |
|---------|---------------|
| [クイックスタート](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | インストール → セットアップ → 2分で最初の会話 |
| [CLIの使い方](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | コマンド、キーバインド、パーソナリティ、セッション |
| [設定](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | 設定ファイル、プロバイダ、モデル、全オプション |
| [メッセージングゲートウェイ](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram、Discord、Slack、WhatsApp、Signal、Home Assistant |
| [セキュリティ](https://hermes-agent.nousresearch.com/docs/user-guide/security) | コマンド承認、DMペアリング、コンテナ隔離 |
| [ツールとツールセット](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | 40以上のツール、ツールセットシステム、ターミナルバックエンド |
| [スキルシステム](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | 手続き的記憶、Skills Hub、スキルの作成 |
| [メモリ](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | 永続メモリ、ユーザープロファイル、ベストプラクティス |
| [MCP統合](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | 任意のMCPサーバーを接続して機能を拡張 |
| [Cronスケジューリング](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | プラットフォーム配信付きのスケジュールタスク |
| [コンテキストファイル](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | 全会話に影響するプロジェクトコンテキスト |
| [アーキテクチャ](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | プロジェクト構造、エージェントループ、主要クラス |
| [コントリビューション](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) | 開発セットアップ、PRプロセス、コードスタイル |
| [CLIリファレンス](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) | 全コマンドとフラグ |
| [環境変数](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | 環境変数の完全リファレンス |

---

## OpenClawからの移行

OpenClawからの移行であれば、Hermesは設定、メモリ、スキル、APIキーを自動的にインポートできます。

**初回セットアップ時:** セットアップウィザード（`hermes setup`）は自動的に`~/.openclaw`を検出し、設定開始前に移行を提案します。

**インストール後はいつでも:**

```bash
hermes claw migrate              # インタラクティブ移行（フルプリセット）
hermes claw migrate --dry-run    # 何が移行されるかをプレビュー
hermes claw migrate --preset user-data   # シークレットなしで移行
hermes claw migrate --overwrite  # 既存の競合を上書き
```

インポート対象:

- **SOUL.md** — ペルソナファイル
- **メモリ** — MEMORY.mdとUSER.mdのエントリ
- **スキル** — ユーザー作成スキル → `~/.hermes/skills/openclaw-imports/`
- **コマンド許可リスト** — 承認パターン
- **メッセージング設定** — プラットフォーム設定、許可ユーザー、作業ディレクトリ
- **APIキー** — 許可リストに含まれるシークレット（Telegram、OpenRouter、OpenAI、Anthropic、ElevenLabs）
- **TTSアセット** — ワークスペースの音声ファイル
- **ワークスペース指示** — AGENTS.md（`--workspace-target`使用時）

全オプションは`hermes claw migrate --help`を参照するか、`openclaw-migration`スキルを使用すると、ドライランプレビュー付きのエージェント主導インタラクティブ移行が可能です。

---

## コントリビューション

コントリビューションを歓迎します！開発セットアップ、コードスタイル、PRプロセスについては[コントリビューションガイド](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)を参照してください。

コントリビュータ向けのクイックスタート — 標準のインストーラを使い、それが作成する`$HERMES_HOME/hermes-agent`（通常は`~/.hermes/hermes-agent`）の完全なgitチェックアウトから作業してください。これは`hermes update`、管理されたvenv、遅延依存関係、ゲートウェイ、ドキュメントツールが使用するレイアウトと一致します。

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

手動クローンのフォールバック（管理インストールのレイアウトを意図的に使わない使い捨てクローンやCI向け）:

venvはクローンしたソースツリーの外に作成してください — エージェントが動作するディレクトリ内のvenvは、エージェントが自身のチェックアウトに対して実行する相対パスのコマンドによって消去され、実行中のランタイムをセッションの途中で破壊する可能性があります。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.hermes/venvs/hermes-dev --python 3.11
source ~/.hermes/venvs/hermes-dev/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

---

## コミュニティ

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — Hermesや他のMCPホスト向けのLinuxデスクトップ制御MCPサーバー。AT-SPIアクセシビリティツリー、Wayland/X11入力、スクリーンショット、コンポジタのウィンドウターゲティングに対応。
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — コミュニティ製WeChatブリッジ: 同じWeChatアカウントでHermes AgentとOpenClawを動かせます。

---

## ライセンス

MIT — [LICENSE](LICENSE)を参照。

[Nous Research](https://nousresearch.com)が構築。
