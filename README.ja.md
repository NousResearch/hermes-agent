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
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Espa%C3%B1ol-red?style=for-the-badge" alt="Español"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-%E4%B8%AD%E6%96%87-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-%D8%A7%D8%B1%D8%AF%D9%88-green?style=for-the-badge" alt="اردو"></a>
</p>

**Nous Researchによって構築された自己進化型AIエージェント。** 経験からスキルを作成し、使用の中でスキルを改善し、知識を積極的に持続させ、過去の会話を検索し、セッションをまたいであなたへの深い理解を段階的に構築する——唯一の内蔵学習ループを持つエージェントです。$5のVPSでも、GPUクラスタでも、あるいはほぼゼロコストのサーバーレスインフラでも動作します。あなたのノートPCに縛られません——Telegramで対話しながら、クラウドVMで作業させることができます。

任意のモデルをサポート——[Nous Portal](https://portal.nousresearch.com)、[OpenRouter](https://openrouter.ai)（200以上のモデル）、[NVIDIA NIM](https://build.nvidia.com)（Nemotron）、[Xiaomi MiMo](https://platform.xiaomimimo.com)、[z.ai/GLM](https://z.ai)、[Kimi/Moonshot](https://platform.moonshot.ai)、[MiniMax](https://www.minimax.io)、[Hugging Face](https://huggingface.co)、OpenAI、またはカスタムエンドポイント。`hermes model` ですぐに切り替え可能——コード変更不要、ロックインなし。

<table>
<tr><td><b>本物のターミナルUI</b></td><td>複数行編集、スラッシュコマンド自動補完、会話履歴、中断リダイレクト、ストリーミングツール出力を備えた完全なTUI。</td></tr>
<tr><td><b>どこでも使える</b></td><td>Telegram、Discord、Slack、WhatsApp、Signal、CLI——すべて単一のゲートウェイプロセスから実行。音声メモ書き起こし、クロスプラットフォーム会話継続性。</td></tr>
<tr><td><b>閉ループ学習</b></td><td>エージェントが記憶を管理し、定期的に自己復習。複雑なタスク後に自動でスキルを作成。スキルは使用の中で自己改善。FTS5セッション検索とLLM要約によるクロスセッション回顧。<a href="https://github.com/plastic-labs/honcho">Honcho</a>による弁証法的ユーザーモデリング。<a href="https://agentskills.io">agentskills.io</a>オープン標準と互換。</td></tr>
<tr><td><b>定時自動化</b></td><td>内蔵cronスケジューラ、任意プラットフォームへの配信対応。日報、夜間バックアップ、週次監査——すべて自然言語で記述、無人実行。</td></tr>
<tr><td><b>委任と並列実行</b></td><td>隔離された子エージェントを生成し並列ワークフローを処理。PythonスクリプトでRPC経由でツールを呼び出し、複数ステップのパイプラインをゼロコンテキストオーバーヘッドのターンに圧縮。</td></tr>
<tr><td><b>どこでも実行</b></td><td>6種類のターミナルバックエンド——ローカル、Docker、SSH、Daytona、Singularity、Modal。DaytonaとModalでサーバーレス永続化——エージェント環境はアイドル時にスリープ、オンデマンドでウェイク、アイドル期間はほぼゼロコスト。$5のVPSからGPUクラスタまで対応。</td></tr>
<tr><td><b>研究対応</b></td><td>バッチ軌道生成、軌道圧縮——次世代ツール呼び出しモデルの訓練用。</td></tr>
</table>

---

## クイックインストール

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

Linux、macOS、WSL2、Android (Termux) に対応。インストーラーがプラットフォーム固有の設定を自動処理。

> **Android / Termux:** 検証済みの手動インストール手順は [Termux ガイド](https://hermes-agent.nousresearch.com/docs/getting-started/termux) を参照。Termuxでは、Hermesは選抜された `.[termux]` 拡張をインストールします。完全な `.[all]` 拡張はAndroid非互換の音声依存を引き込むためです。
>
> **Windows:** PowerShellで実行:
> ```powershell
> iex (irm https://hermes-agent.nousresearch.com/install.ps1)
> ```
> インストール後、ターミナルを再起動する必要がある場合があります。その後 `hermes` で会話を開始。

インストール後:

```bash
source ~/.bashrc    # shellを再読み込み (または: source ~/.zshrc)
hermes              # 会話開始！
```

---

## クイックスタート

```bash
hermes              # インタラクティブ CLI — 会話を開始
hermes model        # LLMプロバイダとモデルを選択
hermes tools        # 有効なツールを設定
hermes config set   # 単一の設定項目を設定
hermes gateway      # メッセージングゲートウェイを起動 (Telegram、Discord等)
hermes setup        # 完全セットアップウィザードを実行 (一度きりの全設定)
hermes claw migrate # OpenClawからの移行 (OpenClawからの場合)
hermes update       # 最新版に更新
hermes doctor       # 問題を診断
```

📖 **[完全なドキュメント →](https://hermes-agent.nousresearch.com/docs/)**

---

## APIキーの収集を省略 — Nous Portal

Hermesは常に任意のプロバイダを使用可能にします。これは変わりません。しかし、モデル、ウェブ検索、画像生成、TTS、クラウドブラウザのために5つの異なるAPIキーを個別に取得したくない場合、**[Nous Portal](https://portal.nousresearch.com)** なら1つのサブスクリプションですべてをカバー:

- **300以上のモデル** — `/model <name>` で即座に切り替え
- **Tool Gateway** — ウェブ検索 (Firecrawl)、画像生成 (FAL)、テキスト音声変換 (OpenAI)、クラウドブラウザ (Browser Use) をすべてサブスクリプション経由でホスト。追加のアカウント登録は一切不要。

新規インストールなら1コマンドで:

```bash
hermes setup --portal
```

OAuthでログインし、Nousを推論プロバイダとして設定し、Tool Gatewayを有効化。`hermes portal info` でいつでもルーティング状況を確認。詳細は [Tool Gateway ドキュメント](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway) を参照。

ツール単位でいつでも独自のAPIキーに戻せます — Gatewayはツール粒度で有効/無効、一括ではない。

---

## CLI とメッセージングプラットフォーム 早見表

Hermesには2つのエントリーポイントがあります: `hermes` でターミナルUIを起動、またはゲートウェイを実行してTelegram、Discord、Slack、WhatsApp、Signal、Emailから対話。会話に入れば、多くのスラッシュコマンドが両UIで共通。

| アクション | CLI | メッセージングプラットフォーム |
|------|-----|----------|
| 会話開始 | `hermes` | `hermes gateway setup` + `hermes gateway start` を実行し、ボットにメッセージ送信 |
| モデル切り替え | `hermes model` | `/model` |
| ツール設定 | `hermes tools` | `/tools` |
| 設定変更 | `hermes config set` | `/config set` |
| スキル管理 | `hermes skills` | `/skills` |
| セッション検索 | `hermes session_search` | `/search` |
| ヘルプ表示 | `/help` | `/help` |

---

## アーキテクチャ概要

```
┌─────────────────────────────────────────────────────────────┐
│                      Hermes Agent                           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   CLI/TUI    │  │   Gateway    │  │  ACP/Editor  │       │
│  │  (ink/react) │  │ (Telegram,   │  │ (VS Code,    │       │
│  │              │  │  Discord,    │  │  Zed,        │       │
│  │              │  │  Slack, etc) │  │  JetBrains)  │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                │
│         └─────────────────┼─────────────────┘                │
│                           ▼                                  │
│              ┌────────────────────────┐                     │
│              │      AIAgent Core      │                     │
│              │  (run_agent.py)        │                     │
│              │  - Conversation loop   │                     │
│              │  - Tool orchestration  │                     │
│              │  - Memory integration  │                     │
│              │  - Context compression │                     │
│              └───────────┬────────────┘                     │
│                          │                                   │
│         ┌────────────────┼────────────────┐                 │
│         ▼                ▼                ▼                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Model     │  │   Tools     │  │  Providers  │         │
│  │  Providers  │  │  (terminal, │  │  (OpenAI,   │         │
│  │  (OpenAI,   │  │  web_search,│  │   Anthropic,│         │
│  │   etc.)     │  │  file, etc) │  │   custom)   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## 貢献

貢献を歓迎します！詳細については [CONTRIBUTING.md](CONTRIBUTING.md) をご覧ください。

---

## ライセンス

[MIT License](LICENSE) © Nous Research

---

## コミュニティ

- **Discord:** [Nous Research](https://discord.gg/NousResearch)
- **ドキュメント:** [hermes-agent.nousresearch.com](https://hermes-agent.nousresearch.com/docs/)
- **問題報告:** [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)

---

*自己進化型エージェントで、あなたのワークフローを加速させましょう。*