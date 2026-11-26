---
name: notion-patrol
description: Use when running or maintaining the read-only Notion Patrol workflow that recursively scans Notion pages/databases for external URLs, checks link health, writes CSV results, and reports weekly Slack summaries.
version: 1.1.0
author: Legal AI / Hermes Agent
license: MIT
platforms: [linux]
language: javascript
runtime: node18+
dependencies: none
metadata:
  hermes:
    tags: [notion, link-checker, url-patrol, read-only, slack, cron]
    related_skills: [notion, slack-mention-replies]
---

# Notionパトロール君

## Overview

Notionの指定ルート（ページまたはデータベース）配下を再帰的に読み取り、外部URLだけを抽出してHTTPステータスを確認するスキルです。Notion APIには **読み取りリクエストのみ** を行い、Notionへの作成・更新・削除・コメント投稿は行いません。

主な用途:

- 法務内マニュアルや法務関連DBに含まれる外部リンクの定期点検
- リンク切れ・疎通エラーのCSV化
- 週次CronによるSlack通知

## When to Use

- NotionページまたはNotionデータベース配下の外部リンクを網羅的に洗い出すとき
- Notionリンクチェック結果をCSVで残したいとき
- 法務Notionのリンクチェックを週次Cronで運用・保守するとき
- Slack通知文面、CSV出力、対象Notionルートを変更・検証するとき

Do **not** use this skill for:

- Notionページ・DBの作成、更新、削除、コメント投稿
- Notion内部リンクの疎通確認
- 機密トークンやAPIキーの収集・表示

## 既定の対象ルート

| 種別 | 名称 | Notion ID |
|---|---|---|
| Page | 法務内マニュアル | `106173b008788028ac4efd380a88308c` |
| Database | 法務関連DB | `8db170cdc5ba4c488ef9302e4e58cede` |

`--root` を指定しない場合は、上記2ルートを対象にします。`--root` は複数回指定できます。Notion ID、UUID形式、Notion URLのいずれも受け付けます。

## 実行方法

```bash
cd /home/kawazoe_taishi/.hermes/skills/productivity/notion-patrol
node scripts/patrol.js --output-dir /home/kawazoe_taishi
```

主なオプション:

```bash
node scripts/patrol.js \
  --root <NotionページIDまたはURL> \
  --root <追加IDまたはURL> \
  --output-dir <CSV出力先> \
  --concurrency 10 \
  --timeout-ms 10000 \
  --notion-version 2022-06-28
```

| オプション | 内容 |
|---|---|
| `--root <id-or-url>` | 巡回対象ルート。複数指定可。未指定時は既定2ルート。 |
| `--output-dir <dir>` | CSV出力先。 |
| `--concurrency <n>` | 外部URLチェックの並列数。既定 `10`。 |
| `--timeout-ms <ms>` | URLごとのタイムアウト。既定 `10000`。 |
| `--notion-version <version>` | Notion APIバージョン。既定 `2022-06-28`。 |
| `--no-check` | URL抽出のみ行い、外部URLへのHEAD/GET確認をスキップ。 |
| `--help` | ヘルプ表示。 |

## 環境変数

以下のいずれかにNotion integration tokenを設定してください。スクリプトはトークンを表示しません。

- `NOTION_API_KEY`
- `NOTION_TOKEN`
- `NOTION_API_TOKEN`

`.env` は次の順に読み込み対象です。

1. カレントディレクトリの `.env`
2. Hermes標準の `~/.hermes/.env`
3. このスキルディレクトリの `.env`

Secrets handling:

- トークン値は `SKILL.md`、Notion仕様ページ、GitHub、Slack本文、ログに記録しない。
- トークンを標準出力・標準エラーに出さない。

## URL抽出・判定仕様

抽出対象:

- Rich text の `href`
- Rich text本文中の raw URL
- `bookmark` / `embed` / `link_preview`
- `image` / `video` / `file` / `pdf` 等の外部ファイルURL

除外対象:

- `notion.so`
- `app.notion.com`
- `http` / `https` 以外のURL

判定:

- `HEAD` で確認する。
- `HEAD` が `405` の場合のみ `GET` にフォールバックする。
- `2xx` / `3xx` は `OK`。
- その他のHTTPステータス、ネットワークエラー、タイムアウトは `NG`。
- `--no-check` 時はステータス `SKIPPED`、判定 `OK` とする。

## 出力

コンソール:

- 件数サマリー
- NGリンク一覧
- CSVパス

CSV:

- ファイル名: `link_check_test_YYYYMMDD.csv`
- 形式: BOM付きUTF-8
- ヘッダー:
  1. `ページ名`
  2. `URL`
  3. `ステータスコード`
  4. `判定`
  5. `Context（文脈）`

CSVはExcelで開きやすいようにBOM付きUTF-8で出力し、カンマ・引用符をCSV escapingします。

## Cron / Slack通知での運用

定期実行にする場合も、Notionパトロール本体は読み取り専用のまま維持します。Cron用ラッパーは `~/.hermes/scripts/notion_patrol_weekly.py` に置き、巡回・集計・Slack通知整形だけを行います。

現行の週次運用設定:

| 項目 | 値 |
|---|---|
| ジョブ名 | `Notionパトロール君 週次リンクチェック` |
| 実行頻度 | 毎週月曜 11:00 JST |
| Cron式（UTC運用時） | `0 2 * * 1` |
| Slack通知先 | `G01ACN6N2HW` |
| 通知メンション | `<!subteam^S02FLCBKU0P>` |
| 実行モード | `--no-agent` |
| プロファイル | `default` |
| 出力先 | `~/.hermes/cron/output/notion-patrol` |

登録例:

```bash
hermes cron create '0 2 * * 1' \
  --name 'Notionパトロール君 週次リンクチェック' \
  --deliver 'slack:G01ACN6N2HW' \
  --script notion_patrol_weekly.py \
  --no-agent \
  --profile default
```

JSTとしてCron時刻が解決される環境では、月曜11:00は `0 11 * * 1` です。`HERMES_TIMEZONE` 未設定かつホストUTC運用の場合は、月曜11:00 JST相当として `0 2 * * 1` を使います。

通知本文には次を含めます。

- `<!subteam^S02FLCBKU0P>`
- 実行日時
- 対象Notionルート
- チェック件数
- OK件数
- NG件数
- CSVパス
- `MEDIA:<csv_path>` によるCSV添付指定
- NGリンク最大20件

Slack上での視認性確認として、Block Kit形式のテスト通知は成功確認済みです。定期実行の基本系は `--no-agent` でスクリプト出力を配送する構成です。Block Kitを本番経路に組み込む場合は、Slack API送信とHermes Cron配送の二重投稿を避けるため、送信成功時は `[SILENT]` または `{"wakeAgent": false}` 相当でCron配送を抑止してください。

## テスト

```bash
node --test /home/kawazoe_taishi/.hermes/skills/productivity/notion-patrol/scripts/patrol.test.js
node --check /home/kawazoe_taishi/.hermes/skills/productivity/notion-patrol/scripts/patrol.js
node /home/kawazoe_taishi/.hermes/skills/productivity/notion-patrol/scripts/patrol.js --help
python -m py_compile /home/kawazoe_taishi/.hermes/scripts/notion_patrol_weekly.py
python /home/kawazoe_taishi/.hermes/scripts/notion_patrol_weekly.py --test-notification
hermes cron list --all
```

## 実装メモ

- Node.js 18+ 標準 `fetch` のみ使用（npm依存なし）。
- 外部リンクチェックは `HEAD`、405時のみ `GET` にフォールバックします。
- 既定タイムアウトは10秒、既定並列数は10です。
- `notion.so` / `app.notion.com` のNotion内部リンクは対象外です。
- ページ、ブロック子要素、データベースクエリを防御的に扱います。
- 保守・拡張時の注意点は `references/implementation-notes.md` を参照してください。
- 仕様書は `references/specification.md` を参照してください。
- 週次Cron実行・Slack通知に配線する場合は、`references/cron-weekly-slack-notification.md` の運用メモを参照してください。

## Common Pitfalls

1. **Notion書き込みを混ぜること。** 本スキルは読み取り専用です。仕様変更がない限り、ページ作成・更新・削除・コメント投稿はしないでください。
2. **トークンをログや仕様書へ出すこと。** 環境変数名だけを記載し、値は記録しないでください。
3. **JST指定をUTCのまま登録すること。** 環境がUTCなら月曜11:00 JSTは `0 2 * * 1` です。
4. **Block Kit送信とCron配送を二重化すること。** Slack APIで直接投稿する場合はCron側の通常配送を抑止してください。
5. **Notion内部リンクを外部リンクとして扱うこと。** `notion.so` / `app.notion.com` は除外対象です。
6. **現在セッションのスキル一覧キャッシュに依存すること。** 新規・更新スキルは新しいセッションまたはリロード後に見える場合があります。

## Verification Checklist

- [ ] `node --test .../patrol.test.js` が成功する。
- [ ] `node --check .../patrol.js` が成功する。
- [ ] `node .../patrol.js --help` が成功する。
- [ ] `python -m py_compile ~/.hermes/scripts/notion_patrol_weekly.py` が成功する。
- [ ] `python ~/.hermes/scripts/notion_patrol_weekly.py --test-notification` がメンション付き文面を出す。
- [ ] `hermes cron list --all` で `Next run` と `Deliver` が意図どおりである。
- [ ] 仕様書・GitHub・NotionDBに秘密値が含まれていない。
