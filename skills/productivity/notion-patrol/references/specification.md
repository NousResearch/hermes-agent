# Notionパトロール君 仕様書

## 1. 目的

Notionパトロール君は、Notionの指定ルート（ページまたはデータベース）配下を再帰的に読み取り、外部URLを抽出してリンク疎通を確認するローカルHermesスキルです。

Notion APIに対して行う操作は読み取りのみです。Notionページ・データベースの作成、更新、削除、コメント投稿は行いません。

## 2. スコープ

### 対象

- Notionページ配下のブロック
- Notionデータベース配下のページ
- ブロック内の外部URL
- 添付・埋め込み系ブロックに含まれる外部URL
- 抽出URLのHTTPステータス確認
- CSV出力
- 週次CronによるSlack通知

### 対象外

- Notion内部リンク（`notion.so` / `app.notion.com`）の疎通確認
- Notionの作成・更新・削除・コメント投稿
- APIトークン値の表示・保存
- 外部リンク先コンテンツの本文解析

## 3. 既定の巡回対象

| 種別 | 名称 | Notion ID |
|---|---|---|
| Page | 法務内マニュアル | `106173b008788028ac4efd380a88308c` |
| Database | 法務関連DB | `8db170cdc5ba4c488ef9302e4e58cede` |

`--root` を指定しない場合、上記2ルートを使用します。

## 4. 実行環境

| 項目 | 内容 |
|---|---|
| 実装言語 | JavaScript |
| 実行環境 | Node.js 18+ |
| 依存 | npm依存なし |
| HTTP実装 | Node.js標準 `fetch` |
| テスト | `node:test` |
| Cron wrapper | Python 3 |

## 5. CLI仕様

基本実行:

```bash
cd /home/kawazoe_taishi/.hermes/skills/productivity/notion-patrol
node scripts/patrol.js --output-dir /home/kawazoe_taishi
```

オプション:

| オプション | 内容 | 既定値 |
|---|---|---|
| `--root <id-or-url>` | 巡回対象ルート。複数指定可。 | 既定2ルート |
| `--output-dir <dir>` | CSV出力先。 | カレントディレクトリ |
| `--concurrency <n>` | 外部URLチェック並列数。 | `10` |
| `--timeout-ms <ms>` | URLごとのタイムアウト。 | `10000` |
| `--notion-version <version>` | Notion APIバージョン。 | `2022-06-28` |
| `--no-check` | URL抽出のみ。疎通確認をスキップ。 | false |
| `--help` | ヘルプ表示。 | - |

## 6. 認証・秘密情報

Notion tokenは以下のいずれかから読み込みます。

- `NOTION_API_KEY`
- `NOTION_TOKEN`
- `NOTION_API_TOKEN`

`.env` 読み込み対象:

1. カレントディレクトリの `.env`
2. `~/.hermes/.env`
3. スキルディレクトリの `.env`

秘密情報の運用:

- トークン値はコード、仕様書、NotionDB、GitHub、Slack本文、ログに記録しません。
- 記録するのは環境変数名のみです。

## 7. Notion読み取り仕様

読み取り対象:

- Page metadata
- Block children
- Database query
- Database metadata

読み取り専用を維持するため、以下は行いません。

- Page create/update/delete
- Database create/update/delete
- Block append/update/delete
- Comment create

## 8. URL抽出仕様

抽出するURL:

- Rich textの `href`
- Rich text本文中の raw URL
- `bookmark` の `url`
- `embed` の `url`
- `link_preview` の `url`
- `image` / `video` / `file` / `pdf` 等の外部URL

除外するURL:

- `notion.so`
- `app.notion.com`
- `http` / `https` 以外

## 9. リンクチェック判定

1. `HEAD` で確認します。
2. `HEAD` が `405` の場合のみ `GET` にフォールバックします。
3. `2xx` / `3xx` は `OK` とします。
4. その他のHTTPステータス、ネットワークエラー、タイムアウトは `NG` とします。
5. `--no-check` 時はステータス `SKIPPED`、判定 `OK` とします。

## 10. CSV出力仕様

ファイル名:

```text
link_check_test_YYYYMMDD.csv
```

文字コード:

```text
BOM付きUTF-8
```

列:

| 列 | ヘッダー |
|---|---|
| 1 | `ページ名` |
| 2 | `URL` |
| 3 | `ステータスコード` |
| 4 | `判定` |
| 5 | `Context（文脈）` |

CSV escapingを行い、Excelで開きやすい形式にします。

## 11. Cron / Slack運用仕様

Cron wrapper:

```text
/home/kawazoe_taishi/.hermes/scripts/notion_patrol_weekly.py
```

Cron出力先:

```text
/home/kawazoe_taishi/.hermes/cron/output/notion-patrol
```

現行登録:

| 項目 | 値 |
|---|---|
| ジョブ名 | `Notionパトロール君 週次リンクチェック` |
| 実行頻度 | 毎週月曜 11:00 JST |
| Cron式 | `0 2 * * 1` |
| Cron式の前提 | `HERMES_TIMEZONE` 未設定・UTC運用 |
| Slack通知先 | `G01ACN6N2HW` |
| メンション | `<!subteam^S02FLCBKU0P>` |
| 実行モード | `--no-agent` |
| プロファイル | `default` |

登録例:

```bash
hermes cron create '0 2 * * 1' \
  --name 'Notionパトロール君 週次リンクチェック' \
  --deliver 'slack:G01ACN6N2HW' \
  --script notion_patrol_weekly.py \
  --no-agent \
  --profile default
```

## 12. Slack通知仕様

通知本文に含める内容:

- `<!subteam^S02FLCBKU0P>`
- 実行日時
- 対象Notionルート
- チェック件数
- OK件数
- NG件数
- CSVパス
- `MEDIA:<csv_path>` によるCSV添付指定
- NGリンク最大20件

失敗時は、以下を区別して通知します。

- `patrol.js` 不在
- タイムアウト
- 非ゼロ終了
- CSV未検出

Block Kit形式のテスト通知は成功確認済みです。Block Kitを本番経路に入れる場合は、Slack APIでの直接投稿とHermes Cron配送が二重投稿にならないよう、送信成功時にCron通常配送を抑止します。

## 13. テスト・検証

```bash
node --test /home/kawazoe_taishi/.hermes/skills/productivity/notion-patrol/scripts/patrol.test.js
node --check /home/kawazoe_taishi/.hermes/skills/productivity/notion-patrol/scripts/patrol.js
node /home/kawazoe_taishi/.hermes/skills/productivity/notion-patrol/scripts/patrol.js --help
python -m py_compile /home/kawazoe_taishi/.hermes/scripts/notion_patrol_weekly.py
python /home/kawazoe_taishi/.hermes/scripts/notion_patrol_weekly.py --test-notification
hermes cron list --all
```

確認観点:

- Nodeテストが成功すること
- `patrol.js` の構文チェックが成功すること
- `--help` が表示されること
- Cron wrapperのPython構文チェックが成功すること
- テスト通知にメンション、対象、件数、注記が含まれること
- Cronの `Next run` と `Deliver` が意図どおりであること
- 秘密情報が出力・保存されていないこと

## 14. 運用上の注意

- 本スキルは読み取り専用です。Notion書き込み処理を追加しないでください。
- APIトークン等の秘密値を記録しないでください。
- JST指定のCronは、実行環境のタイムゾーンを確認してから登録してください。
- Slack APIでBlock Kit投稿を行う場合、Cron配送との二重投稿を防止してください。
- 現在セッションのスキル一覧はキャッシュされる場合があります。更新確認は新規セッションまたはファイル実体で行ってください。
