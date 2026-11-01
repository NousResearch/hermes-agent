# Cron週次実行 + Slack通知の運用メモ

Notionパトロール君を定期実行に配線する場合の再利用パターン。

## 目的

- Notionパトロール本体は読み取り専用のまま維持する。
- Cron側では薄いラッパースクリプトを使い、標準出力にSlack通知本文を出す。
- `hermes cron create ... --no-agent --deliver slack:<channel_id>` で、スクリプト出力をそのままSlackへ配送する。
- Slack上での視認性確認にはBlock Kit形式のテスト通知を使える。ただし本番CronでSlack API直接投稿を使う場合は、Hermes Cron配送との二重投稿を抑止する。

## 現行運用値

| 項目 | 値 |
|---|---|
| ジョブ名 | `Notionパトロール君 週次リンクチェック` |
| 通知先 | `slack:G01ACN6N2HW` |
| メンション | `<!subteam^S02FLCBKU0P>` |
| 実行時刻 | 毎週月曜 11:00 JST |
| Cron式（UTC運用時） | `0 2 * * 1` |
| 実行モード | `--no-agent` |
| プロファイル | `default` |
| ラッパー | `~/.hermes/scripts/notion_patrol_weekly.py` |
| 出力先 | `~/.hermes/cron/output/notion-patrol` |

## 推奨手順

1. パトロール本体の健全性を確認する。

```bash
node --test /home/kawazoe_taishi/.hermes/skills/productivity/notion-patrol/scripts/patrol.test.js
node --check /home/kawazoe_taishi/.hermes/skills/productivity/notion-patrol/scripts/patrol.js
node /home/kawazoe_taishi/.hermes/skills/productivity/notion-patrol/scripts/patrol.js --help
```

2. Cron用ラッパーは `~/.hermes/scripts/` に置く。

- Notionトークンは環境変数または `~/.hermes/.env` から読む。
- トークン値はログ・Slack本文・スキルファイルに出さない。
- 通知本文には、メンション、対象、チェック件数、OK/NG件数、CSVパス、NGリンクの抜粋を含める。
- CSV添付は `MEDIA:<csv_path>` で指定する。

3. 先にテスト通知を出力して、Slack上の見た目を確認する。

```bash
python ~/.hermes/scripts/notion_patrol_weekly.py --test-notification
python -m py_compile ~/.hermes/scripts/notion_patrol_weekly.py
```

4. Cron登録する。

```bash
hermes cron create '0 2 * * 1'   --name 'Notionパトロール君 週次リンクチェック'   --deliver 'slack:G01ACN6N2HW'   --script notion_patrol_weekly.py   --no-agent   --profile default
```

5. 登録後に必ず確認する。

```bash
hermes cron list --all
```

確認する項目:

- `Next run`
- `Deliver`
- `Script`
- `Profile`
- `Status`

## タイムゾーン注意

Hermes Cronの時刻解決は、実行環境の設定に依存する。少なくとも以下を確認する。

```bash
python - <<'PY'
import os
print(os.environ.get('HERMES_TIMEZONE') or 'missing')
PY
```

`HERMES_TIMEZONE` や `config.yaml` の `timezone:` が未設定で、ホスト時刻がUTCとして運用されている場合、JST月曜11:00はUTC月曜02:00なので、Cron式は次のようにする。

```text
0 2 * * 1
```

JSTとして解決される環境では次の式になる。

```text
0 11 * * 1
```

`hermes cron list --all` の `Next run` を見て、意図した時刻になっているか確認する。

## Slack通知形式の例

```text
<!subteam^S02FLCBKU0P>

【Notionパトロール君】週次リンクチェック結果

実行日時: YYYY-MM-DD HH:MM:SS UTC

対象:
- 法務内マニュアル（106173b008788028ac4efd380a88308c）
- 法務関連DB（8db170cdc5ba4c488ef9302e4e58cede）

チェック件数: <n>
OK: <n>
NG: <n>

CSV: `<path>`
MEDIA:<path>

NGリンク（最大20件）:
- [<status>] <url> / <page> / <context>
```

## Block Kit通知を使う場合

Block Kit形式のテスト通知はSlack API応答 `ok: true` で成功確認済み。実装する場合は次を守る。

- fallback `text` にも必ず `<!subteam^S02FLCBKU0P>` を含める。
- blocksの先頭セクションにもメンションを含める。
- header、divider、summary fields、NGリンク一覧、context noteの構成にする。
- Slack APIで直接投稿に成功した場合、Hermes Cronの通常配送を `[SILENT]` または `{"wakeAgent": false}` で抑止する。
- 直接投稿が失敗した場合は、標準出力に通常テキスト通知を出してHermes Cron配送にフォールバックする。
- Slack token値は出力・保存しない。

## Pitfalls

- Notion書き込みはしない。Cronラッパーも巡回・集計・通知に限定する。
- `.env` や環境変数の秘密値をスキル、ログ、Slack本文に保存しない。
- `--no-agent` を使う場合、Slackに届く本文はスクリプトの標準出力そのものになる。文面確認用の `--test-notification` を先に用意する。
- JST指定の依頼でも、Cron実行環境がUTCなら `0 11 * * 1` ではなく `0 2 * * 1` が必要になる。
- Slack APIでBlock Kitを直接投稿する場合、Hermes Cron配送と二重投稿しない。
