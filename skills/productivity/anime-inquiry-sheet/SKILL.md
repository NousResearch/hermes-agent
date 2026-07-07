---
name: anime-inquiry-sheet
description: Use when the user asks whether an anime/manga/game work is recommended for inquiry, or provides a source URL containing multiple works; dedupe against the Google Sheet, fill new rows, then answer with conclusion, reason, and sheet URL.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [anime, licensing, google-sheets, research, japanese]
    related_skills: [google-workspace]
---

# Anime Inquiry Sheet

## Overview

指定された作品タイトルを調査し、Google Sheets `作品リストfor問い合わせ` に問い合わせ判断用の1行を作る。

対象スプレッドシート:

- Sheet ID: `1ZZ9qF5UFr2GGO5IGiWJCBgcUqAhu3PDEQYD-m5NlD-g`
- Sheet name: `シート1`

## When to Use

- ユーザーが作品タイトルを出して「調べて」「シートに入れて」「問い合わせ判断して」と依頼したとき。
- ユーザーが「『鬼滅の刃』という作品は問い合わせ推奨？」のように、作品の問い合わせ推奨度を聞いたとき。
- ユーザーがAnimeTimes等のURLを共有し、「この中のすべての作品を問い合わせ判断して」と依頼したとき。
- 漫画・ゲーム・アニメ・小説などのIPについて、SoulChat等の問い合わせ優先度を整理するとき。

## Columns

必ずこの順番で埋める。

1. `タイトル` — 作品の正式タイトル。
2. `原作の形態` — 漫画、ゲーム、アニメ、小説など。
3. `原作所有社` — 集英社、講談社、KADOKAWAなど。出版社・権利元・原作会社を優先。
4. `原作販売数` — 例: `200万部`。不明なら `調査してもヒットしなかった`。
5. `アニメステータス` — 例: `2027年1月に放送予定`, `アニメ化予定なし`, `2期まで放送済み＆3期予定なし`。
6. `アニメXフォロワー数` — 例: `12.4万人`。公式Xがない/不明ならその旨を書く。
7. `主なファン層` — 女性、10代女性、男性など。根拠が弱ければ `要判断` を含める。
8. `ジャンル` — 恋愛、BL、バトル、歴史など。
9. `あらすじ` — 作品内容が分かる短い説明。
10. `問い合わせ推奨度` — `推奨` / `非推奨` / `要判断` のいずれか。
11. `(非)推奨理由` — 推奨・非推奨・要判断にした理由。
12. `関連するURL` — 調査で参照したURLを基本的に全て入れる。アニメ公式XアカウントURLがあれば必ず含める。
13. `問い合わせ先` — アニメ化済みならアニメ会社。未アニメ化なら原作会社の商品化・タイアップ問い合わせURL。
14. `調査時点` — 調査した日付・時刻。

不明項目は空欄にせず、`調査してもヒットしなかった` のように書く。

## Decision Rules

### 共通

- アニメ公式Xアカウントのフォロワー数が `50万人超` の場合は、他条件に関わらず `要判断`。

### アニメリリース前

- アニメ公式Xフォロワー数が `1万人以上` なら `推奨`。
- 女性ファンが多い内容なら `5000人以上` で `推奨`。
- 集英社作品はフォロワー数に関わらず原則 `推奨`。
- 上記以外は基本 `非推奨`。

### アニメリリース後・放送中

- アニメ公式Xフォロワー数が `9万人以上` なら `推奨`。
- 女性ファンが多い内容なら `5万人以上` で `推奨`。
- 放送中で、上記の推奨基準には満たないが `1万人以上` なら `要判断`。
- 上記の推奨基準を満たしていても、原作が集英社の場合は `要判断`。
- 上記以外は基本 `非推奨`。

## Speed Rules

- 通常調査は `3〜5分以内` を目標に完了する。
- 判定に必要な十分条件が揃ったら早期終了し、細部を深追いしない。
- 5分以内に取れない項目は空欄にせず `調査してもヒットしなかった` または `短時間調査では確認できなかった` と書く。
- 検索エンジンは最後の手段。まず公式サイト、出版社/権利元、アニメ公式、公式X、既知の信頼ソースを直接確認する。
- 取得はできるだけ並列で行う。公式サイト・出版社/権利元・公式X・販売数/ニュース・問い合わせ先を順番待ちにしない。

### Research Priority

1. アニメ公式Xフォロワー数
2. アニメステータス
3. 原作所有社
4. 公式X URL
5. 問い合わせ先
6. 原作販売数、ファン層、詳細URL

### Early Exit Conditions

- 公式Xフォロワー数が `50万人超` → 即 `要判断`。
- アニメリリース前かつ集英社作品 → 即 `推奨`。
- 放送中で、推奨基準未満だが公式Xフォロワー数が `1万人以上` → 即 `要判断`。
- 早期終了後も14カラムは埋めるが、不明項目は深追いせず上記の不明テキストで処理する。

## Duplicate Check Before Research

作品調査を始める前に、単作品依頼でもURL一括依頼でも必ず対象Spreadsheetを一度だけ読み込み、既存調査済みタイトルを確認する。

1. `シート1!A:N` を読み込む。
2. A列 `タイトル` から `正規化タイトル → 行番号` の索引を作る。
3. 正規化ルール:
   - 前後空白除去。
   - Unicode NFKC正規化。
   - `「」『』\"'` 等の引用符を除去。
   - 連続空白を1つに圧縮。
   - 英字は小文字化。
4. 完全一致なら調査・追記しない。単作品依頼でもその時点で停止し、ユーザーへ `「<タイトル>」は調査済みです。Spreadsheetの<行番号>行目にあります。` と伝える。
5. 強い類似一致なら重複候補として扱い、原則追記せずユーザーに候補行番号を伝える。
6. 単作品依頼では、既存一致なしを確認してから調査を開始する。
7. 一括URLモードでは、最初に1回だけ索引を作り、全候補を `調査済み` / `新規` / `重複候補` に分類してから新規だけ調査する。

## URL Batch Mode

ユーザーがAnimeTimes等のURLを共有し、「この中のすべての作品を問い合わせ判断して」と依頼した場合はURL一括モードで処理する。

1. **URL取得** — ページ本文を取得する。生HTML全体を会話コンテキストへ貼らない。
2. **候補抽出** — 作品タイトル候補だけを抽出する。AnimeTimesでは `tag/details.php?id=...` の作品タグを優先し、ニュース記事・声優名・季節まとめ・カテゴリ・ナビゲーションを除外する。
3. **候補整形** — 重複を消し、明らかな非作品を除外する。抽出が曖昧な場合だけ、候補一覧をユーザーへ確認する。
4. **既存チェック** — Duplicate Check Before Researchを必ず先に行う。
5. **新規のみ調査** — 調査済み・重複候補は追記しない。新規作品だけ単作品Workflowに渡す。
6. **バッチ追記** — 複数の新規行は可能な限りまとめてappendし、追加範囲の背景色クリア・文字色黒をまとめて適用する。
7. **最終報告** — 新規追加数、調査済みスキップ数、重複候補、既存行番号、Spreadsheet URLを簡潔に返す。

## Context Management for Batch Mode

一括URL処理では、メイン会話コンテキストを膨らませない。

- メインセッションは **オーケストレーター** として、URL抽出結果・既存行索引・最終追記結果だけを保持する。
- 各作品の詳細調査は、件数が多い場合は別セッション/サブタスクまたは小さなチャンクで処理する。目安は新規5件以下なら同一セッション、6件以上なら3〜5件単位に分割する。
- 別セッション/サブタスクに渡す情報は `作品タイトル`, `元URL`, `14カラム定義`, `Decision Rules`, `Speed Rules` のみに絞る。
- 別セッション/サブタスクにはSpreadsheetへ書き込ませない。返却は14カラムJSON、推奨度、理由、参照URLだけにする。
- Spreadsheetの読み取り・重複判定・append・書式調整・読み返し検証はメインセッションだけが行う。
- 生HTML、検索結果全文、長い調査ログは会話に入れない。必要なら一時ファイルやスクリプト内で処理し、会話には抽出タイトルリストと最終行データだけを残す。
- 各作品の調査メモは最終的な1行データに圧縮し、調査途中のページ本文は保持しない。

## Workflow

1. **入力判定** — 作品タイトル単体か、複数作品を含むURLかを判定する。同名作品が複数ある場合だけ確認する。
2. **既存チェック** — Duplicate Check Before Researchを実行する。単作品でも必ず先に確認し、既存一致したら調査せず行番号を返して完了する。
3. **URL一括処理** — URL入力ならURL Batch Modeに従い、候補抽出・既存分類・新規のみ調査へ進む。
4. **高速調査** — Speed Rulesに従い、公式サイト・出版社/権利元・アニメ公式・公式X・販売数/ニュース・問い合わせ先を並列で調べる。
5. **早期判断** — Early Exit Conditionsに該当したら即判断し、残りの項目は深追いしない。
6. **根拠整理** — 各カラムに対応する値と参照URLをまとめる。不明項目は不明テキストで埋める。
7. **判断** — Decision Rulesで `推奨` / `非推奨` / `要判断` を決める。
8. **行作成** — 14カラム順の1行を作る。
9. **書き込み** — Google Workspace skillを使い、対象シートに追記する。このスキルでは、ユーザーの「問い合わせ推奨？」という依頼自体をシート追記の承認として扱い、追加確認なしで先に記入する。
10. **書式調整** — 追記行の背景色を必ずクリアし、文字色を黒にする。ヘッダー行と同じ背景色や文字色をデータ行へ継承させない。
11. **検証** — 追記後に同じ範囲を読み返し、行が入ったことを確認する。
12. **回答** — ユーザーに結論、理由、Spreadsheet URLを簡潔に伝える。

## User-Facing Response

シート追記・検証後、ユーザーにはこの形式で返す。

```text
結論：<推奨 / 非推奨 / 要判断>
理由：<Decision Rulesと調査結果に基づく短い理由>
Spreadsheet：https://docs.google.com/spreadsheets/d/1ZZ9qF5UFr2GGO5IGiWJCBgcUqAhu3PDEQYD-m5NlD-g/edit?usp=sharing
```

長い調査ログは不要。必要なら「主な根拠URL」だけ追加する。

単作品で既存調査済みだった場合は、調査・追記せずこの形式で返す。

```text
「<タイトル>」は調査済みです。Spreadsheetの<行番号>行目にあります。
Spreadsheet：https://docs.google.com/spreadsheets/d/1ZZ9qF5UFr2GGO5IGiWJCBgcUqAhu3PDEQYD-m5NlD-g/edit?usp=sharing
```

URL一括モードではこの形式で返す。

```text
完了。

新規追加：<件数>件
調査済みのためスキップ：<件数>件
重複候補：<件数>件

調査済み：
- <タイトル>：<行番号>行目

重複候補：
- <タイトル>：既存の「<既存タイトル>」<行番号>行目と関連可能性あり

Spreadsheet：https://docs.google.com/spreadsheets/d/1ZZ9qF5UFr2GGO5IGiWJCBgcUqAhu3PDEQYD-m5NlD-g/edit?usp=sharing
```

## Google Sheets Commands

読み取り:

```bash
GAPI="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/google_api.py"
$GAPI sheets get 1ZZ9qF5UFr2GGO5IGiWJCBgcUqAhu3PDEQYD-m5NlD-g 'シート1!A:N'
```

追記:

```bash
GAPI="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/google_api.py"
$GAPI sheets append 1ZZ9qF5UFr2GGO5IGiWJCBgcUqAhu3PDEQYD-m5NlD-g 'シート1!A:N' --values '[["タイトル", "原作の形態", "原作所有社", "原作販売数", "アニメステータス", "アニメXフォロワー数", "主なファン層", "ジャンル", "あらすじ", "問い合わせ推奨度", "(非)推奨理由", "関連するURL", "問い合わせ先", "調査時点"]]'
```

追記後の背景色クリア・文字色黒:

```bash
# 追記後にA:Nを読み、追加された最終行番号を特定する。
# 例: ROW=2, SHEET_GID=0 の場合、A2:N2の背景色をクリアし文字色を黒にする。
gws sheets spreadsheets batchUpdate \
  --params '{"spreadsheetId":"1ZZ9qF5UFr2GGO5IGiWJCBgcUqAhu3PDEQYD-m5NlD-g"}' \
  --json '{"requests":[{"repeatCell":{"range":{"sheetId":0,"startRowIndex":1,"endRowIndex":2,"startColumnIndex":0,"endColumnIndex":14},"cell":{"userEnteredFormat":{"textFormat":{"foregroundColorStyle":{"rgbColor":{"red":0,"green":0,"blue":0}}}}},"fields":"userEnteredFormat.backgroundColor,userEnteredFormat.backgroundColorStyle,userEnteredFormat.textFormat.foregroundColorStyle"}}]}'
```

`startRowIndex` / `endRowIndex` は0始まり。実行時は追記された行に合わせる。

## Common Pitfalls

1. **URL不足** — 参照URLは省略しない。特に公式X URLを忘れない。
2. **空欄** — 不明なら空欄ではなく `調査してもヒットしなかった` と書く。
3. **フォロワー数の古さ** — Xフォロワー数は調査時点とセットで扱う。
4. **50万人超の見落とし** — 公式Xフォロワー数が50万人超なら必ず `要判断`。
5. **承認待ちで停止** — ユーザーが作品の問い合わせ推奨度を聞いたら、追加承認を待たずにシートへ追記する。
6. **調査の深追い** — 5分以内に取れない情報を探し続けない。十分条件で判断できるなら早期終了する。
7. **検索エンジン依存** — Google/Bing検索を最初に使わない。公式URL・出版社・公式X・既知ソースを先に使う。
8. **直列調査** — 公式サイト、出版社、X、ニュース、問い合わせ先を1つずつ待たない。可能な限り並列化する。
9. **ヘッダー色の継承** — 追記行に背景色やヘッダー由来の文字色が付いたままにしない。追記直後に追加行の `userEnteredFormat.backgroundColor` / `backgroundColorStyle` をクリアし、`textFormat.foregroundColorStyle` を黒にする。
10. **既存チェック漏れ** — 単作品でも一括URLでも、作品調査の前にSpreadsheetを読まないのはNG。既存なら調査せず行番号を返す。
11. **URL内リンクの過剰採用** — AnimeTimes等のURLで声優名・カテゴリ・季節まとめ・ニュース記事を作品として扱わない。
12. **一括処理でのコンテキスト肥大** — 生HTMLや検索結果全文を会話へ入れない。抽出タイトル、14カラムJSON、最終結果だけに圧縮する。
13. **サブタスクの直接書き込み** — 別セッション/サブタスクにSpreadsheetを書かせない。書き込みはメインセッションに集約する。
14. **回答漏れ** — シート追記後は、結論・理由・Spreadsheet URLを必ずユーザーに返す。
15. **同名作品混同** — 同名作品がある場合はユーザー確認。

## Verification Checklist

- [ ] 14カラムが順番通りに埋まっている。
- [ ] 不明項目が空欄ではない。
- [ ] 単作品依頼でも一括URL依頼でも、調査前にSpreadsheet既存行を確認し、既存一致なら行番号を返している。
- [ ] URL一括モードでは候補抽出後に `調査済み` / `新規` / `重複候補` に分類している。
- [ ] 一括処理で生HTML・検索結果全文・長い調査ログを会話コンテキストへ入れていない。
- [ ] 別セッション/サブタスクを使った場合、Spreadsheet書き込みはメインセッションだけが行っている。
- [ ] 通常調査が3〜5分以内、または不明項目を深追いせず処理している。
- [ ] 判定に十分な条件が揃った時点で早期終了している。
- [ ] 公式URL・出版社/権利元・公式Xを検索エンジンより優先している。
- [ ] 関連URLに主要な参照元と公式Xが含まれている。
- [ ] 推奨度がDecision Rulesに沿っている。
- [ ] 調査時点が入っている。
- [ ] 追記行の背景色がクリアされ、文字色が黒になっている。
- [ ] 追記後にシートを読み返して確認した。
- [ ] ユーザーに結論・理由・Spreadsheet URLを返した。
