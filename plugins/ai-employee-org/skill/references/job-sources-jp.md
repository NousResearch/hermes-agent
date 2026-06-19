# 求職監視先（日本）

永続ワークスペース: `dir:C:/Users/downl/Documents/ops/job-seeker`（Kanban タスクに必ず指定。scratch は完了時に消える）。

## 監視キーワード

| カテゴリ | キーワード |
|----------|------------|
| AIエンジニア | `AIエンジニア`, `AI Engineer`, `生成AI`, `LLM`, `RAG` |
| 機械学習 | `機械学習エンジニア`, `MLエンジニア`, `MLOps`, `深層学習` |
| クラウドワークス（外注） | `ソフトウェア開発`, `Python`, `Webアプリ`, `AI開発`, `業務システム` |

## サイト別手順

### ビズリーチ (BizReach)

- ログイン必須 → `browser_navigate` で `https://www.bizreach.jp/` またはスカウト/求人メールを Gmail で追う。
- Gmail: `from:(bizreach.jp OR ビズリーチ) (スカウト OR 求人 OR AI OR 機械学習) newer_than:3d`
- `web_search`: `site:bizreach.jp AIエンジニア OR 機械学習`

### ファインディ (Findy)

- `https://findy-code.io/` — エンジニア向け。ログイン後のおすすめ・マッチ通知を Gmail で補完。
- Gmail: `from:findy-code.io (求人 OR マッチ OR スカウト) newer_than:3d`
- `web_search`: `site:findy-code.io AIエンジニア OR MLOps`

### LAPRAS（表記: LAPLAS / Lapras）

- API: `LAPRAS_API_KEY` を `.env` に設定済みなら `scripts/lapras_jobs.py search` を実行。
- ブラウザ: `https://lapras.com/` ログイン後、求人・スカウト画面を `browser_navigate` + `vision_analyze`。
- Gmail: `from:lapras.com OR from:lapras.jp newer_than:3d`

### クラウドワークス（ソフトウェア外注）

- `https://crowdworks.jp/` — 「ソフトウェア開発」「システム開発」「AI」カテゴリ。
- `web_search`: `site:crowdworks.jp ソフトウェア 開発 募集`
- `browser_navigate` で案件一覧を開き、単価・納期・リモート可否を `seen.json` に追記。

## Gmail（google-workspace skill）

```bash
export PYTHONPATH="C:/Users/downl/Documents/New project/hermes-agent"
GAPI="py -3 C:/Users/downl/.hermes/skills/productivity/google-workspace/scripts/google_api.py"

$GAPI gmail search '(ビズリーチ OR bizreach OR findy OR lapras OR クラウドワークス OR crowdworks) (AI OR 機械学習 OR MLOps OR スカウト OR 求人) newer_than:3d' --max 20
$GAPI gmail search '(面談 OR 面接 OR interview OR オファー) newer_than:7d' --max 10
```

himalaya CLI は使わない（Windows cron で PATH/OAuth が壊れやすい）。

## 重複排除（idempotency）

`C:/Users/downl/Documents/ops/job-seeker/seen.json` に URL または `employer|title|posted_date` ハッシュを記録。既出は `kanban_create` しない。

## 人間承認ゲート

- 応募送信・スカウト返信・クラウドワークス入札 → 必ず `kanban_block(reason="要承認: ...")`。
- ドラフトのみ `kanban_complete` 可。成果物は `applications/` と `reports/` 配下に保存。

## モデル（429 時）

1. 主: **NVIDIA** `nvidia/nemotron-3-super-120b-a12b`
2. 429 後: Nous Free 連鎖（`config.yaml` の `fallback_providers`）
3. Nous Free も尽きたらのみ: `http://127.0.0.1:8080/v1` の llama（最終ロールバック）

非同期サブエージェントは親の同じチェーンを継承する。
