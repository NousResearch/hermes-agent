---
name: warashibe-reselling
description: "わらしべ長者式せどり：1万円スタート→ブラウザ自動化でリサーチ/仕入れ/出品/発送/古物台帳を回すエンドツーエンドスキル"
version: 1.1.0
author: hakua
license: MIT
platforms: [windows, linux, macos]
metadata:
  hermes:
    tags: [reselling, sedori, browser-automation, antique-dealer, moa, kanban]
    category: autonomous-ai-agents
    related_skills: [ai-employee-org, hermes-agent, moa-fugu-orchestrator]
---

# Warashibe Reselling Skill

## サブコマンド
| コマンド | 内容 |
|---|---|
| `hermes warashibe status` | 全体ステータス表示 |
| `hermes warashibe license -o <dir>` | 古物商許可申請パッケージ生成 |
| `hermes warashibe reroll --execute` | AI社員をsedori- prefixでリロール |
| `hermes warashibe research -k "Switchソフト" -b 10000` | リサーチ試算シート生成 |
| `hermes warashibe price -k "Switchソフト" --platforms mercari yahoo_auction --dry-run` | CloakBrowserで公開価格調査（dry-runはURLのみ） |
| `hermes warashibe price -k "Switchソフト" --platforms mercari` | CloakBrowser実ブラウザで価格取得 |
| `hermes warashibe shipping` | 発送API検証 |
| `hermes warashibe platforms` | プラットフォーム比較表 |
| `hermes warashibe sop -o <dir>` | 梱包/返品SOPテンプレ生成 |
| `hermes warashibe ledger` | 古物台帳CSV初期化 |
| `hermes warashibe profit --buy 1000 --sell 2000` | 利益計算 |

## Gateway Slash
`/warashibe status` `/warashibe license` `/warashibe reroll` `/warashibe research -k "キーワード"` `/warashibe price -k "キーワード"` `/warashibe shipping` `/warashibe platforms` `/warashibe sop` `/warashibe ledger` `/warashibe profit --buy 1000 --sell 2000`

## CloakBrowser価格調査（v1.1.0）
- ログイン不要の公開ページだけを低頻度で調査する。購入・出品・フォーム送信は行わない。
- Amazon JPはスクレイピングせず、SP-APIなど公式手段を使う。
- `computer-use`（cua-driver OS GUI層）とクラウド`browser-use`（Browserbase CDP）は変更せず維持。
- CloakBrowser v0.4.10 がローカルWebページ操作の既定エンジン。`hermes warashibe price` が該当。
- メルカリ・ヤフオクなどの価格比較は `search_markets()` を経由。各プラットフォームのセレクタは `price_research.py` のSELECTORS定数で管理。

## Kanban構成
```
Board: sedori-ops
Backlog → Research → Buy Decision → Purchased → Inspection → Listing → Sold → Shipped → Closed
```

## プロファイル
| ロール | 担当 |
|---|---|
| sedori-secretary | 全体調整・利益判断・人間承認ゲート |
| sedori-researcher | メルカリ/ヤフオク/Amazonリサーチ・利益試算 |
| sedori-buyer | 仕入れ実行・支払い・在庫登録 |
| sedori-lister | 出品下書き・画像最適化・価格改定 |
| sedori-shipper | 発送ラベル・追跡番号・購入者通知 |
| sedori-ledger | 古物台帳・損益通算・確定申告データ |

## プラットフォーム別手数料
| プラットフォーム | 手数料 | API |
|---|---|---|
| メルカリ | 10% | 非公式(規約リスク) → browser-use推奨 |
| ヤフオク | 8.8% | Yahoo!オークションAPI(要申請) |
| Amazon JP | 8〜15% | SP-API(プロ出品者) |
| eBay | 12.35%+$0.30 | Trading/Browse/Feed API |
| ブックオフ | 買取専門 | 出品不可 |
| ハードオフ | 買取専門 | 出品不可 |
| 楽天ラクマ | 6% | 非公式 |
| PayPayフリマ | 5% | 非公式 |

## 発送API
| API | 用途 | 必須env |
|---|---|---|
| ヤマトB2 | 宅急便/ネコポス | YAMATO_B2_CLIENT_ID/SECRET |
| ゆうプリ | ゆうパケット/ゆうパック | YUUPRI_CLIENT_ID/SECRET |
| ネコポス | ネコポス専用 | B2に含む |

## プラットフォーム別規約リスク
- **Amazon**: スクレイピング厳禁。SP-APIのみ。
- **メルカリ**: 非公式API・スクレイピング=規約違反。browser-use(人間模倣)が安全。
- **ヤフオク**: API要申請。browser-useはグレーゾーン。
- **eBay**: 公式API充実。browser-useも可。

## Pitfalls
- **古物商許可未取得での販売は法律違反**（3年以下懲役or100万円以下罰金）
- **メルカリ規約違反でアカウントBAN** → browser-useで人間模倣操作に留める
- **Amazonスクレイピングは即BAN+法的リスク** → SP-APIのみ使用
- **古物台帳の記帳漏れ** → kanban completeフックで自動追記
- **確定申告** → 売上-経費=利益、白色申告（65万円控除）or青色申告（65万+10万控除）
