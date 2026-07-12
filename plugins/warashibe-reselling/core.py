"""
Warashibe Reselling — core logic
わらしべ長者式せどり プラグイン コア
"""
from __future__ import annotations
import json, os, pathlib, shutil, subprocess, sys, textwrap
from typing import Any

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent
_HERMES_HOME_ENV = os.environ.get("HERMES_HOME")
HERMES_HOME = pathlib.Path(_HERMES_HOME_ENV) if _HERMES_HOME_ENV else (pathlib.Path.home() / ".hermes")
HERMES_REPO = pathlib.Path(os.environ.get("HERMES_REPO", "C:/Users/downl/Documents/New project/hermes-agent"))

# ── Platforms ─────────────────────────────────────────────────────────
PLATFORMS = {
    "mercari":     {"name": "メルカリ",       "fee_pct": 10,  "fee_flat": 0,   "shipping": "ゆうゆう/らくらく", "api": "非公式(規約リスク)", "scrape": "browser-use推奨"},
    "yahoo_auction":{"name": "ヤフオク!",      "fee_pct": 8.8, "fee_flat": 0,   "shipping": "簡単配送(ヤマト/日本郵政)", "api": "Yahoo!オークションAPI(要申請)", "scrape": "browser-use可"},
    "amazon_jp":   {"name": "Amazon JP",     "fee_pct": 8,   "fee_flat": 0,   "shipping": "FBA/自己発送",       "api": "SP-API(プロ出品者)", "scrape": "不可(厳禁)"},
    "ebay":        {"name": "eBay",          "fee_pct": 12.35, "fee_flat": 45, "shipping": "eBay International Shipping", "api": "Trading/Browse/Feed API", "scrape": "browser-use可(公開検索のみ)"},
    "bookoff":     {"name": "ブックオフOL",   "fee_pct": 0,   "fee_flat": 0,   "shipping": "宅配買取(着払い)",   "api": "買取専門(出品不可)", "scrape": "不要"},
    "hardoff":     {"name": "ハードオフ",     "fee_pct": 0,   "fee_flat": 0,   "shipping": "宅配買取(着払い)",   "api": "買取専門(出品不可)", "scrape": "不要"},
    "rakuma":      {"name": "楽天ラクマ",     "fee_pct": 6,   "fee_flat": 0,   "shipping": "らくらくラクマ便",   "api": "非公式", "scrape": "browser-use可"},
    "paypay":      {"name": "PayPayフリマ",   "fee_pct": 5,   "fee_flat": 0,   "shipping": "ゆうゆく/PayPay便",  "api": "非公式", "scrape": "browser-use可"},
}

# ── KPI Defaults ─────────────────────────────────────────────────────
KPI_DEFAULTS = {
    "min_profit_yen": 500,        # KPI: absolute minimum profit per unit
    "min_profit_rate": 0.30,      # KPI: minimum profit margin (30%)
    "max_turnaround_days": 14,    # operational: inventory days target
    "packaging_cost": 80,         # cost model: shipping supplies
    "listing_time_cost": 0,       # automation assumes zero manual time
}

# ── Legacy DEFAULTS (kept for backward compat) ──────────────────────
DEFAULTS = {
    "budget_yen": 10000,
    "min_profit_rate": KPI_DEFAULTS["min_profit_rate"],
    "min_profit_yen": KPI_DEFAULTS["min_profit_yen"],
    "max_turnaround_days": KPI_DEFAULTS["max_turnaround_days"],
    "packaging_cost": KPI_DEFAULTS["packaging_cost"],
    "listing_time_cost": KPI_DEFAULTS["listing_time_cost"],
}

def calc_profit(buy_price: int, sell_price: int, platform: str = "mercari",
                shipping_out: int = 0, packaging: int = 80) -> dict:
    """利益計算"""
    p = PLATFORMS.get(platform, PLATFORMS["mercari"])
    fee = int(sell_price * p["fee_pct"] / 100) + p["fee_flat"]
    total_cost = buy_price + shipping_out + packaging + fee
    profit = sell_price - total_cost
    rate = profit / sell_price if sell_price else 0
    return {
        "buy_price": buy_price,
        "sell_price": sell_price,
        "platform_fee": fee,
        "shipping_out": shipping_out,
        "packaging": packaging,
        "total_cost": total_cost,
        "profit": profit,
        "profit_rate": round(rate, 4),
        "go": rate >= 0.30 and profit >= 500,
    }

# ── License Package ────────────────────────────────────────────────────
LICENSE_DOCS = [
    ("申請書_様式第1号.md",       "古物営業許可申請書（東京都公安委員会）"),
    ("履歴書_様式第2号.md",       "履歴書"),
    ("誓約書_様式第3号.md",       "誓約書"),
    ("身分証明書_チェックリスト.md", "身分証明書・住民票取得チェックリスト"),
    ("営業所図面テンプレート.md",  "営業所（自宅）平面図テンプレート"),
    ("警察署相談予約メール.md",    "警察署生活安全課 相談予約メール文面"),
    ("必要書類チェックリスト.md",  "全必要書類チェックリスト"),
    ("申請手数料メモ.md",         "手数料・収入証紙情報"),
]

LICENSE_CHECKLIST = """\
# 古物商許可申請 必要書類チェックリスト
# 東京都公安委員会（警視庁生活安全総務課 古物係）管轄

## 申請先
- 警視庁本部 生活安全総務課 古物係（申請窓口）
- または管轄警察署（日野警察署等）生活安全課経由
- 住所: 〒100-8929 東京都千代田区霞が関2-1-1
- TEL: 03-3581-4321（代表）

## 必要書類（個人申請の場合）

### 1. 申請書（様式第1号）
- [ ] 警視庁HPからダウンロード・記入
- [ ] 署名・押印

### 2. 履歴書（様式第2号）
- [ ] 5年分の略歴
- [ ] 署名・押印

### 3. 誓約書（様式第3号）
- [ ] 欠格事由に該当しない旨の誓約
- [ ] 署名・押印

### 4. 身分証明書
- [ ] 本籍地市町村長発行の身分証明書（3ヶ月以内）
- [ ] ※本籍地以外の住所に住民票がある場合：居住地市町村長発行も可

### 5. 住民票抄本
- [ ] マイナンバー未記載のもの（3ヶ月以内）
- [ ] 本籍地記載のもの

### 6. 営業所の図面
- [ ] 間取り図（手書き可）※自宅営業所の場合
- [ ] 賃貸借契約書の写し（賃貸の場合）または登記事項証明書（所有の場合）

### 7. 営業方法書
- [ ] URL・SNS等のネット販売説明書
- [ ] 取扱商品カテゴリ

### 8. 手数料
- [ ] 収入証紙 19,000円（東京都分）
- [ ] ※郵送申請の場合：現金書留で送付

### 9. その他（ネット専業の場合）
- [ ] サーバー/ドメインの契約証明書（ある場合）
- [ ] 運営規定（ある場合）

## 申請後の流れ
1. 窓口で受理（即日）
2. 審査期間：約40〜50日
3. 実地確認（立入検査）：事前連絡あり
4. 許可証交付（5年有効）
5. 標識（縦18×横27cm以上）を営業所に掲示

## 許可取得後の義務
- 古物台帳の作成・保存（3年間）
- 盗品発見時の届出義務
- 変更届（14日以内）

## 日野警察署 連絡先（相談窓口）
- 住所: 〒191-8501 東京都日野市神川4-15-1
- 電話: 042-581-0110
- 生活安全課 直通: 042-581-0143
"""

LICENSE_EMAIL = """\
件名: 古物営業許可申請についてのご相談予約

{警察署名} 生活安全課 御中

突然のお手紙（メール）失礼いたします。
下記の内容で古物営業許可の申請を予定しており、事前相談をお願いしたくご連絡いたしました。

【申請者情報】
・氏名: {申請者氏名}
・住所: 〒{郵便番号} {住所}
・電話番号: {電話番号}
・メール: {メールアドレス}

【営業予定内容】
・営業形態: 個人（ネット専業）
・取扱商品: {取扱商品カテゴリ}
・販売先: 主にインターネットオークション・フリマアプリ
・営業所: 自宅（上記住所）

【相談希望事項】
1. 申請書類の確認・記入上の注意点
2. ネット専業での営業所としての要件
3. 申請から許可交付までの標準的な期間
4. その他必要な手続き

ご多忙の折に恐縮ですが、ご都合のよい日時をお知らせいただけますと幸いです。
よろしくお願い申し上げます。

{申請者氏名}
〒{郵便番号} {住所}
TEL: {電話番号}
Email: {メールアドレス}
"""

LICENSE_FLOORPLAN = """\
# 営業所（自宅）平面図テンプレート

## 作成上の注意
- 手書きでOK。A4用紙に間取りを描く
- 間取り: 各部屋の寸法（m単位）
- 「古物取扱場所」「保管場所」を明記
- 玄関・窓の位置を記入

## 記入例（日野市の1Kアパート想定）

```
┌─────────────────────────────┐
│                     │
│   ┌───────────┐   │  ← 玄関
│   │           │   │
│   │  キッチン   │   │
│   │           │   │
│   └───────────┘   │
│                     │
│  ┌──────────────────┐│
│  │                    ││
│  │   居室（6畳）       ││
│  │                    ││
│  │  ★古物保管場所      ││
│  │  （クローゼット内）   ││
│  │                    ││
│  │  ★PC作業スペース     ││
│  │  （出品・記帳）       ││
│  │                    ││
│  └──────────────────┘│
│                     │
└─────────────────────┘

凡例:
  ★ = 古物取扱・保管場所
  ── = 壁・間仕切り
```

## 添付書類
- [ ] 賃貸借契約書の写し（賃貸の場合）
- [ ] 賃貸人の使用許諾書（※必要な場合あり）
"""

def generate_license_package(output_dir: str) -> dict:
    """古物商許可申請パッケージを生成"""
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    # チェックリスト
    p = out / "必要書類チェックリスト.md"
    p.write_text(LICENSE_CHECKLIST, encoding="utf-8")
    written.append(str(p))
    # 相談メール
    p = out / "警察署相談予約メール.md"
    p.write_text(LICENSE_EMAIL, encoding="utf-8")
    written.append(str(p))
    # 図面テンプレ
    p = out / "営業所図面テンプレート.md"
    p.write_text(LICENSE_FLOORPLAN, encoding="utf-8")
    written.append(str(p))
    return {"output_dir": str(out), "files": written, "count": len(written)}

# ── AI Employee Reroll ─────────────────────────────────────────────────
SEDORI_ROLES = {
    "sedori-secretary":  {"desc": "せどり全体調整・タスク分解・利益判断・人間承認ゲート", "model": "moa", "default": "hakuapulse-orchestrator"},
    "sedori-researcher": {"desc": "メルカリ/ヤフオク/Amazon/楽天 スクレイピングリサーチ・利益試算", "model": "moa", "default": "hakuapulse-orchestrator"},
    "sedori-buyer":      {"desc": "仕入れ実行・購入判断・支払い・在庫登録", "model": "moa", "default": "hakuapulse-orchestrator"},
    "sedori-lister":     {"desc": "出品下書き・画像最適化・タイトルSEO・価格改定", "model": "moa", "default": "hakuapulse-orchestrator"},
    "sedori-shipper":    {"desc": "発送ラベル発行・追跡番号登録・購入者通知・梱包指示", "model": "moa", "default": "hakuapulse-orchestrator"},
    "sedori-ledger":     {"desc": "古物台帳自動記帳・損益通算・確定申告データ出力", "model": "moa", "default": "hakuapulse-orchestrator"},
}

def reroll_ai_employees(dry_run: bool = True) -> dict:
    """ai-employees プラグインを sedori- prefix で複製・書き換え"""
    src = PLUGIN_DIR.parent / "ai-employee-org"
    if not src.exists():
        return {"error": f"ai-employee-org not found at {src}"}
    results = {"profiles": [], "kanban": None, "dry_run": dry_run}

    # 1. プロファイル作成
    for name, info in SEDORI_ROLES.items():
        cmd = ["hermes", "profile", "create", name, "--description", info["desc"]]
        if dry_run:
            results["profiles"].append({"name": name, "cmd": " ".join(cmd), "dry_run": True})
        else:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                results["profiles"].append({"name": name, "ok": r.returncode == 0, "out": r.stdout[:200]})
            except Exception as e:
                results["profiles"].append({"name": name, "error": str(e)})

    # 2. Kanbanボード
    cmd = ["hermes", "kanban", "boards", "create", "sedori-ops", "--name", "せどり運営", "--switch"]
    if dry_run:
        results["kanban"] = {"cmd": " ".join(cmd), "dry_run": True}
    else:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            results["kanban"] = {"ok": r.returncode == 0, "out": r.stdout[:200]}
        except Exception as e:
            results["kanban"] = {"error": str(e)}

    return results

# ── Research Sheet ────────────────────────────────────────────────────
def generate_research_sheet(keyword: str, budget: int = 10000, output_path: str = None) -> dict:
    """リサーチ試算シートを生成（Keepa + オークファン想定）"""
    if not output_path:
        output_path = str(pathlib.Path.home() / "Documents" / "ops" / "sedori" / f"research_{keyword}_{budget}yen.csv")
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # ヘッダー
    header = "keyword,platform,item_name,buy_price,sell_price,fee,shipping,packaging,profit,profit_rate,turnaround_days,score,url,retrieved_at\n"
    out.write_text(header, encoding="utf-8")
    return {
        "keyword": keyword,
        "budget": budget,
        "output_path": str(out),
        "note": "browser-use でメルカリ/ヤフオク/Amazon をクロール→ CSV追記する設計。Keepa API・オークファンAPIは .env のキーが必要。",
    }

# ── Shipping Label Verification ───────────────────────────────────────
SHIPPING_APIS = {
    "yamato_b2": {
        "name": "ヤマトB2クラウドAPI",
        "url": "https://cb-api.yamato-hcl.co.jp/api/v1/",
        "auth": "OAuth2 (client_id + client_secret)",
        "env_vars": ["YAMATO_B2_CLIENT_ID", "YAMATO_B2_CLIENT_SECRET"],
        "status": "要契約（ヤマト運輸 B2クラウドAPI 契約書提出が必要）",
    },
    "yuupri": {
        "name": "日本郵便 ゆうプリAPI",
        "url": "https://api.prt.post.japanpost.jp/api/v1/",
        "auth": "OAuth2 (client_id + client_secret)",
        "env_vars": ["YUUPRI_CLIENT_ID", "YUUPRI_CLIENT_SECRET"],
        "status": "要契約（日本郵便 ゆうプリR契約・API利用申請が必要）",
    },
    "necopos": {
        "name": "ヤマトネコポスAPI（B2に含む）",
        "url": "https://cb-api.yamato-hcl.co.jp/api/v1/shipments",
        "auth": "B2と同一",
        "env_vars": ["YAMATO_B2_CLIENT_ID", "YAMATO_B2_CLIENT_SECRET"],
        "status": "B2契約でネコポス発行可。サイズ限定（32×24×3cm）",
    },
}

def verify_shipping_apis() -> dict:
    """発送APIの環境変数・契約状態を検証"""
    results = {}
    for key, info in SHIPPING_APIS.items():
        env_ok = all(os.environ.get(v) for v in info["env_vars"])
        results[key] = {
            "name": info["name"],
            "env_vars_set": env_ok,
            "missing": [v for v in info["env_vars"] if not os.environ.get(v)],
            "status": info["status"],
            "ready": env_ok,
        }
    return results

# ── Platform Comparison ───────────────────────────────────────────────
def platform_comparison() -> dict:
    """プラットフォーム別 手数料・規約・リスク比較"""
    return PLATFORMS

# ── SOP Templates ─────────────────────────────────────────────────────
SOP_PACKING = """\
# 梱包・発送 SOP

## 1. 梱包資材
| 送料サイズ | サイズ目安 | 推奨資材 | コスト |
|---|---|---|---|
| ゆうパケット | 23×34×3cm | クラフト紙袋+緩衝材 | ¥80〜120 |
| ゆうパケットプラス | 34×31×3cm | 専用箱 | ¥150 |
| 宅急便コンパクト | 25×20×5cm | 専用箱 | ¥250 |
| ネコポス | 32×24×3cm | 専用袋/箱 | ¥250 |

## 2. 梱装手順
1. 商品検品（動作確認・傷確認・写真撮影）
2. 緩衝材で包む（プチプチ/緩衝材ロール）
3. 箱/袋に収納（サイズ内であることを確認）
4. 送り状貼り付け（ゆうプリ/ネコポス送り状）
5. 「丁寧に梱包しました」メモ添える（評価アップ）

## 3. 発送方法選択フロー
- 1kg未満・薄型 → ゆうパケット（最安¥250）
- 本・CD → ゆうパケット（厚さ3cm以内）
- 大型・割れ物 → 宅急便（サイズ別料金）
- 海外 → eBay International Shipping / DHL
"""

SOP_CLAIM = """\
# 返品・クレーム対応 SOP

## 1. 初期対応（24時間以内）
1. メッセージ確認 → 謝罪文返信（テンプレ使用）
2. 事実確認（写真要求・状況聞き取り）
3. 責任範囲の判断（出品者責任か配送事故か）

## 2. 返金・返品フロー
1. 返品承諾 → 返送着払いで受領
2. 商品確認後、販売価格全額返金
3. メルカリ: 「取引キャンセル」申請
4. ヤフオク: 「取引ナビ」からキャンセル申請

## 3. 注意事項
- 説明欄に記載した状態と異なる場合は自己責任
- 配送中破損は補償対象（ゆうパック: 最大30万）
- クレーム対応の記録を残す（証拠保全）
- 悪質な場合は事務局通報
"""

def generate_sop_templates(output_dir: str) -> dict:
    """SOPテンプレートを生成"""
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "梱包発送SOP.md").write_text(SOP_PACKING, encoding="utf-8")
    (out / "返品クレームSOP.md").write_text(SOP_CLAIM, encoding="utf-8")
    return {"output_dir": str(out), "files": ["梱包発送SOP.md", "返品クレームSOP.md"]}

# ── Antique Ledger ────────────────────────────────────────────────────
LEDGER_HEADER = "取得日,取得先氏名,取得先住所,品名,品番,数量,取得価格,販売日,販売先,販売価格,利益,プラットフォーム,備考\n"

def init_ledger(path: str = None) -> dict:
    """古物台帳CSVを初期化"""
    if not path:
        path = str(pathlib.Path.home() / "Documents" / "ops" / "sedori" / "kobutsu_ledger.csv")
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(LEDGER_HEADER, encoding="utf-8-sig")
    return {"ledger_path": str(p), "ready": True}

def append_ledger(entry: dict, path: str = None) -> dict:
    """古物台帳に1行追記"""
    if not path:
        path = str(pathlib.Path.home() / "Documents" / "ops" / "sedori" / "kobutsu_ledger.csv")
    p = pathlib.Path(path)
    if not p.exists():
        init_ledger(path)
    row = ",".join([
        entry.get("取得日",""),
        entry.get("取得先氏名",""),
        entry.get("取得先住所",""),
        entry.get("品名",""),
        entry.get("品番",""),
        str(entry.get("数量","")),
        str(entry.get("取得価格","")),
        entry.get("販売日",""),
        entry.get("販売先",""),
        str(entry.get("販売価格","")),
        str(entry.get("利益","")),
        entry.get("プラットフォーム",""),
        entry.get("備考",""),
    ])
    with open(p, "a", encoding="utf-8-sig") as f:
        f.write(row + "\n")
    return {"appended": True, "row": row}
