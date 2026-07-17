# osint-agent (v0.2.0)

統合 OSINT エージェント — SitDeck + World Monitor Free + Computer Use + 多層 Web 検索。

## Enable

```powershell
# プラグイン一式 + toolsets（osint_agent / web / search / computer_use 含む）
hermes osint-agent stack enable

# Computer Use ランタイム確認（cua-driver）
hermes computer-use install
hermes computer-use doctor

# 個別に toolset を足す場合
hermes tools enable computer_use
hermes tools enable web
hermes tools enable osint_agent
```

疎通:

```powershell
hermes osint-agent status
hermes osint-agent brief --slot morning
```

## Operator workflow（推奨順）

### 1. Computer Use playbook → 手動 UI

```text
osint_agent_computer_use_plan  (target=all|worldmonitor|sitdeck)
  → computer_use で実行
```

| Target | URL | 用途 |
|--------|-----|------|
| WorldMonitor | https://worldmonitor.app/ | Free UI / 地図・アラートの目視 |
| SitDeck | https://app.sitdeck.com/ + Global Pulse | ダッシュボード目視・ログイン確認 |

Slash: `/osint-agent cu`

### 2. 多層収集プラン + web_search

```text
osint_agent_multilayer_collect  (topic=…, fetch_wm_free=true)
  → L5 queries を web_search で実行
  → 必要なら L4 を computer_use で補強
  → osint_agent_brief にマージ
```

層構成:

| Layer | 手段 |
|-------|------|
| L1 | WM Free JSON (`free_snapshot` / `worldmonitor_free_crawl`) |
| L2 | 官公庁 RSS（allowlist） |
| L3 | SitDeck（Playwright crawl **または** CU） |
| L4 | Computer Use（WM 手動 UI） |
| L5 | `web_search`（toolset=`web`） |
| L6 | Shinka / MoA（任意・MILSPEC） |

Slash: `/osint-agent multilayer`

### 3. 統合ブリーフ

`osint_agent_brief` — PDB + WM Free + SitDeck + 官公庁 RSS + 厚労省 + 多層メモ。

## Free JSON / Playwright vs Computer Use

| 経路 | いつ使う | ツール |
|------|----------|--------|
| **WM Free JSON** | 一括ヘッドライン・機械可読 | `osint_agent_multilayer_collect` / brief の `include_wm_free` |
| **SitDeck Playwright** | 認証済みヘッドレス巡回（cron/brief） | `osint_agent_brief` → sitdeck-osint crawl |
| **Computer Use** | SPA/地図の目視、crawl 失敗時、オペレータ確認 | `osint_agent_computer_use_plan` → `computer_use` |

要点:

- バルクデータは Free JSON / Playwright を優先（速い・再現しやすい）。
- CU は「目視・UI 検証」用。有料チェックアウトや秘密のチャット出力は禁止。
- 根拠捏造禁止。URL + アクセス日を残す。

## Tools

| Tool | 役割 |
|------|------|
| `osint_agent_status` | スタック準備状況 |
| `osint_agent_brief` | 統合 MD/JSON |
| `osint_agent_computer_use_plan` | CU playbook |
| `osint_agent_multilayer_collect` | L1–L6 プラン + 任意 WM Free |

## Cron

```powershell
hermes osint-agent cron install --deliver telegram,discord
# dry-run:
hermes osint-agent cron install --dry-run --deliver local
```
