# Hermes Go Watchdog（Windows）

Hermes Desktop（`Hermes.exe`）と Desktop が spawn する `hermes serve` バックエンドを**相互監視**する独立プロセスです。  
**Hermes Agent の plugin / tool / skill / MCP / cron には一切登録しません。**

## 隔離（AI から制御不可）

| 項目 | 内容 |
|------|------|
| プロセス | Hermes Python/Electron とは別バイナリ |
| 設定 | `%LOCALAPPDATA%\HermesWatchdog\`（ロック・状態 JSON） |
| ログ | `%HERMES_HOME%\logs\hermes-go-watchdog.log` |
| 変更 API | `HERMES_WATCHDOG_ADMIN_TOKEN` 必須（未設定なら **403**） |
| 読取 API | `GET /health`, `GET /api/status`（ローカル / tailnet） |

## ビルド

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\Build-HermesGoWatchdog.ps1
```

成果物: `scripts\windows\watchdog-go\dist\hermes-watchdog.exe`

## 起動

```powershell
# 環境変数（例）
$env:HERMES_WATCHDOG_ADMIN_TOKEN = "<operator-secret>"
$env:HERMES_WATCHDOG_TS_AUTHKEY = "<ts-authkey>"   # 任意: tsnet 有効化

powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\Start-HermesGoWatchdog.ps1
```

### フラグ（Start スクリプト経由）

| フラグ | 既定 | 説明 |
|--------|------|------|
| `-IntervalSec` | 20 | 監視周期 |
| `-FailThreshold` | 2 | backend 連続失敗で Desktop 再起動 |
| `-Once` | off | 1 周期だけ実行して終了 |
| `-NoTsnet` | off | tsnet を強制 OFF |
| `-Listen` | 127.0.0.1:9920 | ローカル HTTP |

## Tailscale（tsnet）

1. Tailscale 管理画面で **auth key** を発行（推奨: reusable + タグ付き）
2. 環境変数 `HERMES_WATCHDOG_TS_AUTHKEY` または `TS_AUTHKEY` に設定（**リポジトリにコミットしない**）
3. 起動すると tailnet 上で `hermes-watchdog` として `:443` で待受
4. 他ノードから: `curl -k https://hermes-watchdog/health`（MagicDNS / ホスト名）

## HTTP API

| Method | Path | 認証 | 説明 |
|--------|------|------|------|
| GET | `/health` | 不要 | 生存確認 |
| GET | `/api/status` | 不要 | ウォッチドッグ状態 JSON |
| POST | `/api/v1/pause` | Admin | 監視一時停止 |
| POST | `/api/v1/resume` | Admin | 監視再開 |
| POST | `/api/v1/cycle` | Admin | 即時 1 周期 |
| POST | `/api/v1/stop` | Admin | Graceful stop |

Admin 認証: `Authorization: Bearer <token>` または `X-Admin-Token: <token>`

## 監視ロジック

1. **起動時 prewarm** — リポジトリ `.venv` で `hermes serve --port 0` を先に立ち上げ、`%LOCALAPPDATA%\HermesWatchdog\desktop-backend.json` に URL/token/port を公開
2. `Hermes.exe` 不在 → 管理 backend は reaping しない → Desktop 起動（manifest があれば `HERMES_DESKTOP_REMOTE_*` も注入）
3. Desktop 生存 + backend 不在 → **Electron 再起動の前に** managed serve を起動/復旧
4. 連続失敗が `-FailThreshold` 以上 → Desktop 強制再起動
5. 予約 ops ポート (9120/8787/…) は backend 判定・reap 対象外（従来どおり）

### Desktop ショートカット

パッケージ `Hermes.exe` 直起動は `HERMES_DESKTOP_*` を付けない。Go watchdog が prewarm していれば Desktop は `desktop-backend.json` を読んで **15s 以内** に既存 serve へ接続する（`apps/desktop/electron/watchdog-backend.ts`）。

## 追加フラグ（exe / Start スクリプト）

| フラグ | 既定 | 説明 |
|--------|------|------|
| `-prewarm-backend` | on | serve の prewarm / 常時監督 |
| `-managed-backend-port` | 9118 | watchdog 管理の固定 serve ポート（9120/8787/9119 とは別） |
| `-backend-start-timeout` | 120 | `/api/status` 待ち (秒) |
| `-backend-ready-timeout` | 45 | `/api/status` 待ち (秒) |

## 監視ロジック（旧 PowerShell 版との差分）

## 停止

- タスクマネージャで `hermes-watchdog.exe` を終了
- または Admin API: `POST /api/v1/stop` + Bearer token
- ロック: `%LOCALAPPDATA%\HermesWatchdog\watchdog.lock`

## スタック再起動との関係

`restart-hermes-stack.ps1 -StartGoWatchdog` で**明示指定時のみ**起動（既定 OFF）。  
Hermes Agent からは到達不可。
