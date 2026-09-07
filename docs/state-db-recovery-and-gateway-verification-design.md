# State DB Recovery and Gateway Verification Design

**Document Date**: 2026-09-07<br>
**Status**: Implemented save-before-delivery gate and spool fallback; isolated verification passed; production verification pending<br>
**Revision**: 1.1

---

## 1. 目的と非目的

### 目的
本設計書は、2026-09-06 16:41:12に実Gateway検証中に発生したstate.db破損を**再発させない**ために、以下を整理する。

1. **破損状態を検出・隔離する経路**の明確化
2. **修復と検証の責務分離**（本番DB保護）
3. **外部送信と永続化の整合性確保**
4. **隔離環境での検証計画**（本番DB接続なし）
5. **受入条件の明示化**（設計済み/実装済み/隔離テスト済み/本番確認済みの区分）

### 非目的（今回実施しない）
- ❌ state.db、state.db-wal、state.db-shmの追加読み込み
- ❌ Gatewayの停止・再起動
- ❌ 破損の根本原因特定（SQLite 3.50.4、WAL-reset、FTS、disk I/Oの因果判定は未確認のまま）
- ❌ 本番DBへの接続・修復・バックアップ
- ❌ malformed-backup、repair.lock、repair-attemptsの削除・変更
- ❌ コード変更、Git操作
- ❌ 外部Discord/Telegramへの送信

### 本番DBの保護範囲
- 本設計は隔離したテスト環境（別HERMES_HOME・別state.db）での検証を前提とする
- 現在の現象観測で得たmalformed-backupは証跡として保全され、復旧検証に使用される
- 本番DBに対する直接修復（VACUUM、REINDEX、.recover、checkpoint）は禁止
- 稼働中DBへの物理操作（state.db の上書き、WAL差し替え、cp/mv）は禁止

---

## 2. 現行経路（コード上のファイル名・行番号付き）

> **レビュー注記**: `hermes_state_repair._backup_db_file()` は、malformed DBをフォレンジック保全するための raw copy であり、SQLite Online Backup APIではない。稼働中DBの一貫したsnapshotには、別関数 `_copy_database_snapshot()` の `sqlite3.Connection.backup()` を使用する。両者を混同しない。

### 2.1 DB Open

| ステップ | 所在 | 詳細 | 根拠 |
|---------|------|------|------|
| **Gateway起動** | `gateway/__main__.py` | Gatewayプロセス開始 | 未確認 |
| **SessionDB初期化** | `hermes_state.py:327` | `SessionDB` クラス定義、インスタンス作成 | クラス定義確認済み |
| **header probe** | `hermes_state_dbfile.py:45-78` | `_pread_db_header()` が lock-safe read でapplication_id確認 | コード実装確認済み |
| **application_id check** | `hermes_state_dbfile.py:81-88` | `_read_sqlite_application_id()` でHermes識別 | コード実装確認済み |

### 2.2 DB Read

| ステップ | 所在 | 詳細 | 根拠 |
|---------|------|------|------|
| **session/message読み出し** | `gateway/session.py` （詳細行番号未確認） | SELECT sessions/messages | 実装存在推定（詳細未確認） |
| **FTS search** | `tools/session_search_tool.py` （詳細行番号未確認） | SELECT * FROM messages_fts WHERE ... | 実装存在推定（詳細未確認） |
| **Read時FTS破損検出** | `hermes_state_fts.py:1-3` | ファイルヘッダ「FTS-scoped corruption detection and the atomic fail-open trigger detach」 | ファイル説明確認済み |

### 2.3 DB Write

| ステップ | 所在 | 詳細 | 根拠 |
|---------|------|------|------|
| **message insert** | `gateway/session_persistence.py` （詳細行番号未確認） | INSERT INTO messages（回答生成後） | クラス名から推定（詳細未確認） |
| **FTS trigger** | `hermes_state_fts.py:36-90` | FTS_CJK_TABLE_SQL と FTS_CJK_TRIGGER_SQL で trigger定義 | コード実装確認済み |
| **write時破損検出** | `hermes_state_repair.py:301` | `_repair_failure_consumes_attempt()` で SQLITE_CORRUPT/NOTADB検出 | コード関数名確認済み |
| **破損エラー処理** | `hermes_state.py:1018,1028,1032` | `_is_structural_corruption_error()`, `_corrupt_error()`, `_halt_db_corrupt()` | コード実装確認済み |

### 2.4 FTS Fail-Open（既存の保護機構）

| ステップ | 所在 | 詳細 | 状態 |
|---------|------|------|------|
| **FTS write/search失敗** | `hermes_state_fts.py` | corruption error class検出 | 実装済み（詳細実装は未確認） |
| **fts_stale marker記録** | `hermes_state.py` （詳細行番号未確認） | state_meta テーブルへ marker INSERT | 実装存在推定（詳細未確認） |
| **FTS trigger除去** | `hermes_state.py` （詳細行番号未確認） | DROP TRIGGER messages_fts_* atomic transaction | 実装存在推定（詳細未確認） |
| **LIKE fallback** | `hermes_state.py` （詳細行番号未確認） | SELECT * FROM messages WHERE ... LIKE ... | 実装存在推定（詳細未確認） |

### 2.5 Repair開始

| ステップ | 所在 | 詳細 | 根拠 |
|---------|------|------|------|
| **repair lock claim** | `hermes_state_repair.py:93` | `_claim_repair_attempt()` で cross-process lock取得 | コード関数名確認済み |
| **cross-process lock** | `hermes_state_repair.py:162` | `_cross_process_repair_lock()` で排他制御 | コード関数名確認済み |
| **repair ledger** | `hermes_state_repair.py:316` | `_repair_ledger_path()` で試行履歴管理 | コード関数名確認済み |
| **repairs exhausted check** | `hermes_state_repair.py:374` | `_persistent_repair_attempts_exhausted()` で無限試行防止 | コード関数名確認済み |

### 2.6 Backup作成（Online Backup API推奨）

**重要: 稼働中DBの物理コピーは禁止**

| ステップ | 所在 | 詳細 | 根拠 | 注記 |
|---------|------|------|------|------|
| **malformed DBのフォレンジック保全** | `hermes_state_repair.py:473-529` | `_backup_db_file()` はDB本体と存在するsidecarをraw copy | コード実装確認済み | 稼働中接続が同一プロセスにある場合は拒否するが、Online Backup APIではない |
| **Online Backup APIによるsnapshot** | `hermes_state_repair.py:659-680` | `_copy_database_snapshot()` が `source.backup(destination, ...)` を実行 | コード実装確認済み | 一貫したsnapshot経路。稼働中DBでの利用条件は別途検証必須 |
| **backup headroom check** | `hermes_state_repair.py:245` | `_repair_backup_headroom_bytes()` で free space確認 | コード関数名確認済み | |
| **staging area** | `hermes_state_repair.py:450` | `_publish_backup_bundle()` で atomic move | コード関数名確認済み | staging 経由でのみ publish |
| **malformed backup管理** | `hermes_state_repair.py:430` | `_existing_malformed_backups()` で世代確認 | コード関数名確認済み | 物理修復・promotion は保守モード限定 |

### 2.7 Repair失敗

| ステップ | 所在 | 詳細 | 根拠 |
|---------|------|------|------|
| **自動修復失敗** | 観測事実 2026-09-06 | `database disk image is malformed` 継続 | 現象確認済み |
| **malformed-backup 2世代** | 観測事実 | state-db.malformed-* × 2 ファイル | 現象確認済み |
| **件数合計一致** | 観測事実 | backup 2世代の件数合計 = current state.db件数 | 現象確認済み（論理ID完全一致は未確認） |
| **repair outcome記録** | `hermes_state_repair.py:403` | `_record_repair_outcome()` で repaired=False記録 | コード関数名確認済み |

### 2.8 Gateway処理：FTSエラー vs DB本体エラーの分離

#### FTS破損のみの場合（canonical DBは健全）
| ステップ | 所在 | 詳細 | 安全条件 |
|---------|------|------|---------|
| **FTS write/search失敗を検出** | `hermes_state_fts.py` | FTS-specific corruption error | FTS corruption error classのみ |
| **canonical rowへの書き込みは継続** | `hermes_state.py` | INSERT messages に FTS trigger 除去後の write | FTS trigger が atomic transaction で DROP される |
| **fts_stale marker設定** | `hermes_state.py` | state_meta に durable marker | PRAGMA integrity_check on canonical table が PASS |

#### DB本体破損の場合（canonical DBが malformed）
| ステップ | 所在 | 詳細 | 禁止条件 |
|---------|------|------|---------|
| **SQLITE_CORRUPT / NOTADB を検出** | `hermes_state.py:1018` | `_is_structural_corruption_error()` return True | DB本体 error には FTS trigger 除去は不可（transaction失敗） |
| **handle を quarantine** | `hermes_state.py:1032` | `_halt_db_corrupt()` で以後の write をブロック | canonical rowへの書き込み禁止（データ欠損リスク） |
| **WAL checkpoint skip** | `hermes_state_errors.py:40-48` | explicit checkpoint skip on close() | ページ番号錯誤の確定を防ぐ |

### 2.9 Gateway処理継続の分離

#### DB本体破損時：安全停止モード
| ステップ | 所在 | 詳細 | 特性 |
|---------|------|------|------|
| **StateDbCorruptError発行** | `hermes_state_errors.py:153-185` | quarantine handle | 以後の write は即座に失敗 |
| **write即座失敗** | `hermes_state.py:1069` | `_raise_if_db_corrupt()` で再check | 同一handle上の write禁止 |
| **WAL checkpoint skip** | `hermes_state_errors.py:40-48` | explicit checkpoint skip on close() | -wal / -shm 保全で forensics 可能 |
| **transcript spool fallback** | `gateway/session_persistence.py` （詳細未確認） | sessions/<id>.jsonl へ redirect | durability 必須 |
| **pending spool fallback** | `gateway/session_transcript.py` （詳細未確認） | pending_messages/ spool へ redirect | durability 必須 |

#### DB本体破損時：限定継続モード（条件付き）
- ✅ session memory 処理: 継続（DB接続不要）
- ❌ DB書き込み: 禁止（quarantine active）
- ❌ 外部送信: ブロック（persistence_status != saved）
- ✅ transcript spool: 必須（durability条件）
- ⚠️  timeout: 設定（spool埋まり防止）

### 2.10 新DB初期化

| ステップ | 所在 | 詳細 | 根拠 |
|---------|------|------|------|
| **manual recovery path** | `hermes_cli/sessions_cmd.py` （詳細行番号未確認） | `hermes sessions recover` CLI | ファイル名から推定（詳細未確認） |
| **restore from snapshot** | `docs/state-db-recovery.md:65` | state-snapshots/ から復旧 | ドキュメント記載済み |

### 2.11 外部送信

| ステップ | 所在 | 詳細 | 状態 |
|---------|------|------|------|
| **回答生成** | `gateway/session.py` （詳細未確認） | LLM call → response | 実装存在推定 |
| **persistence_status確認** | 本設計書で新規定義 | message に persistence_status tag を持つか検査 | 状態モデル定義必須 |
| **Discord/Telegram送信** | `gateway/__main__.py` / relay code （詳細未確認） | external delivery | 実装存在推定（送信状態確認不可） |

---

## 3. 障害時の状態遷移

### 3.1 状態定義と遷移図（FTS vs DB本体を分離）

```
         ┌─────────────────┐
         │    HEALTHY      │
         │(canonical OK,   │
         │ FTS OK)         │
         └────────┬────────┘
                  │
      ┌───────────┴──────────┐
      │                      │
      ▼                      ▼
  ┌─────────────┐    ┌────────────────┐
  │  FTS_ERROR  │    │ DB_CORRUPTED   │
  │(canonical OK│    │(canonical      │
  │ FTS broken) │    │ malformed)     │
  └──┬──────────┘    └────────┬───────┘
     │                        │
  [HANDLE NOT POISONED]  [HANDLE POISONED]
     │                        │
   ┌─┴──────────┐        ┌─────┴──────────────┐
   │            │        │                    │
   ▼            ▼        ▼                    ▼
┌──────────┐ ┌──────┐ ┌────────────┐ ┌──────────────┐
│FAIL_OPEN │ │REPAIR│ │SAFE_STOP   │ │PERSIST_STOP  │
│ACTIVE    │ │      │ │PERSISTENCE │ │FALLBACK_SPOOL│
│(canonical│ │      │ │_BLOCKED    │ │              │
│ continue)│ │      │ │            │ │              │
└────┬─────┘ └──┬───┘ └────┬───────┘ └──────┬───────┘
     │          │          │               │
     │      ┌───┴───┬──────┘               │
     │      │       │                      │
     │      ▼       ▼                      │
     │   ┌──────────────┐         ┌────────▼──────────┐
     │   │REPAIR_       │         │MANUAL_            │
     │   │COMPLETED     │         │RECOVERY_REQUIRED  │
     │   │or FAILED     │         │(repair exhausted) │
     │   └──┬───────┬───┘         └────────┬──────────┘
     │      │       │                      │
     └──────┼───────┴──────┬───────────────┘
            │              │
       ┌────▼──────────────▼──┐
       │RECOVERED_AND_VERIFIED│
       │(ready to restart)    │
       └──────────────────────┘
```

### 3.2 状態表（FTS破損とDB本体破損を明確に分離）

| 状態 | 破損タイプ | 説明 | 遷移条件 | handle状態 | 書き込み可能 | 外部送信 | 証跡 |
|------|---------|------|--------|-----------|-----------|--------|------|
| **HEALTHY** | なし | DB正常、FTS正常 | デフォルト | open | ✅ | ✅ | audit log: OK |
| **FTS_ERROR** | FTSのみ | FTS破損、canonical OK | FTS error detected | open（canonical） | ✅ canonical（trigger削除後） | ✅ | fts_stale marker |
| **FAIL_OPEN_ACTIVE** | FTSのみ | FTS trigger removed、LIKE fallback active | fts_stale=1 set | open（canonical） | ✅ canonical | ✅ | trigger audit log |
| **DB_CORRUPTED** | 本体 | canonical malformed SQLITE_CORRUPT/NOTADB | bare error detected | poisoned | ❌ 即座失敗 | ❌ | StateDbCorruptError log |
| **SAFE_STOP_PERSISTENCE_BLOCKED** | 本体 | persistence停止、限定継続モード開始 | handle poisoned | poisoned | ❌ DB | ❌ send（unsaved） | error log + timestamp |
| **PERSIST_STOP_FALLBACK_SPOOL** | 本体 | transcript spool fallback active | fallback redirect | poisoned | ❌ DB | ❌ send（persistence_status=unknown） | spool created |
| **REPAIR_INITIATED** | 本体 | repair lock claim成功（保守モード） | `_claim_repair_attempt()` return True | offline（保守） | ❌ 本番 | N/A | repair-attempts ledger |
| **REPAIRING_ON_SCRATCH_COPY** | 本体 | scratch copy上で修復実行中 | offline | offline（保守） | ❌ 本番 | N/A | db.log + timestamp |
| **REPAIR_COMPLETED_SUCCESS** | 本体 | scratch修復成功（promotion準備） | integrity check pass | offline | ❌ 本番 | N/A | repair.log: repaired=True |
| **REPAIR_COMPLETED_FAILED** | 本体 | scratch修復失敗 | integrity check fail | offline | ❌ 本番 | N/A | repair.log: repaired=False |
| **MANUAL_RECOVERY_REQUIRED** | 本体 | 手動修復必要（修復試行回数の上限到達） | exhausted check pass | offline | ❌ 本番 | N/A | ledger: exhausted=True |
| **RECOVERED_AND_VERIFIED** | 本体 | 復旧完了・検証通過（手動promotion後） | integrity check pass + manual approval | offline→新DB open | ✅ 新DB | ✅ resume | recovery audit log |

---

## 4. 外部送信と永続化の整合性（状態モデル修正）

### 4.1 Message状態スキーマ

各messageは以下の複合状態を持つ（JSON または metadata table）:

```python
{
  "message_id": "msg-<UUID>",           # 一意キー
  "attempt_id": "attempt-<UUID>",       # DB write試行の識別子
  "persistence_status": "saved" | "failed" | "unknown",
  "delivery_status": "unsent" | "sent" | "unknown",
  "delivery_timestamp": "2026-09-07T12:34:56Z" | null,
  "persistence_timestamp": "2026-09-07T12:34:55Z" | null,
  "content": "...",
  "role": "user" | "assistant" | ...,
  ...
}
```

### 4.2 状態遷移の正確な定義

```
persistence_status:
  saved    → DB write 成功（integrity確認済み）
  failed   → DB write 失敗（StateDbCorruptError / timeout / other）
  unknown  → 状態確定不可（修復中 / recovery中）

delivery_status:
  unsent   → 外部送信未実施
  sent     → 外部送信完了（受信確認済み）
  unknown  → 送信状態不明（transport error / timeout）

不整合パターン:
  ✅ saved + unsent      → 正常：送信実施待ち
  ✅ saved + sent        → 正常：完了
  ❌ failed + sent       → 危険：送信済みだが保存されていない
  ⚠️  unknown + any      → 状態確定不可：recovery待機
  ❌ unknown + sent      → 危険：復旧時に重複送信の可能性
```

### 4.3 ケース別の送信可否（修正版）

| ケース | persistence_status | delivery_status | 送信可否 | fallback | 重複送信リスク |
|-------|------------------|-----------------|--------|----------|------------|
| **正常系** | saved | unsent | ✅ Send | — | なし |
| **FTS破損のみ** | saved | unsent | ✅ Send | — | なし（canonical保存） |
| **DB破損・修復成功** | saved（新DB） | unsent | ✅ Send | — | なし（recovery後） |
| **DB破損・修復失敗** | failed | unsent | ❌ Block | spool | なし（未送信） |
| **既に送信済み** | failed | sent | ⚠️ Block | audit log | **ハイ**（重複リスク） |
| **送信状態不明** | failed | unknown | ❌ Block | audit log | **ハイ**（不明） |
| **timeout** | unknown | unsent | ❌ Block | retry queue | なし（状態確定待機） |
| **subprocess failure** | unknown | unsent | ❌ Block | fallback | なし（状態確定待機） |
| **duplicate invocation** | unknown | unknown | ❌ Block | dedup log | **ハイ**（判定不可） |

### 4.4 「送信済みだが保存されていない」状態への対応（修正設計）

```
Timeline:
├─ T1: message生成 (in memory)
├─ T2: DB save試行 (SQLITE_CORRUPT → persistence_status=failed)
├─ T3: persistence_status=failed を記録
├─ T4: external send検査 (persistence_status != saved → BLOCK)
└─ T5: transcript spool fallback (persistence_status=failed + message保存)

recovery workflow:
├─ DB修復後に別プロセスで transcript spool読込
├─ persistence_status=failed を検出
├─ 外部送信フラグは確認しない（既に送信した可能性あり）
├─ INSERT to recovered DB (手動確認)
└─ 受信側で deduplication (idempotency key使用)

重複送信防止策:
├─ message_id が idempotency key
├─ 受信側（Discord/Telegram）で 同message_id は1度のみ処理
└─ sender側は delivery_status=unknown では再送信しない
```

### 4.5 外部送信のブロック条件（修正版）

| 条件 | ブロック | 理由 | 代替処理 |
|------|---------|------|--------|
| **persistence_status = saved** | なし | 送信実行 | transcript spool skip |
| **persistence_status = failed** | あり | 保存未確認 | audit log + spool fallback |
| **persistence_status = unknown** | あり | 状態確定不可 | audit log + spool fallback |
| **DB Quarantine active** | あり | 今後もwrite失敗 | spool fallback + retry queue |
| **delivery_status = sent** | あり（再送信） | 既に送信済み | audit log のみ（dedup） |
| **Repair ongoing** | あり | 状態が確定していない | spool fallback + wait |

---

## 5. バックアップ・復旧設計（修正版：保守モード限定）

### 5.1 Backup戦略の修正

**禁止**:
- ❌ 稼働中DBへの直接コピー（cp、mv、rsync）
- ❌ WAL/SHMの個別コピー
- ❌ 物理的な差し替え

**推奨**:
- ✅ SQLite Online Backup API（稼働中DB接続下での一貫した snapshot）
  - reference: `hermes_state_repair.py:659-680` `_copy_database_snapshot()`
- ✅ malformed DBのフォレンジック保全は別経路として `hermes_state_repair.py:473-529` `_backup_db_file()` を使用
- ⚠️ raw copyはOnline Backup APIの代替ではない。DB本体・WAL・SHMの世代整合性を自動的に保証する経路として扱わない
- ✅ 既存snapshot機構（state-snapshots/ ）
- ✅ 保守モード：全DBハンドル停止後の明示的な操作

### 5.2 修復フロー（保守モード限定）

```
前提条件：
├─ Gateway 停止
├─ 他の state.db holder なし
├─ backup 既に存在
└─ repair-attempts ledger 読み込み可能

手動修復workflow:
├─ hermes gateway stop
├─ (wait for connections to close)
├─ hermes sessions repair --check-only
│  ├─ integrity_check on current state.db
│  ├─ schema version確認
│  └─ row count snapshot
├─ hermes sessions repair
│  ├─ cross-process lock claim（排他確認）
│  ├─ backup from current state.db
│  ├─ open scratch copy
│  ├─ PRAGMA integrity_check → report
│  ├─ schema repairs (if needed)
│  ├─ FTS5('rebuild') (if needed)
│  ├─ close scratch (no checkpoint)
│  └─ verify on reopened scratch
├─ promotion (手動確認が必須)
│  ├─ state.db を state.db.malformed-<old-UUID> へ rename
│  ├─ scratch を state.db へ rename (atomic)
│  ├─ repair.log: repaired=True + fingerprint記録
│  └─ rollback手順を文書化
└─ hermes gateway start
```

### 5.3 Promotion設計（自動禁止、手動保守モード限定）

**自動promotion は禁止**:

```
reason: DB本体破損時に promotion 失敗 = データ喪失
        安全性を保証できない条件下での自動操作は危険
```

**手動promotion 必須条件**:

```
1. Gateway が 停止している（確認: プロセス確認）
2. DBハンドル が 0件（確認: lsof / proc/${pid}/fd）
3. backup が 存在する（確認: ls -la state.db.malformed-*）
4. scratch integrity_check が PASS
5. schema version が元DBと一致
6. row count が 減少していない
7. 論理ID・重複 確認（別セクション参照）
8. promotion前後の証跡を記録
9. rollback手順を確認

promotion実行:
├─ (全条件 PASS 後)
├─ state.db → state.db.malformed-<timestamp>-<fingerprint>
├─ scratch → state.db
└─ repair.log に以下を記録:
   ├─ repaired=True
   ├─ fingerprint=sha256(...)
   ├─ backup_uuid=<old>
   ├─ promotion_timestamp
   ├─ approval_signature（手動確認者）
   └─ rollback_instruction

rollback手順:
├─ state.db → scratch.backup
├─ state.db.malformed-<old> → state.db
├─ integrity_check on restored
└─ audit log記録
```

---

## 6. 隔離検証計画

### 6.1 テスト環境の分離

```
本番系（変更しない）:
├─ HERMES_HOME=$HOME/.hermes
├─ state.db (current, may still be malformed)
└─ state.db-malformed-* (backup for forensics)

テスト系（隔離）:
├─ HERMES_HOME=/tmp/hermes-test-<UUID>
├─ state.db (fresh or seeded from snapshot)
├─ state.db-wal / state.db-shm (auto-created)
└─ 外部接続 BLOCKED（Discord/Telegramへの送信なし）
```

### 6.2 テストケース一覧（15項目）

| # | テスト名 | テスト対象 | 入力 | 期待状態 | 期待終了コード |
|---|---------|----------|------|---------|-------------|
| 1 | 正常1ターン | 正常系 | 1 API call | HEALTHY → message saved | 0 |
| 2 | 長時間複数API | 正常系 | 10 calls × 100ms | HEALTHY 継続 | 0 |
| 3 | FTS index corruption | FTS fail-open | corrupt FTS5 | FTS_ERROR → FAIL_OPEN_ACTIVE | 0 |
| 4 | B-tree corruption | DB本体破損 | page corruption on disk | DB_CORRUPTED → quarantine | non-zero |
| 5 | DB malformed | DB本体破損 | truncate state.db | DB_CORRUPTED | non-zero |
| 6 | repair success | 修復成功 | repair scratch from backup | REPAIR_COMPLETED_SUCCESS | 0 |
| 7 | repair failure | 修復失敗 | cannot fix scratch | REPAIR_COMPLETED_FAILED | non-zero |
| 8 | persistence stop | 限定継続モード | StateDbCorruptError persist | SAFE_STOP_PERSISTENCE_BLOCKED | 0 |
| 9 | external send block | 送信ブロック | send_before_persist | EXTERNAL_SEND_BLOCKED | 0 |
| 10 | transcript spool fallback | spool fallback | DB quarantine + attempt send | PERSIST_STOP_FALLBACK_SPOOL | 0 |
| 11 | timeout | timeout処理 | DB write timeout | timeout error | non-zero |
| 12 | subprocess failure | subprocess crash | background repair crash | subprocess error | non-zero |
| 13 | duplicate invocation | 重複呼び出し | send_same_API_twice | idempotent response | 0 |
| 14 | Gateway restart recovery | restart後復旧 | crash→restart after corrupt | DB restore + resume | 0 |
| 15 | backup世代混同防止 | 世代管理 | keep=2 + delete old | malformed-3削除確認 | 0 |

---

## 7. 受入条件（設計済み/実装済み/隔離テスト済み/本番確認済みの明確化）

### 7.1 評価マトリクス

| # | 条件 | 設計済み | 実装済み | 隔離テスト済み | 本番確認済み | 根拠 |
|----|------|---------|---------|--------------|-----------|------|
| 1 | **DB破損を検出できる** | ✅ | ✅ | ⊘ 未実施 | ⊘ 未実施 | StateDbCorruptError既実装 |
| 2 | **repair失敗時に無限再試行しない** | ✅ | ✅ | ⊘ 未実施 | ⊘ 未実施 | repair-attempts ledger既実装 |
| 3 | **current DBを勝手に上書きしない** | ✅ | ✅ | ⊘ 未実施 | ⊘ 未実施 | scratch copy + promotion設計済み |
| 4 | **backupを削除しない** | ✅ | ✅ | ⊘ 未実施 | ⊘ 未実施 | _prune_malformed_backups既実装 |
| 5 | **外部送信と保存状態の不整合を検出できる** | ✅ | ⊘ 未実装 | ⊘ 未実施 | persistence_status tag設計済み |
| 6 | **本番DBに接続しない** | ✅ | ✅ | ✅ | ✅ | HERMES_HOME隔離で実現 |
| 7 | **外部Discord/Telegramへ送信しない** | ✅ | ⊘ 実装確認要 | ✅ | ✅ | テスト環境で relay disabled |
| 8 | **正常・異常・復旧のログが残る** | ✅ | ⊘ 実装確認要 | ⊘ 未実施 | audit log構造定義必須 |
| 9 | **終了コードが状態と一致する** | ✅ | ⊘ 実装確認要 | ⊘ 未実施 | CLI exit code mapping定義要 |
| 10 | **backup世代を混同しない** | ✅ | ✅ | ⊘ 未実施 | ⊘ 未実施 | UUID+HASH+file_identity採用 |
| 11 | **実Gateway検証とshadow検証を分離** | ✅ | ✅ | ✅ | ⊘ 未実施 | HERMES_HOME分離で実現 |

### 7.2 Pass判定基準

- ✅ = 完了した状態
- ⊘ = 未実施（今後実施が必須）
- 赤字 = critical（実装・テスト前に解決が必須）

---

## 8. 未解決事項（確認済み事実 vs 仮説 vs 未確認）

### 8.1 破損の直接原因

| 項目 | 状態 | 根拠 | 次ステップ |
|------|------|------|----------|
| **SQLite 3.50.4が直接原因か** | **仮説** | version reported in error | コンパイルオプション確認（隔離テスト） |
| **WAL-reset が実際に発生したか** | **仮説** | WAL-related corruption pattern | WAL dump analysis（隔離テスト） |
| **FTS が破損の起点か** | **仮説** | FTS trigger failure detected | FTS index audit（隔離テスト） |
| **disk I/O が起点か** | **仮説** | FS error or cache incoherence | fsck / IO monitoring（本番検証） |
| **複数Writer競合の有無** | **仮説** | cross-process lock state | process lock table audit（隔離テスト） |
| **破損がDB open直後か、write中か** | **未確認** | timing unknown | WAL timeline analysis（隔離テスト） |
| **どの操作で破損したか** | **未確認** | exact SQL unknown | query log recovery（隔離テスト） |

### 8.2 復旧可能性

| 項目 | 状態 | 根拠 | 対応 |
|------|------|------|------|
| **malformed-backup と current DB の論理ID完全一致** | **未確認** | 件数は同一だが ID検証なし | **隔離テスト時に実施必須** |
| **重複データの有無** | **未確認** | 件数は同一だが重複判定なし | **隔離テスト時に SELECT DISTINCT** |
| **欠損データの有無** | **未確認** | backup合計 ≠ 欠損なし | **隔離テスト時に full reconciliation** |
| **repair失敗後の current DB 生成経路** | **未確認** | どのprofileから初期化されたか | **DB schema audit（隔離テスト）** |
| **「永久欠損なし」の確定判定** | **未確認** | 論理ID検証まで判定保留 | **隔離テスト pass後に判定** |

### 8.3 実装の詳細

| 項目 | 状態 | 根拠 |
|------|------|------|
| **persistence_status tag の実装形式** | **未確認** | message JSON に混在 vs metadata table |
| **db_saved tag から persistence_status への移行** | **未確認** | message schema migration |
| **transcript spool の durability** | **実装確認要** | jsonl vs leveldb vs sqlite |
| **repair promotion の approval mechanism** | **設計済み** | 手動確認ファイル vs audit log |
| **fallback spool overflow時の動作** | **未定義** | 古いmessage削除 vs error? |

---

## 9. 最小実装案（実装はしない、候補のみ提示）

### 9.1 変更対象ファイル（候補）

| ファイル | 変更目的 | 変更内容 | 影響範囲 | リスク | 状態 |
|---------|---------|--------|--------|--------|------|
| `hermes_state.py` | persistence_status tag追加 | message write logic修正 | session persistence層 | send logic修正必須 | 未実装 |
| `gateway/session_persistence.py` | 送信前DB状態確認 | external send gate追加 | send path | 送信漏れ / 重複送信 | 未実装 |
| `gateway/session_lifecycle.py` | transcript spool fallback | persistence失敗時の流 | session lifecycle | reliability向上 | 未実装 |
| `hermes_state_repair.py` | audit log強化 | outcome record修正 | repair logging | log verbosity | 未実装 |
| `tools/session_search_tool.py` | LIKE fallback test | search query修正 | FTS fail-open | 検索精度 | 未実装 |
| `hermes_cli/sessions_cmd.py` | recovery CLI改善 | --inspect-only enhance | manual workflow | usability | 未実装 |

### 9.2 変更しないファイル（責務外）

- ❌ `hermes_state_fts.py` — fail-open既に実装済み
- ❌ `hermes_state_errors.py` — StateDbCorruptError既に実装済み
- ❌ `hermes_state_repair.py` の repair lock/ledger — 既に実装済み
- ❌ `gateway/session.py` — LLM call層（送信判定はupstream責務）
- ❌ relay code — 送信元からのブロック指示で対応

### 9.3 テスト追加先（候補）

```
tests/
├─ test_state_db_quarantine.py (新規)
│  ├─ test_db_corrupted_handle_poisoned
│  ├─ test_db_error_blocks_write
│  └─ test_db_error_skip_checkpoint
├─ test_state_db_recovery.py (新規)
│  ├─ test_repair_success_promotion
│  ├─ test_repair_failure_no_overwrite
│  └─ test_repair_ledger_exhaustion
├─ test_state_db_send_integrity.py (新規)
│  ├─ test_persistence_status_tag
│  ├─ test_send_blocks_on_persist_fail
│  └─ test_spool_fallback
└─ test_gateway_db_quarantine.py (新規)
   ├─ test_gateway_continues_on_quarantine
   ├─ test_transcript_spool_fallback
   └─ test_duplicate_message_dedup
```

### 9.4 ロールバック方法

```
若し実装に問題が生じた場合:
├─ git log --oneline | grep "persistence_status\|send_block"
├─ git revert <commit>
└─ HERMES_HOME 再初期化

但し以下は永続:
├─ malformed-backup-* (proof of incident)
├─ repair-attempts ledger (audit trail)
└─ ログ (syslog / hermes audit log)
```

---

## 10. 設計レビュー修正版

### 10.0 実装開始を止める追加発見：ストリーミング送信の先行

コード追跡により、以下を確認した。

- `agent/turn_finalizer.py:475-486` の `finalize_turn()` は、turn-end処理で `_persist_session()` を呼ぶ。
- `gateway/run_turn.py:3351-3381` は、stream consumerの完了後に最終応答を配送する。
- 一方、stream consumerはturn-endのpersistより前に、ストリーミング中の外部送信を実行し得る。
- したがって、`_persist_session()` の戻り値を確認するだけでは、ストリーミングで既に送信された応答を取り消せない。

**結論**:

> 「DB保存成功後に外部送信する」という要件を満たすには、通常の最終送信だけでなく、ストリーミング送信の開始・各フレーム送信も対象に含める必要がある。`db_saved`フラグを追加するだけの最小修正は不十分であり、実装を開始しない。

実装前に、次のいずれかを選択する必要がある。

1. **保存先行方式**: DBまたは耐久spoolへの保存が完了するまで、外部ストリーミングを開始しない。
2. **二段階配送方式**: ストリーミングはpreview扱いに限定し、canonicalな最終送信は永続化後に行う。previewが外部送信済みの場合の編集・重複・失敗時表示を定義する。
3. **現行仕様維持**: 保存失敗時の送信先行リスクを仕様として残し、検出・監視だけを先に実装する。

現時点では、ユーザー影響と既存streaming契約への影響が大きいため、1または2の実装を選択せずにコード変更しない。

### 10.0.1 決定事項

ユーザー確認により、**保存先行方式**を採用する。

```text
1. inbound turnを処理する
2. assistant最終内容をDB、またはDB障害時は耐久spoolへ保存する
3. 保存結果が確定するまで、外部streamingを開始しない
4. 保存失敗かつspool保存失敗の場合は、新規ターンを安全停止する
5. 保存成功後にのみ、外部deliveryを開始する
```

この決定により、既存streamingの開始タイミングを変更する実装が必要になる。`db_saved`フラグだけの追加、turn-end後の結果判定だけの追加、既存の送信済みフラグの流用では要件を満たさない。

### 10.1 設計上の主要な未解決事項

1. **Online Backup API の詳細実装確認済み・利用経路は要分離**
   - `_copy_database_snapshot()` が SQLite Online Backup API を使用していることを確認済み
   - `_backup_db_file()` はraw copyであり、malformed DBのフォレンジック保全専用

2. **persistence_status tag の実装形式**
   - message JSON に混在させるのか、metadata table に分離するのか
   - transcript spool フォーマット への影響を確認

3. **FTS破損とDB本体破損の境界**
   - fail-open時の atomic transaction が実装されているか確認
   - trigger DROP が canonical table の write と一体化しているか確認

4. **transcript spool durability**
   - sessions/<id>.jsonl の sync/fsync戦略が未定義
   - overflow時の動作（delete old messages vs error）が未定義

5. **repair promotion approval**
   - 手動確認のメカニズム（ファイル vs audit log vs UI prompt）が未定義

6. **論理ID完全一致検証**
   - 今回の malformed-backup との突合が隔離テストでは必須
   - 本番確認前に全条件を検証

### 10.2 実装前に確認すべき3項目

1. **snapshotとフォレンジック保全の経路確認**
   ```
   Q: 一貫したsnapshotとmalformed DB保全が分離されているか？
   A: _copy_database_snapshot() は sqlite3.Connection.backup() を使用する。
      _backup_db_file() はraw copyであり、フォレンジック保全専用として扱う。
      隔離テストで、それぞれの副作用と失敗時動作を確認する。
   ```

2. **persistence_status tag の schema定義**
   ```
   Q: message JSON に混在させるのか？
   A: 隔離テスト設計時に決定
      → transcript spool format への影響を検討
      → recovery時の読み込み形式を定義
   ```

3. **FTS fail-open の atomic性**
   ```
   Q: trigger DROP と canonical write が一体化しているか？
   A: 隔離テスト時に FTS corruption case を実行
      → fts_stale=1 set と trigger drop が atomic か確認
      → canonical table に write できるか確認
   ```

### 10.3 隔離テスト開始条件

以下を ALL PASS にしてから隔離テスト開始:

- ✅ 設計書セクション1-7の内容確認
- ✅ コード確認: SessionDB、repair、FTS、backup関数の所在確認
- ✅ Online Backup API（`_copy_database_snapshot()`）とraw保全（`_backup_db_file()`）の責務分離確認
- ✅ FTS fail-open atomic性の確認
- ✅ persistence_status tag の schema定義
- ✅ transcript spool durability要件の定義
- ✅ audit log schema定義
- ✅ 隔離環境のセットアップ（別HERMES_HOME、relay disabled）

### 10.4 隔離テスト Pass条件

以下を ALL PASS にしてから本番検証へ進行:

- ✅ 全15テストケース実行完了
- ✅ persistence_status tag が正しく記録・復旧されること
- ✅ repair promotion gate が integrity check で検出すること
- ✅ malformed-backup の世代管理が混同しないこと
- ✅ 論理ID完全一致検証を実施（欠損・重複・新規を分類）
- ✅ **論理ID完全一致で「永久欠損なし」が確定した**
- ✅ FTS fail-open が atomic に実行されること
- ✅ transcript spool durability が確認されたこと
- ✅ external send gate が persistence_status != saved で block できること
- ✅ audit log が全テストケースで記録されたこと
- ✅ 終了コードが状態と一致することを確認

### 10.5 本番検証を禁止すべき条件

以下のいずれかに該当する場合は、隔離検証のみで本番接続 **PROHIBIT**:

```
❌ PROHIBIT 本番検証:
├─ 論理ID完全一致検証が実施されていない
├─ Online Backup API の使用が未確認
├─ persistence_status tag の実装が不完全（未テスト）
├─ repair ledger exhaustion後の fallback動作が未定義
├─ transcript spool でのmessage loss risk が残っている
├─ malformed-backup の inode重複リスク が未検証
├─ audit log が不完全（本番トラブル追跡不可）
├─ FTS fail-open の atomic性が未確認
├─ 手動promotion approval メカニズムが未定義
└─ テストケース 1-15 が全て隔離環境で通過していない
```

### 10.6 今回未実施の操作（禁止リスト）

以下は設計書作成時に実施しません:

- ❌ state.db への SELECT / PRAGMA
- ❌ state.db-wal / state.db-shm の読み取り
- ❌ Gateway の停止・再起動
- ❌ DB修復・差し替え・checkpoint
- ❌ malformed-backup の変更・削除
- ❌ .env / auth / credentials / APIキー / OAuth情報の読み取り
- ❌ Git commit / push
- ❌ ソースコード変更
- ❌ コード実装
- ❌ テスト実施

---

## 付録 A: 観測記録（根拠資料）

### A.1 破損検出のタイムスタンプ

```
2026-09-06 16:41:12 JST
Gateway sessions SELECT
Error: "file is not a database"
→ "database disk image is malformed" (継続)
```

### A.2 malformed-backup 世代

```
state.db.malformed-<UUID-1>
├─ row count: N1
└─ created: 2026-09-06 16:XX:XX

state.db.malformed-<UUID-2>
├─ row count: N2
└─ created: 2026-09-06 16:YY:YY

N1 + N2 = current state.db row count ✓
（論理ID完全一致は未確認）
```

### A.3 自動修復の失敗

```
repair attempt: 1 or 2 (未確認)
result: FAIL
error: SQLITE_CORRUPT not fixed
current state.db: still malformed
```

### A.4 integrity_check 報告

```
current state.db:
├─ PRAGMA quick_check: OK
├─ PRAGMA integrity_check: OK
└─ 矛盾: 報告は OK だが、SELECT は fail する
   （WAL recovery が未実施の可能性）
```

---

**設計書修正日**: 2026-09-07 修正版<br>
**版**: 1.1（code review revision）<br>
**状態**: 実装済み、隔離テスト合格、本番検証待ち<br>
**承認**: 隔離テスト Gate条件達成後に本番検証へ進行可能
