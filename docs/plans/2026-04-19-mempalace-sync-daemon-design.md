# MemPalace Sync Daemon — Design Doc

> **Status:** Draft
> **Author:** Jethro + Hermes
> **Date:** 2026-04-19
> **Related:** `2026-04-19-mempalace-sync-daemon-plan.md`（實作 plan）
> **Supersedes:** 上一個 session 提出的 Path 2A（直接改 `hermes_state.py`）

---

## 1. 目標與限制

### 1.1 問題陳述

Hermes 目前的 session recall 只靠 SQLite FTS5（BM25 關鍵字）。它對精準關鍵字很好，但：

- 語意相近但用詞不同的 query 搜不到（e.g.「資料庫遷移」找不到 `postgres_migration`）
- 無 entity / temporal / mining pipeline — 「Jethro 2025 年用的框架」這種時間性 ER 查詢做不到
- 無 wing / room 結構 — 無法把跨 session 的同一專案自動聚類

MemPalace 提供了這些能力（convo_miner、knowledge_graph、wings/rooms/drawers）。但上一個 session 的 spike 確認：

- ❌ 沒有 `from mempalace import Palace` facade
- ❌ Chroma 併發寫入缺 jitter retry（跟 Hermes gateway + CLI + worktree + cron 四路 writer 撞 lock）
- ❌ 直接整合 `hermes_state.py`（Path 2A）= 150 LOC +700 新碼 + 45~65 tests + 日後 rebase 惡夢

### 1.2 設計目標

| 目標 | 約束 |
|---|---|
| **G1.** 對話歷史自動進入 MemPalace，無須人工 | auto-hook |
| **G2.** Agent 可透過 MCP 查 MemPalace | pass-through（2A-lite 已定） |
| **G3.** Hermes 主程式 0 改動 | 便於跟上游 rebase |
| **G4.** 寫入失敗絕不 block Hermes | fire-and-forget + retry queue |
| **G5.** 併發安全 | 只有 daemon 一個 writer 進程 |
| **G6.** 可觀測 | 有 log、有 status 指令、有 metrics |
| **G7.** 可關閉 / 可回退 | systemd/cron 關掉就回到 FTS5-only |

### 1.3 非目標（YAGNI）

- 實時（< 1s 延遲）同步 — 對 session recall 來說沒必要
- 同步歷史 session 的 delete/update — MemPalace 設計就是 append-only，Hermes 的 delete 不鏡像過去（可日後做 tombstone）
- 取代 `hermes_state.py` 或 FTS5 — 兩者並存，MemPalace 僅擴增查詢能力
- 改動 `hermes_state.py` — 一行都不碰

---

## 2. 架構

### 2.1 資料流

```
┌──────────────────────────────────┐
│ Hermes 主程式（cli / gateway /   │
│ batch_runner / acp_adapter）     │  ← 不改動
└──────────────┬───────────────────┘
               │ SessionDB.append_message()
               ▼
    ┌────────────────────┐
    │ ~/.hermes/state.db │ ← 唯一 source of truth
    │ (SQLite + FTS5)    │
    └──────────┬─────────┘
               │ read-only poll
               │ WHERE messages.id > last_synced_id
               ▼
  ┌───────────────────────────────────┐
  │ mempalace-sync daemon             │
  │  • 每 N 分鐘 (預設 5 min) 輪詢    │
  │  • 匯出新 messages 為 JSONL       │
  │  • 寫入 ~/.hermes/mempalace-sync/ │
  │      inbox/<session_id>.jsonl     │
  │  • 呼叫 convo_miner ingest        │
  │  • 更新 cursor                    │
  │  • 失敗 → retry queue             │
  └──────────────┬────────────────────┘
                 │ mempalace convo-mine <dir>
                 ▼
  ┌───────────────────────────────────┐
  │ MemPalace palace                  │
  │ ~/.mempalace/default/             │
  │   (ChromaDB + SQLite KG)          │
  └──────────────┬────────────────────┘
                 │ reads
                 ▼
  ┌───────────────────────────────────┐
  │ MemPalace MCP server              │  ← stdio，YAML 3 行接入
  │ (spawned by Hermes mcp_tool.py)   │
  └──────────────┬────────────────────┘
                 │ search_memories / recall_entity / ...
                 ▼
          Hermes Agent
```

### 2.2 組件

| 組件 | 位置 | 語言 | 約 LOC |
|---|---|---|---|
| `mempalace-sync` daemon | `~/repo/jethro/mempalace-sync/` (新 repo) | Python 3.11 | ~300 |
| Systemd unit（Linux） | `mempalace-sync/deploy/mempalace-sync.service` | — | ~20 |
| Hermes MCP config | `~/.hermes/config.yaml` | YAML | +10 |
| 無 Hermes 主程式改動 | — | — | **0** |

### 2.3 為什麼獨立 repo

- Hermes 不依賴它 → Hermes 升級不受影響
- 它可以自由演進 / 迭代實驗
- 未來若 MemPalace 改 schema，改 daemon 一個地方
- 可以同時給別的 Hermes-like 工具用（multi-agent 共享 palace）

---

## 3. Daemon 內部設計

### 3.1 狀態檔

`~/.hermes/mempalace-sync/state.json`

```json
{
  "last_synced_message_id": 12843,
  "last_run_at": 1745094432.12,
  "total_synced": 7329,
  "failed_session_ids": ["20260418_xyz"],
  "schema_version": 1
}
```

`last_synced_message_id` 是 cursor — `messages.id` 是 AUTOINCREMENT，單調遞增，完美。

### 3.2 主迴圈（偽代碼）

```python
def run_once(state_db, palace_dir, state_path, inbox_dir) -> SyncStats:
    state = load_state(state_path)
    cursor = state["last_synced_message_id"]

    # 1. Read-only query — never writes to state.db
    new_msgs = query_new_messages(state_db, since_id=cursor)
    if not new_msgs:
        return SyncStats(new=0)

    # 2. Group by session_id → one JSONL file per session
    by_session = group_by_session(new_msgs)
    written_files = []
    max_id = cursor
    for sid, msgs in by_session.items():
        fp = inbox_dir / f"{sid}.jsonl"
        append_jsonl(fp, msgs)          # idempotent append
        written_files.append(fp)
        max_id = max(max_id, max(m["id"] for m in msgs))

    # 3. Invoke convo_miner (subprocess or in-process API)
    try:
        run_convo_miner(inbox_dir, palace_dir)
    except Exception as e:
        # Fire-and-forget: don't update cursor so next run retries
        log_error(e)
        state["failed_session_ids"] = list(by_session.keys())
        save_state(state_path, state)
        return SyncStats(new=len(new_msgs), success=False)

    # 4. Update cursor ONLY after successful mining
    state["last_synced_message_id"] = max_id
    state["total_synced"] += len(new_msgs)
    state["last_run_at"] = time.time()
    state["failed_session_ids"] = []
    save_state(state_path, state)

    # 5. Cleanup: move processed JSONLs to archive/
    archive_processed(written_files, inbox_dir / "archive")

    return SyncStats(new=len(new_msgs), success=True)
```

### 3.3 JSONL 格式（給 convo_miner 吃）

每行一則 message，用 MemPalace `normalize.py` 已支援的通用 chat format：

```json
{"ts": "2026-04-19T04:01:12Z", "role": "user", "content": "...", "session_id": "20260419_024424_91a40131", "source": "telegram", "model": "claude-opus-4-7"}
{"ts": "2026-04-19T04:01:45Z", "role": "assistant", "content": "...", "session_id": "20260419_024424_91a40131"}
```

檔名 `<session_id>.jsonl` → convo_miner 會把整個 session 當成一個對話流 → 自動 chunk 成 drawers，分到對應 wing/room。

### 3.4 CLI 介面

```bash
mempalace-sync run           # 跑一次然後退出（給 cron 用）
mempalace-sync watch         # 常駐，每 N 分鐘跑一次（給 systemd 用）
mempalace-sync status        # 印 state.json + inbox/archive 大小
mempalace-sync reset         # 清 cursor（小心，會重 ingest 全部）
mempalace-sync backfill      # 從 state.db 頭開始灌入（一次性）
```

### 3.5 併發保護

- 用 `fcntl.flock` on `~/.hermes/mempalace-sync/sync.lock`
- 確保同時只有一個 daemon instance 在跑
- `state.db` 是 read-only 開啟（`?mode=ro`），不可能寫壞 Hermes 資料

---

## 4. Hermes 側接入（MCP pass-through）

### 4.1 `~/.hermes/config.yaml`

```yaml
mcp_servers:
  mempalace:
    command: "mempalace"
    args: ["mcp-server", "--palace", "/home/jethro/.mempalace/default"]
    env: {}
    timeout: 120
    connect_timeout: 60
    tools:
      # 可選：只暴露 read 用工具，避免 LLM 亂寫
      include: ["search_memories", "recall_entity", "list_wings", "list_rooms_in_wing", "read_drawer"]
```

### 4.2 使用者體驗

- Agent 多出 `search_memories(query)` 工具
- 原有 `session_search(query)` 保留不變（FTS5 關鍵字）
- LLM 會依任務自選（短關鍵字用 FTS5 快，語意模糊用 MemPalace）
- 之後可以加 prompt 引導：「若 FTS5 搜不到再試 MemPalace」

---

## 5. Failure Modes & Recovery

| 失敗模式 | 偵測 | 復原 |
|---|---|---|
| state.db busy / locked | SQLite `database is locked` | daemon retry 3 次，指數退避；不更新 cursor |
| convo_miner crash | subprocess exit != 0 | 不更新 cursor，下次 run 從同 cursor 重試；warning to stderr |
| JSONL 寫入壞掉 | fsync 失敗 | delete partial file，不更新 cursor |
| cursor 檔損毀 | `json.JSONDecodeError` | fallback to 0 + log warning；需手動 `backfill` |
| 兩個 daemon 同時跑 | flock 失敗 | 第二個 exit 0 + log warning |
| MemPalace palace 不存在 | convo_miner 自動建立 | no-op |
| Chroma SQLite 損毀 | convo_miner crash | 靠 MemPalace 自己的 repair / backup；daemon 照常 retry |
| state.db 被搬走 / 刪除 | open() FileNotFoundError | daemon exit 1 + log error；systemd restart 會自動復原 |

### 5.1 資料一致性

- **嚴格 at-least-once**：同一 message 可能被 ingest 多次（例如 convo_miner 跑完後 crash 在更新 cursor 前）
- **MemPalace 原生 dedup**：`convo_miner` 本身有 `file_already_mined()` 檢查（by file hash），所以重 ingest 是 no-op
- **不保證**：delete from Hermes 的 messages 不會從 palace 刪除（MemPalace 是 append-only，這是 design trade-off）

---

## 6. 效能預算

| 指標 | 目標 |
|---|---|
| Hermes 寫入延遲影響 | **0**（read-only 不 block） |
| Daemon 佔用 CPU | 閒時 < 0.1%，mining 時爆 1 core ~秒級 |
| Daemon 記憶體 | < 200 MB（含 ChromaDB embedding model） |
| 典型同步間隔 | 5 分鐘 |
| Query latency（via MCP） | < 1 s（MemPalace 自己的效能預算） |
| 磁碟用量 | `~/.hermes/mempalace-sync/inbox/` 定期清理，不累積 |

---

## 7. Rollout

1. **Week 0:** 單機 cron 跑 `mempalace-sync run --every 5m`
2. **Week 1:** 人工檢查 palace 內容、試用 MCP query、比對 FTS5 結果
3. **Week 2:** 切 systemd，加 metric 檢查
4. **Week 3+:** 若穩定，可以加 `mempalace-sync backfill` 灌入全部歷史 state.db

### 7.1 退場條件

隨時 `systemctl stop mempalace-sync` 即可停止同步。Hermes 完全不受影響，MemPalace MCP 可在 config 註解掉。

---

## 8. Open Questions

1. **MemPalace MCP server 是用 stdio 還是 HTTP？** → 預設 stdio（upstream 主推）
2. **要不要額外鏡像 `tool_calls` / `reasoning`？** → 先**不**鏡像，只鏡像 `role + content`（避免 palace 被 tool noise 淹沒）
3. **MemPalace embedder 是英文 only（`all-MiniLM-L6-v2`）— 中文查得到嗎？** → **Phase 0 實測 PASS**（2026-04-19）：真實 `~/.hermes/state.db`（263 msgs），zh-TW 8/8 命中，英文 4/4 命中。Hybrid BM25 + vector 救回了純 embedder 不足。不需要 patch ChromaBackend。`MEMPALACE_EMBED_MODEL` env var 不存在（先前設計是基於錯誤假設）。
   - 若未來大規模 dogfood 發現語意弱（e.g. paraphrase 查詢失敗），可選擇：(a) 替 MemPalace 發 PR 加 `embedding_function` 設定，(b) 切 sqlite-vss 自建。
4. **是否需要 per-source filtering？** → Phase 2 再考慮（e.g. 「只搜 telegram session」）

---

## 9. 決策紀錄

| 決策 | 替代方案 | 選擇理由 |
|---|---|---|
| 獨立 daemon，不改 Hermes | Session-close hook / Turn-level hook | G3（Hermes 0 改動）、G5（併發安全只一個 writer） |
| 用 convo_miner 而非 `palace.upsert()` | 手刻 60 LOC upsert 邏輯 | MemPalace supported path、內建 mining pipeline（wing/room/entity） |
| JSONL 匯出而非 in-memory API | 直接 import mempalace | 沒 Python facade（上次 spike 確認） |
| at-least-once | exactly-once | convo_miner 本身 idempotent；實作簡單 |
| cursor = `messages.id` | timestamp / session_id+idx | AUTOINCREMENT 單調、SQLite 原生索引 |

---

## 10. 相關文件

- 實作 plan：`docs/plans/2026-04-19-mempalace-sync-daemon-plan.md`
- Spike 報告：`~/tmp/mempalace-spike/INTEGRATION_COST.md`、`INSTALL_BENCHMARK.md`
- MemPalace AGENTS.md：`~/tmp/mempalace-spike/repo/AGENTS.md`
- Hermes schema：`hermes_state.py:37-110`
