# Subagent Progress → TTL 24h Telegram Topic 轉發

**Date:** 2026-04-19
**Workspace:** hermes-agent (`/home/jethro/repo/agent/hermes-agent`)
**Triggering chat:** Telegram group 小蟹助手群 (`-1003837358001`), thread `1981` (🔀 subagent 進度轉發)

---

## 1. Goal

當 Telegram 上的 parent session 呼叫 `delegate_task` 時：

1. **自動為該 parent session 建立一個專屬 forum topic**（延遲建立：第一筆 progress 才建）
2. **把 subagent.* 事件流（start / thinking / tool / progress / complete）轉發到該 topic**，不污染主對話
3. **記錄每個 topic 的 "最後訊息時間"**
4. **背景 cron job 巡檢**：凡是超過 24h 沒有新訊息的 mapped topic → 自動 `telegram_topic delete`

## 2. Current context / key findings

### 2.1 Progress event 的真正路徑（上次 session summary 有誤判）

- **產生端:** `tools/delegate_tool.py:158-262` `_build_child_progress_callback`
  - 事件類型: `subagent.start`, `subagent.thinking`, `subagent.tool`, `subagent.progress`, `subagent.complete`
  - 透過 `parent_cb(...)` relay 到 parent agent 的 `tool_progress_callback`
- **Parent 的 callback 綁定:** `gateway/run.py:9152`
  - `agent.tool_progress_callback = progress_callback if tool_progress_enabled else None`
- **Telegram 真正在用的 callback:** `gateway/run.py:8685-8748` `progress_callback` (定義在 `_run_agent` 內)
  - **目前只處理 `tool.started`，其他 event_type 直接 return** (line 8691-8692)
  - 這就是為什麼 subagent.* 永遠不會出現在 Telegram
- **訊息發送 loop:** `gateway/run.py:8764-8883` `send_progress_messages()`
  - 用 `adapter.edit_message()` 持續更新**同一則**訊息
  - 發到 `source.chat_id` + `metadata={"thread_id": source.thread_id}`（parent 的 topic）
- **api_server.py:2078 那條** 是 **OpenAI-compatible API server** 的 callback，不是 Telegram 路徑 —
  上次 session summary 誤導，此改動**不需要動 api_server.py**。

### 2.2 Telegram topic 工具已就緒

`tools/telegram_topic_tool.py`:
- `create_forum_topic` → 回傳 `thread_id`
- `delete_forum_topic(chat_id, message_thread_id)`
- 內部呼叫 `python-telegram-bot` 的 `Bot(token).create_forum_topic(...)` 等

Token 載入: `gateway.config.load_gateway_config().platforms[Platform.TELEGRAM].token`

### 2.3 send_message / adapter 面

- `gateway/platforms/telegram.py` adapter: `send(chat_id, content, metadata={"thread_id": ...})`
- Topic 權限要求: bot 須為 forum supergroup 的 admin + `can_manage_topics`

### 2.4 Assumptions

- Bot 已有 `can_manage_topics` 權限（用戶已在 thread 1981 驗證過 `telegram_topic` 工具可用）
- 只在 `source.platform == TELEGRAM` 且 parent 所在 chat 是 forum supergroup 時啟用
- 非 forum chat（一般 group / DM）→ 功能自動靜默停用，progress 仍照舊留在主對話
- Subagent progress 只轉發到 progress topic，**不再**出現在 parent 的對話串（避免重複轉發）

## 3. Proposed approach

### 3.1 架構

```
[subagent]
  └─ _build_child_progress_callback relays subagent.* events
      └─ parent.tool_progress_callback  (gateway/run.py progress_callback)
          ├─ if event_type.startswith("subagent."):
          │    └─ SubagentTopicRouter.route(session_id, source, event) ──┐
          └─ else (existing tool.started handling)                        │
                                                                          ▼
                                                    ┌─────────────────────────────────┐
                                                    │ SubagentTopicRouter             │
                                                    │ - lazy create topic per session │
                                                    │ - send via Telegram adapter     │
                                                    │ - update last_message_ts        │
                                                    │ - persist mapping JSON          │
                                                    └─────────────────────────────────┘
                                                                          │
                                                                          ▼
                                                    [~/.hermes/state/subagent_topics.json]
                                                                          ▲
                                                                          │ read
                                  [cron: cron/subagent_topic_reaper.py]  │ (every 15 min)
                                  deletes topics with last_message_ts > 24h old
```

### 3.2 State store

**路徑:** `get_hermes_home() / "state" / "subagent_topics.json"`

**Schema:**
```json
{
  "version": 1,
  "topics": {
    "<session_id>": {
      "chat_id": "-1003837358001",
      "thread_id": "2345",
      "topic_name": "🔀 SA · gpt-5.4 · 03:14",
      "created_ts": 1745036000.0,
      "last_message_ts": 1745036123.4,
      "parent_chat_id": "-1003837358001",
      "parent_thread_id": "1981"
    }
  }
}
```

**併發控制:** 使用 `portalocker` 或 `fcntl` 做檔案鎖（gateway 已經依賴哪一個要先確認；若都沒有則用 threading.Lock + atomic rename）。

### 3.3 Lazy create 邏輯

第一次遇到 `subagent.start` 時（此 event 的 payload 中 `goal` 即 `delegate_task` 的 goal 參數）:

1. 檢查 state store 是否已有 `session_id` → 有就用舊的
2. 沒有 → 透過 **`_derive_topic_name(goal)`** 產生短名:
   - 規則: `"SA " + _summarize(goal, max_chars=20)`
   - `_summarize` 策略（由簡到複雜，取第一個 match 的）:
     a. 取 goal 第一行/第一句 → strip → 若 ≤20 chars 直接用
     b. 否則截到 20 chars 並尾加 `…`
     c. 移除控制字元、換行、markdown 符號
   - 總長度 hard-cap 64 chars（Telegram 限制）
   - **這是 agent 已經下達 delegate_task 時提供的 goal，等於 "agent 給的短描述"** — 不需要額外呼叫 LLM，零 latency 零成本
3. 呼叫 `create_forum_topic(chat_id=source.chat_id, name=topic_name)`
4. 寫回 state store
5. **在 parent 主對話發 pointer 訊息（一次性）**:
   ```
   🔀 Subagent 進度轉發至 → #{topic_name}
   （此 topic 在無新訊息 24h 後自動刪除）
   ```
   發到 `source.chat_id` + `source.thread_id`（parent 所在 topic，如本 thread 1981）
6. **在新 topic 發 opening 訊息**:
   ```
   🔀 Session {session_id[:8]}
   Goal: {goal[:300]}
   — 無新訊息 24h 後此 topic 將被自動刪除
   ```

**命名衝突處理:** Telegram 允許 topic 同名，不做去重。若同一 parent session 做多次 delegate_task（我們以 session_id 為 key，不是 delegate call 為 key），所有事件都流入**同一個 topic**，topic 名沿用第一次 delegate 的 goal — 這是刻意設計，一個 parent session 一個 topic。

**Q: 名字過於籠統怎麼辦（例如 goal = "do the task"）？**
→ 降級為 `SA {session_id[:8]} · {HH:MM}`。閾值：若 `_summarize(goal)` 結果 < 4 chars 或為空，用 fallback。

### 3.4 事件格式化

| event_type | 轉發格式 |
|---|---|
| `subagent.start` | `🔀 Started: {goal}` |
| `subagent.thinking` | `💭 {preview[:200]}` |
| `subagent.tool` | `{emoji} {tool_name}` + optional preview |
| `subagent.progress` | `⏳ {preview}` (已是 batched summary) |
| `subagent.complete` | `✅ Done: {preview[:300]}` |

**速率控制:** 沿用現有 `_PROGRESS_EDIT_INTERVAL = 1.5s` pattern，但用**新訊息 append**（不是 edit），因為 topic 內事件流是線性歷史。若單次 burst 超過 5 筆，合併為一則 multi-line 訊息送出。

### 3.5 Reaper cron job

**腳本:** `cron/subagent_topic_reaper.py`

```python
# 偽碼
state = load_state()
now = time.time()
TTL = 24 * 3600
expired = [(sid, t) for sid, t in state["topics"].items()
           if now - t["last_message_ts"] > TTL]
for sid, t in expired:
    try:
        telegram_topic_delete(chat_id=t["chat_id"], thread_id=t["thread_id"])
    except TopicNotFound:
        pass  # already gone
    except Exception as e:
        log(f"Failed to delete {sid}: {e}")
        continue
    del state["topics"][sid]
save_state(state)
```

**排程:** 用 Hermes 既有的 `cronjob` 工具（`mcp_cronjob action=create`），每 15 分鐘跑一次；或直接寫成可獨立跑的 CLI (`python -m cron.subagent_topic_reaper`) 並透過系統 cron 觸發。

**偏好:** 用 `mcp_cronjob`，因為它跟 agent 同生命週期、有 log 回饋，也避免用戶再去設 crontab。

## 4. Step-by-step plan

### Phase A — Forwarding backbone (可單獨 ship)

1. **新增 `gateway/subagent_topic_router.py`**
   - class `SubagentTopicRouter`:
     - `__init__(state_path, adapter_provider, logger)`
     - `async route(session_id, source, event_type, tool_name, preview, goal, **kw)`
     - private: `_ensure_topic(session_id, source, first_goal) -> thread_id`
     - private: `_load_state()` / `_save_state()` with file lock
     - private: `_format_event(event_type, ...) -> str`
   - Singleton 透過 `get_subagent_topic_router()` 取得

2. **修改 `gateway/run.py` `progress_callback` (line 8685)**
   - 加 branch: `if event_type.startswith("subagent."):`
     - 把事件放到 `progress_queue` 的**第二個 queue** (`subagent_queue`) 或直接 `asyncio.run_coroutine_threadsafe(router.route(...), _loop)`
     - **不要**讓 subagent.* 事件走進既有的 tool.started 行為（避免重複）
   - 既有 `if event_type not in ("tool.started",): return` 保留，確保 subagent.* **不會**同時進入 parent progress message

3. **在 `_run_agent` 開頭條件啟用 router**
   - 只有 `source.platform == Platform.TELEGRAM` 時才把 router 綁上 callback
   - 只有用戶 config 啟用（新 flag: `display.subagent_progress_topic: true`, 預設 `true`）時啟用

4. **Forum supergroup 偵測**
   - Telegram Bot API 無法 query chat 是否 forum → **惰性偵測**：第一次 `create_forum_topic` 失敗（`no_rights` / `chat not forum`）就把該 chat_id 加入 runtime-cache blocklist，該 parent session 後續 subagent progress 走舊路徑

5. **Unit tests**
   - `tests/test_subagent_topic_router.py`:
     - Lazy create on first event
     - State file round-trip
     - Event formatting
     - Non-TELEGRAM platform → no-op
     - Topic creation failure → fallback to parent thread

### Phase B — Reaper

6. **新增 `cron/subagent_topic_reaper.py`**
   - 純函式 `reap_expired_topics(now=None, ttl_seconds=86400) -> list[dict]`
   - CLI entrypoint: `python -m cron.subagent_topic_reaper`
   - 讀 same state file, 呼叫 `telegram_topic_tool` 內部函式（或重用 `_run_topic_op`）
   - 刪除後 update state 並 log 總結

7. **Tests**
   - `tests/test_subagent_topic_reaper.py`:
     - TTL boundary (23.9h keep, 24.1h delete)
     - Graceful handling of already-deleted topic
     - 空 state file

8. **部署 cron job** (執行層，非 code):
   - `mcp_cronjob action=create schedule="every 15m" prompt="Run subagent topic reaper" script="<path>/cron/subagent_topic_reaper.py"`
   - 或直接在 agent 啟動時註冊到 `cron/scheduler.py` 的 in-process scheduler（較佳，省一層外部依賴）

### Phase C — 整合驗證 (手測)

9. 在這個 thread (1981) 真的觸發一次 `delegate_task`，觀察：
   - 是否建立新 topic
   - 事件是否正確進入新 topic
   - 主對話不再被 subagent progress 洗
   - State file 內容正確
10. 手動把 state file 的 `last_message_ts` 改成 25h 前，跑 reaper，驗證 topic 被刪、state 被清

## 5. Files likely to change

**新增:**
- `gateway/subagent_topic_router.py`
- `cron/subagent_topic_reaper.py`
- `tests/test_subagent_topic_router.py`
- `tests/test_subagent_topic_reaper.py`
- (state) `~/.hermes/state/subagent_topics.json` (runtime only)

**修改:**
- `gateway/run.py` (progress_callback branching around line 8685; router wiring around line 9152)
- `cron/scheduler.py` (optional: 註冊 in-process reaper job)
- `docs/user-guide/features/delegation.md` (文件更新)
- config schema doc: 新增 `display.subagent_progress_topic` flag

**不動:**
- `gateway/platforms/api_server.py` (跟 Telegram 無關)
- `tools/delegate_tool.py` (現有 callback 已夠用)
- `tools/telegram_topic_tool.py` (會被 router 與 reaper import，不需改動)

## 6. Tests / validation

**自動:**
- `pytest tests/test_subagent_topic_router.py tests/test_subagent_topic_reaper.py -v`
- 既有 gateway tests 應全數通過（progress path 有 regression 風險）

**手動:**
1. Restart gateway → `/reload-mcp` 或整個重起
2. 在 thread 1981 發 `請 delegate_task 做 X` → 預期建新 topic
3. 到新 topic 看到事件流
4. 修改 state 時間戳 → 執行 reaper → topic 被刪、state 清空

## 7. Risks / tradeoffs / open questions

| 風險 | 緩解 |
|---|---|
| Telegram flood control (大量 subagent 事件) | Batch + 1.5s throttle 沿用；超限降級為只送 progress/complete |
| `create_forum_topic` 失敗卡住 parent turn | 用 try/except 包起來，失敗走 legacy 路徑 |
| State file race (多 parent turn 同時寫) | 檔案鎖 + atomic rename |
| Reaper 刪到使用者手動建的 topic | Router 只管自己寫入 state 的 topic；reaper 也只讀該 state，不碰其他 |
| Bot 被踢出 group 時 reaper 失敗 | `telegram_topic delete` 回 `chat_not_found` → 視為清除成功（removes from state） |
| TTL 指「最後事件時間」vs「最後一則訊息送出成功時間」 | 用「送出成功」為準（寫入 state 前先送訊息） |

**Open questions:**
- Q1: topic 名稱要不要包含 parent 對話的提問摘要？目前只用 session_id 前 8 碼 + 時間，比較穩定但較難辨識。
- Q2: 是否要在 parent 主對話放一則「🔀 progress → 見 topic #xxx」指引訊息？
- Q3: in-process scheduler (`cron/scheduler.py`) vs `mcp_cronjob` 哪個？in-process 簡單但 gateway 重啟會 reset 計時；cronjob 較穩健但多一層 IPC。

## 8. Out of scope

- CLI/TUI mode 的 subagent progress（它已有 tree view，不需要 topic 分流）
- 其他平台（Discord thread / Slack thread）的類似功能 — 不同 API 面，另案規劃
- Subagent final result 摘要到 parent — 目前 `_flush()` 已處理，不動
