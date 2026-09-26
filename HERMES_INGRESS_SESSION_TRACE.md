# HERMES 訊息入口與 Session 身分追溯分析報告 (Telegram DM 文字訊息)

- **目標 Commit SHA**: `9fd30b85f15f4f07c413691b2f4a169990bbb3f0`
- **核心假設**: 本追蹤分析**嚴格假設為新建 Session (Fresh Session Creation Path)** 的情境。若為既有 Session 續用或歷史 Session 恢復/重置路徑 (`_route_recover`)，標示為待查邊界。
- **範疇說明**: 本文件僅針對 **Telegram 私聊 (DM) 的一則一般文字訊息**，記錄從平台 Adapter 接收 Update、身份判斷、Session key/id 計算，直到完成第一次 Session 持久化寫入 (State DB / JSON) 的完整調用鏈路與欄位細節。
- **免責與限制**:
  1. `HERMES_ARCHITECTURE_INDEX.md` 僅用作檔案搜尋索引，非核實之資料流依據。
  2. `gateway/stream_dispatch.py` 處理輸出串流，非輸入入口。
  3. 不包含模型執行、Turn 執行、結果送出、Memory 處理或其他平台邏輯。

---

## 流程總覽與追蹤步驟

| 步驟 | 階段說明 | 呼叫關係 (Caller → Callee) | 檔案路徑與關鍵符號/行號 | 使用身分欄位 | 寫入位置與資料類型 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 1** | PTB 訊息接收與 Event 建構 | `TelegramMessageHandler` → `_handle_text_message` | `plugins/platforms/telegram/adapter.py`<br>`_handle_text_message` (L6446) | `msg.from_user.id`<br>`msg.from_user.username`<br>`msg.chat.id`<br>`msg.chat.type` | **無持久化寫入**<br>(記憶體建立 `MessageEvent`) |
| **Step 2** | Adapter 文字批次緩衝與佇列 | `_handle_text_message` → `_enqueue_text_event` → `_flush_text_batch` → `handle_message` | `plugins/platforms/telegram/adapter.py`<br>`_enqueue_text_event` (L6519)<br>`_flush_text_batch` (L6571)<br>`gateway/platforms/base.py`<br>`handle_message` (L3954) | `event.source.platform`<br>`event.source.chat_id`<br>`event.source.user_id`<br>`event.source.chat_type` | **無持久化寫入**<br>(記憶體字典 `_pending_text_batches` 緩衝) |
| **Step 3** | Base Adapter 身分計算與 Session Task 啟動 | `handle_message` → `_event_session_key` → `_start_session_processing` | `gateway/platforms/base.py`<br>`handle_message` (L3954)<br>`_start_session_processing` (L3866) | `session_key`<br>(例如 `agent:main:telegram:dm:<chat_id>`) | **無持久化寫入**<br>(記憶體字典 `_active_sessions` 加鎖與 Task 建立) |
| **Step 4** | 異步 Task 執行 Handler 調用 | `_process_message_background` → `_message_handler` | `gateway/platforms/base.py`<br>`_process_message_background` (L4458) | `event`, `session_key` | **無持久化寫入** |
| **Step 5** | Profile Scope 封裝與 Gateway 切換 | `_make_default_profile_message_handler` → `_handle_message` | `gateway/run_adapters.py`<br>`_make_default_profile_message_handler` (L1607) | `event.source`, `profile_home` | **無持久化寫入** |
| **Step 6** | Gateway Inbound 准入與 Session Slot 鎖定 | `_handle_message` → `_hm_admit_event` → `_claim_active_session_slot` → `_persist_active_agents` → `_handle_message_with_agent` | `gateway/run_inbound.py`<br>`_handle_message` (L1275)<br>`gateway/run.py`<br>`_persist_active_agents` (L3989) | `_quick_key`, `source` | **Gateway 全域運行狀態寫入（非 Session 資料）**<br>檔案：`gateway_state.json`<br>欄位：`active_agents` (Integer, 活躍 Agent 數) |
| **Step 7** | Session 身分確定與 查/建 請求 | `_handle_message_with_agent` → `_hmwa_resolve_session` → `async_session_store.get_or_create_session` | `gateway/run_turn.py`<br>`_handle_message_with_agent` (L2161)<br>`_hmwa_resolve_session` (L404) | `source` (`platform`, `user_id`, `chat_id`, `chat_type="dm"`) | **無持久化寫入**<br>(發起 Session 解析流程) |
| **Step 8** | Single-Flight Session 建立決策 | `get_or_create_session` → `_get_or_create_session_impl` → `_route_create` | `gateway/session.py`<br>`get_or_create_session` (L881)<br>`_get_or_create_session_impl` (L912)<br>`_route_create` (L1020) | `session_key` (str)<br>`session_id` (str, 生成新 ID)<br>`origin` (SessionSource) | **記憶體狀態更新**<br>(設定 `decision.needs_save=True` 與 `decision.metadata_only_save=False`) |
| **Step 9** | **第一次 Session 本體持久化寫入** (Routing Index, Session Row & Peer) | `_save_entries` / `_finish_route_transition` → `create_session` / `record_gateway_session_peer` | `gateway/session.py`<br>`_get_or_create_session_impl` (L949)<br>`gateway/session_persistence.py`<br>`_save_entries` (L478) / `_persist_routing_data` (L440)<br>`gateway/session_recovery.py`<br>`_finish_route_transition` (L387)<br>`hermes_state_sessions.py`<br>`_insert_session_row` (L297)<br>`hermes_state_gateway.py`<br>`record_gateway_session_peer` (L226) | `session_id` (str)<br>`session_key` (str)<br>`source` ("telegram")<br>`user_id` (str)<br>`chat_id` (str)<br>`chat_type` ("dm")<br>`transport_profile` (str)<br>`origin_json` (str, JSON)<br>`display_name` (str) | **第一次 Session 持久化寫入（核心）**<br>1. SQLite 表 `gateway_routing` (TEXT/REAL)<br>2. Legacy JSON `sessions.json` (JSON 鏡像)<br>3. SQLite 表 `sessions` (TEXT/REAL，含主列與 Peer 更新) |

---

## 各步驟詳細數據流與身分欄位追蹤

### Step 1: Telegram Handler Entry
- **呼叫者/被呼叫者**: python-telegram-bot 事件循環 → `plugins/platforms/telegram/adapter.py:TelegramAdapter._handle_text_message()` (L6446)
- **傳入參數**: `update: Update`, `context: ContextTypes.DEFAULT_TYPE`
- **身分提取與轉換**:
  - 從 `update` 取得 `msg = self._effective_update_message(update)` (L6420)。
  - `_is_user_authorized_from_message(msg)` (L6450) 驗證身份，提取身分欄位：
    - `msg.from_user.id` (`user_id`: str)
    - `msg.from_user.username` (`user_name`: str)
    - `msg.chat.id` (`chat_id`: str)
    - `msg.chat.type` (`chat_type`: str，Telegram 私聊為 `"private"`，後續映射為 `"dm"`)
  - `_build_message_event(msg, MessageType.TEXT)` (L7047) 建立 `MessageEvent` 物件，包含 `SessionSource`:
    - `platform`: `Platform.TELEGRAM` (`"telegram"`)
    - `user_id`: `str(msg.from_user.id)`
    - `chat_id`: `str(msg.chat.id)`
    - `chat_type`: `"dm"`
- **持久化寫入**: **無**。

---

### Step 2: Adapter Text Batching & Debounce Buffer
- **呼叫者/被呼叫者**: `_handle_text_message()` → `_enqueue_text_event(event)` (L6519) → `_flush_text_batch(key)` (L6571) → `_flush_buffered()` (L6527) → `self.handle_message(event)` (L6550)
- **身分欄位**: `event.source` (`platform="telegram"`, `chat_id`, `user_id`, `chat_type="dm"`)
- **機制說明**:
  - `_text_batch_key(event)` (L6514) 計算分批 key，觸發 `_apply_topic_recovery(event)` (L6516)。
  - 事件進入記憶體字典 `self._pending_text_batches[key]` 緩衝，並由 `asyncio.create_task` 延遲調度 `_flush_text_batch`。
- **持久化寫入**: **無**。

---

### Step 3: Base Adapter Ingress & Active Session Task Spawn
- **呼叫者/被呼叫者**: `BasePlatformAdapter.handle_message(event)` (L3954) → `_start_session_processing(event, session_key)` (L3866)
- **身分與 Routing Key 確定**:
  - 檢測 `_drop_unresolved(event)` (L3974)。
  - 呼叫 `session_key = self._event_session_key(event)` (L3982)，計算出 Session 路由鍵（例如 `"agent:main:telegram:dm:<chat_id>"`）。
- **機制說明**:
  - 將 Session 鎖註冊至記憶體字典 `self._active_sessions[session_key] = guard` (L3872)。
  - 啟動異步 Task: `asyncio.create_task(self._process_message_background(event, session_key))` (L3873)。
- **持久化寫入**: **無**。

---

### Step 4: Background Message Handler Execution
- **呼叫者/被呼叫者**: `_process_message_background(event, session_key)` (L4458) → `response = await self._message_handler(event)` (L4475)
- **身分欄位**: `event`, `session_key`
- **持久化寫入**: **無**。

---

### Step 5: Gateway Profile Runtime Scope Wrapper
- **呼叫者/被呼叫者**: `_make_default_profile_message_handler()` (L1607 in `gateway/run_adapters.py`) → `GatewayRunnerInboundMixin._handle_message(event)` (L1618)
- **身分欄位**: `event.source`, `profile_home`
- **機制說明**:
  - `_admit_primary_source(event.source, default_home)` (L1615) 正規化 `source` 並決定 profile runtime 目錄。
  - 切換至 profile runtime 作用域 `async with _async_profile_runtime_scope(profile_home):` (L1617)。
- **持久化寫入**: **無**。

---

### Step 6: Gateway Inbound Ingress Processing (全域運行狀態記錄)
- **呼叫者/被呼叫者**: `GatewayRunnerInboundMixin._handle_message(event)` (L1275 in `gateway/run_inbound.py`) → `_handle_message_with_agent(...)` (L1349)
- **身分欄位**: `source` (`platform`, `chat_id`, `user_id`, `chat_type`), `_quick_key` (str)
- **機制說明**:
  - `_hm_admit_event(event)` (L1277) 通過准入門閥。
  - `_claim_active_session_slot(_quick_key, source)` (L1332) 搶佔 Session 執行槽。
  - 設定 Turn 狀態標記：`_claim_state.turn.agent = _AGENT_PENDING_SENTINEL` (L1339)。
  - 調用 `self._persist_active_agents()` (L1343)。
- **持久化寫入**:
  - **位置**: `<hermes_home>/gateway_state.json` (由 `_write_runtime_status_quiet` 寫入，`gateway/run.py` L3989)。
  - **資料類型/欄位**: JSON 格式，更新 `active_agents` (Integer, 代表當前全域活躍 Agent 數)。
  - **說明**: 此寫入僅為 Gateway 全域系統運行續計數，**並非 Session 本體資料或對話身分之持久化寫入**。

---

### Step 7: Session Determination
- **呼叫者/被呼叫者**: `GatewayRunnerTurnMixin._handle_message_with_agent(...)` (L2161 in `gateway/run_turn.py`) → `_hmwa_resolve_session(event, source)` (L2173) → `async_session_store.get_or_create_session(source, ...)` (L435)
- **身分欄位**: `source` (`platform="telegram"`, `user_id`, `chat_id`, `chat_type="dm"`, `thread_id=None`)
- **持久化寫入**: **無**。

---

### Step 8: Single-Flight Session Creation Execution
- **呼叫者/被呼叫者**: `AsyncSessionStore.get_or_create_session(source)` (L881 in `gateway/session.py`) → `_get_or_create_session_impl(source)` (L912) → `_route_create(...)` (L1020)
- **身分與物件生成**:
  - 計算 `session_key = self._generate_session_key(source)` (L887)。
  - 生成新 `session_id = _new_session_id(now)` (格式例: `YYYYMMDD-HHMMSS-xxxx`)。
  - 建立記憶體 `SessionEntry` (L1022) 並寫入記憶體快取 `self._entries[session_key]`。
  - 標記寫入需求：`decision.needs_save = True` (L1031)。由於為新建 Session，`decision.metadata_only_save` 保持為 `False`。
  - 組裝 DB 建立參數 `create_kwargs = self._session_create_kwargs(...)` (`gateway/session_recovery.py` L402)。
- **持久化寫入**: **無** (此步驟進行記憶體狀態準備，緊接著於 Step 9 觸發寫入)。

---

### Step 9: 第一次 Session 本體持久化寫入 (First Session Persistence Write)

在新建 Session 情境下，`_get_or_create_session_impl` 檢測到 `decision.needs_save == True` 且 `decision.metadata_only_save == False`，調用 `self._save_entries()` (L949 in `gateway/session.py`) 進行全量快照寫入，並接著執行 `_finish_route_transition()` (L959)：

#### 寫入 A: Session 路由索引寫入 (`gateway_routing` 與 `sessions.json`)
- **呼叫鏈路**: `_save_entries()` (L949) → `_persist_routing_data(data, generation)` (L440 in `gateway/session_persistence.py`)
- **寫入位置 1 (Primary)**: SQLite 數據庫 `state.db` 中的 `gateway_routing` 資料表
  - **方法**: `replace_gateway_routing_entries()` (L301 in `hermes_state_gateway.py`)
  - **寫入欄位與資料類型**:
    - `scope` (`TEXT`): Profile 作用域識別符 (如 `""` 或 `"main"`)
    - `session_key` (`TEXT`): 例如 `"agent:main:telegram:dm:12345678"`
    - `entry_json` (`TEXT`): 序列化之 `SessionEntry` JSON 字串，內含 `session_id`, `created_at`, `origin`, `platform`, `chat_type` 等
    - `updated_at` (`REAL` / `TIMESTAMP`): Epoch 時間戳記 (float)
- **寫入位置 2 (Legacy Mirror)**: 檔案 `<sessions_dir>/sessions.json`
  - **條件與失敗處置**:
    - **寫入條件**: 在 `_persist_routing_data()` (L456 in `gateway/session_persistence.py`) 中，先執行 SQLite `gateway_routing` 表寫入。僅當 `self._write_sessions_json == True` (預設) 或 SQLite 寫入失敗 (`not db_saved`) 時，才會呼叫 `_save_sessions_json()`。
    - **失敗情況處置**: 若 SQLite 已成功 commit (`db_saved == True`)，即使寫入 `sessions.json` 拋出 Exception（如權限不足或磁碟空間滿），系統會 catch 該例外並發出 warning (L466)，**不中斷對話流程且視為成功**，因為 `state.db` 是權威資料源；若 SQLite 與 JSON 皆寫入失敗才會拋出例外。

#### 寫入 B: Session 主記錄與 Peer 更新寫入 (`sessions` 資料表)
- **呼叫鏈路**: `_finish_route_transition(...)` (L959 in `gateway/session.py`) → `_create_session_row(...)` (L436 in `gateway/session_recovery.py`)
- **寫入位置**: SQLite 數據庫 `state.db` 中的 **`sessions` 資料表**（**無獨立之 `session_peers` 資料表**）。
- **子寫入 1 (主列建立)**:
  - **方法**: `SessionDB.create_session()` (L397 in `hermes_state_sessions.py`) → `_insert_session_row()` (L297 in `hermes_state_sessions.py`)
  - **SQL 語句**: `INSERT INTO sessions (id, source, user_id, session_key, chat_id, chat_type, thread_id, model, model_config, system_prompt_hash, parent_session_id, cwd, profile_name, transport_profile, git_repo_root, origin_json, display_name, started_at) VALUES (...) ON CONFLICT(id) DO UPDATE ...`
  - **寫入欄位與資料類型**:
    - `id` (`TEXT`): `session_id` (如 `"20260330-120000-abcd"`)
    - `source` (`TEXT`): `"telegram"`
    - `user_id` (`TEXT`): Telegram 發送者 User ID (如 `"12345678"`)
    - `session_key` (`TEXT`): 路由鍵 (如 `"agent:main:telegram:dm:12345678"`)
    - `chat_id` (`TEXT`): Chat ID (如 `"12345678"`)
    - `chat_type` (`TEXT`): `"dm"`
    - `thread_id` (`TEXT`): `NULL`
    - `profile_name` (`TEXT`): 所屬 Profile 名稱 (如 `"main"`)
    - `transport_profile` (`TEXT`): 傳輸 Profile 名稱 (如 `"main"`)
    - `origin_json` (`TEXT`): 序列化 `SessionSource` JSON 字串
    - `display_name` (`TEXT`): 顯示名稱
    - `started_at` (`REAL`): float epoch 時間戳記
- **子寫入 2 (Peer 欄位更新與自癒)**:
  - **方法**: `_record_gateway_session_peer()` (L442 in `gateway/session_recovery.py`) → `SessionDB.record_gateway_session_peer()` (L226 in `hermes_state_gateway.py`)
  - **SQL 語句**: `UPDATE sessions SET session_key = ?, source = ?, user_id = ?, chat_id = ?, chat_type = ?, thread_id = ?, display_name = COALESCE(?, display_name), origin_json = COALESCE(?, origin_json), transport_profile = COALESCE(?, transport_profile) WHERE id = ?`
- **寫入條件與失敗處置**:
  - `_create_session_row()` (L436 in `gateway/session_recovery.py`) 將 `create_session()` 與 `record_gateway_session_peer()` 包裹在 `try...except` 中。若寫入 SQLite `sessions` 表失敗，系統僅記錄 warning (L445: `"Failed to create session row ... deferring to the self-healing peer refresh on the next turn"`)，**捕獲例外不拋出**。缺失的資料列將在下次 Turn 的 peer refresh 自癒補齊。

---

## 第一次持久化寫入細節總結

1. **首次 Session 持久化寫入觸發點**: `gateway/session.py` 中的 `_get_or_create_session_impl()`，新建 Session 時呼叫 `_save_entries()` 與 `_finish_route_transition()`。
2. **寫入目標與實體位置**:
   - **SQLite 資料庫 (`state.db`)**:
     - 資料表 `gateway_routing`: 儲存路由鍵 (`session_key`) 到 SessionEntry JSON 的全量與增量寫入。
     - 資料表 `sessions`: 儲存 Session 實體主檔列，並經由 `record_gateway_session_peer()` 更新同表內的 peer 路由屬性欄位。
   - **JSON 檔案 (`sessions.json`)**:
     - 檔案位置: `<sessions_dir>/sessions.json`，作為 `gateway_routing` 的 Legacy 鏡像，寫入失敗不中斷主流程。
   - **系統狀態檔 (`gateway_state.json`)**:
     - 於 Step 6 准入時寫入全域 `active_agents` 計數，為 Gateway 運作狀態，非 Session 身分/資料記錄。

---

## 仍待查的邊界 (Unchecked / Pending Boundaries)

1. **既有 Session (Existing Session) 與 Session 恢復路徑 (`_route_recover`)**:
   - 若 Session 已存在，`decision.metadata_only_save` 為 `True`，會走單筆快寫 `_save_entry(session_key)` (L948)。若資料庫中存在未載入的 Session，會進入 `_route_recover()` (L1008) 進行 `state.db` 撈取與重開，此路徑暫列待查邊界。
2. **Telegram Multi-Account / Multi-Bot Profile 分流細節**:
   - 本追蹤假設 Telegram 使用預設或單一 Profile。若系統設定多 Bot 帳號 multiplex 模式，Profile 導向 (`_admit_primary_source`) 的 `transport_profile` 與 `profile_name` 配對邏輯涉及額外 Yaml 配置讀取與多 Profile 空間切換。
3. **Topic Mode / DM Topic Recovery 的邊界**:
   - 若 Telegram 使用者開啟了 DM Topic 模式 (`_recover_telegram_topic_thread_id`)，`thread_id` 可能會被替換為上次網絡活躍的 Topic ID，使 `session_key` 生成與 Topic binding 表格 (`telegram_topic_bindings`) 產生互動。
