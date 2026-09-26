# Hermes 架構索引 (Commit: d0288be5b3330d2442e3907185b8e9d0958297bb)

## 1. 主要入口與模組索引
*   **命令列/UI 進入點** [程式已實作]
    *   **模組責任**: 處理終端機命令、引數解析，決定啟動單次請求 (oneshot) 或 TUI/CLI 模式，並處理中斷與生命週期管理。
    *   **主要輸入/輸出**: CLI 命令與引數 -> 設定環境並觸發 Agent 核心。
    *   **檔案路徑與關鍵符號**: `cli.py` (`def main`, `class HermesCLI`), `hermes_cli/main.py` (`def main`)
*   **Gateway / 常駐服務邊界** [程式已實作]
    *   **模組責任**: 常駐服務模式，處理跨平台串接 (Telegram, WhatsApp, Web, 等)、群組與使用者對話狀態機、並負責非同步訊息接收與分發。
    *   **主要輸入/輸出**: 從外部 API (Webhook/Polling) 接收請求 -> 透過 `run_turn_runner.py` 啟動對應 Session 的 Agent -> 發送訊息回外部平台。
    *   **檔案路徑與關鍵符號**: `gateway/run.py` (`def main`), `gateway/stream_dispatch.py` (`def dispatch`), `gateway/run_turn_runner.py`
*   **排程與自動化 (Cron)** [程式已實作]
    *   **模組責任**: 背景排程任務觸發 (如定期總結、報告或定期檢查)。
    *   **主要輸入/輸出**: 時間觸發 -> 初始化特定環境與 Prompt (`scheduler_prompt.py`) -> 執行排程作業腳本。
    *   **檔案路徑與關鍵符號**: `cron/scheduler.py`
*   **代理核心 (Agent Core)** [程式已實作]
    *   **模組責任**: 建立 `AIAgent` 執行個體，管理對話迴圈、呼叫工具 (Tool execution)、委派任務 (Delegation) 以及向模型 (Provider) 請求。
    *   **主要輸入/輸出**: 接收對話 Context (messages) -> 與 API Provider 互動 -> 取得模型回應並決定下一步動作。
    *   **檔案路徑與關鍵符號**: `run_agent.py` (`class AIAgent`, `def _execute_tool_calls`)
*   **狀態與持久化 (State & Database)** [程式已實作]
    *   **模組責任**: 整合對 SQLite 資料庫 (WAL mode) 的讀寫操作，包含會話元資料、歷史對話、全域設定、使用量及 FTS5 全文檢索索引。
    *   **主要輸入/輸出**: 接收 Agent、Gateway 或 CLI 的儲存與查詢請求 -> 寫入 `state.db`。
    *   **檔案路徑與關鍵符號**: `hermes_state.py` (`class SessionDB`), 各式 `hermes_state_*.py`
*   **訊息鏡像與對外送出 (Delivery & Mirror)** [程式已實作]
    *   **模組責任**: 處理跨平台或外部發送機制的訊息傳遞，並同時將「已發送訊息」鏡像寫回 SQLite 的目標 Session (使 Agent 知悉已發送)。
    *   **主要輸入/輸出**: 本地產生的訊息 -> 平台適配器發送 -> 調用 state 寫入。
    *   **檔案路徑與關鍵符號**: `gateway/delivery.py` (`class DeliveryRouter`), `gateway/mirror.py` (`def mirror_to_session`)

## 2. 高層資料流
1.  **訊息進入與路由** [程式已實作]: 使用者命令或訊息可從 `cli.py` (本機端) 進入，或是從 `gateway/run.py` (遠端平台) 進入。Gateway 會根據來源建立或尋找 Session (`_find_session_id`)。
2.  **執行與結果送出** [程式已實作]: `run_agent.py` 將歷史與新訊息組裝為 Prompt 傳給 LLM。LLM 返回的文字會透過 `gateway/delivery.py` 回傳給外部；若是工具呼叫，由 Agent 在本地執行。
3.  **記憶/Context 使用** [程式已實作]: Agent 處理時透過 `SessionDB` (`hermes_state.py`) 載入對應會話的歷史。若透過 Cron 或 Gateway 主動發送訊息，會透過 `mirror.py` 把訊息補登入歷史，保持 Context 同步。

## 3. 持久化位置索引
*   **主要資料庫 (寫入檔案)** [程式已實作]:
    *   **位置**: 透過 `hermes_constants.py` 的 `get_hermes_home()` 與 `_get_platform_default_hermes_home()` 動態決定，通常為 `~/.hermes/state.db` (若設定 HERMES_HOME 或在 sudo 環境會改寫路徑)。
    *   **保存狀態類型**: 會話 Metadata、歷史對話 (Messages)、工具使用紀錄、FTS5 檢索索引。
*   **記憶體狀態** [程式已實作]:
    *   **Token 用量佇列**: `hermes_state_usage.py` 的 `_token_writer_loop` 會將統計數值暫存於 Queue 中批次寫入。
    *   **Gateway 快取**: 包含 Session Router / Broker 在執行期的路由表，存在於記憶體中 (依賴 `state.db` 來持久化識別碼)。
*   **其他外掛、Hooks 或設定** [待查]:
    *   外部模組 (`optional-skills` 等) 是否有在其獨立的目錄產生 artifacts、Cache，或在 Hook 觸發時有其他外部落盤點尚未確認。

## 4. 後續值得拆開查的問題清單
*   **Agent 的工具呼叫與防護邊界**：在 `run_agent.py` 內，Tool calls 的執行與 guardrail 控制 (`_execute_tool_calls`, `_set_tool_guardrail_halt`) 流程細節？模型如何與各類本機 Tool 通訊？ (建議先讀 `run_agent.py`)
*   **狀態庫的高併發寫入鎖定機制**：在多執行緒與 Gateway 模式下，`SessionDB` 如何透過鎖確保安全與不產生寫入衝突？ (建議先讀 `hermes_state.py`、`hermes_state_lockowners.py`)
*   **對話壓縮機制 (Compression) 啟動與落盤**：當對話過長時，背景總結與對話壓縮是如何啟動且確保不會阻擋主對話流程？ (建議先讀 `hermes_state_compression.py` 與 `trajectory_compressor.py`)
