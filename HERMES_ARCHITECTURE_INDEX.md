# Hermes 架構索引 (Commit: d0288be5b3330d2442e3907185b8e9d0958297bb)

## 1. 主要入口與模組索引
*   **命令列/UI 進入點** [程式已實作]
    *   **模組責任**: 處理終端機命令、引數解析，決定啟動單次請求 (oneshot) 或 TUI/Gateway，並處理中斷與生命週期管理。
    *   **主要輸入/輸出**: CLI 命令與引數 -> 設定環境並觸發 Agent 或子系統，輸出至終端。
    *   **與其他模組連接**: 處理完引數後，呼叫 `run_agent.py` 啟動 Agent，或讀取 `hermes_state.py` 獲取狀態。
    *   **檔案路徑與關鍵符號**:
        *   `cli.py` (`def main` 行 1653, `class HermesCLI` 行 883)
        *   `hermes_cli/main.py` (`def main` 行 3555)

*   **代理核心 (Agent Core)** [程式已實作]
    *   **模組責任**: 建立 `AIAgent` 執行個體，管理對話迴圈、呼叫工具 (Tool execution)、委派任務 (Delegation) 以及 Provider (模型) 請求。
    *   **主要輸入/輸出**: 接收對話 Context (messages) -> 與 API Provider 互動 -> 取得模型回應並決定下一步動作 (回應使用者或呼叫工具)。
    *   **與其他模組連接**: 處理請求時會讀寫 `hermes_state.py` 保存或讀取紀錄，並在需要時載入工具。
    *   **檔案路徑與關鍵符號**:
        *   `run_agent.py` (`class AIAgent` 行 240, `def main` 行 1496, `def _execute_tool_calls` 行 1320)

*   **狀態與持久化 (State & Database)** [程式已實作]
    *   **模組責任**: 整合對 SQLite 資料庫 (WAL mode) 的操作，涵蓋會話 (Session) 管理、歷史對話 (Messages)、使用量 (Usage)、全文檢索 (FTS5) 與鎖定守衛 (Lockguard)。
    *   **主要輸入/輸出**: 接收來自 Agent 或 CLI 的儲存/查詢請求 -> 讀寫本機 `state.db` -> 回傳資料行或確認。
    *   **與其他模組連接**: 被 `run_agent.py`、`cli.py` 等模組直接呼叫來進行記憶狀態同步；藉由多個 Mixin 分別管理不同資料表的狀態。
    *   **檔案路徑與關鍵符號**:
        *   `hermes_state.py` (`class SessionDB` 行 451, `class AsyncSessionDB` 行 1648)
        *   `hermes_state_sessions.py` (`class SessionSessionsMixin` 行 265)
        *   `hermes_state_messages.py` (`class SessionMessagesMixin` 行 122)

## 2. 高層資料流
1.  **訊息進入與路由** [程式已實作]: 使用者命令由 `cli.py` 進入，依照傳輸途徑決定路由，初始化環境並實例化 `AIAgent` (`hermes_cli/main.py` -> `run_agent.py`)。
2.  **執行與結果送出** [程式已實作]: `AIAgent` 將 Context 組裝為提示 (Prompt)，發送給 LLM Provider (`run_agent.py` 內)。若 LLM 返回工具呼叫，由 Agent 攔截於本地執行 (`_execute_tool_calls`)，將結果作為新訊息再度送回 LLM，直到最終結果產出並回傳給傳輸介面。
3.  **記憶/Context 使用** [程式已實作]: Agent 啟動或處理請求時，透過 `SessionDB` (`hermes_state.py` 內 `_read_ctx` 等方法) 載入特定 Session 的歷史紀錄以建立 Context。

## 3. 持久化位置索引
*   **主要資料庫 (寫入檔案)** [程式已實作]:
    *   **位置**: 預設為 `~/.hermes/state.db` (定義於 `hermes_constants.py` 的 `get_hermes_home()` 行 111)。
    *   **保存狀態類型**: 會話 Metadata (對話標題、設定)、歷史對話 (Messages)、工具使用紀錄、API Token 用量統計、FTS5 檢索索引 (`hermes_state_*.py` 中諸多 `def update*` 或 `def write*` 實作)。
*   **記憶體狀態** [程式已實作]:
    *   **部分佇列與統計區**: Token 用量的統計佇列 (定義於 `hermes_state_usage.py` 的 `_token_writer_loop`) 在批次落盤前只暫時存在於記憶體 (`queue.Queue`) 中。
*   **其他外掛或技能設定** [推論／未確認/待查]: 對於 `optional-skills` 或 `plugins` 目錄中的擴充套件，是否有在 `state.db` 以外寫入獨立快取或設定檔，目前尚未核實。

## 4. 後續值得拆開查的問題清單
*   **Agent 的工具呼叫與防護邊界**：在 `run_agent.py` 內，Tool calls 的執行與 guardrail 控制 (`_execute_tool_calls`, `_set_tool_guardrail_halt`) 流程細節？模型如何與各類本機 Tool 通訊？ (建議先讀 `run_agent.py`)
*   **狀態庫的高併發寫入鎖定機制**：在多執行緒與 Gateway 模式下，`SessionDB` 如何透過鎖確保安全與不產生寫入衝突？ (建議先讀 `hermes_state.py`、`hermes_state_lockowners.py`)
*   **對話壓縮機制 (Compression) 啟動與落盤**：當對話過長時，背景總結與對話壓縮是如何啟動且確保不會阻擋主對話流程？ (建議先讀 `hermes_state_compression.py` 與 `trajectory_compressor.py`)
