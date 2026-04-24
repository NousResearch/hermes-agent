# Hermes-First 綜合實施計劃（整合版）

> **文檔版本**：v1.0  
> **創建日期**：2026-04-24  
> **狀態**：執行中  
> **用途**：Yao 的唯一執行參考文檔

---

## 一、系統拓撲與三節點角色定義

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Hermes-First 系統拓撲                              │
└─────────────────────────────────────────────────────────────────────────────┘

                            ┌─────────────────────┐
                            │   Cloud Ubuntu      │
                            │   (Hermes Primary)  │
                            │                     │
                            │  • Agent Core       │
                            │  • Knowledge Base   │
                            │  • Memory Provider  │
                            │  • Skills Manager   │
                            │  • Gateway (主)     │
                            │                     │
                            │  路徑:              │
                            │  /hermes/hermes-    │
                            │  agent/             │
                            └──────────┬──────────┘
                                       │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
        ┌───────────────────┐ ┌───────────────┐ ┌───────────────────┐
        │   MacBook         │ │  OpenClaw     │ │   Discord         │
        │   (localhost)     │ │  (Backup)     │ │   (UI Layer)      │
        │                   │ │               │ │                   │
        │  • Flask Dashboard│ │  • 熱備份     │ │  • 主交互介面     │
        │  • launchd 守護    │ │  • 故障轉移   │ │  • 消息通道       │
        │  • 本地監控        │ │  • 異地冗餘   │ │  • 指令路由       │
        │                   │ │               │ │                   │
        │  Port: 8080       │ │  路徑:        │ │  Webhook: 3000    │
        │  路徑: ~/hermes   │ │  /openclaw/   │ │                   │
        └───────────────────┘ └───────────────┘ └───────────────────┘

三節點職責矩陣：

┌──────────────┬────────────────────────────┬────────────────────────────┐
│    節點      │           職責              │           狀態             │
├──────────────┼────────────────────────────┼────────────────────────────┤
│ Cloud        │ • Hermes 主進程運行        │ ✅ 正常運行                 │
│ Ubuntu       │ • Knowledge Base 存儲      │                            │
│ (Primary)    │ • Memory Provider         │                            │
│              │ • Skills 加載與執行        │                            │
│              │ • Gateway 主實例           │                            │
├──────────────┼────────────────────────────┼────────────────────────────┤
│ MacBook      │ • Flask Dashboard (8080)   │ ✅ 正常運行                 │
│ (Local)      │ • launchd 進程管理         │                            │
│              │ • 本地開發調試             │                            │
│              │ • 圖形化監控界面           │                            │
├──────────────┼────────────────────────────┼────────────────────────────┤
│ OpenClaw     │ • 熱備份/故障轉移          │ ⚠️  待配置                  │
│ (Backup)     │ • 異地冗餘存儲             │                            │
│              │ • 峰值負載分流             │                            │
├──────────────┼────────────────────────────┼────────────────────────────┤
│ Discord      │ • 主 UI 交互介面           │ ✅ 正常運行                 │
│ (Interface)  │ • 消息路由與指令解析       │                            │
│              │ • 會話管理                 │                            │
└──────────────┴────────────────────────────┴────────────────────────────┘

通信流向：

Discord  ←──Webhook──→  Gateway (Cloud)  ──→  Agent Core
                                       │
                                       ▼
                               ┌───────────────┐
                               │ Context Engine │
                               └───────┬───────┘
                                       │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
   ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
   │ Knowledge Base│          │ Skills Base   │          │ Memory Base   │
   │   (靜態)      │          │   (技能)      │          │   (動態)      │
   └───────────────┘          └───────────────┘          └───────────────┘
```

---

## 二、 Constitution / Contracts / Implementation 三層分離架構

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           三層分離架構                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  第一層：Constitution（憲法層）                                              │
│  ────────────────────────────────────────────────────────────────────────── │
│  定義：系統的根本規則、價值觀、行為邊界                                       │
│  位置：/knowledge/facts/SOUL.md（靈魂定義）                                   │
│        /knowledge/policies/*.md（策略規則）                                  │
│                                                                             │
│  內容示例：                                                                  │
│  • AGENTS.md：Agent 的角色定義與權限邊界                                     │
│  • SOUL.md：系統核心價值觀與行為準則                                         │
│  • policies/：各類業務規則與約束                                             │
│                                                                             │
│  特性：                                                                      │
│  ✅ 人類可讀，易於審查                                                       │
│  ✅ 穩定，修改需要明確的變更流程                                              │
│  ✅ 是 Contracts 層的最終解釋依據                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  第二層：Contracts（契約層）                                                 │
│  ────────────────────────────────────────────────────────────────────────── │
│  定義：組件之間的介面約定、數據格式、交互協議                                 │
│  位置：/knowledge/docs/specs/（規格文檔）                                    │
│        /agent/adapters/base.py（適配器介面）                                │
│        /gateway/platforms/base.py（平台適配基類）                            │
│        /skills/.index.toml（技能索引契約）                                  │
│        /knowledge/index.json（知識庫索引契約）                              │
│                                                                             │
│  內容示例：                                                                  │
│  • context_engine.py：與 Memory Provider 的介面約定                        │
│  • memory_provider.py：記憶提供者必須實現的抽象介面                          │
│  • tools/registry.py：工具註冊的合約規範                                     │
│                                                                             │
│  特性：                                                                      │
│  ✅ 機器可解析，程序員友好                                                   │
│  ✅ 修改需要向後兼容或明確版本標注                                            │
│  ✅ Implementation 層的實現必須符合這些約定                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  第三層：Implementation（實現層）                                            │
│  ────────────────────────────────────────────────────────────────────────── │
│  定義：具體的代碼實現、功能細節                                               │
│  位置：所有 /agent/*.py（除 base.py 外）                                    │
│        所有 /gateway/*.py（除 base.py 外）                                  │
│        所有 /tools/*.py                                                     │
│        所有 /skills/*/                                                      │
│                                                                             │
│  特性：                                                                      │
│  ✅ 可以根據需要重構、優化、替換                                             │
│  ✅ 不影響 Contracts 層的約定                                                │
│  ✅ 不影響 Constitution 層的規則                                              │
│  ✅ 遵循「里氏替換原則」：任何實現都可以被符合介面的其他實現替換               │
└─────────────────────────────────────────────────────────────────────────────┘

三層依賴規則：

┌─────────────────────────────────────────────────────────────────────────────┐
│                           依賴方向                                          │
│                                                                             │
│   Constitution                                                               │
│       │                                                                     │
│       ▼ (解釋與約束)                                                        │
│   Contracts                                                                  │
│       │                                                                     │
│       ▼ (定義介面)                                                          │
│   Implementation                                                             │
│                                                                             │
│   依賴規則：Implementation → Contracts → Constitution                       │
│   嚴禁反向依賴！                                                             │
│                                                                             │
│   示例：                                                                     │
│   • memory_provider.py (Implementation) 實現 memory_provider.py::BaseMemory │
│     (Contracts) 中定義的介面                                                 │
│   • 永遠不允許在 memory_provider.py 中直接引用 SOUL.md 的內容作為判斷邏輯    │
│     （應該通過 Contracts 層暴露的 API 訪問）                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、Knowledge Base + Skills Base + Memory Base 整合設計

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     三庫整合架構                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           Knowledge Base                                    │
│  路徑：/home/ubuntu/.hermes/hermes-agent/knowledge/                         │
│  職責：靜態事實、策略規則、歷史文檔、系統憲法                                 │
│  特性：讀取為主，少量寫入（學習新知識時）                                    │
│                                                                             │
│  目錄結構：                                                                  │
│  knowledge/                                                                  │
│  ├── index.json           ← 統一索引（被所有組件引用）                      │
│  ├── docs/                ← 文檔（從 docs/ 遷移）                          │
│  │   ├── plans/                                                          │
│  │   ├── specs/                                                          │
│  │   └── migration/                                                      │
│  ├── facts/                ← 事實性知識                                     │
│  │   ├── SOUL.md          ← 系統靈魂/價值觀                                 │
│  │   ├── learnings.md     ← 學習記錄                                       │
│  │   └── incident-log/    ← 事故記錄                                       │
│  └── policies/             ← 策略規則                                      │
│      └── *.md                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
┌────────────────────────────────────────────┐
│              Skills Base                    │
│  路徑：/skills/                           │
│  職責：可執行技能、工具箱                   │
│  特性：按需加載，技能執行後寫入記憶庫       │
└────────────────────────────────────────────┘
         │                                    │                     │
         │                                    │                     │
         └────────────────┬───────────────────┘                     │
                          ▼                                         │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Context Engine                                     │
│  文件：/home/ubuntu/.hermes/hermes-agent/agent/context_engine.py           │
│  職責：協調三庫，構建每次對話的上下文                                       │
│                                                                             │
│  核心流程：                                                                  │
│                                                                             │
│  1. Memory.prefetch(query) ──→ 獲取會話歷史相關記憶                          │
│         │                                                                 │
│         ▼                                                                 │
│  2. Knowledge.search(query) ──→ 檢索相關靜態知識                            │
│         │                                                                 │
│         ▼                                                                 │
│  3. Skills.load_needed(skills) ──→ 載入必要技能                            │
│         │                                                                 │
│         ▼                                                                 │
│  4. 組裝上下文並返回                                                        │
│                                                                             │
│  關鍵約束：                                                                  │
│  • 三庫無循環依賴                                                            │
│  • Memory 是動態的，Knowledge 是靜態的                                     │
│  • Skills 執行結果可寫入 Memory                                              │
└─────────────────────────────────────────────────────────────────────────────┘

三庫交叉引用關係圖：

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  knowledge/index.json                                                       │
│      │                                                                      │
│      ├─→ agent/context_engine.py::build_knowledge_prompt()                 │
│      │                                                                      │
│      ├─→ agent/knowledge_memory_interface.py::search_knowledge()          │
│      │                                                                      │
│      └─→ tools/session_search_tool.py (語義檢索時)                         │
│                                                                             │
│  skills/.index.toml                                                         │
│      │                                                                      │
│      ├─→ agent/skill_commands.py::load_skill()                             │
│      │                                                                      │
│      ├─→ tools/skills_tool.py::skills_list()                               │
│      │                                                                      │
│      └─→ tools/skills_hub.py (hub 安裝時)                                   │
│                                                                             │
│  memory/MEMORY.md                                                           │
│      │                                                                      │
│      ├─→ agent/builtin_memory_provider.py (內建實現)                       │
│      │                                                                      │
│      ├─→ agent/memory_manager.py (統一調度)                                │
│      │                                                                      │
│      └─→ agent/knowledge_memory_interface.py (關聯知識庫)                  │
│                                                                             │
│  數據流向：                                                                  │
│  ┌──────────┐    讀取     ┌──────────┐    讀取     ┌──────────┐            │
│  │ Knowledge │───────────→│ Context  │───────────→│  LLM     │            │
│  │   Base    │            │  Engine  │            │          │            │
│  └──────────┘            └────┬─────┘            └──────────┘            │
│       ▲                       │                                         │
│       │  關聯寫入              │  技能執行                                 │
│       │                       ▼                                         │
│       │                ┌──────────────┐                                 │
│       └────────────────│    Memory    │◄─── Skills Base                  │
│                        │     Base     │       執行結果                    │
│                        └──────────────┘                                 │
│                              ▲                                           │
│                              │                                           │
│                        skills/.index.toml ──→ Skills                      │
│                                                 Base                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 四、三時間視角執行計劃（Horizons）

### 4.1 Horizon 1：Now（1-2 天）— 立即行動

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Horizon 1：Now（1-2 天）                                 │
│                    核心目標：建立目錄結構，確認系統可運行                      │
└─────────────────────────────────────────────────────────────────────────────┘

絕對優先順序（按順序執行）：

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 0（先行）：代碼修補（已執行，記錄在此）                                │
└─────────────────────────────────────────────────────────────────────────────┘
以下代碼修補已在本次 session 完成，Horizon 1 開始前已生效：

  A. SOUL.md 路徑向後兼容修補（4 個檔案）：
     • agent/prompt_builder.py
     • hermes_cli/config.py
     • hermes_cli/doctor.py
     • plugins/memory/retaindb/__init__.py
     → 嘗試新路徑 knowledge/facts/SOUL.md，fallback 到 SOUL.md

  B. delegate_task routing 修補（2 個檔案）：
     • tools/delegate_tool.py（_load_config merge 邏輯 + 單一任務 acp_args 傳遞）
     • tests/tools/test_delegate.py（新增回歸測試）

  C. config.yaml delegation 修補：
     • delegation.model: gpt-5.4
     • delegation.provider: openai-codex

---
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1：創建知識庫目錄結構（含所有子目錄）                                  │
└─────────────────────────────────────────────────────────────────────────────┘
命令：
```bash
mkdir -p /home/ubuntu/.hermes/hermes-agent/knowledge/{docs/{plans,specs,migration},facts,policies}
```
風險：無
回滾：rm -rf /home/ubuntu/.hermes/hermes-agent/knowledge/

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2：遷移文檔到知識庫                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
命令：
```bash
# 遷移 docs/ 下的內容
mv /home/ubuntu/.hermes/hermes-agent/docs/plans/* \
   /home/ubuntu/.hermes/hermes-agent/knowledge/docs/plans/
mv /home/ubuntu/.hermes/hermes-agent/docs/specs/* \
   /home/ubuntu/.hermes/hermes-agent/knowledge/docs/specs/
mv /home/ubuntu/.hermes/hermes-agent/docs/migration/* \
   /home/ubuntu/.hermes/hermes-agent/knowledge/docs/migration/

# 遷移根目錄文件
mv /home/ubuntu/.hermes/hermes-agent/AGENTS.md \
   /home/ubuntu/.hermes/hermes-agent/knowledge/docs/AGENTS.md
mv /home/ubuntu/.hermes/hermes-agent/SOUL.md \
   /home/ubuntu/.hermes/hermes-agent/knowledge/facts/SOUL.md
```
風險：低
回滾：反向 mv 即可（見 STEP 6）

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3：遷移記憶庫事實到知識庫                                               │
└─────────────────────────────────────────────────────────────────────────────┘
命令：
```bash
# 遷移 learnings 和 incident-log
mv /home/ubuntu/.hermes/hermes-agent/memory/learnings.md \
   /home/ubuntu/.hermes/hermes-agent/knowledge/facts/learnings.md
mv /home/ubuntu/.hermes/hermes-agent/memory/learnings-v2.md \
   /home/ubuntu/.hermes/hermes-agent/knowledge/facts/learnings-v2.md
mv /home/ubuntu/.hermes/hermes-agent/memory/OPTIMIZATION-HISTORY.md \
   /home/ubuntu/.hermes/hermes-agent/knowledge/facts/OPTIMIZATION-HISTORY.md
mv /home/ubuntu/.hermes/hermes-agent/memory/incident-log/ \
   /home/ubuntu/.hermes/hermes-agent/knowledge/facts/incident-log/
```
風險：低
回滾：反向 mv 即可

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4：創建知識庫索引                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
文件：/home/ubuntu/.hermes/hermes-agent/knowledge/index.json
```json
{
  "version": "1.0.0",
  "last_updated": "2026-04-24",
  "sections": {
    "docs": ["plans", "specs", "migration"],
    "facts": ["SOUL.md", "learnings.md", "OPTIMIZATION-HISTORY.md"],
    "policies": []
  }
}
```
風險：無
回滾：刪除該文件

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5：驗證系統可運行（含 Migration 正確性）                                │
└─────────────────────────────────────────────────────────────────────────────┘
命令：
```bash
cd /home/ubuntu/.hermes/hermes-agent

# 基本可運行驗證
python3 -c "from agent.context_engine import ContextEngine; print('OK')"
python3 -c "import os; assert os.path.exists('knowledge/index.json'); print('Knowledge index OK')"

# Migration 正確性驗證（全部存在才成功，任一失敗整體失敗）
python3 -c "
import os
checks = [
  ('knowledge/facts/SOUL.md', 'SOUL.md'),
  ('knowledge/docs/AGENTS.md', 'AGENTS.md'),
  ('knowledge/docs/plans', 'plans/'),
  ('knowledge/docs/specs', 'specs/'),
  ('knowledge/docs/migration', 'migration/'),
  ('knowledge/facts/learnings.md', 'learnings.md'),
  ('knowledge/facts/OPTIMIZATION-HISTORY.md', 'OPTIMIZATION-HISTORY.md'),
  ('memory/incident-log', 'incident-log/'),
]
for path, name in checks:
  assert os.path.exists(path), f'MISSING: {name}'
print('Migration: all OK')
"
```
風險：無

---

### 4.2 Horizon 2：Soon（1-2 週）— 技能庫與記憶庫整合 + OpenClaw 配置

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Horizon 2：Soon（1-2 週）                                 │
│  核心目標：建立技能索引，完善記憶-知識介面，配置 OpenClaw 熱備份             │
└─────────────────────────────────────────────────────────────────────────────┘

【OpenClaw 配置須知】
系統拓撲中 OpenClaw 承擔熱備份/故障轉移職責，但目前狀態為「待配置」。
以下 STEP 6-9 專注技能/記憶庫整合，OpenClaw 配置另需獨立的基礎設施規劃：
  • 同步方式：rsync / git pull / 共享儲存？
  • 故障轉移觸發條件：heartbeat 多久？自動還是半自動？
  • 同步頻率：即時？每分鐘？
  這些是 OpensClaw 配置的必備參數，不在此次重構計劃範圍內。

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6：創建技能索引（詳細實作）                                            │
└─────────────────────────────────────────────────────────────────────────────┘

目標：建立技能索引 TOML，替代現有的 skill_commands.py 硬編碼列表

前置條件：STEP 1-5 完成（knowledge/ 目錄已建立）

實作步驟：
1. 創建 /home/ubuntu/.hermes/hermes-agent/skills/.index.toml（內容見上）
2. 驗證 skills/ 下實際目錄存在：
   ```bash
   ls -d /home/ubuntu/.hermes/hermes-agent/skills/*/ | head -10
   ```
   確認 TOML 中引用的路徑（`skills.github`、`skills.software-development` 等）與實際目錄對應
3. 修改 agent/skill_commands.py：新增 read_index_toml() 函數，fallback 到現有列表
4. 修改 tools/skills_tool.py：skills_list() 改為讀取 .index.toml
5. 驗證：python3 -c "from skills import list_skills; print(list_skills())"

風險：低（fallback 保持向後兼容）
文件：/home/ubuntu/.hermes/hermes-agent/skills/.index.toml
```toml
[skills.github]
path = "github"
description = "GitHub code review, issues, PR workflow"
trust_level = "trusted"

[skills.software-development]
path = "software-development"
description = "Code development, debugging, deployment"
trust_level = "trusted"

[skills.research]
path = "research"
description = "Web research, information gathering"
trust_level = "trusted"

[skills.productivity]
path = "productivity"
description = "Calendar, email, task management"
trust_level = "trusted"

[skills.email]
path = "email"
description = "Email composition and management"
trust_level = "trusted"

[skills.feeds]
path = "feeds"
description = "RSS/Atom feed consumption"
trust_level = "trusted"

[skills.note-taking]
path = "note-taking"
description = "Note creation and organization"
trust_level = "trusted"

[skills.mcp]
path = "mcp"
description = "MCP protocol tools"
trust_level = "experimental"
```
修改文件：
• agent/skill_commands.py：新增從 .index.toml 讀取技能列表
• tools/skills_tool.py：使用 .index.toml 作為技能列表來源

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7：實現知識-記憶介面（詳細實作）                                        │
└─────────────────────────────────────────────────────────────────────────────┘

目標：建立 Knowledge 與 Memory 的關聯橋樑

前置條件：STEP 4（index.json 已建立）

實作步驟：
1. 新建 agent/knowledge_memory_interface.py
2. 實現 SearchResult dataclass、link_memory_to_knowledge()
3. 與 Memory Manager 集成
4. 驗證：python3 -c "from agent.knowledge_memory_interface import KnowledgeMemoryInterface; print('OK')"

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 8：增強 BuiltinMemoryProvider（詳細實作）                                │
└─────────────────────────────────────────────────────────────────────────────┘

目標：將內存提供者從 memory_provider.py 拆分出來，獨立維護

前置條件：STEP 7 完成（知識-記憶介面已建立）

實作步驟：
1. 新建 agent/builtin_memory_provider.py（從 memory_provider.py 抽取邏輯）
2. 修改 agent/memory_manager.py：使用新的 BuiltinMemoryProvider
3. 驗證：python3 -c "from agent.builtin_memory_provider import BuiltinMemoryProvider; print('OK')"

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 9：實現 Context Coordinator（新建而非改造現有 context_engine.py）      │
└─────────────────────────────────────────────────────────────────────────────┘

目標：新建獨立的 Context Coordinator，實現三庫協調邏輯

前置條件：STEP 6-8 完成（Skills 索引、知識-記憶介面、內存提供者已建立）

說明：現有 context_engine.py 是上下文壓縮基類，三庫協調是不同職責。因此新建 context_coordinator.py 而非改造現有檔。

實作步驟：
1. 新建 agent/context_coordinator.py：實現 ContextCoordinator 類
2. 實現 Memory.prefetch() → Knowledge.search() → Skills.load_needed() 流程
3. 在 agent/context_engine.py 同一目錄下作為獨立模組維護
4. 驗證：python3 -c "from agent.context_coordinator import ContextCoordinator; print('OK')"

---

### 4.3 Horizon 3：Later（1 個月+）— Agent Core 拆分與 Gateway 瘦身

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Horizon 3：Later（1 個月+）                                │
│  核心目標：重構核心組件，建立長期可維護架構                   │
└─────────────────────────────────────────────────────────────────────────────┘

⚠️ 前置相依：STEP 9（Context Coordinator）必須完成後才能開始 STEP 10-12。
   原因：兩者都操作 agent/ 目錄下的文件。STEP 9 完成後，
   agent/context_coordinator.py 與 STEP 10 的 anthropic_adapter 拆分不會衝突。
   嚴禁並發執行 STEP 9 和 STEP 10。

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 10：Agent Core 拆分（高風險，需謹慎）                                   │
└─────────────────────────────────────────────────────────────────────────────┘
備份：
```bash
cp agent/anthropic_adapter.py agent/anthropic_adapter.py.backup
cp agent/prompt_builder.py agent/prompt_builder.py.backup
```

拆分：
```
agent/anthropic_adapter.py → agent/adapters/
                              ├── base.py      (公共介面)
                              ├── anthropic.py (API 調用)
                              └── openai.py    (OpenAI 適配)

agent/prompt_builder.py     → agent/prompts/
                              ├── system_prompt.py
                              ├── user_prompt.py
                              └── injection.py
```

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 11：Gateway 瘦身（高風險，需完整備份）                                 │
└─────────────────────────────────────────────────────────────────────────────┘
備份：
```bash
cp gateway/run.py gateway/run.py.backup
```

拆分：
```
gateway/run.py → gateway/
                 ├── session_manager.py  (Session 管理)
                 ├── display.py          (Display 邏輯)
                 └── run.py              (瘦身後的核心調度)
```

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 12：工具層分組重構                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
新建文件：/home/ubuntu/.hermes/hermes-agent/tools/tool_groups.py
修改文件：/home/ubuntu/.hermes/hermes-agent/tools/registry.py（延遲加載）

---

## 五、變更紀律（Change Discipline）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         變更紀律章節                                          │
│                                                                             │
│  目的：確保未來任何更新都不需要重新設計整個系統                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 5.1 核心原則                                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

  ① 三層分離不變：
     Constitution → Contracts → Implementation
     任何變更只能在 Implementation 層進行，
     除非是新增 Constitution 規則（需人類審批）。

  ② Contracts 層向後兼容：
     任何 Contracts 層的修改必須保持向後兼容。
     如果無法向後兼容，必須新增版本號（如 base_v2.py）。

  ③ 禁止循環依賴：
     Knowledge → Memory → Skills → Context Engine
     這個方向是允許的，反向則嚴禁。

┌─────────────────────────────────────────────────────────────────────────────┐
│ 5.2 變更分類                                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

  A. 小改動（可以直接執行）：
     • 修復 bug
     • 優化性能
     • 添加注釋
     • 重命名不影響介面的變數

  B. 中改動（需要測試）：
     • 新增工具到現有組
     • 修改已有函數的實現方式
     • 新增可選參數
     • 修改工具分組

  C. 大改動（需要完整流程）：
     • 新增或刪除 Contracts 層的介面
     • 移動或重命名目錄結構
     • 修改三層之間的依賴關係
     • 新增或刪除技能

┌─────────────────────────────────────────────────────────────────────────────┐
│ 5.3 大改動的變更流程                                                         │
└─────────────────────────────────────────────────────────────────────────────┘

  步驟：
  1. 在 /knowledge/docs/plans/ 下創建變更提案（change-proposal-YYYY-MM-DD.md）
  2. 說明：
     - 為什麼需要這個變更
     - 影響範圍
     - 回滾方案
     - 測試計劃
  3. 獲得 Yao 的批准
  4. 創建備份
  5. 執行變更
  6. 執行測試
  7. 更新相關文檔
  8. 合併到主分支

┌─────────────────────────────────────────────────────────────────────────────┐
│ 5.4 未來新增功能時的集成檢查清單                                             │
└─────────────────────────────────────────────────────────────────────────────┘

  在合併任何新功能前，必須確認：

  □ 新功能位於 Implementation 層
  □ 不會引入與 Constitution 層的直接依賴
  □ 遵循現有 Contracts 層的介面約定
  □ 不會引入循環依賴
  □ 有明確的回滾方案
  □ 已更新相關索引文件（knowledge/index.json, skills/.index.toml）

┌─────────────────────────────────────────────────────────────────────────────┐
│ 5.5 版本標注規則                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

  當 Contracts 層發生不兼容變更時：
  • 保留舊版本（base.py）並標注廢棄時間
  • 新增新版本（base_v2.py）
  • 在 /knowledge/docs/plans/ 中記錄版本變更
  • 設置6個月的過渡期

```

---

## 六、回滾計劃（Rollback Plan）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           回滾計劃                                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 6.1 即時回滾（Horizon 1 階段）                                              │
│     風險：低，無需備份即可恢復                                               │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1：如有 knowledge/ 目錄，先恢復遷移的文檔（使用前先檢查目錄是否存在）
```bash
cd /home/ubuntu/.hermes/hermes-agent

# 先檢查目錄是否存在
if [ -d "knowledge/" ]; then
  # 恢復遷移的文檔（每個 mv 前先檢查檔案是否存在）
  [ -f knowledge/docs/plans/* ] && mv knowledge/docs/plans/* docs/plans/
  [ -f knowledge/docs/specs/* ] && mv knowledge/docs/specs/* docs/specs/
  [ -f knowledge/docs/migration/* ] && mv knowledge/docs/migration/* docs/migration/

  # 恢復根目錄文件
  [ -f knowledge/docs/AGENTS.md ] && mv knowledge/docs/AGENTS.md AGENTS.md
  [ -f knowledge/facts/SOUL.md ] && mv knowledge/facts/SOUL.md SOUL.md

  # 恢復記憶庫事實
  [ -f knowledge/facts/learnings.md ] && mv knowledge/facts/learnings.md memory/learnings.md
  [ -f knowledge/facts/learnings-v2.md ] && mv knowledge/facts/learnings-v2.md memory/learnings-v2.md
  [ -f knowledge/facts/OPTIMIZATION-HISTORY.md ] && mv knowledge/facts/OPTIMIZATION-HISTORY.md memory/OPTIMIZATION-HISTORY.md
  [ -d knowledge/facts/incident-log/ ] && mv knowledge/facts/incident-log/ memory/incident-log/

  # 刪除知識庫目錄
  rm -rf knowledge/
else
  echo "knowledge/ 目錄不存在，跳過回滾"
fi
```

STEP 2：如有代碼修補需要回滾，恢復原始版本
```bash
cd /home/ubuntu/.hermes/hermes-agent

# 恢復 SOUL.md 路徑修補
git checkout -- agent/prompt_builder.py hermes_cli/config.py \
                  hermes_cli/doctor.py plugins/memory/retaindb/__init__.py

# 恢復 delegate_tool 修補
git checkout -- tools/delegate_tool.py
git checkout -- tests/tools/test_delegate.py

# 恢復 config.yaml delegation（如有必要）
git checkout -- /home/ubuntu/.hermes/config.yaml

echo "回滾完成"
```

┌─────────────────────────────────────────────────────────────────────────────┐
│ 6.2 中期回滾（Horizon 2 階段）                                              │
│     風險：中，需恢復技能索引和介面                                           │
└─────────────────────────────────────────────────────────────────────────────┘

命令序列：
```bash
cd /home/ubuntu/.hermes/hermes-agent

# 刪除新建的索引文件
rm -f skills/.index.toml
rm -f agent/knowledge_memory_interface.py
rm -f agent/builtin_memory_provider.py

# 恢復修改的代碼文件
git checkout -- agent/skill_commands.py
git checkout -- agent/memory_manager.py
git checkout -- agent/context_engine.py

echo "回滾完成"
```

┌─────────────────────────────────────────────────────────────────────────────┐
│ 6.3 高風險回滾（Horizon 3 階段）                                             │
│     風險：高，必須有完整備份                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1：保留 .backup 備份（階段開始時已創建）

STEP 2：執行回滾
```bash
cd /home/ubuntu/.hermes/hermes-agent

# 恢復舊版文件
mv agent/anthropic_adapter.py.backup agent/anthropic_adapter.py
mv agent/prompt_builder.py.backup agent/prompt_builder.py
mv gateway/run.py.backup gateway/run.py

# 刪除拆分出來的新目錄
rm -rf agent/adapters/
rm -rf agent/prompts/

echo "回滾完成"
```

┌─────────────────────────────────────────────────────────────────────────────┐
│ 6.4 緊急完全回滾（萬能的 git 方案）                                          │
│     風險：極低，但會丟失所有未提交的更改                                      │
└─────────────────────────────────────────────────────────────────────────────┘

```bash
cd /home/ubuntu/.hermes/hermes-agent

# 方案A：保留當前更改到 stash
git stash
git checkout HEAD -- .
echo "已恢復到 git 版本，當前更改已保存到 stash"

# 方案B：直接覆蓋（危險）
git checkout HEAD -- .
echo "已恢復到 git 版本，所有未提交的更改已丟失"
```

┌─────────────────────────────────────────────────────────────────────────────┐
│ 6.5 回滾後驗證                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

無論使用哪種回滾方案，回滾後必須執行：

```bash
cd /home/ubuntu/.hermes/hermes-agent

# 1. 驗證目錄結構
ls -la knowledge/ 2>/dev/null && echo "ERROR: knowledge/ 仍存在" || echo "OK: knowledge/ 已刪除"

# 2. 驗證關鍵文件
test -f agent/anthropic_adapter.py && echo "OK: anthropic_adapter.py 存在" || echo "ERROR: 缺少 anthropic_adapter.py"
test -f agent/prompt_builder.py && echo "OK: prompt_builder.py 存在" || echo "ERROR: 缺少 prompt_builder.py"
test -f gateway/run.py && echo "OK: gateway/run.py 存在" || echo "ERROR: 缺少 gateway/run.py"

# 3. 驗證系統可運行
python3 -c "from agent.context_engine import ContextEngine; print('OK: ContextEngine 可導入')"

# 4. 驗證 Discord 連接（如有條件）
# python3 -c "from gateway.platforms.discord import DiscordPlatform; print('OK')"
```

---

<!-- SECTION 7 DELETED — 內容已整合至 Section 四的 Horizon 1，保留以避免衝突 -->

<!-- Section 7 removed. All content unified into Horizon 1 (Section 四, STEP 0-5). -->

---

## 八、執行檢查清單

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Horizon 1 完成檢查                                    │
└─────────────────────────────────────────────────────────────────────────────┘

□ 知識庫目錄已創建：knowledge/{docs/{plans,specs,migration},facts,policies}
□ SOUL.md 已遷移到 knowledge/facts/SOUL.md
□ AGENTS.md 已遷移到 knowledge/docs/AGENTS.md
□ 文檔已遷移：plans/, specs/, migration/
□ learnings.md 已遷移到 knowledge/facts/
□ OPTIMIZATION-HISTORY.md 已遷移到 knowledge/facts/
□ incident-log/ 已遷移到 knowledge/facts/
□ knowledge/index.json 已創建
□ 系統可正常導入 ContextEngine
□ Discord 連接測試通過

代碼修補（已在本次 session 完成，無需重做）：
✅ SOUL.md 路徑向後兼容修補（4 個檔案）
✅ delegate_task _load_config merge 修補
✅ config.yaml delegation: gpt-5.4 / openai-codex
✅ 回歸測試新增（test_delegate.py）

┌─────────────────────────────────────────────────────────────────────────────┐
│                        Horizon 2 完成檢查                                    │
└─────────────────────────────────────────────────────────────────────────────┘

□ skills/.index.toml 已創建
□ agent/knowledge_memory_interface.py 已創建
□ agent/builtin_memory_provider.py 已創建
□ agent/skill_commands.py 已更新（使用 .index.toml）
□ tools/skills_tool.py 已更新（使用 .index.toml）
□ agent/context_engine.py 已更新（支持三庫協調）
□ Memory 和 Knowledge 的關聯功能已實現

┌─────────────────────────────────────────────────────────────────────────────┐
│                        Horizon 3 完成檢查                                    │
└─────────────────────────────────────────────────────────────────────────────┘

□ agent/anthropic_adapter.py 已拆分到 agent/adapters/
□ agent/prompt_builder.py 已拆分到 agent/prompts/
□ gateway/run.py 已瘦身
□ tools/tool_groups.py 已創建
□ tools/registry.py 已支持延遲加載
□ 所有舊文件 .backup 已清理或保留

┌─────────────────────────────────────────────────────────────────────────────┐
│                        最終狀態                                              │
└─────────────────────────────────────────────────────────────────────────────┘

□ 三層分離架構已建立
□ 三庫整合已實現
□ 變更紀律已定義
□ 回滾方案已測試
□ 文檔已更新
□ 系統穩定運行
```

---

## 九、附錄：相關文件路徑索引

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        文件路徑索引                                          │
└─────────────────────────────────────────────────────────────────────────────┘

【新建文件】
• /home/ubuntu/.hermes/hermes-agent/knowledge/index.json         （知識庫統一索引）
• /home/ubuntu/.hermes/hermes-agent/skills/.index.toml           （技能索引）
• /home/ubuntu/.hermes/hermes-agent/agent/builtin_memory_provider.py
• /home/ubuntu/.hermes/hermes-agent/agent/knowledge_memory_interface.py
• /home/ubuntu/.hermes/hermes-agent/agent/adapters/base.py
• /home/ubuntu/.hermes/hermes-agent/agent/adapters/anthropic.py
• /home/ubuntu/.hermes/hermes-agent/agent/adapters/openai.py
• /home/ubuntu/.hermes/hermes-agent/agent/prompts/system_prompt.py
• /home/ubuntu/.hermes/hermes-agent/agent/prompts/user_prompt.py
• /home/ubuntu/.hermes/hermes-agent/agent/prompts/injection.py
• /home/ubuntu/.hermes/hermes-agent/gateway/session_manager.py
• /home/ubuntu/.hermes/hermes-agent/gateway/platforms/base.py
• /home/ubuntu/.hermes/hermes-agent/tools/tool_groups.py

【修改文件】
• /home/ubuntu/.hermes/hermes-agent/agent/context_engine.py       （三庫協調）
• /home/ubuntu/.hermes/hermes-agent/agent/skill_commands.py         （技能索引）
• /home/ubuntu/.hermes/hermes-agent/agent/memory_manager.py         （記憶管理）
• /home/ubuntu/.hermes/hermes-agent/tools/skills_tool.py            （技能列表）
• /home/ubuntu/.hermes/hermes-agent/tools/registry.py               （延遲加載）
• /home/ubuntu/.hermes/hermes-agent/gateway/run.py                   （瘦身）

【遷移後需刪除的舊文件】（待遷移，Horizon 1 STEP 2-3 執行後生效）
• /home/ubuntu/.hermes/hermes-agent/docs/plans/*                   （已遷移）
• /home/ubuntu/.hermes/hermes-agent/docs/specs/*                    （已遷移）
• /home/ubuntu/.hermes/hermes-agent/docs/migration/*                （已遷移）
• /home/ubuntu/.hermes/hermes-agent/AGENTS.md                       （已遷移）
• /home/ubuntu/.hermes/hermes-agent/SOUL.md                         （已遷移）
• /home/ubuntu/.hermes/hermes-agent/memory/learnings.md             （已遷移）
• /home/ubuntu/.hermes/hermes-agent/memory/learnings-v2.md          （已遷移）
• /home/ubuntu/.hermes/hermes-agent/memory/OPTIMIZATION-HISTORY.md （已遷移）
• /home/ubuntu/.hermes/hermes-agent/memory/incident-log/            （已遷移）

【Backup 文件（回滾用）】
• /home/ubuntu/.hermes/hermes-agent/agent/anthropic_adapter.py.backup
• /home/ubuntu/.hermes/hermes-agent/agent/prompt_builder.py.backup
• /home/ubuntu/.hermes/hermes-agent/gateway/run.py.backup
```

---

**文檔結束**

*本文檔是 Yao 執行 Hermes-First 重構的唯一參考文檔。*
*所有變更必須遵循「變更紀律」章節的規定。*
*如有任何疑問，請先查閱 Constitution / Contracts / Implementation 三層分離架構。*
