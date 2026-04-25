# Hermes-First 架構重設計：實施計劃

## 現狀診斷

| 層面 | 現有路徑 | 問題 |
|------|----------|------|
| 知識庫 | `memory/`（MEMORY.md、learnings.md、incident-log/） | 散落於多個 Markdown 檔案，無統一索引 |
| 技能庫 | `skills/` | 與 OpenClaw 共用，無版本控制，無明確繼承關係 |
| 記憶庫 | `agent/memory_provider.py` + `BuiltinMemoryProvider` | 僅支援一個外部 provider，缺乏持久化查詢介面 |
| Agent 核心 | `agent/anthropic_adapter.py`（57KB）、`agent/prompt_builder.py`（46KB） | 高度耦合，缺乏職責分離 |
| Gateway | `gateway/run.py`（448KB） | 單體過大，平台適配邏輯分散 |
| 工具層 | `tools/`（60+ 工具） | 工具 schema 膨脹，無分組機制 |

---

## 目標架構

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Hermes Core                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │  Knowledge   │  │   Skills     │  │        Memory             │ │
│  │    Base      │  │    Base      │  │        Base              │ │
│  │              │  │              │  │                          │ │
│  │ knowledge/   │  │ skills/      │  │ memory_provider.py       │ │
│  │ - docs/      │  │ - github/     │  │ BuiltinMemoryProvider    │ │
│  │ - facts/     │  │ - mcp/       │  │ External providers:       │ │
│  │ - policies/  │  │ - software/  │  │   - Honcho               │ │
│  │ - index.json │  │ - DESCR.toml │  │   - (pluggable)          │ │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘ │
│         │                 │                       │                │
│         └─────────────────┼───────────────────────┘                │
│                           ▼                                        │
│              ┌────────────────────────┐                            │
│              │   Context Engine       │                            │
│              │ (agent/context_engine) │                            │
│              └───────────┬────────────┘                            │
│                          │                                          │
│  ┌───────────────────────┼───────────────────────────────────────┐  │
│  │              Agent Core (agent/)                              │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │  │
│  │  │ Anthropic  │ │  Prompt    │ │  Memory    │ │  Skill     │  │  │
│  │  │ Adapter    │ │  Builder   │ │  Manager   │ │  Commands  │  │  │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    Gateway Layer (gateway/)                     │  │
│  │  run.py (瘦核心) │ platforms/ │ builtin_hooks/                 │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    Tools Layer (tools/)                         │  │
│  │  registry.py │ skill_manager_tool.py │ memory_tool.py          │  │
│  └────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三系統整合設計

### 知識庫（Knowledge Base）
- **路徑**：`knowledge/`（新建）
- **內容**：`docs/`（從 `docs/` 遷移）、`facts/`（事實性知識）、`policies/`（策略規則）、`index.json`（統一索引）
- **被引用方式**：通過 `agent/context_engine.py` 的 `ContextEngine.build_knowledge_prompt()` 注入系統提示
- **與記憶庫的關係**：知識庫提供靜態事實，記憶庫提供會話級別上下文

### 技能庫（Skills Base）
- **路徑**：`skills/`（現有結構）
- **內容**：各技能目錄下的 `SKILL.md` + `references/` + `templates/`
- **被引用方式**：通過 `agent/skill_commands.py` 和 `tools/skills_tool.py` 的 `skill_view()` 動態載入
- **與記憶庫的關係**：技能執行結果寫入記憶庫（通過 `MemoryManager.on_memory_write()` hook）

### 記憶庫（Memory Base）
- **路徑**：`memory/`（現有）
- **核心**：`agent/memory_provider.py`（抽象介面）+ `BuiltinMemoryProvider`（內建實現）
- **被引用方式**：`MemoryManager` 在每次 turn 調用 `prefetch()` 和 `sync_turn()`
- **與知識庫的關係**：記憶庫的 `prefetch()` 可查詢知識庫索引以實現語義檢索

---

## 分階段遷移計劃

### 第一階段：目錄重構與基礎設施（1-2 天）

**目標**：建立新目錄結構，隔離 concerns

#### 1.1 新建知識庫目錄
```bash
mkdir -p /home/ubuntu/.hermes/hermes-agent/knowledge/{docs,facts,policies}
```

#### 1.2 遷移檔案（保持不變，只搬動位置）
| 操作 | 源 | 目標 | 風險 |
|------|-----|------|------|
| 遷移 | `docs/plans/` | `knowledge/docs/plans/` | 低 |
| 遷移 | `docs/specs/` | `knowledge/docs/specs/` | 低 |
| 遷移 | `docs/migration/` | `knowledge/docs/migration/` | 低 |
| 遷移 | `AGENTS.md` | `knowledge/docs/AGENTS.md` | 低 |
| 遷移 | `SOUL.md` | `knowledge/facts/SOUL.md` | 低 |
| 遷移 | `MEMORY.md` | `memory/MEMORY.md`（已存在，跳過） | - |
| 遷移 | `learnings.md` | `knowledge/facts/learnings.md` | 低 |
| 遷移 | `incident-log/` | `knowledge/facts/incident-log/` | 低 |
| 遷移 | `memory/OPTIMIZATION-HISTORY.md` | `knowledge/facts/OPTIMIZATION-HISTORY.md` | 低 |

**涉及檔案**：
- `docs/plans/2026-03-16-pricing-accuracy-architecture-design.md`
- `docs/specs/container-cli-review-fixes.md`
- `docs/migration/openclaw.md`
- `AGENTS.md`
- `SOUL.md`
- `memory/learnings.md`
- `memory/learnings-v2.md`
- `memory/incident-log/`
- `memory/OPTIMIZATION-HISTORY.md`

#### 1.3 建立知識庫索引
**新建檔案**：`knowledge/index.json`
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

**涉及檔案**：`knowledge/index.json`（新建）

#### 1.4 更新 import 路徑
- `agent/context_engine.py`：新增 `build_knowledge_prompt()` 方法，從 `knowledge/` 讀取
- `run_agent.py`：在初始化時加載知識庫索引

**涉及檔案**：
- `agent/context_engine.py`（修改）
- `run_agent.py`（修改）

---

### 第二階段：技能庫重構（2-3 天）

**目標**：將技能庫從 OpenClaw 遷移的殘留中解放，建立 hermes-native 技能管理

#### 2.1 技能結構標準化
**保持**：`skills/github/`、`skills/software-development/`、`skills/research/` 等 8 個技能分類

**遷移**：
| 操作 | 源 | 目標 | 風險 |
|------|-----|------|------|
| 遷移 | `skills/email/` | `skills/email/`（已在正確位置） | - |
| 遷移 | `skills/feeds/` | `skills/feeds/`（已正確） | - |

**新建**：`skills/.index.toml`（技能索引）
```toml
[skills.github]
path = "github"
description = "GitHub code review, issues, PR workflow"
trust_level = "trusted"

[skills.software-development]
path = "software-development"
description = "Code development, debugging, deployment"
trust_level = "trusted"
```

**涉及檔案**：
- `skills/.index.toml`（新建）
- `skills/DESCRIPTION.md`（若存在，遷至 `knowledge/docs/skills-DESCRIPTION.md`）

#### 2.2 技能與知識庫的關聯
在 `knowledge/index.json` 中新增 `linked_skills` 欄位：
```json
{
  "docs": {
    "linked_skills": ["software-development", "github"]
  }
}
```

#### 2.3 技能執行鉤子與記憶庫整合
修改 `agent/skill_commands.py`：在技能執行後調用 `MemoryManager.on_memory_write()`

**涉及檔案**：`agent/skill_commands.py`（修改）

---

### 第三階段：記憶庫增強（2-3 天）

**目標**：實現可插拔的外部記憶 provider，支援語義檢索

#### 3.1 增强 BuiltinMemoryProvider
**新建**：`agent/builtin_memory_provider.py`（從 `agent/memory_provider.py` 拆分）

**保持功能**：
- `MEMORY.md` 讀寫
- `USER.md` 讀寫
-  session-scoped 上下文

**新增功能**：
- `prefetch()` 實現（調用 `session_search_tool.py` 的語義搜索）
- 與 `knowledge/index.json` 的索引查詢介面

#### 3.2 實現外部 Provider 介面
**保持不變**：`agent/memory_provider.py`（抽象基類）

**新建 provider 示例**：`plugins/memory/honcho/`（若啟用 Honcho）

#### 3.3 知識庫-記憶庫查詢介面
**新建**：`agent/knowledge_memory_interface.py`
```python
class KnowledgeMemoryInterface:
    def search_knowledge(self, query: str) -> List[SearchResult]:
        """記憶庫查詢知識庫索引"""

    def link_memory_to_knowledge(self, memory_id: str, knowledge_ref: str):
        """將記憶条目關聯到知識庫條目"""
```

**涉及檔案**：
- `agent/builtin_memory_provider.py`（新建）
- `agent/knowledge_memory_interface.py`（新建）

---

### 第四階段：Agent Core 拆分（3-5 天）

**目標**：將 57KB 的 `anthropic_adapter.py` 和 46KB 的 `prompt_builder.py` 拆分為職責清晰的模組

#### 4.1 Anthropic Adapter 拆分
| 現有檔案 | 拆分後 | 職責 |
|----------|--------|------|
| `agent/anthropic_adapter.py` | `agent/adapters/anthropic.py` | API 調用 |
| `agent/anthropic_adapter.py` | `agent/adapters/openai.py` | OpenAI 適配 |
| `agent/anthropic_adapter.py` | `agent/adapters/base.py` | 公共介面 |

**涉及檔案**：
- `agent/adapters/`（新建目錄）
- `agent/anthropic_adapter.py`（刪除，遷移至 `agent/adapters/`）

#### 4.2 Prompt Builder 重構
| 現有檔案 | 拆分後 | 職責 |
|----------|--------|------|
| `agent/prompt_builder.py` | `agent/prompts/system_prompt.py` | 系統提示構建 |
| `agent/prompt_builder.py` | `agent/prompts/user_prompt.py` | 用戶提示構建 |
| `agent/prompt_builder.py` | `agent/prompts/injection.py` | 上下文注入 |

**涉及檔案**：
- `agent/prompts/`（新建目錄）
- `agent/prompt_builder.py`（刪除）

#### 4.3 Context Engine 增强
將 `agent/context_engine.py` 設為調度中心：
```python
class ContextEngine:
    def __init__(self, memory_manager, knowledge_base, skills_base):
        self.memory = memory_manager
        self.knowledge = knowledge_base
        self.skills = skills_base

    def build_turn_context(self, turn_num, user_msg):
        # 1. 從記憶庫 prefetch
        # 2. 從知識庫檢索相關文檔
        # 3. 從技能庫加載必要技能說明
        # 4. 組裝並返回上下文
```

**涉及檔案**：`agent/context_engine.py`（重寫）

---

### 第五階段：Gateway 瘦身（2-3 天）

**目標**：將 `gateway/run.py`（448KB）拆分為獨立模組

#### 5.1 拆分方案
| 現有內容 | 遷移至 | 風險 |
|----------|--------|------|
| Session 管理 | `gateway/session_manager.py` | 中 |
| Platform 適配 | `gateway/platforms/`（已有） | 低 |
| Hook 系統 | `gateway/builtin_hooks/`（已有） | 低 |
| Config 處理 | `gateway/config.py`（已有） | 低 |
| Display | `gateway/display.py` | 低 |

#### 5.2 平臺適配器標準化
**保持**：`gateway/platforms/discord.py`（已有）
**新建**：`gateway/platforms/base.py`（統一介面）

**涉及檔案**：
- `gateway/session_manager.py`（新建）
- `gateway/platforms/base.py`（新建）
- `gateway/run.py`（重寫，只保留核心調度邏輯）

---

### 第六階段：工具層重構（2-3 天）

**目標**：控制工具 schema 膨脹，建立分組機制

#### 6.1 工具註冊表重構
**保持**：`tools/registry.py`

**新增**：`tools/tool_groups.py`
```python
TOOL_GROUPS = {
    "file_operations": ["file_operations", "file_tools"],
    "memory": ["memory_tool", "session_search_tool"],
    "skills": ["skill_manager_tool", "skills_tool"],
    "communication": ["send_message_tool"],
}
```

#### 6.2 工具延迟加載
修改 `tools/registry.py`：根據 `toolset_distributions.py` 配置實現按需加載

**涉及檔案**：
- `tools/tool_groups.py`（新建）
- `tools/registry.py`（修改）
- `toolset_distributions.py`（修改）

---

## 遷移序列與風險級別

| 階段 | 任務 | 風險 | 回滾方案 |
|------|------|------|----------|
| 1.1 | 新建目錄 | 無 | `rm -rf knowledge/` |
| 1.2 | 遷移 docs/ | 低 | `mv knowledge/docs/* docs/` |
| 1.3 | 建立 index.json | 無 | `rm knowledge/index.json` |
| 1.4 | 更新 context_engine | 中 | git checkout 恢復 |
| 2.1 | 技能結構標準化 | 低 | 恢復 skills/ 目錄 |
| 2.3 | 技能-記憶整合 | 中 | 註釋掉 hook 調用 |
| 3.1 | 拆分記憶 provider | 中 | 合併回 memory_provider.py |
| 3.3 | 知識-記憶介面 | 中 | 刪除該模組 |
| 4.1 | Adapter 拆分 | **高** | 保留舊檔案作為 fallback |
| 4.2 | Prompt Builder 重構 | **高** | 保留舊檔案作為 fallback |
| 4.3 | Context Engine 重寫 | **高** | git checkout 恢復 |
| 5.1 | Gateway 拆分 | **高** | 保留完整 run.py backup |
| 6.1 | 工具分組 | 低 | 回滾 tool_groups.py |

---

## 詳細檔案操作清單

### 新建目錄
```bash
mkdir -p knowledge/{docs,facts,policies}
mkdir -p agent/adapters
mkdir -p agent/prompts
mkdir -p agent/knowledge_memory_interface.py  # 檔案非目錄
mkdir -p gateway/session_manager.py  # 檔案非目錄
mkdir -p tools/tool_groups.py  # 檔案非目錄
```

### 新建檔案
| 檔案路徑 | 內容 | 階段 |
|----------|------|------|
| `knowledge/index.json` | 統一索引 | 1 |
| `knowledge/docs/plans/2026-03-16-pricing-accuracy-architecture-design.md` | 從 docs/plans 遷移 | 1 |
| `knowledge/docs/specs/container-cli-review-fixes.md` | 從 docs/specs 遷移 | 1 |
| `knowledge/docs/migration/openclaw.md` | 從 docs/migration 遷移 | 1 |
| `knowledge/docs/AGENTS.md` | 從根目錄遷移 | 1 |
| `knowledge/facts/learnings.md` | 從 memory/ 遷移 | 1 |
| `knowledge/facts/learnings-v2.md` | 從 memory/ 遷移 | 1 |
| `knowledge/facts/OPTIMIZATION-HISTORY.md` | 從 memory/ 遷移 | 1 |
| `knowledge/facts/incident-log/` | 從 memory/incident-log/ 遷移 | 1 |
| `skills/.index.toml` | 技能索引 | 2 |
| `agent/builtin_memory_provider.py` | 從 memory_provider.py 拆分 | 3 |
| `agent/knowledge_memory_interface.py` | 知識-記憶介面 | 3 |
| `agent/adapters/base.py` | 適配器基類 | 4 |
| `agent/adapters/anthropic.py` | Anthropic 專用適配 | 4 |
| `agent/adapters/openai.py` | OpenAI 適配 | 4 |
| `agent/prompts/system_prompt.py` | 系統提示構建 | 4 |
| `agent/prompts/user_prompt.py` | 用戶提示構建 | 4 |
| `agent/prompts/injection.py` | 上下文注入 | 4 |
| `gateway/session_manager.py` | Session 管理 | 5 |
| `gateway/platforms/base.py` | 平臺適配基類 | 5 |
| `tools/tool_groups.py` | 工具分組配置 | 6 |

### 修改檔案
| 檔案路徑 | 修改內容 | 階段 |
|----------|----------|------|
| `agent/context_engine.py` | 新增 build_knowledge_prompt() 和與記憶/技能庫的整合 | 1, 4 |
| `run_agent.py` | 新增知識庫索引初始化和路徑引用 | 1 |
| `agent/skill_commands.py` | 新增 MemoryManager.on_memory_write() hook 調用 | 2 |
| `agent/memory_manager.py` | 新增與 BuiltinMemoryProvider 和外部 provider 的協調邏輯 | 3 |
| `agent/prompt_builder.py` | 重構為 agent/prompts/ 三個子模組的 facade | 4 |
| `agent/anthropic_adapter.py` | 拆分為 agent/adapters/ 三個子模組的 facade | 4 |
| `gateway/run.py` | 瘦身，只保留核心調度，拆分到各專門模組 | 5 |
| `gateway/config.py` | 與 session_manager.py 整合 | 5 |
| `tools/registry.py` | 新增工具分組和延迟加載邏輯 | 6 |
| `toolset_distributions.py` | 新增工具分組配置引用 | 6 |

### 刪除檔案（遷移後）
| 檔案路徑 | 原因 | 風險 |
|----------|------|------|
| `docs/plans/2026-03-16-pricing-accuracy-architecture-design.md` | 已遷移至 knowledge/docs/plans/ | 低 |
| `docs/specs/container-cli-review-fixes.md` | 已遷移至 knowledge/docs/specs/ | 低 |
| `docs/migration/openclaw.md` | 已遷移至 knowledge/docs/migration/ | 低 |
| `AGENTS.md` | 已遷移至 knowledge/docs/AGENTS.md | 低 |
| `SOUL.md` | 已遷移至 knowledge/facts/SOUL.md | 低 |
| `memory/learnings.md` | 已遷移至 knowledge/facts/learnings.md | 低 |
| `memory/learnings-v2.md` | 已遷移至 knowledge/facts/learnings-v2.md | 低 |
| `memory/OPTIMIZATION-HISTORY.md` | 已遷移至 knowledge/facts/OPTIMIZATION-HISTORY.md | 低 |
| `memory/incident-log/` | 已遷移至 knowledge/facts/incident-log/ | 低 |
| `agent/memory_provider.py` | 功能已拆分至 builtin_memory_provider.py + 各 provider | **高**（需確認新provider正常工作） |
| `agent/prompt_builder.py` | 功能已拆分至 agent/prompts/ | **高**（需確認新模組正常工作） |
| `agent/anthropic_adapter.py` | 功能已拆分至 agent/adapters/ | **高**（需確認新模組正常工作） |
| `gateway/run.py`（舊） | 重寫瘦身版本 | **高**（需完整備份） |

---

## 回滾計劃

### 即時回滾（階段 1-3）
```bash
# 恢復遷移的檔案
mv knowledge/docs/* docs/
mv knowledge/facts/* memory/
rm -rf knowledge/
git checkout -- agent/context_engine.py run_agent.py
```

### 中期回滾（階段 4-5）
```bash
# 保留 fallback：舊檔案重命名為 .backup
cp agent/anthropic_adapter.py agent/anthropic_adapter.py.backup
cp agent/prompt_builder.py agent/prompt_builder.py.backup
cp gateway/run.py gateway/run.py.backup

# 恢復
mv agent/anthropic_adapter.py.backup agent/anthropic_adapter.py
mv agent/prompt_builder.py.backup agent/prompt_builder.py
mv gateway/run.py.backup gateway/run.py
```

### 緊急回滾（全階段）
```bash
cd /home/ubuntu/.hermes/hermes-agent
git stash  # 保留當前更改
git checkout HEAD -- .  # 完全恢復到 git 版本
```

---

## 執行優先順序

```
第一優先（立即執行）:
1. mkdir knowledge/{docs,facts,policies}
2. 遷移 docs/ → knowledge/docs/
3. 遷移 SOUL.md, learnings.md → knowledge/facts/
4. 新建 knowledge/index.json
5. 更新 run_agent.py 的 HERMES_KNOWLEDGE_PATH 常量

第二優先（1周內）:
6. 建立 skills/.index.toml
7. 新建 agent/builtin_memory_provider.py
8. 新建 agent/knowledge_memory_interface.py
9. 更新 agent/skill_commands.py 整合記憶庫

第三優先（2周內）:
10. 拆分 agent/anthropic_adapter.py → agent/adapters/
11. 重構 agent/prompt_builder.py → agent/prompts/
12. 重寫 agent/context_engine.py
13. 拆分 gateway/run.py

第四優先（1個月內）:
14. 新建 tools/tool_groups.py
15. 更新 tools/registry.py 延迟加載
16. 清理遷移後的舊檔案
```

---

## 三系統交叉引用關係

```
knowledge/index.json
    │  (被以下讀取)
    ├─► agent/context_engine.py::build_knowledge_prompt()
    ├─► agent/knowledge_memory_interface.py::search_knowledge()
    └─► tools/session_search_tool.py (語義檢索時)

skills/.index.toml
    │  (被以下讀取)
    ├─► agent/skill_commands.py::load_skill()
    ├─► tools/skills_tool.py::skills_list()
    └─► tools/skills_hub.py (hub 安裝時)

memory/MEMORY.md
    │  (被以下讀寫)
    ├─► agent/builtin_memory_provider.py (內建實現)
    ├─► agent/memory_manager.py (統一調度)
    └─► agent/knowledge_memory_interface.py (關聯知識庫)
```

---

*計劃版本：v1.0.0*
*創建時間：2026-04-24*
*下次審查：2026-05-01*
