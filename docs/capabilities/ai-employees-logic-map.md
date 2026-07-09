# AI 员工页面逻辑地图

> 状态：MVP 实施版  
> 范围：Hermes Desktop `AI 员工` 页面 + profile-backed Agent registry API  
> 约束：每个 AI 员工对应一个 Hermes profile/Agent；中文名只是显示/训练元数据，系统调度 ID 必须保持英文 slug。

## 0. 编号体系

| 前缀 | 类型 | 说明 |
| --- | --- | --- |
| `CAP-*` | 能力 | 用户可感知的产品能力 |
| `PAGE-*` | 页面 | Desktop 页面/区域 |
| `BTN-*` | 按钮 | 用户可点击控件 |
| `ACT-*` | 动作 | UI 或 API 触发的动作 |
| `FLOW-*` | 流程 | 多动作组成的业务链路 |
| `DATA-*` | 数据 | 员工、profile、SOUL、registry 等数据对象 |
| `API-*` | 接口 | Hermes Desktop/backend API |
| `RULE-*` | 规则 | 架构、安全、交互和边界规则 |
| `TEST-*` | 测试 | 自动/手动回归测试项 |

---

## 1. 能力地图

| ID | 能力 | 描述 | 页面 | 流程 | API | 测试 |
| --- | --- | --- | --- | --- | --- | --- |
| `CAP-001` | AI 员工目录 | 展示所有 profile-backed Agent，并优先显示中文员工名 | `PAGE-001`, `PAGE-002` | `FLOW-001` | `API-001` | `TEST-001`, `TEST-004` |
| `CAP-002` | 员工身份训练 | 修改中文名、岗位、任务说明和 `SOUL.md` | `PAGE-003` | `FLOW-002` | `API-002`, `API-003`, `API-004` | `TEST-002`, `TEST-004` |
| `CAP-003` | 员工启动 | 选择某个员工/profile 并新建会话 | `PAGE-004` | `FLOW-003` | 无直接 REST；走 profile/new-chat store | `TEST-003`, `TEST-004` |
| `CAP-004` | 能力摘要 | 展示模型、provider、技能数、Gateway 状态和文件路径 | `PAGE-002`, `PAGE-005` | `FLOW-001` | `API-001` | `TEST-001`, `TEST-004` |

---

## 2. 页面地图

| ID | 页面/区域 | 文件 | 包含按钮 | 读写数据 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `PAGE-001` | AI 员工页面壳 | `apps/desktop/src/app/ai-employees/index.tsx` | `BTN-001` | `DATA-001` | `/ai-employees` 全页面工作台，类似 Video Studio，不是 overlay |
| `PAGE-002` | 员工列表 + 概览 | `apps/desktop/src/app/ai-employees/index.tsx` | `BTN-002`, `BTN-003` | `DATA-001`, `DATA-002` | 左侧员工 roster，右侧员工详情 |
| `PAGE-003` | 训练员工 Tab | `apps/desktop/src/app/ai-employees/index.tsx` | `BTN-004`, `BTN-005` | `DATA-002`, `DATA-003`, `DATA-004` | 编辑中文元数据和 SOUL.md |
| `PAGE-004` | 试用员工 Tab | `apps/desktop/src/app/ai-employees/index.tsx` | `BTN-006` | `DATA-005` | 用所选员工新建聊天 |
| `PAGE-005` | 技能与能力 Tab | `apps/desktop/src/app/ai-employees/index.tsx` | 后续扩展 | `DATA-006` | 当前 MVP 展示摘要，后续接技能/工具/模型训练 |
| `PAGE-006` | 左侧导航入口 | `apps/desktop/src/app/chat/sidebar/index.tsx` | `BTN-007` | route | Sidebar 进入 `/ai-employees` |
| `PAGE-007` | 命令面板入口 | `apps/desktop/src/app/command-palette/index.tsx` | `BTN-008` | route | Command Palette 进入 `/ai-employees` |

---

## 3. 按钮地图

| ID | 按钮 | 所在页面 | 触发动作 | 接口 | 规则 |
| --- | --- | --- | --- | --- | --- |
| `BTN-001` | 刷新员工 | `PAGE-001` | `ACT-001` | `API-001` | `RULE-001`, `RULE-003` |
| `BTN-002` | 员工列表行 | `PAGE-002` | `ACT-002` | 无 | `RULE-001` |
| `BTN-003` | 用这个员工新建会话 | `PAGE-002` | `ACT-006` | 无 | `RULE-002` |
| `BTN-004` | 训练员工 Tab | `PAGE-003` | `ACT-003` | `API-003` | `RULE-004` |
| `BTN-005` | 保存训练 | `PAGE-003` | `ACT-004`, `ACT-005` | `API-002`, `API-004` | `RULE-001`, `RULE-003`, `RULE-004`, `RULE-005` |
| `BTN-006` | 试用员工新建会话 | `PAGE-004` | `ACT-006` | 无 | `RULE-002` |
| `BTN-007` | Sidebar AI 员工 | `PAGE-006` | `ACT-007` | 无 | `RULE-001` |
| `BTN-008` | Command Palette AI 员工 | `PAGE-007` | `ACT-007` | 无 | `RULE-001` |

---

## 4. 动作地图

| ID | 动作 | 输入 | 输出 | 触发方 | 后续流程 |
| --- | --- | --- | --- | --- | --- |
| `ACT-001` | 拉取员工列表 | 无 | `DATA-001[]` | 页面加载 / `BTN-001` | `FLOW-001` |
| `ACT-002` | 选择员工 | `profile_id` | selected employee | `BTN-002` | `FLOW-001` |
| `ACT-003` | 读取员工 SOUL.md | selected `profile_id` | `DATA-004` | 选中员工 / 训练 Tab | `FLOW-002` |
| `ACT-004` | 保存员工显示/岗位元数据 | `DATA-002` | updated employee | `BTN-005` | `FLOW-002` |
| `ACT-005` | 保存员工 SOUL.md | `DATA-004` | write ok | `BTN-005` | `FLOW-002` |
| `ACT-006` | 用员工新建会话 | selected `profile_id` | `$newChatProfile` + route `/` | `BTN-003`, `BTN-006` | `FLOW-003` |
| `ACT-007` | 打开 AI 员工页面 | route | `/ai-employees` | `BTN-007`, `BTN-008` | `FLOW-001` |

---

## 5. 流程地图

### `FLOW-001` 页面进入 / 员工选择流程

```text
Sidebar/Command Palette → ACT-007 → routes.ts /ai-employees → DesktopController lazy load AiEmployeesView
→ ACT-001 → API-001 → DATA-001[] → 员工列表 + 详情
→ ACT-002 → selected profile_id → PAGE-002 概览刷新
```

相关对象：`DATA-001`, `DATA-002`  
规则：`RULE-001`, `RULE-003`

### `FLOW-002` 训练员工流程

```text
PAGE-003 → ACT-003 → API-003 getProfileSoul → DATA-004
用户编辑中文名/岗位/任务说明/SOUL.md
BTN-005 → ACT-004 → API-002 update metadata
        → ACT-005 → API-004 update SOUL.md
        → ACT-001 refresh → DATA-001[]
```

相关对象：`DATA-002`, `DATA-003`, `DATA-004`  
规则：`RULE-001`, `RULE-003`, `RULE-004`, `RULE-005`

### `FLOW-003` 员工启动流程

```text
BTN-003/BTN-006 → ACT-006 → $newChatProfile = selected.profile_id
→ requestFreshSession() → navigate('/') → 下一次发送消息时用该 profile 创建会话
```

相关对象：`DATA-005`  
规则：`RULE-002`

---

## 6. 数据地图

| ID | 数据 | TypeScript/Python 位置 | 生产方 | 消费方 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `DATA-001` | `AgentEmployeeInfo[]` | `apps/desktop/src/types/hermes.ts` / `hermes_cli/ai_employees.py` | `API-001` | `PAGE-001`, `PAGE-002` | 员工 roster；合并 registry 与 live profiles |
| `DATA-002` | 员工显示元数据 | `agents/registry.json`, `profile.yaml`, `SOUL.md` metadata block | `API-002` | `PAGE-002`, `PAGE-003` | 中文名、岗位、任务说明、分类、emoji |
| `DATA-003` | `profile.yaml` | `$HERMES_HOME` per profile | `API-002` | profile list / kanban routing / Desktop | 训练写入的 profile 元数据 |
| `DATA-004` | `SOUL.md` | `$HERMES_HOME/SOUL.md` or profile `SOUL.md` | `API-003`, `API-004` | Agent system prompt on new session | 员工身份训练主体 |
| `DATA-005` | `$newChatProfile` | `apps/desktop/src/store/profile.ts` | `ACT-006` | session creation path | 下一次新聊天目标员工 profile |
| `DATA-006` | 能力摘要 | `AgentEmployeeInfo` | `API-001` | `PAGE-005` | 模型/provider/技能数/Gateway 状态 |

---

## 7. 接口地图

| ID | Hermes API | 方法 | 使用方 | 状态 |
| --- | --- | --- | --- | --- |
| `API-001` | `/api/ai-employees` | `GET` | `AiEmployeesView` | 已实现 |
| `API-002` | `/api/ai-employees/{profile_id}/metadata` | `PUT` | `AiEmployeesView` | 已实现 |
| `API-003` | `/api/profiles/{name}/soul` | `GET` | `AiEmployeesView` | 已存在，复用 |
| `API-004` | `/api/profiles/{name}/soul` | `PUT` | `AiEmployeesView` | 已存在，复用 |

---

## 8. 规则地图

| ID | 规则 | 说明 | 覆盖对象 |
| --- | --- | --- | --- |
| `RULE-001` | profile id 不可中文化 | `profile_id` 必须匹配 `[a-z0-9][a-z0-9_-]{0,63}`；中文名只用于显示和训练 | `DATA-001`, `DATA-002`, `API-002` |
| `RULE-002` | 新会话才完全生效 | 员工切换/训练不修改当前已缓存系统提示词；训练保存后提示新会话生效 | `FLOW-002`, `FLOW-003` |
| `RULE-003` | registry 与 live profiles 合并 | 列表以真实 profile 为准，过滤不存在的 registry 条目 | `API-001` |
| `RULE-004` | 不读取密钥 | 页面不读取 `.env` / auth 文件，不展示 secrets | `PAGE-003`, `API-002` |
| `RULE-005` | SOUL body 保留 | 更新 metadata block 时保留原 SOUL.md 主体内容 | `API-002`, `API-004` |

---

## 9. 测试地图

| ID | 测试 | 命令/位置 | 覆盖 |
| --- | --- | --- | --- |
| `TEST-001` | 后端列表合并 registry + profiles | `scripts/run_tests.sh tests/hermes_cli/test_web_server_ai_employees.py` | `API-001`, `RULE-003` |
| `TEST-002` | 后端 metadata 写入 registry/profile.yaml/SOUL.md | 同上 | `API-002`, `RULE-001`, `RULE-005` |
| `TEST-003` | 前端选择员工并设置 `$newChatProfile` | `npm run --workspace apps/desktop test:ui -- src/app/ai-employees/index.test.tsx` | `FLOW-003` |
| `TEST-004` | 前端页面渲染、训练保存 | 同上 | `FLOW-001`, `FLOW-002` |
| `TEST-005` | TypeScript 回归 | `npm run --workspace apps/desktop typecheck` | route/types/page wiring |
| `TEST-006` | Python 编译回归 | `python3 -m py_compile hermes_cli/web_server.py hermes_cli/ai_employees.py` | backend import/syntax |
