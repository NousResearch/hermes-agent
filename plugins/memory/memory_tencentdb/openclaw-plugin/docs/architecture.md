# TencentDB Agent Memory Client — OpenClaw 记忆插件（客户端接入版）

> 创建: 2026-05-17 | 状态: 开发中
> 插件 ID: `memory-tencentdb-client`
> 显示名称: Memory TencentDB (Client)

## 1. 背景

服务化改造完成后，四层记忆数据（L0 对话/L1 原子/L2 场景/L3 画像）全部托管在远端 Gateway：
- **数据存储**: TCVDB (向量) + COS (文件) + Redis (状态)
- **Pipeline**: Gateway Worker 自动完成 L1→L2→L3 抽取
- **API**: 15 个 v2 REST 端点覆盖全部 CRUD + Search

**原插件（memory-tencentdb）**是"全栈"架构：本地 SQLite/VDB + 本地 Pipeline + 本地 Embedding + OpenClaw Hooks + CLI，~15000 行。

**新插件（memory-tencentdb-client）**是纯客户端：只注册 OpenClaw hooks + tools，所有数据操作通过 `@tencentdb-agent-memory/memory-sdk-ts-v2` 委托给远端 Gateway。

## 2. 三层架构

```
┌───────────────────────────────────────────────────────┐
│  OpenClaw Plugin (memory-tencentdb-client)            │  框架适配层
│  hooks (recall/capture) + tools + prompt 注入          │  只依赖 SDK，不碰 HTTP/存储
│  └─ import { MemoryClient, MemoryFileReader } from SDK│
├───────────────────────────────────────────────────────┤
│  @tencentdb-agent-memory/memory-sdk-ts-v2 (独立包)                             │  通用 SDK 层
│  MemoryClient (14 API) + MemoryFileReader (STS 直读)   │  零框架依赖，纯 fetch
│  以后 Dify / AutoGen / LangChain 也用这个              │
├───────────────────────────────────────────────────────┤
│  Gateway v2 API                                        │  远端服务
│  VDB + COS + Redis + Pipeline Worker                   │
└───────────────────────────────────────────────────────┘
```

## 3. 插件职责（只做框架适配层）

| 功能 | Hook/Tool | 实现 |
|------|-----------|------|
| **对话捕获** | `agent_end` hook | SDK `client.addConversation()` |
| **记忆召回** | `before_prompt_build` hook | 并行: `client.searchAtomic()` + `client.readCore()` + `client.listScenarios()` |
| **标签清理** | `before_message_write` hook | 剥离 `<relevant-memories>` 标签 |
| **L1 搜索** | `tdai_memory_search` tool | SDK `client.searchAtomic()` |
| **L0 搜索** | `tdai_conversation_search` tool | SDK `client.searchConversation()` |
| **文件读取** | `tdai_read_cos` tool | SDK `MemoryFileReader.read()` (STS 直读对象存储) |
| **Prompt 注入** | recall 内部 | 格式化: Persona + L1 记忆 + Scene Navigation + 工具引导 |

### 不做的事

- ❌ 不启动 VectorStore / SQLite / TCVDB
- ❌ 不启动 EmbeddingService
- ❌ 不启动 Pipeline / Timer / Worker
- ❌ 不做 L1/L2/L3 抽取
- ❌ 不管 COS 存储后端
- ❌ 不管 Redis 状态
- ❌ 不做本地 Checkpoint

## 4. 配置项

```jsonc
{
  // Gateway 连接
  "gateway.url": "http://127.0.0.1:8420",
  "gateway.apiKey": "",
  "gateway.instanceId": "default",

  // 召回
  "recall.maxResults": 5,
  "recall.includePersona": true,
  "recall.includeSceneNav": true,

  // 捕获
  "capture.enabled": true
}
```

## 5. 文件结构

```
memory-tencentdb-client/
├── openclaw.plugin.json       # 插件清单
├── package.json               # deps: { "@tencentdb-agent-memory/memory-sdk-ts-v2": "1.0.0-beta.2" }
├── index.ts                   # 入口：初始化 SDK + 注册 hooks/tools
├── src/
│   ├── hooks/
│   │   ├── recall.ts          # before_prompt_build → SDK 召回 → prompt 注入
│   │   └── capture.ts         # agent_end → SDK addConversation
│   ├── tools/
│   │   ├── memory-search.ts   # tdai_memory_search → SDK searchAtomic
│   │   ├── conversation-search.ts  # → SDK searchConversation
│   │   └── read-cos.ts        # tdai_read_cos → SDK MemoryFileReader.read
│   └── format.ts              # 召回结果格式化 + 工具引导注入
├── tests/
│   └── sdk-cos.ts             # SDK COS 直读手动测试
├── .gitignore
└── README.md
```

## 6. SDK 依赖策略

```jsonc
"dependencies": {
  "@tencentdb-agent-memory/memory-sdk-ts-v2": "1.0.0-beta.2"
}
```

SDK 已发布到 npm registry：[`@tencentdb-agent-memory/memory-sdk-ts-v2@1.0.0-beta.2`](https://www.npmjs.com/package/@tencentdb-agent-memory/memory-sdk-ts-v2/v/1.0.0-beta.2)，由 `npm install` 自动拉取，不再走 vendor / 本地 `file:` / tgz。
SDK 保持独立包，不绑定任何框架，以后出 Dify 插件、Python 版等都复用。

## 7. read_cos 工具设计

### COS 直读（STS）

- SDK 的 `MemoryFileReader` 通过 Gateway `/v2/cos/secret` 获取 STS 临时凭证
- 凭证自动缓存，过期前 2 分钟刷新
- 直接 GET COS 对象（COS V5 签名），不经 Gateway 代理中转

### AI 如何知道可以调 read_cos

1. **Persona 末尾的 Scene Navigation**：
   ```
   ## 🗺️ Scene Navigation
   ### Path: scene_blocks/职业发展与技术实践.md
   **热度**: 3 | Summary: 后端工程师，Go + TypeScript...
   ```
   AI 看到路径后主动调 `tdai_read_cos` 读取详情。

2. **工具引导（format.ts 注入）**：
   ```
   <memory-tools-guide>
   - tdai_memory_search: 搜索结构化记忆
   - tdai_conversation_search: 搜索原始对话
   - tdai_read_cos: 读取场景文件（使用 Scene Navigation 中的路径）
   </memory-tools-guide>
   ```

3. **工具 description**：
   ```
   "Read a file from cloud storage. Use paths from Scene Navigation
    (e.g. 'scene_blocks/xxx.md') or 'persona.md'."
   ```

## 8. 关键设计决策

### Q1: session_id 怎么确定？

直接使用 OpenClaw 框架传入的 `ctx.sessionKey`（hook context 自带），与原插件行为一致。不需要自己生成或拼接。

### Q2: 离线/断连降级？

第一版不做——Gateway 不可达时 hook 返回空（不注入记忆），capture 失败记 warn。后续可加本地 fallback。

### Q3: 和原插件冲突吗？

插件 ID 不同（`memory-tencentdb-client` vs `memory-tencentdb`），不冲突。但同时启用会重复捕获/注入，建议只启用一个。

## 9. 实现步骤

| # | 任务 | 预计 |
|---|------|------|
| 1 | `package.json` + `openclaw.plugin.json` + `.gitignore` + `README.md` | 15 min |
| 2 | `index.ts` — 初始化 SDK Client/MemoryFileReader + 注册 hooks/tools | 30 min |
| 3 | `hooks/capture.ts` — agent_end → addConversation | 20 min |
| 4 | `hooks/recall.ts` + `format.ts` — 并行召回 + prompt 格式化 | 45 min |
| 5 | `tools/*.ts` — 3 个工具转发 | 30 min |
| 6 | SDK 测试脚本 (`tests/sdk-cos.ts`) | 15 min |
| 7 | 本地联调测试 | 30 min |

**总计**: ~3 小时
