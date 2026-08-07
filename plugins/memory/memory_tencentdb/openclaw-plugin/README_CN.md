# OpenClaw 适配：TencentDB Agent Memory（v3）

简体中文 · [English](./README.md)

本目录是连接 Memory Gateway **`/v3/*`** 的 **OpenClaw 客户端适配插件**。插件本身不做抽取、索引、场景归纳或画像生成，而是连接已运行的 Memory Gateway，通过 npm TypeScript SDK 完成对话捕获、记忆召回与工具暴露。

| 项 | 说明 |
|----|------|
| 插件 ID | `memory-tencentdb-client`（历史 ID；本分支实现为 v3） |
| SDK | [`@tencentdb-agent-memory/memory-sdk-ts-v2@1.0.0-beta.2`](https://www.npmjs.com/package/@tencentdb-agent-memory/memory-sdk-ts-v2/v/1.0.0-beta.2)（npm；根入口即 v3） |
| 数据面 | `MemoryClient` → `/v3/*` + isolation（`teamId` / `agentId` / `userId`） |
| COS 读文件 | 工具 `tdai_read_cos`，`createMemoryFileReader`（`POST /v2/cos/secret` + STS） |
| 不含 | Offload / Context Engine |

本地 standalone Gateway 推荐：`http://127.0.0.1:8420`，`apiKey = "local"`，`instanceId = "default"`，isolation 三元组均为 `"default"`（对齐 `DEFAULT_ISOLATION_ID`）。若 Gateway 启用了 `TDAI_GATEWAY_API_KEY`，`server.apiKey` 须一致。

更多设计说明见：`../docs/design/2026-07-20-v3-plugin-install-dual-mode-design.md`、`docs/architecture.md`。

## 架构

```text
OpenClaw runtime
  └─ memory-tencentdb-client plugin
       ├─ hooks/capture.ts             agent_end → addConversation (L0, /v3)
       ├─ hooks/recall.ts              before_prompt_build → 搜索 + prompt 注入
       ├─ tools/memory-search.ts       tdai_memory_search → searchAtomic (L1)
       ├─ tools/conversation-search.ts tdai_conversation_search → searchConversation (L0)
       └─ tools/read-cos.ts            tdai_read_cos → MemoryFileReader（COS STS）
            │
            ▼
       @tencentdb-agent-memory/memory-sdk-ts-v2
            │  MemoryClient            →  HTTP /v3/*
            │  createMemoryFileReader  →  /v2/cos/secret + STS GET
            ▼
       TencentDB Agent Memory Gateway（standalone :8420 或远端服务）
```

## 工具

| 工具 | 用途 |
|------|------|
| `tdai_memory_search` | 搜索 L1 结构化记忆 |
| `tdai_conversation_search` | 搜索 L0 原始对话 |
| `tdai_read_cos` | 按相对路径读记忆文件（`scene_blocks/…`、`persona.md` 等） |

## 双模式（Gateway 部署形态）

| 模式 | 含义 |
|------|------|
| `local` | 本机 Gateway `:8420`，占位鉴权与 isolation |
| `server` | 远端 Gateway，真实 Key / 实例 / isolation 必填 |

两种模式都用**同一套 npm SDK**（`memory-sdk-ts-v2`）。「在线/离线」指 Gateway 部署形态（local vs server），不是 SDK 包获取方式。

## 快速开始

推荐在 `MemoryCore/` 下执行 v3 安装脚本（封装：`npm install` 拉 SDK + 构建插件 + `openclaw plugins install -l` + 写配置）：

```bash
# 本地 Gateway（默认）
bash scripts/install-openclaw-plugin.sh

# 远端 Gateway（严格）
MEMORY_INSTALL_MODE=server \
  TDAI_MEMORY_ENDPOINT="https://memory.example.com" \
  TDAI_MEMORY_API_KEY="<instance-api-key>" \
  TDAI_MEMORY_INSTANCE_ID="<instance-id>" \
  TDAI_MEMORY_TEAM_ID="team-..." \
  TDAI_MEMORY_AGENT_ID="agent-..." \
  TDAI_MEMORY_USER_ID="user-..." \
  bash scripts/install-openclaw-plugin.sh
```

脚本会安装并构建本插件（`npm install` 从 registry 拉取 SDK）、执行 `openclaw plugins install -l`，并更新 `~/.openclaw/openclaw.json`：设置 `plugins.slots.memory = "memory-tencentdb-client"`，启用插件，写入 `server`（含 isolation）、`recall`、`capture`。**OpenClaw >= 2026.4.24** 时写入 `hooks.allowPromptInjection` / `hooks.allowConversationAccess`；更老版本会自动跳过这两个字段。

常用环境变量：`OPENCLAW_CONFIG_FILE`、`TDAI_MEMORY_ENDPOINT`、`TDAI_MEMORY_API_KEY`、`TDAI_MEMORY_INSTANCE_ID`、`TDAI_MEMORY_TEAM_ID`、`TDAI_MEMORY_AGENT_ID`、`TDAI_MEMORY_USER_ID`、`WRITE_OPENCLAW_CONFIG=0`。

OpenClaw **不会**自动调用本脚本。也可跳过脚本、按下面步骤手动安装。

### 1. 安装 OpenClaw CLI

```bash
curl -fsSL https://get.openclaw.dev | bash
openclaw --version
```

### 2. 安装插件依赖并构建

```bash
cd MemoryCore/openclaw-plugin
npm install
npm run build
```

`package.json` 依赖 npm 包 [`@tencentdb-agent-memory/memory-sdk-ts-v2@1.0.0-beta.2`](https://www.npmjs.com/package/@tencentdb-agent-memory/memory-sdk-ts-v2/v/1.0.0-beta.2)（根入口即 v3）。

### 3. 安装到 OpenClaw

```bash
openclaw plugins install -l .
```

### 4. 配置插件

使用安装脚本时默认会写配置。仅在 `WRITE_OPENCLAW_CONFIG=0` 或完全手动安装时需要自己改。

> **重要 —— `hooks.*` 与 OpenClaw 版本强相关。**  
> `hooks.allowPromptInjection` / `hooks.allowConversationAccess` 从 **OpenClaw `2026.4.24`** 起才被 gateway schema 接受。更老版本使用 strict schema，**带上这两个字段会导致 gateway 启动失败**。请用 `openclaw --version` 确认版本。

**示例 A —— OpenClaw `>= 2026.4.24`（推荐）：**

```jsonc
{
  "plugins": {
    "slots": {
      "memory": "memory-tencentdb-client"
    },
    "entries": {
      "memory-tencentdb-client": {
        "enabled": true,
        "hooks": {
          "allowPromptInjection": true,
          "allowConversationAccess": true
        },
        "config": {
          "server": {
            "url": "http://127.0.0.1:8420",
            "apiKey": "local",
            "instanceId": "default",
            "teamId": "default",
            "agentId": "default",
            "userId": "default"
          },
          "recall": {
            "maxResults": 5,
            "includePersona": true,
            "includeSceneNav": true
          },
          "capture": {
            "enabled": true
          }
        }
      }
    }
  }
}
```

**示例 B —— OpenClaw `< 2026.4.24`：** 完全省略 `hooks` 块；`config`（含 isolation）保持一致。

`server.instanceId` 作为 `x-tdai-service-id` 发送。isolation 在构造 v3 SDK 时传入。远端（`server` 安装模式）不要使用占位 `local` / `default`，除非启用安装脚本逃生开关。

- `hooks.allowPromptInjection`：允许 `before_prompt_build` 注入召回上下文。  
- `hooks.allowConversationAccess`：2026.4.24+ 上 non-bundled 插件必须为 `true`，否则 L0 捕获会被静默拦截。

### 5. 重启 OpenClaw Gateway

```bash
openclaw gateway restart
```

## 适配职责

| 模块 | 实现文件 | 说明 |
|------|----------|------|
| 对话捕获 | `src/hooks/capture.ts` | 结束后经 v3 `addConversation()` 写 L0 |
| 记忆召回 | `src/hooks/recall.ts` | 构建 prompt 前搜索并注入简洁上下文 |
| L1 工具 | `src/tools/memory-search.ts` | Agent 搜索结构化记忆 |
| L0 工具 | `src/tools/conversation-search.ts` | Agent 搜索原始对话 |
| 文件读取 | `src/tools/read-cos.ts` | Agent 经 COS STS 读 pipeline 产物 |

## 配置项

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `server.url` | `http://127.0.0.1:8420` | Memory Gateway 地址 |
| `server.apiKey` | `local` | Bearer；standalone 默认可用 `local` |
| `server.instanceId` | `default` | 经 `x-tdai-service-id` 发送 |
| `server.teamId` | `default` | v3 isolation `team_id` |
| `server.agentId` | `default` | v3 isolation `agent_id` |
| `server.userId` | `default` | v3 isolation `user_id` |
| `server.rejectUnauthorized` | `true` | 是否校验 HTTPS 证书 |
| `recall.maxResults` | `5` | 每轮最多注入的 L1 条数 |
| `recall.includePersona` | `true` | 是否注入 L3 |
| `recall.includeSceneNav` | `true` | 是否注入 L2 场景导航 |
| `capture.enabled` | `true` | 是否自动捕获对话 |
| `hooks.allowPromptInjection` | `true` | 仅在 OpenClaw `>= 2026.4.24` 写入 |
| `hooks.allowConversationAccess` | `true` | 仅在 OpenClaw `>= 2026.4.24` 写入；non-bundled 上 L0 必需 |

## 文件结构

```text
openclaw-plugin/
├── openclaw.plugin.json       # 插件清单（id = memory-tencentdb-client）
├── package.json               # npm: @tencentdb-agent-memory/memory-sdk-ts-v2
├── index.ts                   # OpenClaw 入口（v3 client + COS reader）
├── src/hooks/capture.ts       # L0 捕获
├── src/hooks/recall.ts        # 召回与 prompt 注入
├── src/tools/                 # Agent 工具（含 read-cos）
├── src/format.ts              # prompt 格式化
└── docs/architecture.md       # 更细的架构说明
```

## 作为其它 Agent 的适配模板

适配其它框架时可复用同一模式：

1. 一轮用户/助手对话结束后调用 v3 `addConversation()`（带 isolation）。
2. 下一轮 prompt 构建前调用 `searchAtomic()`、`readCore()`，必要时 `listScenarios()`。
3. 只注入简洁、带标签的记忆上下文。
4. 暴露 L1 搜索、L0 对话搜索、COS 文件读取（`MemoryFileReader`）。
5. Adapter 保持无状态；存储与异步 L1/L2/L3 处理交给 Memory Gateway。

## 注意

- 仅客户端 adapter：不要在插件内启动 Memory Gateway 子进程，也不要在本地实现抽取逻辑。
- 无 COS/STS 的 standalone：`tdai_read_cos` 应返回可读错误；capture/recall 不得依赖它。
- Gateway 启动与更广 SDK 示例见仓库根目录 README。
