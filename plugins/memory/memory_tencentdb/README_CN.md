# MemoryCore

MemoryCore 是 TencentDB Agent Memory 的**记忆与元数据核心**，统一存储并提供三类数据能力：

- **Memory**：L0 对话、L1 原子记忆、L2 场景记忆和 L3 核心画像。
- **Knowledge 元信息**：Wiki、Code Graph 等知识源的标识、类型、状态、关联关系和服务地址。
- **资产管理元信息**：User、Team、Agent、Task、Skill、Knowledge Asset，以及成员、归属和访问关系。

MemoryCore 独立运行，通过 HTTP Gateway 对外提供这些能力。OpenClaw、Hermes 和自定义应用通过轻量 Adapter 或 SDK 接入。Agent 是调用方，也可以作为一种被管理的元数据实体；MemoryCore 不负责托管、调度或运行 Agent 本身。

> MemoryCore 保存 Knowledge 元信息，不保存或处理 Knowledge 内容。Wiki 解析、代码图谱构建、索引和内容检索由 `MemoryKnowledge/` 提供。

## 核心能力

- **Memory 存储与处理**：记录 L0 对话，并维护 L1 原子记忆、L2 场景记忆和 L3 核心画像。
- **Memory 召回**：支持关键词、Embedding 与混合检索；没有 Embedding Provider 时仍可使用 BM25。
- **Knowledge 元信息登记**：登记知识源并维护其标识、类型、状态、关联关系和服务地址。
- **资产元信息管理**：管理 User、Team、Agent、Task、Skill、Knowledge Asset，以及成员、归属和访问关系。
- **Skill Memory**：支持 Skill 创建、版本、资源、搜索、路由和对话抽取。
- **统一访问接口**：通过 HTTP API 和 TypeScript/Python SDK 为 Adapter 与应用提供能力。

## 架构

```text
OpenClaw / Hermes / 自定义应用
                │
                │ HTTP API / SDK
                ▼
        MemoryCore Gateway :8420
        ├─ Memory
        │  └─ L0 / L1 / L2 / L3
        ├─ Knowledge 元信息
        ├─ 资产管理元信息
        └─ SQLite + 本地文件

MemoryKnowledge
      └─ Knowledge 解析 / 索引 / 检索
```

## 运行方式

MemoryCore 以 Standalone Runtime 形式开源，适合本地开发、单机部署和 Agent sidecar：

- 默认监听 `127.0.0.1:8420`。
- 使用 SQLite、本地文件和进程内状态。
- 除 LLM API 外没有必需的外部服务。
- 默认关闭远程 Embedding，使用 BM25 召回。
- 数据默认写入 `~/.memory-tencentdb/memory-tdai`。

## 环境要求

- Node.js `>= 22.16.0`
- npm
- 一个 OpenAI-compatible LLM API；只读查询可以不触发 LLM，但记忆抽取和归纳需要有效凭证

## 快速开始

### 1. 安装与构建

```bash
cd MemoryCore
npm install
npm run build
```

### 2. 启动 Standalone Gateway

```bash
export TDAI_GATEWAY_CONFIG="$PWD/tdai-gateway.standalone.yaml"
export TDAI_LLM_API_KEY="your-api-key"
export TDAI_LLM_BASE_URL="https://api.openai.com/v1"
export TDAI_LLM_MODEL="gpt-4o-mini"

node --import tsx src/gateway/server.ts
```

Gateway 启动后访问：

```bash
curl http://127.0.0.1:8420/health
```

如需从其他机器或容器访问，必须同时设置监听地址和鉴权：

```bash
export TDAI_GATEWAY_HOST="0.0.0.0"
export TDAI_GATEWAY_API_KEY="replace-with-a-strong-random-token"
```

除 `/health` 和 CORS 预检外，启用鉴权后所有接口均需携带：

```text
Authorization: Bearer <TDAI_GATEWAY_API_KEY>
x-tdai-service-id: <memory-instance-id>
```

## 从旧版升级

如果从 v1.x或v0.x（数据格式 v2）升级到 v2.0.0+（数据格式 v3），**启动新版 Gateway 前**需要先运行数据迁移脚本。

> ⚠️ 迁移前请务必备份整个数据目录。

```bash
# dry-run 检查
python scripts/migrate-v2-to-v3/v2-to-v3-migrate.py ~/.memory-tencentdb/memory-tdai --dry-run

# 执行迁移
python scripts/migrate-v2-to-v3/v2-to-v3-migrate.py ~/.memory-tencentdb/memory-tdai
```

详见 [迁移脚本说明](scripts/migrate-v2-to-v3/README_CN.md)。

## Docker

在 `MemoryCore/` 目录构建：

```bash
docker build -t memory-core:local .
```

启动 Standalone 容器：

```bash
docker run --rm \
  -p 8420:8420 \
  -e TDAI_LLM_API_KEY="your-api-key" \
  -e TDAI_GATEWAY_API_KEY="replace-with-a-strong-random-token" \
  -v "$PWD/tdai-gateway.standalone.yaml:/data/config/tdai-gateway.yaml:ro" \
  -v memory-core-data:/data/tdai-memory \
  memory-core:local
```

通过环境变量或 Secret Manager 注入凭证，不要把 API Key 或其他凭证写入镜像和配置仓库。

## Agent 接入

### OpenClaw

推荐使用 `openclaw-plugin/` 中的轻量客户端 Adapter。它连接已运行的 MemoryCore Gateway，不在 OpenClaw 进程内重复运行记忆管线。

从仓库根目录执行：

```bash
bash MemoryCore/scripts/install-openclaw-plugin.sh
```

常用连接参数：

```text
TDAI_MEMORY_ENDPOINT=http://127.0.0.1:8420
TDAI_MEMORY_API_KEY=<与 Gateway 相同的 API Key>
TDAI_MEMORY_INSTANCE_ID=default
```

### Hermes

`hermes-plugin/` 提供 Hermes Memory Provider。它遵循同样的 Adapter 模式，通过 Gateway 完成对话写入与记忆召回。

### 自定义 Agent

自定义 Runtime 可以直接使用仓库中的 SDK：

- `../sdk/memory-core/typescript/`
- `../sdk/memory-core/python/`

一个 Adapter 通常只需要完成三件事：

1. 会话结束或每轮完成后写入 L0。
2. 构造 Prompt 前召回 L1/L2/L3。
3. 将召回结果以有边界、可识别的上下文注入 Agent。

## API 范围

| API | 用途 | 状态 |
| --- | --- | --- |
| `/capture`、`/recall`、`/search/*` | 早期 Gateway 兼容接口 | 兼容保留 |
| `/v2/conversation/*` | L0 写入、查询、搜索、删除和计数 | 稳定 |
| `/v2/atomic/*` | L1 查询、搜索、更新、删除和计数 | 稳定 |
| `/v2/scenario/*`、`/v2/core/*` | L2/L3 读写 | 稳定 |
| `/v3/conversation/*`、`/v3/atomic/*`、`/v3/scenario/*`、`/v3/core/*` | 强隔离的 L0–L3 数据面 | 推荐新接入使用 |
| `/v3/skill/*` | Skill 管理、检索、版本、资源和抽取 | 稳定 |
| `/v3/meta/*` | User、Team、Agent、Task、Asset 和权限关系 | 管理面 |
| `/v3/knowledge/*` | 知识资产元数据登记 | 管理面 |
| `/health` | 健康检查 | 公共 |

v3 记忆数据面要求 `team_id`、`agent_id`、`user_id`，可以通过请求体或对应的 `x-tdai-*` Header 传入；`session_id` 可选，用于限定会话范围。

## 配置

Gateway 按以下优先级加载配置：

1. `TDAI_GATEWAY_CONFIG` 指定的 YAML 或 JSON。
2. 当前目录下的 `tdai-gateway.yaml` 或 `tdai-gateway.json`。
3. 数据目录下的 `tdai-gateway.yaml` 或 `tdai-gateway.json`。
4. 环境变量和内置默认值。

环境变量覆盖配置文件。常用配置：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `TDAI_GATEWAY_CONFIG` | 自动发现 | 配置文件路径 |
| `TDAI_GATEWAY_HOST` | `127.0.0.1` | Gateway 监听地址 |
| `TDAI_GATEWAY_PORT` | `8420` | Gateway 端口 |
| `TDAI_GATEWAY_API_KEY` | 未设置 | HTTP Bearer 鉴权；非回环监听必须设置 |
| `TDAI_CORS_ORIGINS` | 空 | 允许的 Origin，逗号分隔 |
| `TDAI_DATA_DIR` | `~/.memory-tencentdb/memory-tdai` | 本地数据目录 |
| `TDAI_LLM_API_KEY` | 空 | LLM API Key |
| `TDAI_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API 地址 |
| `TDAI_LLM_MODEL` | `gpt-4o` | LLM 模型 |
| `TDAI_SKILL_ENABLED` | 配置文件值 | 强制启用 Skill 模块 |

配置模板：

- `tdai-gateway.standalone.yaml`：最小单机 Memory 配置。
- `tdai-gateway.yaml`：Standalone + Skill 默认配置。
- `tdai-gateway.proxy.yaml`：通过 OpenAI-compatible Proxy 调用模型。

## 存储与隔离

- Memory 和 Metadata 使用 SQLite 存储。
- 文件与大对象保存在本地数据目录。
- Pipeline State 由当前进程维护。
- BM25 无需外部 Embedding 服务；需要时可配置 OpenAI-compatible Embedding API。

所有业务调用都应明确 `x-tdai-service-id`。新 Adapter 建议使用 v3 数据面，并始终提供 Team、Agent、User 隔离维度。

## 目录结构

```text
MemoryCore/
├── src/core/              L0–L3 Memory、Skill、Store 和 Storage 抽象
├── src/gateway/           HTTP Gateway 与 v2/v3 Router
├── src/services/          Pipeline Scanner、Worker 和调度服务
├── openclaw-plugin/       OpenClaw 轻量客户端 Adapter
├── hermes-plugin/         Hermes Memory Provider
├── scripts/               安装、构建、迁移和运维工具
│   ├── install-hermes-plugin.sh         Hermes provider 安装脚本
│   ├── install-openclaw-plugin-v2.sh    OpenClaw 插件安装脚本
│   └── migrate-v2-to-v3/                数据迁移工具（v2 → v3）
├── Dockerfile             MemoryCore Gateway 镜像
├── tdai-gateway*.yaml     Gateway 配置模板
└── package.json           Node.js 包与构建命令
```

## 本地数据工具

```bash
npm run read-local-memory
npm run seed-v2
```

## 安全建议

- 非回环地址监听时必须配置 `TDAI_GATEWAY_API_KEY`。
- CORS 默认关闭；只允许明确可信的 Origin，不要在生产环境使用 `*`。
- 所有 Secret 通过环境变量或 Secret Manager 注入。
- 不要提交 `.env`、数据库文件、日志、导出数据或真实服务配置。
- 每个请求都应校验实例和 Team/User/Agent 归属，避免跨租户访问。

## License

MIT
