# Changelog

本文件记录 **TencentDB Agent Memory** 的显著变更，格式遵循
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循
[Semantic Versioning](https://semver.org/)。

覆盖仓库全部开源模块：`MemoryCore` / `MemoryPanel` / `MemoryKnowledge` /
`MemoryProxy` / SDK。

---

## [2.0.0] — 2026-08-03

> **产品定位**：让 Agent 的经验、文档、代码沉淀成可复用资产，让下一位 Agent
> 直接读档。详见 [README_CN.md](./README_CN.md)。

### 🧠 四种记忆资产 · 首次完整开源

四类资产从"对话/工作痕迹"里自动沉淀出来：

- **Chat Memory** — 从对话中逐层提取 L0 原始记录 → L1 事实 → L2 场景 → L3
  长期认知；跨会话保留偏好、决策、交互历史。
- **Skill** — 从跑通的任务里提炼可复用 SOP，附版本 / 资源文件 / 触发边界 /
  执行步骤 / 验证规则。新增 Skill 强制归档功能。
- **Wiki** — 把文档变成结构化页面 + 链接图谱（灵感来自 Karpathy 的 LLM 知识库
  实践）。
- **CodeGraph** — 索引仓库的符号 / 文件 / 调用关系 / 影响路径，Agent 改代码
  前先做 impact analysis。新增定时自动同步代码库功能。

### 🎛️ Memory Hub · 面向团队的操作台

管控面板（`agentmemory/memory-hub` 镜像，含 Panel + Knowledge Service）：

- 建 Team / Agent，把资产按 Owner / 版本 / 状态 / 可见性统一管理
- 三级可见性：`private` / `team` / `restricted`（User / Role / Agent ACL），
  外加 `agent` 定向装配
- Agent Loadout：给不同 Agent 绑定不同资产、调整优先级和使用方式
- Wiki + CodeGraph 工坊内置在 Hub，导入代码库/文档就能自动构建
- 管理员（System Admin）现在也可使用资产管理功能
- 面板全面支持中英文切换；统一页面设计风格，优化列表交互和分页体验

### 🔀 Memory Proxy · Agent 挂上记忆的通道

`agentmemory/memory-proxy` 让 Claude Code 等 coding agent 直接用上团队记忆：

- **Anthropic / OpenAI 双协议**：`/claude-code/<spaceId>/v1/messages` 和
  `/v1/chat/completions` 都接
- **首轮引导**：sessionInit 通过 `AskUserQuestion` 让用户选 team / agent /
  task，proxy 记住绑定
- **每轮注入**：把该 agent 的 L2/L3 记忆、matched skill、wiki/code-graph
  拼进 system prompt，转发上游 LLM
- **鉴权**：`x-tdai-user-key` → 内核 `/v3/meta/auth/verify` 换 `user_id`，
  按用户维度控制资产可见性
- Cost Guard 支持为不同 Agent 配置不同模型以降低成本

### 🚀 一条命令拉起完整三件套

三个镜像多架构（`linux/amd64` + `linux/arm64`）已发布到
[Docker Hub `agentmemory`](https://hub.docker.com/u/agentmemory)，公开可拉、
无需登录：

```bash
git clone https://github.com/Tencent/TencentDB-Agent-Memory.git
cd TencentDB-Agent-Memory/deploy/global-images
cp .env.example .env && $EDITOR .env    # 填入两组 LLM 参数
./start-all.sh                          # 一键起
```

`start-all.sh` 首次启动会自动 `init-admin`、生成 admin `sk-mem-...` 并落盘
`.admin-key`；自检 `/v3/meta/auth/verify` 后打印可复制的 `claude` 启动命令。
`stop-all.sh --purge` 彻底清 volume + admin key，方便重置。

详见 [INSTALL_CN.md](./INSTALL_CN.md) / [INSTALL.md](./INSTALL.md)。

### 🧰 官方 SDK

- **TypeScript** — `@tencentdb-agent-memory/memory-sdk-ts-v2`

  ```ts
  import { MemoryClient, SkillClient, MetadataClient } from "@tencentdb-agent-memory/memory-sdk-ts-v2";

  const memory = new MemoryClient({
    endpoint, apiKey, serviceId,
    teamId, agentId, userId,     // v3 严格 isolation：三项必填
  });
  ```

  顶级 export 就是 v3 严格 isolation 版本；老代码走 `.../v2/v3` 子路径也
  能继续用（子路径保留为向后兼容别名）。

- **Python** — `pip install tencentdb-agent-memory-sdk-python`

  ```python
  from tencentdb_agent_memory import MemoryClient                     # 默认（v2 兼容）
  from tencentdb_agent_memory.v3 import MemoryClient, MetadataClient, SkillClient
  ```

### 📖 文档

- 新增 CodeBuddy / Hermes / OpenClaw 接入指南
- 更新安装指南中的角色权限说明

---

## [2.0.0-beta.1] — 2026-07-21

首次公开发布。SemVer 从 `2.0.0-beta.1` 起步（npm 包名迁移到 `-v2` 后缀：
`@tencentdb-agent-memory/memory-tencentdb-v2`、`memory-sdk-ts-v2`）。
Docker 镜像 tag 独立于 npm 版本，本次镜像发的是 `:1.0.0-beta.1`。

> **产品定位**：让 Agent 的经验、文档、代码沉淀成可复用资产，让下一位 Agent
> 直接读档。详见 [README_CN.md](./README_CN.md)。

### 🧠 四种记忆资产 · 首次完整开源

四类资产从"对话/工作痕迹"里自动沉淀出来：

- **Chat Memory** — 从对话中逐层提取 L0 原始记录 → L1 事实 → L2 场景 → L3
  长期认知；跨会话保留偏好、决策、交互历史。
- **Skill** — 从跑通的任务里提炼可复用 SOP，附版本 / 资源文件 / 触发边界 /
  执行步骤 / 验证规则。
- **Wiki** — 把文档变成结构化页面 + 链接图谱（灵感来自 Karpathy 的 LLM 知识库
  实践）。
- **CodeGraph** — 索引仓库的符号 / 文件 / 调用关系 / 影响路径，Agent 改代码
  前先做 impact analysis。

### 🎛️ Memory Hub · 面向团队的操作台

管控面板（`agentmemory/memory-hub` 镜像，含 Panel + Knowledge Service）：

- 建 Team / Agent，把资产按 Owner / 版本 / 状态 / 可见性统一管理
- 三级可见性：`private` / `team` / `restricted`（User / Role / Agent ACL），
  外加 `agent` 定向装配
- Agent Loadout：给不同 Agent 绑定不同资产、调整优先级和使用方式
- Wiki + CodeGraph 工坊内置在 Hub，导入代码库/文档就能自动构建

### 🔀 Memory Proxy · Agent 挂上记忆的通道

`agentmemory/memory-proxy` 让 Claude Code 等 coding agent 直接用上团队记忆：

- **Anthropic / OpenAI 双协议**：`/claude-code/<spaceId>/v1/messages` 和
  `/v1/chat/completions` 都接
- **首轮引导**：sessionInit 通过 `AskUserQuestion` 让用户选 team / agent /
  task，proxy 记住绑定
- **每轮注入**：把该 agent 的 L2/L3 记忆、matched skill、wiki/code-graph
  拼进 system prompt，转发上游 LLM
- **鉴权**：`x-tdai-user-key` → 内核 `/v3/meta/auth/verify` 换 `user_id`，
  按用户维度控制资产可见性

### 🚀 一条命令拉起完整三件套

三个镜像多架构（`linux/amd64` + `linux/arm64`）已发布到
[Docker Hub `agentmemory`](https://hub.docker.com/u/agentmemory)，公开可拉、
无需登录：

```bash
git clone https://github.com/Tencent/TencentDB-Agent-Memory.git
cd TencentDB-Agent-Memory/deploy/global-images
cp .env.example .env && $EDITOR .env    # 填入两组 LLM 参数
./start-all.sh                          # 一键起
```

`start-all.sh` 首次启动会自动 `init-admin`、生成 admin `sk-mem-...` 并落盘
`.admin-key`；自检 `/v3/meta/auth/verify` 后打印可复制的 `claude` 启动命令。
`stop-all.sh --purge` 彻底清 volume + admin key，方便重置。

详见 [INSTALL_CN.md](./INSTALL_CN.md) / [INSTALL.md](./INSTALL.md)。

### 🧰 官方 SDK

- **TypeScript** — `@tencentdb-agent-memory/memory-sdk-ts-v2`

  ```ts
  import { MemoryClient, SkillClient, MetadataClient } from "@tencentdb-agent-memory/memory-sdk-ts-v2";

  const memory = new MemoryClient({
    endpoint, apiKey, serviceId,
    teamId, agentId, userId,     // v3 严格 isolation：三项必填
  });
  ```

  顶级 export 就是 v3 严格 isolation 版本；老代码走 `.../v2/v3` 子路径也
  能继续用（子路径保留为向后兼容别名）。

- **Python** — `pip install tencentdb-agent-memory-sdk-python`

  ```python
  from tencentdb_agent_memory import MemoryClient                     # 默认（v2 兼容）
  from tencentdb_agent_memory.v3 import MemoryClient, MetadataClient, SkillClient
  ```
