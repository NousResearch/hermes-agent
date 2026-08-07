---
name: openclaw-memory-tencentdb-setup
description: 用于在 OpenClaw 环境中安装、配置并验证 @tencentdb-agent-memory/memory-tencentdb 插件。当用户提到"安装记忆插件""配置 memory-tencentdb""开启长期记忆/召回"或出现相关报错时应触发。
version: 1.0.0
---

## 目的

在不依赖外部托管记忆服务的前提下，为 OpenClaw 提供可持续的本地长期记忆能力（L0→L1→L2→L3），并完成从安装、配置到验收的一次性闭环。

## 适用场景

- 用户要求在 OpenClaw 中安装或启用 `memory-tencentdb`
- 用户需要配置召回、提取、画像、清理等参数
- 用户反馈"插件已装但无记忆 / 无召回 / 无向量检索"

## 不适用场景

- 用户只需要解释 memory 理念，不要求实际落地
- 用户要接入非 OpenClaw 宿主（先确认目标框架）

## 标准工作流

### 1) 环境预检

先确认基础版本满足要求：

- OpenClaw: `>= 2026.3.13`
- Node.js: `>= 22.16.0`

执行：

```bash
openclaw --version
node -v
```

若版本不满足，先升级再继续。

### 2) 安装插件

执行安装命令：

```bash
openclaw plugins install @tencentdb-agent-memory/memory-tencentdb
```

如已安装则执行更新：

```bash
openclaw plugins update memory-tencentdb
```

### 3) 写入最小配置

编辑 `~/.openclaw/openclaw.json`，确保存在：

```json
{
  "memory-tencentdb": {
    "enabled": true
  }
}
```

说明：该插件支持零配置启动；不补充其它字段也能运行基础能力。

### 4) 按需追加推荐配置（生产常用）

根据用户需求补充如下分组：

- `capture`: 对话捕获与保留策略
- `extraction`: L1 提取与去重
- `pipeline`: L1→L2→L3 调度
- `recall`: 召回数量、阈值、策略
- `persona`: 场景与画像触发参数
- `embedding`: 向量检索配置（远端 OpenAI 兼容）

推荐模板：

```json
{
  "memory-tencentdb": {
    "capture": {
      "enabled": true,
      "excludeAgents": [],
      "l0l1RetentionDays": 90,
      "cleanTime": "03:00"
    },
    "extraction": {
      "enabled": true,
      "enableDedup": true,
      "maxMemoriesPerSession": 10,
      "model": "provider/model"
    },
    "pipeline": {
      "everyNConversations": 5,
      "enableWarmup": true,
      "l1IdleTimeoutSeconds": 600,
      "l2DelayAfterL1Seconds": 10,
      "l2MinIntervalSeconds": 900,
      "l2MaxIntervalSeconds": 3600,
      "sessionActiveWindowHours": 24
    },
    "recall": {
      "enabled": true,
      "maxResults": 5,
      "scoreThreshold": 0.3,
      "strategy": "hybrid"
    },
    "persona": {
      "triggerEveryN": 50,
      "maxScenes": 15,
      "backupCount": 3,
      "sceneBackupCount": 10,
      "model": "provider/model"
    },
    "embedding": {
      "enabled": true,
      "provider": "openai",
      "baseUrl": "https://api.openai.com/v1",
      "apiKey": "${EMBEDDING_API_KEY}",
      "model": "text-embedding-3-small",
      "dimensions": 1536,
      "conflictRecallTopK": 5
    }
  }
}
```

### 5) 关键配置规则（避免隐性失败）

- `embedding.provider = "none"` 时，向量能力会禁用，仅保留关键词路径。
- 若配置远端 `provider`（如 `openai` / `deepseek`），必须同时提供：
  - `apiKey`
  - `baseUrl`
  - `model`
  - `dimensions`
- 上述任一缺失时，插件会继续运行，但自动降级为非向量模式。
- `l0l1RetentionDays`：
  - `0` 表示不清理
  - 非 `0` 时建议 `>=3`
  - 若设为 `1~2`，需显式开启 `allowAggressiveCleanup`

### 6) 重启并验证生效

执行：

```bash
openclaw gateway restart
```

检查项：

- Gateway 日志中出现 `[memory-tdai]` 前缀
- 数据目录已创建：`~/.openclaw/state/memory-tdai/`
- 至少包含：`conversations/`、`records/`、`scene_blocks/`、`vectors.db`

### 7) 功能冒烟测试

执行一次最小对话回路并验证：

1. 连续对话 2~3 轮，提供可记忆信息（偏好、约束、背景）。
2. 发起新一轮对话，观察是否出现召回上下文注入。
3. 在 Agent 中调用：
   - `tdai_memory_search`
   - `tdai_conversation_search`
4. 确认能检索到刚刚产生的内容。

## 故障排查速查

- 插件无日志：检查 `openclaw.json` 中 `memory-tencentdb.enabled` 是否为 `true`，并确认已重启 Gateway。
- 有记录无召回：检查 `recall.enabled`、`scoreThreshold` 是否过高。
- 无向量结果：检查 `embedding` 四元组（`apiKey/baseUrl/model/dimensions`）是否齐全。
- 清理过猛导致历史过少：检查 `l0l1RetentionDays` 与 `allowAggressiveCleanup`。
- 配置已改但行为不变：确认修改的是 `~/.openclaw/openclaw.json`，并再次重启 Gateway。

## 安全与合规约束

- 将 `apiKey` 视为敏感信息；不在聊天、日志、截图中明文扩散。
- 优先使用环境变量注入密钥；配置示例中仅保留占位符。
- 仅修改 `memory-tencentdb` 对应配置段，避免覆盖用户其它插件配置。

## 完成定义（Definition of Done）

在结束任务前，必须同时满足：

- 插件安装/更新命令执行成功
- `openclaw.json` 已存在有效 `memory-tencentdb` 配置
- Gateway 已重启
- `[memory-tdai]` 日志可见
- 数据目录与关键文件已生成
- 至少 1 次检索工具调用成功返回结果

## 交付话术模板

可在完成后向用户输出：

- 已完成 `memory-tencentdb` 安装与配置，并重启 Gateway。
- 已验证日志与数据目录生效，记忆链路可用。
- 如需下一步优化，可继续调优 `recall.scoreThreshold`、`pipeline.everyNConversations`、`persona.triggerEveryN` 与 `embedding` 模型参数。