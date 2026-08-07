# memory-tdai CLI

`openclaw memory-tdai` 命令空间，提供离线数据管理工具。

## seed — 导入历史对话数据

将历史对话 JSON 文件导入到记忆管线中，完整执行 L0→L1→L2→L3 流程。适用于：

- 将已有对话数据灌入记忆系统
- 批量测试记忆提取效果
- 迁移/恢复记忆数据

### 用法

```bash
openclaw memory-tdai seed --input <file> [options]
```

### 参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `--input <file>` | ✅ | 输入 JSON 文件路径 |
| `--output-dir <dir>` | — | 输出目录（默认自动生成带时间戳的目录） |
| `--session-key <key>` | — | 回退 session key（当输入数据缺少时使用） |
| `--config <file>` | — | 配置覆盖文件（JSON，与 openclaw.json 插件配置深度合并） |
| `--strict-round-role` | — | 严格校验每轮对话必须包含 user 和 assistant 消息 |
| `--yes` | — | 跳过交互确认（如时间戳自动填充确认） |

### 示例

```bash
# 基本用法
openclaw memory-tdai seed --input conversations.json

# 指定输出目录
openclaw memory-tdai seed --input data.json --output-dir ./seed-output

# 使用自定义配置覆盖（如调整 pipeline 参数）
openclaw memory-tdai seed --input data.json --config seed-config.json

# 跳过所有确认
openclaw memory-tdai seed --input data.json --yes

# 严格模式 + 自定义配置
openclaw memory-tdai seed --input data.json --config seed-config.json --strict-round-role --yes
```

### 输入文件格式

支持两种 JSON 格式：

#### Format A：对象包装

```json
{
  "sessions": [
    {
      "sessionKey": "user-alice",
      "sessionId": "conv-001",
      "conversations": [
        [
          { "role": "user", "content": "你好", "timestamp": 1711929600000 },
          { "role": "assistant", "content": "你好！有什么可以帮你？", "timestamp": 1711929601000 }
        ],
        [
          { "role": "user", "content": "今天天气怎么样？" },
          { "role": "assistant", "content": "今天晴天，适合出门。" }
        ]
      ]
    }
  ]
}
```

#### Format B：顶层数组

```json
[
  {
    "sessionKey": "user-alice",
    "conversations": [
      [
        { "role": "user", "content": "你好" },
        { "role": "assistant", "content": "你好！" }
      ]
    ]
  }
]
```

#### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `sessionKey` | string | ✅ | Session 标识（如用户 ID、频道名） |
| `sessionId` | string | — | 会话实例 ID（同一 sessionKey 下可有多个 sessionId） |
| `conversations` | message[][] | ✅ | 对话轮次数组，每个轮次是一组消息 |
| `role` | string | ✅ | 消息角色：`user` 或 `assistant` |
| `content` | string | ✅ | 消息内容 |
| `timestamp` | number \| string | — | 时间戳：epoch 毫秒或 ISO 8601 字符串。缺失时 seed 会提示自动填充 |

### 配置覆盖

`--config` 接受一个 JSON 文件，与 `openclaw.json` 中的插件配置**两级深度合并**：

- 顶层 key 如果两边都是对象 → 浅合并（保留 base 中未覆盖的字段）
- 其他类型 → 直接覆盖

常见场景：seed 时使用更激进的 pipeline 参数以加速处理：

```json
{
  "pipeline": {
    "everyNConversations": 3,
    "enableWarmup": false,
    "l1IdleTimeoutSeconds": 2,
    "l2DelayAfterL1Seconds": 1,
    "l2MinIntervalSeconds": 1,
    "l2MaxIntervalSeconds": 10
  }
}
```

如果需要 seed 到独立的 TCVDB 数据库：

```json
{
  "storeBackend": "tcvdb",
  "tcvdb": {
    "database": "my_seed_test_db"
  },
  "pipeline": {
    "everyNConversations": 3,
    "enableWarmup": false,
    "l1IdleTimeoutSeconds": 2
  }
}
```

### 输出目录结构

```
<output-dir>/
├── conversations/          — L0 JSONL 文件
├── records/                — L1 JSONL 文件
├── scene_blocks/           — L2 场景块
├── vectors.db              — SQLite 向量数据库（仅 sqlite 后端）
├── .metadata/
│   ├── manifest.json       — 元数据（store 绑定 + seed 运行记录）
│   └── checkpoint.json     — 管线进度
└── .backup/                — 滚动备份
```

Seed 完成后，`manifest.json` 会记录本次运行信息：

```json
{
  "version": 1,
  "createdAt": "2026-04-01T22:00:00.000Z",
  "store": {
    "type": "sqlite",
    "sqlite": { "path": "vectors.db" }
  },
  "seed": {
    "inputFile": "conversations.json",
    "sessions": 3,
    "rounds": 42,
    "messages": 128,
    "startedAt": "2026-04-01T22:00:00.000Z",
    "completedAt": "2026-04-01T22:05:30.000Z"
  }
}
```
