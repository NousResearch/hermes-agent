---
name: openclaw-memory-tencentdb-migration
description: 帮助存量用户将 OpenClaw 记忆插件从旧包 @tdai/memory-tdai 迁移到新包 @tencentdb-agent-memory/memory-tencentdb。当用户提到"插件迁移""更换记忆插件包名""memory-tdai 升级""包名变更"或出现旧包相关安装报错时应触发。
version: 1.0.0
---

## 目的

帮助已安装 `@tdai/memory-tdai`（旧包名）的存量用户，平滑迁移到 `@tencentdb-agent-memory/memory-tencentdb`（新包名），确保已有记忆数据不丢失、配置完整还原。

## 背景

- **旧包名**：`@tdai/memory-tdai`（插件 ID：`memory-tdai`）
- **新包名**：`@tencentdb-agent-memory/memory-tencentdb`（插件 ID：`memory-tencentdb`）
- 新旧插件共用相同的数据目录（`~/.openclaw/memory-tdai/`），卸载旧插件**不会删除数据目录**，已有记忆数据不受影响
- 卸载旧插件**会删除** `openclaw.json` 中该插件的配置段，需提前备份

## 适用场景

- 用户已安装 `@tdai/memory-tdai`，需迁移到新包名
- 用户执行 `openclaw plugins install @tdai/memory-tdai` 报 404 / not found
- 用户被告知旧包已废弃，需要迁移

## 不适用场景

- 用户从未安装过记忆插件（应使用 `openclaw-memory-tencentdb-setup` skill）
- 用户使用的是其他记忆插件（如 `openclaw-mem0`）

## 标准工作流

### 1) 确认当前状态

确认旧插件是否已安装：

```bash
openclaw plugins list | grep -i memory
```

预期看到 `memory-tdai` 或 `@tdai/memory-tdai` 处于 loaded 状态。

如果未看到旧插件，跳过迁移流程，直接使用 `openclaw-memory-tencentdb-setup` skill 进行全新安装。

### 2) 备份现有配置（关键步骤）

卸载旧插件会删除 `openclaw.json` 中的配置段。**必须先备份**。

执行以下命令提取旧插件配置：

```bash
cat ~/.openclaw/openclaw.json | python3 -c "
import sys, json
cfg = json.load(sys.stdin)
plugins = cfg.get('plugins', {}).get('entries', {})
old_cfg = plugins.get('memory-tdai', {})
if old_cfg:
    print(json.dumps(old_cfg, indent=2, ensure_ascii=False))
    with open('/tmp/memory-tdai-config-backup.json', 'w') as f:
        json.dump(old_cfg, f, indent=2, ensure_ascii=False)
    print('\n✅ 配置已备份到 /tmp/memory-tdai-config-backup.json')
else:
    print('⚠️ 未找到 memory-tdai 配置段（可能使用默认配置）')
"
```

**特别关注以下配置是否存在（如有则必须记录）**：

- `embedding` 配置（`provider`、`baseUrl`、`apiKey`、`model`、`dimensions`、`proxyUrl`）
- `extraction.model`（提取使用的模型）
- `persona.model`（画像使用的模型）
- `capture.excludeAgents`（排除的 agent 列表）
- `capture.l0l1RetentionDays`（数据保留天数）

### 3) 确认数据目录存在

```bash
ls -la ~/.openclaw/memory-tdai/
```

预期看到：`conversations/`、`records/`、`scene_blocks/`、`vectors.db`、`persona.md` 等文件。

记录当前数据量作为迁移后验证依据：

```bash
echo "=== 迁移前数据统计 ==="
wc -l ~/.openclaw/memory-tdai/conversations/*.jsonl 2>/dev/null || echo "无对话数据"
wc -l ~/.openclaw/memory-tdai/records/*.jsonl 2>/dev/null || echo "无记录数据"
ls ~/.openclaw/memory-tdai/scene_blocks/*.md 2>/dev/null | wc -l | xargs -I{} echo "场景块: {} 个"
wc -c ~/.openclaw/memory-tdai/persona.md 2>/dev/null || echo "无 persona"
```

### 4) 卸载旧插件

```bash
openclaw plugins uninstall memory-tdai
```

执行后确认：

- `openclaw.json` 中 `memory-tdai` 配置段已被删除（预期行为）
- `~/.openclaw/memory-tdai/` 数据目录**仍然存在**（不会被删除）

```bash
# 验证数据目录仍在
ls ~/.openclaw/memory-tdai/ && echo "✅ 数据目录完好" || echo "❌ 数据目录丢失！"
```

### 5) 安装新插件

```bash
openclaw plugins install @tencentdb-agent-memory/memory-tencentdb
```

### 6) 还原配置

将步骤 2 备份的配置写回 `openclaw.json`，注意新插件的配置 key 是 `memory-tencentdb`：

```bash
python3 -c "
import json, os

# 读取备份配置
backup_path = '/tmp/memory-tdai-config-backup.json'
if os.path.exists(backup_path):
    with open(backup_path) as f:
        old_cfg = json.load(f)
    print('📋 备份配置内容：')
    print(json.dumps(old_cfg, indent=2, ensure_ascii=False))
else:
    old_cfg = {'enabled': True}
    print('⚠️ 未找到备份，使用最小配置')

# 读取当前 openclaw.json
config_path = os.path.expanduser('~/.openclaw/openclaw.json')
with open(config_path) as f:
    cfg = json.load(f)

# 写入新插件配置
cfg.setdefault('plugins', {}).setdefault('entries', {})['memory-tencentdb'] = old_cfg

with open(config_path, 'w') as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)

print('\n✅ 配置已写入 memory-tencentdb')
"
```

如果备份丢失或用户需要手动恢复，至少确保写入最小配置：

```json
{
  "memory-tencentdb": {
    "enabled": true
  }
}
```

### 7) 重启 Gateway 并验证

```bash
openclaw gateway restart
```

检查项：

- Gateway 日志中出现 `[memory-tdai]` 前缀（注：日志标签仍为 memory-tdai，这是正常的）
- 数据目录内容未变化

```bash
echo "=== 迁移后验证 ==="
# 确认新插件已加载
openclaw plugins list | grep -i memory

# 确认数据量与迁移前一致
wc -l ~/.openclaw/memory-tdai/conversations/*.jsonl 2>/dev/null
wc -l ~/.openclaw/memory-tdai/records/*.jsonl 2>/dev/null
```

### 8) 功能冒烟验证

执行一次对话确认记忆链路正常：

1. 发送一条包含个人信息的消息（如偏好、习惯）
2. 确认日志中有 `[before_prompt_build]` 和 `[agent_end]` 相关输出
3. 如有 embedding 配置，确认向量检索正常（日志无 embedding 报错）

## 回滚方案

如迁移后出现问题，可快速回滚：

```bash
# 1. 卸载新插件
openclaw plugins uninstall memory-tencentdb

# 2. 重新安装旧插件（如 npm 源仍可用）
openclaw plugins install @tdai/memory-tdai

# 3. 手动还原配置（从备份）
# 将 /tmp/memory-tdai-config-backup.json 内容写回 openclaw.json 的 memory-tdai 段

# 4. 重启
openclaw gateway restart
```

## 故障排查

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| 新插件无日志输出 | 配置中 `enabled` 未设为 `true` | 检查 `openclaw.json` 中 `memory-tencentdb.enabled` |
| 安装新插件报错 | npm 源不可用 | 检查网络 / npm registry 配置 |
| 迁移后无历史记忆 | 配置还原不完整 | 对比 `/tmp/memory-tdai-config-backup.json` 与当前配置 |
| embedding 报错 | `apiKey` 等配置丢失 | 从备份中还原 `embedding` 配置段 |
| 数据目录为空 | 卸载时异常删除（极少见） | 检查 `~/.openclaw/memory-tdai/` 是否存在 |

## 安全与合规约束

- 备份文件 `/tmp/memory-tdai-config-backup.json` 可能包含 `apiKey`，迁移完成后建议删除：`rm /tmp/memory-tdai-config-backup.json`
- 不在聊天、日志中明文展示 `apiKey`
- 仅修改 `memory-tencentdb` 配置段，不影响用户其它插件

## 完成定义（Definition of Done）

迁移完成需同时满足：

- [x] 旧插件 `@tdai/memory-tdai` 已卸载
- [x] 新插件 `@tencentdb-agent-memory/memory-tencentdb` 已安装并加载
- [x] `openclaw.json` 中存在完整的 `memory-tencentdb` 配置（含用户自定义的 embedding 等配置）
- [x] Gateway 已重启
- [x] 日志中出现 `[memory-tdai]` 前缀
- [x] 数据目录完好，数据量与迁移前一致
- [x] 至少 1 次对话验证记忆链路正常
- [x] 已清理备份文件中的敏感信息

## 交付话术模板

> 已完成记忆插件迁移：
> - 旧插件 `@tdai/memory-tdai` → 新插件 `@tencentdb-agent-memory/memory-tencentdb`
> - 已有记忆数据完整保留（对话/记录/场景块/向量库均未受影响）
> - 配置已从旧插件完整还原（含 embedding / extraction / persona 等自定义配置）
> - Gateway 已重启，记忆链路验证正常
