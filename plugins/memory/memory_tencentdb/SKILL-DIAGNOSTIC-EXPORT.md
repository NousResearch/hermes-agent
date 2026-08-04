---
name: openclaw-diagnostic-export
description: 帮助用户导出 OpenClaw + memory-tencentdb（原 memory-tdai）记忆插件的现场诊断数据，用于排查问题。当用户提到"导出诊断数据""export diagnostic""现场数据""排查问题""导出日志""收集现场""打包现场数据"时应触发。
version: 1.0.0
---

## 目的

将 OpenClaw 日志、记忆插件数据（L0~L3）、脱敏后的配置打包为本地压缩包，由用户确认后手动发送给研发团队排查问题。

> **名称说明**：插件已从 `@tdai/memory-tdai` 更名为 `@tencentdb-agent-memory/memory-tencentdb`，但数据目录始终为 `~/.openclaw/memory-tdai/`（代码中硬编码）。本 skill 中所有对 `memory-tdai` 目录的引用均指实际数据目录路径，与插件 ID 无关。

## 导出工作流

### Step 1: 确认环境

在导出前，先确认 OpenClaw 工作目录存在且可访问：

```bash
# 探测工作目录（优先级：环境变量 > ~/.openclaw > ~/.clawdbot）
OPENCLAW_DIR="${OPENCLAW_STATE_DIR:-$HOME/.openclaw}"
[ -d "$OPENCLAW_DIR" ] || OPENCLAW_DIR="$HOME/.clawdbot"
ls -la "$OPENCLAW_DIR/" 2>/dev/null && echo "✅ 找到: $OPENCLAW_DIR" || echo "❌ 未找到 OpenClaw 工作目录"
```

确认 memory-tdai 子目录存在：

```bash
ls -la "$OPENCLAW_DIR/memory-tdai/" 2>/dev/null
```

### Step 2: 执行导出脚本

运行项目 `scripts/` 目录下的导出脚本：

```bash
bash scripts/export-diagnostic.sh
```

> 脚本位于本项目的 `scripts/export-diagnostic.sh`，如果通过 `pnpm` 或其他方式运行，需确保工作目录在项目根目录下。

脚本默认将压缩包输出到 `~/Downloads/openclaw-diagnostic-<timestamp>.tar.gz`。

如需指定其他输出目录：

```bash
bash scripts/export-diagnostic.sh /tmp
```

### Step 3: 确认导出结果

脚本执行完成后，检查输出：

1. **确认压缩包已生成** — 脚本末尾会打印压缩包路径和大小
2. **向用户说明包含内容**：

| 文件/目录 | 内容 | 隐私风险 |
|-----------|------|---------|
| `env-info.txt` | 系统版本、OpenClaw 版本、目录结构、磁盘占用 | 低 |
| `logs/` | OpenClaw 网关日志 + 滚动日志（最近 3 天，每文件最多 5000 行） | 低 |
| `memory-tdai/` | 记忆插件全量数据：L0 对话、L1 记忆、L2 场景、L3 画像、SQLite 数据库、checkpoint | **高** — 包含用户对话原文 |
| `openclaw-config-redacted.json` | 脱敏后的配置（已移除 API Key/Token/Password/Secret，models/channels/env 整体替换） | 低 |
| `plugins-info.txt` | 已安装插件列表和版本 | 低 |

3. **提醒用户**：
   - 配置文件已自动脱敏，API Key、Token 等敏感信息已被替换为 `***REDACTED***`
   - **记忆数据（memory-tdai/）包含用户对话原文**，请确认可以分享后再发送
   - 压缩包存放在本地，**不会自动上传**，需要用户手动发送给研发团队

### Step 4: 告知用户后续操作

导出完成后，告知用户：

1. 压缩包已保存在本地（打印具体路径）
2. 请检查内容后，通过企微/邮件等方式手动发送给研发团队
3. 如只需部分数据（如仅日志或仅配置），可解压后选择性发送

## 导出内容详解

### OpenClaw 日志位置

| 日志类型 | 路径 | 说明 |
|---------|------|------|
| 网关 stdout | `~/.openclaw/logs/gateway.log` | 网关守护进程标准输出 |
| 网关 stderr | `~/.openclaw/logs/gateway.err.log` | 网关守护进程错误输出 |
| 滚动日志 | `/tmp/openclaw/openclaw-YYYY-MM-DD.log` | 按日期滚动，JSON Lines 格式，24h 自动清理 |
| 配置审计 | `~/.openclaw/logs/config-audit.jsonl` | 配置写入审计记录 |
| 命令日志 | `~/.openclaw/logs/commands.log` | 命令事件日志（hook 可选） |

### 记忆插件数据结构

```
~/.openclaw/memory-tdai/
├── conversations/          — L0 原始对话（每日 JSONL 分片）
├── records/                — L1 结构化记忆（每日 JSONL 分片）
├── scene_blocks/           — L2 场景 Markdown 文件
├── persona.md              — L3 用户画像
├── vectors.db              — SQLite 数据库（向量 + 全文索引）
├── .metadata/              — checkpoint、scene_index.json
└── .backup/                — 滚动备份
```

### 配置脱敏规则

导出脚本对 `openclaw.json` 执行以下脱敏：

| 规则 | 处理方式 |
|------|---------|
| 字段名匹配 `apiKey/token/password/secret/credential` 且值为字符串 | 替换为 `***REDACTED(Nchars)***` |
| SecretRef 对象（含 source/provider/id） | id 替换为 `***REDACTED***` |
| 顶层 `models`、`secrets`、`channels`、`env` 块 | 整体替换为 `***REDACTED_SECTION***` |
| `gateway.auth` 下的 token/password | 替换为 `***REDACTED***` |
| 其余字段（含 `plugins` 完整配置） | **保留原样**（插件配置是排查重点） |

## 手动导出（脚本不可用时的备选方案）

如果导出脚本无法执行（如 Node.js 不可用），按以下步骤手动收集：

```bash
# 1. 创建导出目录
EXPORT_DIR=~/Downloads/openclaw-diagnostic-$(date +%Y%m%d-%H%M%S)
mkdir -p "$EXPORT_DIR"

# 2. 复制日志
cp -r ~/.openclaw/logs/ "$EXPORT_DIR/logs/" 2>/dev/null
cp /tmp/openclaw/openclaw-$(date +%Y-%m-%d).log "$EXPORT_DIR/" 2>/dev/null

# 3. 复制记忆插件数据
cp -r ~/.openclaw/memory-tdai/ "$EXPORT_DIR/memory-tdai/" 2>/dev/null

# 4. 手动脱敏配置（⚠️ 必须手动删除敏感字段！）
# 复制配置并用编辑器删除 models/secrets/channels 块和所有 apiKey/token 值
cp ~/.openclaw/openclaw.json "$EXPORT_DIR/openclaw-config-NEEDS-MANUAL-REDACTION.json"

# 5. 打包
cd ~/Downloads && tar -czf "$EXPORT_DIR.tar.gz" "$(basename $EXPORT_DIR)"

echo "⚠️ 请务必在发送前手动检查并删除配置中的敏感信息！"
```

## 常见问题排查线索

导出数据后，研发团队通常关注以下方面：

| 排查方向 | 查看文件 | 关键信息 |
|---------|---------|---------|
| 插件是否加载 | `logs/` 中搜索 `[memory-tdai]` | 插件注册、配置解析日志（注：日志标签仍为 `[memory-tdai]`，与插件 ID 无关） |
| 记忆召回是否工作 | `logs/` 中搜索 `[recall]` | 搜索策略、耗时、命中数 |
| L1 提取是否触发 | `logs/` 中搜索 `[pipeline]` | 调度触发、L1/L2/L3 执行状态 |
| 向量搜索是否可用 | `openclaw-config-redacted.json` 的 `plugins.entries` | embedding 配置是否正确 |
| 数据量/磁盘占用 | `env-info.txt` | du 输出、文件数量 |
| checkpoint 状态 | `memory-tdai/.metadata/recall_checkpoint.json` | 进度、游标、计数器 |
