# 数据迁移工具

## 使用场景

从 MemoryCore v1.x或 v0.x 升级到 v2.0.0+ 时，数据面的表结构和文件布局发生了变化，需要先运行本迁移脚本将存量数据升级到 v3 格式，再启动新版 Gateway。

**受影响的数据：**

- `vectors.db` — 新增 `team_id`、`task_id`、`user_id`、`agent_id`、`version` 等租户隔离字段
- `scene_blocks/`、`persona.md`、`.metadata/` — L2/L3 文件迁移到 `profiles/` 子目录下的 scoped 路径

**⚠️ 迁移前请务必备份整个数据目录**，避免意外数据丢失。

## 前置条件

- Python 3.8+
- 数据目录路径（默认为 `~/.memory-tencentdb/memory-tdai/`）

## 用法

```bash
# 1. 先 dry-run 检查，不实际修改数据
python v2-to-v3-migrate.py /path/to/memory-tdai --dry-run

# 2. 确认无误后执行迁移
python v2-to-v3-migrate.py /path/to/memory-tdai

# 3. 仅迁移数据库（跳过 L2/L3 文件）
python v2-to-v3-migrate.py /path/to/memory-tdai --db-only
```

### 参数说明

| 参数 | 说明 |
|---|---|
| `/path/to/memory-tdai` | 数据目录路径，必填 |
| `--dry-run` | 仅检查，不实际修改 |
| `--db-only` | 仅迁移 `vectors.db` 表结构，跳过 L2/L3 文件 |
| `--no-backup` | 跳过自动备份（默认会自动创建 `.bak` 文件） |

## 迁移内容

### 1. 数据库表结构升级

| 表 | 变更 |
|---|---|
| `l1_records` | 新增 `team_id`、`task_id`、`user_id`、`agent_id`、`version` 字段 |
| `l0_conversations` | 新增 `team_id`、`task_id`、`user_id`、`agent_id` 字段 |
| `l1_fts` / `l0_fts` | 重建 FTS5 索引，增加租户隔离列 |
| `memory_audit` | 新增审计表 |
| `skills` | 新增技能表 |
| `skill_fts` | 新增技能全文索引表 |

### 2. L2/L3 文件迁移

| 源路径 | 目标路径 |
|---|---|
| `{data_dir}/scene_blocks/` | `{data_dir}/profiles/team%3Adefault%7Cagent%3Adefault/scene_blocks/` |
| `{data_dir}/persona.md` | `{data_dir}/profiles/team%3Adefault%7Cagent%3Adefault/persona.md` |
| `{data_dir}/.metadata/` | `{data_dir}/profiles/team%3Adefault%7Cagent%3Adefault/.metadata/` |

## 示例

```bash
# Hermes 场景
python scripts/migrate-v2-to-v3/v2-to-v3-migrate.py ~/.memory-tencentdb/memory-tdai --dry-run
python scripts/migrate-v2-to-v3/v2-to-v3-migrate.py ~/.memory-tencentdb/memory-tdai

# OpenClaw 场景
python scripts/migrate-v2-to-v3/v2-to-v3-migrate.py ~/.memory-tencentdb/memory-tdai --dry-run
python scripts/migrate-v2-to-v3/v2-to-v3-migrate.py ~/.memory-tencentdb/memory-tdai
```

## 常见问题

**Q: 迁移失败了怎么办？**

脚本默认会在迁移前自动备份 `vectors.db`（生成 `.bak.{timestamp}` 文件）。L2/L3 文件采用复制而非移动，源文件不会被删除。如果迁移失败，直接用备份恢复即可。

**Q: 可以重复执行吗？**

可以。脚本是幂等的——已存在的字段和文件会跳过，不会重复处理。

**Q: 全新安装需要跑迁移吗？**

不需要。迁移脚本仅用于从旧版（v1.x）升级到新版（v2.0.0+）的存量用户。全新安装的新版 Gateway 会自动创建 v3 格式的数据。
