# SQLite → 腾讯云向量数据库迁移工具

离线迁移工具，用于将 memory-tdai 的数据从本地 SQLite 存储迁移到腾讯云向量数据库（TCVDB）。

## 前置条件

- Node.js >= 22.16.0
- 插件已通过 `openclaw plugins install` 安装
- 迁移脚本已编译（见下方）

## 编译

迁移脚本使用 TypeScript 编写，运行前需要先编译：

```bash
npm run build:migrate-sqlite-to-vdb
```

编译产物输出到 `scripts/migrate-sqlite-to-tcvdb/dist/`，可直接用 Node 运行。

## 使用方法

```bash
# 预检模式（仅查看源数据，不执行写入）
npm run migrate:sqlite-to-tcvdb -- \
  --plugin-data-dir ~/.openclaw/memory-tdai \
  --openclaw-config-path ~/.openclaw/openclaw.json \
  --tcvdb-url http://127.0.0.1:80 \
  --tcvdb-username root \
  --tcvdb-api-key-env TCVDB_API_KEY \
  --tcvdb-database agent_memory_prod \
  --tcvdb-embedding-model bge-large-zh \
  --dry-run

# 正式迁移
npm run migrate:sqlite-to-tcvdb -- \
  --plugin-data-dir ~/.openclaw/memory-tdai \
  --openclaw-config-path ~/.openclaw/openclaw.json \
  --tcvdb-url http://127.0.0.1:80 \
  --tcvdb-username root \
  --tcvdb-api-key-env TCVDB_API_KEY \
  --tcvdb-database agent_memory_prod \
  --tcvdb-embedding-model bge-large-zh \
  --yes
```

### 更多示例

```bash
# 直接传入 API Key（不通过环境变量）
npm run migrate:sqlite-to-tcvdb -- \
  --plugin-data-dir ~/.openclaw/memory-tdai \
  --openclaw-config-path ~/.openclaw/openclaw.json \
  --tcvdb-url http://127.0.0.1:80 \
  --tcvdb-username root \
  --tcvdb-api-key 'your-api-key-here' \
  --tcvdb-database agent_memory_prod \
  --tcvdb-embedding-model bge-large-zh \
  --yes
```

```bash
# 指定自定义 SQLite 路径（数据库不在默认 vectors.db 位置时）
npm run migrate:sqlite-to-tcvdb -- \
  --plugin-data-dir ~/.openclaw/memory-tdai \
  --sqlite-path /backup/2026-04/vectors-snapshot.db \
  --openclaw-config-path ~/.openclaw/openclaw.json \
  --tcvdb-url http://127.0.0.1:80 \
  --tcvdb-username root \
  --tcvdb-api-key-env TCVDB_API_KEY \
  --tcvdb-database agent_memory_prod \
  --tcvdb-embedding-model bge-large-zh \
  --yes
```

```bash
# 只迁移 L1 记忆层（跳过 L0 原始消息和 Profile）
npm run migrate:sqlite-to-tcvdb -- \
  --plugin-data-dir ~/.openclaw/memory-tdai \
  --openclaw-config-path ~/.openclaw/openclaw.json \
  --tcvdb-url http://127.0.0.1:80 \
  --tcvdb-username root \
  --tcvdb-api-key-env TCVDB_API_KEY \
  --tcvdb-database agent_memory_prod \
  --tcvdb-embedding-model bge-large-zh \
  --layers l1 \
  --yes
```

```bash
# 只迁移 L0 和 L1（不迁移 Profile）
npm run migrate:sqlite-to-tcvdb -- \
  --plugin-data-dir ~/.openclaw/memory-tdai \
  --openclaw-config-path ~/.openclaw/openclaw.json \
  --tcvdb-url http://127.0.0.1:80 \
  --tcvdb-username root \
  --tcvdb-api-key-env TCVDB_API_KEY \
  --tcvdb-database agent_memory_prod \
  --tcvdb-embedding-model bge-large-zh \
  --layers l0,l1 \
  --yes
```

```bash
# 英文语料场景：使用英文 BM25 分词
npm run migrate:sqlite-to-tcvdb -- \
  --plugin-data-dir ~/.openclaw/memory-tdai \
  --openclaw-config-path ~/.openclaw/openclaw.json \
  --tcvdb-url http://127.0.0.1:80 \
  --tcvdb-username root \
  --tcvdb-api-key-env TCVDB_API_KEY \
  --tcvdb-database agent_memory_prod \
  --tcvdb-embedding-model bge-large-en-v1.5 \
  --bm25-language en \
  --yes
```

```bash
# 禁用 BM25 稀疏向量（仅使用密集向量检索）
npm run migrate:sqlite-to-tcvdb -- \
  --plugin-data-dir ~/.openclaw/memory-tdai \
  --openclaw-config-path ~/.openclaw/openclaw.json \
  --tcvdb-url http://127.0.0.1:80 \
  --tcvdb-username root \
  --tcvdb-api-key-env TCVDB_API_KEY \
  --tcvdb-database agent_memory_prod \
  --tcvdb-embedding-model bge-large-zh \
  --no-bm25-enabled \
  --yes
```

```bash
# 仅迁移数据，不自动更新 openclaw.json 和 manifest（手动管理配置）
npm run migrate:sqlite-to-tcvdb -- \
  --plugin-data-dir ~/.openclaw/memory-tdai \
  --openclaw-config-path ~/.openclaw/openclaw.json \
  --tcvdb-url http://127.0.0.1:80 \
  --tcvdb-username root \
  --tcvdb-api-key-env TCVDB_API_KEY \
  --tcvdb-database agent_memory_prod \
  --tcvdb-embedding-model bge-large-zh \
  --no-apply-config \
  --no-rewrite-manifest \
  --yes
```

```bash
# 追加迁移：允许目标库已有数据，跳过非空检查
npm run migrate:sqlite-to-tcvdb -- \
  --plugin-data-dir ~/.openclaw/memory-tdai \
  --openclaw-config-path ~/.openclaw/openclaw.json \
  --tcvdb-url http://127.0.0.1:80 \
  --tcvdb-username root \
  --tcvdb-api-key-env TCVDB_API_KEY \
  --tcvdb-database agent_memory_prod \
  --tcvdb-embedding-model bge-large-zh \
  --no-fail-if-target-nonempty \
  --no-verify-counts \
  --yes
```

```bash
# 输出迁移摘要到 JSON 文件（适合 CI/自动化流水线）
npm run migrate:sqlite-to-tcvdb -- \
  --plugin-data-dir ~/.openclaw/memory-tdai \
  --openclaw-config-path ~/.openclaw/openclaw.json \
  --tcvdb-url http://127.0.0.1:80 \
  --tcvdb-username root \
  --tcvdb-api-key-env TCVDB_API_KEY \
  --tcvdb-database agent_memory_prod \
  --tcvdb-embedding-model bge-large-zh \
  --summary-json-path ./migration-report.json \
  --job-id "migrate-2026-04-13" \
  --yes
```

```bash
# 设置自定义超时和别名
npm run migrate:sqlite-to-tcvdb -- \
  --plugin-data-dir ~/.openclaw/memory-tdai \
  --openclaw-config-path ~/.openclaw/openclaw.json \
  --tcvdb-url http://10.0.1.50:80 \
  --tcvdb-username admin \
  --tcvdb-api-key-env TCVDB_API_KEY \
  --tcvdb-database agent_memory_prod \
  --tcvdb-embedding-model bge-large-zh \
  --tcvdb-alias "生产环境-主库" \
  --tcvdb-timeout-ms 30000 \
  --yes
```

## 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `--plugin-data-dir` | 是 | — | 插件数据目录路径 |
| `--openclaw-config-path` | 是 | — | `openclaw.json` 配置文件路径 |
| `--sqlite-path` | 否 | `<plugin-data-dir>/vectors.db` | SQLite 数据库文件路径（默认取数据目录下的 `vectors.db`） |
| `--plugin-id` | 否 | `memory-tencentdb` | 写入配置时使用的插件 ID |
| `--tcvdb-url` | 是 | — | TCVDB 服务地址 |
| `--tcvdb-username` | 是 | — | TCVDB 用户名 |
| `--tcvdb-api-key` | * | — | TCVDB API 密钥（明文） |
| `--tcvdb-api-key-env` | * | — | 包含 API 密钥的环境变量名 |
| `--tcvdb-database` | 是 | — | TCVDB 数据库名 |
| `--tcvdb-embedding-model` | 是 | — | Embedding 模型名称 |
| `--tcvdb-alias` | 否 | `""` | 用户自定义别名 |
| `--tcvdb-timeout-ms` | 否 | `10000` | 请求超时时间（毫秒） |
| `--layers` | 否 | `l0,l1,l2,l3` | 要迁移的层（逗号分隔） |
| `--dry-run` | 否 | `false` | 仅预览，不执行写入 |
| `--yes` | 否 | `false` | 跳过交互确认 |
| `--apply-config` | 否 | `true` | 迁移后更新 openclaw.json |
| `--config-backup` | 否 | `true` | 写入配置前先备份原配置文件 |
| `--rewrite-manifest` | 否 | `true` | 将 manifest.json 更新为 tcvdb |
| `--fail-if-target-nonempty` | 否 | `true` | 目标库非空时中止 |
| `--verify-counts` | 否 | `true` | 迁移后校验记录数 |
| `--summary-json-path` | 否 | — | 将迁移摘要写入此文件 |
| `--job-id` | 否 | — | 迁移任务 ID（用于追踪） |
| `--bm25-enabled` | 否 | `true` | 启用 BM25 稀疏向量 |
| `--bm25-language` | 否 | `zh` | BM25 语言（`zh` 或 `en`） |

\* `--tcvdb-api-key` 和 `--tcvdb-api-key-env` 二选一，必须提供其中一个。

## 目录结构

```
scripts/migrate-sqlite-to-tcvdb/
├── cli-entry.ts          # CLI 入口
├── sqlite-to-tcvdb.ts    # 迁移核心逻辑（参数解析、预检、数据迁移）
├── config-write.ts       # OpenClaw 配置更新（JSON5，自包含）
├── manifest-write.ts     # Manifest 重写
├── *.test.ts             # 就近放置的测试文件
├── tsconfig.json         # 迁移脚本编译配置
├── dist/                 # 编译产物（已 gitignore）
└── README.md             # 本文件

bin/migrate-sqlite-to-tcvdb.mjs     # 极薄 bin 包装 → dist/
```

迁移脚本通过 `../../src/` 引用存储实现（VectorStore、TcvdbMemoryStore 等），但**不依赖 `openclaw/plugin-sdk`**。配置写回直接使用 `json5`。
