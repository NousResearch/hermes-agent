# Changelog

本文件记录 `@tencentdb-agent-memory/memory-tencentdb` 插件的所有显著变更，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [Semantic Versioning](https://semver.org/)。

---

## [1.0.1] - 2026-07-13

### 🐛 修复

- **Gateway local L2 定时器未走场景提取** ([#227](https://github.com/Tencent/TencentDB-Agent-Memory/pull/227))：本地 timer 扫描将 L2 schedule 错路由到非 scene extraction 任务类型，导致 L2 场景提取定时触发失效。抽出 `timer-routing.ts`，按 timer member 前缀正确映射 `offload-l1` / `offload-l15` / `offload-l2` 与 `L1` / `L2` / `L3` 任务。
- **升级 `cos-nodejs-sdk-v5` 至 3.0.0**：旧版 COS SDK 存在安全风险，升级从而修复该问题。

---

## [1.0.0] - 2026-06-11

> **正式版发布**：从 OpenClaw 专属插件演进为**面向所有 Agent 的通用记忆服务**。完整的 Gateway 独立服务 + v2 HTTP API + 官方 TypeScript / Python SDK，任何 Agent 框架均可接入完整的多层记忆与上下文压缩能力。

### ⚠️ Breaking Changes

- **客户端/服务端架构拆分**：记忆引擎从 OpenClaw 嵌入式插件拆分为独立 Gateway 服务进程，部署方式与接入方式发生变化。
- **配置结构变更**：插件配置结构扁平化重构，原 `gateway` 字段迁移为 `server` 嵌套。
- **插件入口模式变更**：支持 `local`（进程内本地运行，默认）和 `client`（连接外部 Memory Gateway）两种接入模式。

### ✨ 新功能

#### 独立 Gateway 服务（v2 API）

记忆能力不再绑定 OpenClaw 宿主，以独立服务形式运行，通过 v2 HTTP API 为任意 Agent 提供记忆读写与管线管理：

- **完整 v2 API**：14 条标准路由覆盖记忆 CRUD、原子更新、场景索引、管线状态查询等全部操作。
- **管线状态查询（`/v2/pipeline/status`）**：实时获取 L1/L2/L3 各阶段运行状态与进度。
- **实例生命周期管理（`/v2/instance/destroy`）**：支持外部系统主动创建/销毁记忆实例。
- **可选 Bearer 鉴权 + CORS 白名单**：保护对外暴露的 API 安全。
- **请求体校验**：强制 1 MiB 上限，防止异常请求。

#### 官方 SDK

- **TypeScript SDK 1.0.0**（`@tencentdb-agent-memory/memory-sdk-ts`）：类型安全，覆盖全部 v2 API，npm 安装即用。
- **Python SDK**（`tencentdb-agent-memory-sdk-python`）：pip wheel 安装，同步/异步双模式，覆盖全部 v2 API。

#### 通用 Agent 框架适配

- **OpenClaw 插件适配**：支持 `local`（进程内本地运行，默认）和 `client`（连接外部 Memory Gateway）两种接入模式。`local` 模式保持原有体验。
- **Hermes Agent 适配**：`memory_tencentdb_v2` adapter，支持 Hermes 框架多租户场景。
- **通用接入**：任何能发 HTTP 请求的 Agent（LangChain、AutoGPT、自研框架等）均可通过 SDK 或裸 API 接入。

#### 可观测性

- **OpenTelemetry 全链路 Trace**：支持 OTLP 协议上报，可对接 Jaeger / Grafana Tempo 等后端。
- **Langfuse 集成**：仅转发 LLM 相关 span，适合评估记忆提取质量。
- **管线评测指标**：L1 提取率、去重决策分布、各阶段 token 消耗、recall 延迟等。

#### 部署

- **Standalone Docker 镜像**：单容器即可运行完整记忆服务，无需外部依赖。
- **Hermes + Memory All-in-One 镜像**：Agent + 记忆服务一体化部署方案。
- **`memory-tencentdb-ctl` CLI**：运维命令行工具，支持配置管理、存储引擎切换等操作。
- **v2 安装器脚本**：一键配置插件适配层。

### 📖 文档

- SDK 接入指南（TypeScript / Python 示例）。
- Gateway 独立部署文档。
- OpenClaw / Hermes 插件安装说明。
- Docker 部署与配置参考。

---

<details>
<summary>1.0.0 预发布版本</summary>

## [1.0.0-beta.2] - 2026-06-05

### ✨ 新功能

- 统一插件入口，支持 local（进程内本地运行，默认）和 client（连接外部 Memory Gateway）两种接入模式。
- 新增 Offload Server V2：L1/L1.5/L2 异步执行器、L3 压缩处理器、MMD 注入；Gateway 集成 /v2/offload/* 路由；

## [1.0.0-beta.1] - 2026-05-29

> **架构大版本**：从 OpenClaw 专属插件演进为**面向所有 Agent 的通用记忆服务**。核心能力通过独立 Gateway 服务 + 标准化 v2 HTTP API 对外暴露，配套官方 TypeScript / Python SDK，任何 Agent 框架均可接入完整的多层记忆管线。

### ⚠️ Breaking Changes

- **客户端/服务端架构拆分**：记忆引擎从 OpenClaw 嵌入式插件拆分为独立 Gateway 服务进程，部署方式与接入方式发生变化。
- **配置结构变更**：插件配置结构扁平化重构，原 `gateway` 字段迁移为 `server` 嵌套；Offload 路径从 `/api/v1` 变更为 `/offload/v1`。

### ✨ 新功能

#### 独立 Gateway 服务（v2 API）

记忆能力不再绑定 OpenClaw 宿主，以独立服务形式运行，通过 v2 HTTP API 为任意 Agent 提供记忆读写与管线管理：

- **完整 v2 API**：14 条标准路由覆盖记忆 CRUD、原子更新、场景索引、管线状态查询等全部操作。
- **管线状态查询（`/v2/pipeline/status`）**：实时获取 L1/L2/L3 各阶段运行状态与进度。
- **实例生命周期管理（`/v2/instance/destroy`）**：支持外部系统主动创建/销毁记忆实例。
- **可选 Bearer 鉴权 + CORS 白名单**：保护对外暴露的 API 安全。
- **请求体大小限制**：强制 1 MiB 上限，防止异常请求。

#### 官方 SDK

- **TypeScript SDK 1.0.0**（`@tencentdb-agent-memory/memory-sdk-ts`）：类型安全，覆盖全部 v2 API，npm 安装即用。
- **Python SDK**（`tencentdb-agent-memory-sdk-python`）：pip wheel 安装，同步/异步双模式，适配主流 Python Agent 框架。

#### 通用 Agent 框架适配

- **OpenClaw 插件适配**：通过 client plugin 接入 Gateway 服务，保持原有开发体验。
- **Hermes Agent 适配**：`memory_tencentdb_v2` adapter，支持 Hermes 框架多租户场景。
- **通用接入**：任何能发 HTTP 请求的 Agent（LangChain、AutoGPT、自研框架等）均可通过 SDK 或裸 API 接入。

#### 可观测性

- **OpenTelemetry 全链路 Trace**：支持 OTLP 协议上报，可对接 Jaeger / Grafana Tempo 等后端。
- **Langfuse 集成**：仅转发 LLM 相关 span，适合评估记忆提取质量。
- **管线评测指标**：L1 提取率、去重决策分布、各阶段 token 消耗、recall 延迟等。

#### 部署

- **Standalone Docker 镜像**：单容器即可运行完整记忆服务，无需外部依赖。
- **Hermes + Memory All-in-One 镜像**：Agent + 记忆服务一体化部署方案。
- **`memory-tencentdb-ctl` CLI**：运维命令行工具，支持配置管理、存储引擎切换等操作。
- **v2 安装器脚本**：`install-openclaw-plugin-v2.sh`，一键配置插件适配层。

### 📖 文档

- SDK 接入指南（TypeScript / Python 示例）。
- Gateway 独立部署文档。
- OpenClaw / Hermes 插件安装说明。
- Docker 部署与配置参考。

</details>

---

## [0.3.6] - 2026-05-27

### ✨ 新功能

- **Recall 上下文预算控制** ([#71](https://github.com/Tencent/TencentDB-Agent-Memory/pull/71) / [#70](https://github.com/Tencent/TencentDB-Agent-Memory/issues/70))：新增 `recall.maxCharsPerMemory` 与 `recall.maxTotalRecallChars` 两项配置，默认 `0` 不改变现有行为；设置为正整数后，会在 L1 召回完成、注入 `<relevant-memories>` 之前按分数顺序裁剪超长条目并丢弃溢出部分，避免长会话因记忆膨胀挤占上下文。已在 README、README_CN 与 `openclaw.plugin.json` 同步说明。
- **L1 / L2 / L3 提示词按用户输入语言自适应** ([#38](https://github.com/Tencent/TencentDB-Agent-Memory/issues/38))：`l1-extraction`、`l1-dedup`、`scene-extraction`、`persona-generation` 四个 prompt 中所有自由文本字段（`scene_name`、记忆 `content`、scene `.md` 标题/正文、`persona.md` 各章节）现在跟随用户消息的主导语言书写；JSON 字段名、枚举值、ISO 时间戳、`persona.md` 等结构化文件名继续保持英文作为稳定契约。无需配置 `locale`，任意语言（en / fr / ja / es / …）均可直接使用。
- **Embedding `sendDimensions` 可选关闭**：`OpenAIEmbeddingService` 默认仍会在请求体携带 `dimensions` 字段（兼容 OpenAI `text-embedding-3-*` Matryoshka 截断）；新增 `embedding.sendDimensions` 配置项，设置为 `false` 时省略该字段，可对接 BGE-M3 等不支持自定义维度的固定维度模型（原会被服务端 HTTP 400 拒绝 `does not support matryoshka representation`）。
- **Gateway 可选 Bearer 鉴权 + CORS 白名单**：新增 `server.apiKey` / `TDAI_GATEWAY_API_KEY` 配置项，设置后所有非 `/health` 路由需携带 `Authorization: Bearer <key>`（`crypto.timingSafeEqual` 防时序攻击）；新增 `server.corsOrigins` / `TDAI_CORS_ORIGINS` 配置项，显式指定允许的 CORS 来源列表（空列表 = 不发送 CORS 头，`"*"` = 保留旧版宽松行为）。启动时打印安全态势摘要，非回环地址 + 无 apiKey 时输出 WARN。两项均默认关闭，现有部署无需改动。Hermes Python 客户端同步支持 `MEMORY_TENCENTDB_GATEWAY_API_KEY` 环境变量自动附加 Bearer 头。
- **Offload `collect` 模式**：新增 `offload.mode: "collect"` 配置，仅执行数据采集（L0 捕获 + 向量写入）而不触发 L3 压缩，适用于纯数据积累阶段或调试场景。

### 🐛 修复

#### 数据安全 / 数据隔离
- **L2 LLM 提取失败导致 `scene_blocks/` 被清空 / 半写入** ([#88](https://github.com/Tencent/TencentDB-Agent-Memory/issues/88))：Phase 1 已对 `scene_blocks/` 做完整快照，但 LLM 抛错时 `catch` 直接 `return`，沙箱里的部分写入 / 删除不会回滚，后续 recall 因此看不到场景导航，降级为碎片召回。新增 `BackupManager.findLatestBackup` + `restoreLatestDirectory`，在 LLM 失败时自动从最新备份恢复；采用 fail-soft 设计：无备份时不动目标目录，恢复过程自身的错误也不会替换原始 LLM 错误。
- **Cleaner 安全加固**：`computeCutoffMsByLocalDay` 拒绝无效 cutoff（未来时间 / 距今不足 24h）；SQLite 与 TCVDB 在 `expired/total > 80%` 时阻止删除；`runOnce` 增加最小保留护栏（L0:50 / L1:20）并产出 `cleaner_summary` JSON 审计日志；新增 `__tests__/cleaner/verify-cleaner-safety.ts` E2E 校验。
- **场景文件名含空格导致 Persona Scene Navigation 引用失效**：LLM 在 L2 偶发用 `Daily Rhythm in Shanghai.md` 这类含空格的名字创建 scene block，导致 `persona.md` 中 `### Path: scene_blocks/<name>.md` 引用无法被下游 `\S+\.md` 风格解析器（health-checker 等）识别，soak 后健康检查必报 `Scene Navigation 存在但无场景引用`。修复：(1) `scene-extraction` prompt 增加"📛 文件命名规范（强制）"段，禁止空格 / 括号 / 引号等标点；(2) 新增 `core/scene/filename-normalizer.ts`，在 `SceneExtractor.extract` Phase 5b（cleanup 后、`syncSceneIndex` 前）自动归一化文件名（空格 → `-`、剥离危险标点、冲突时追加 `-2` 后缀），下游 PersonaGenerator / recall / profile-sync 自动使用干净名，无需改动。

#### OpenClaw 宿主兼容
- **`api.runtime.state` 在新版 OpenClaw 上为 `undefined` 导致注册时崩溃** ([#78](https://github.com/Tencent/TencentDB-Agent-Memory/issues/78) / [#85](https://github.com/Tencent/TencentDB-Agent-Memory/pull/85) / [#79](https://github.com/Tencent/TencentDB-Agent-Memory/pull/79))：为两处调用点加可选链 + fallback，优先调用宿主 `runtimeState.resolveStateDir()`，缺失时退到 `OPENCLAW_STATE_DIR` 环境变量，再退到 `~/.openclaw`。同时修复 cli-metadata 注册模式下 `runtime` 为空对象 `{}` 仍会触发 `TypeError` 的问题（在该模式下提前 return，只调用 `registerCli`）。
- **`contextEngine` slot ID 与插件名不一致**：`registerContextEngine` 的 ID 从 `openclaw-context-offload` 改为 `memory-tencentdb`，避免 `openclaw doctor --fix` 把 slot 重置；`setup-offload.sh` 中的 `CONTEXT_ENGINE_ID` 同步更新。
- **L1.5 settle 永不返回导致 L2 卡死**：在 L2 poll 中为 L1.5 settle 增加 60s 超时，当未配置 Context Engine slot 导致 `assemble` 永不被调用时，自动 force-settle 解锁 L2。
- **Standalone 文本任务仍暴露工具导致 DeepSeek 等后端 L1 抽取不稳定** ([#58](https://github.com/Tencent/TencentDB-Agent-Memory/issues/58) / [#59](https://github.com/Tencent/TencentDB-Agent-Memory/pull/59))：`enableTools=false` 时彻底不传工具列表（此前即便只读子集也会鼓励 OpenAI 兼容后端尝试 tool calling，DeepSeek 上尤其明显）。
- **L2 cold-start skip 被错误地更新 `l2LastRunTime`**：导致首次 skip 后必须等满 `l2MaxInterval` 才会真正运行 L2；现在仅在确实跑过的情况下更新时间戳。

#### Offload（Context Engine）稳定性
- **`sanitizeText` 误删 emoji / CJK Extension B / Math Bold 等非 BMP 字符** ([#30](https://github.com/Tencent/TencentDB-Agent-Memory/issues/30) / [#31](https://github.com/Tencent/TencentDB-Agent-Memory/pull/31))：`UNSAFE_CHAR_RE` 含 `[\uD800-\uDFFF]` 但缺 `u` flag，JS 按 UTF-16 code unit 处理时会把每个非 BMP 码点的两个 surrogate 各自 strip 掉。加上 `u` flag 后只匹配孤立（畸形）surrogate，emoji / 扩展 CJK / 数学加粗等正常恢复；新增 vitest 套件覆盖保留与剥离两类 case。
- **Emergency 截断在 `MIN_KEEP` 拒绝下死锁**：`EMERGENCY_MIN_MESSAGES_TO_KEEP` 由 4 降到 2；新增 `_emergencyTruncateOversized`，当 head/tail 删除均被阻塞时就地截断超大消息（保留 `tool_use` 块结构），最后兜底强制删除并配对清理 `toolResult`，在 LLM 可见的内容里加截断告示。
- **多轮 aggressive compression 累计耗时**：由 6 轮 `O(N × rounds)` 全量 tiktoken 改为单趟 `O(N × 1)` 直接计算到目标阈值的精确切点。615 条消息从 84s 降到 ~14s；tool 配对、user 消息保护、MMD 保留、stall 检测等安全机制全部保留。
- **FP-HEAD-DELETE 在多轮 FAST-SKIP 后误删新消息**：移除该路径，改用 `FP-BOUNDARY-DELETE`（基于上一轮 aggressive 边界的 O(1) 头部删除，index + fingerprint 双重验证、tool-pair 安全）+ `BOUNDARY-INCR-SKIP`（增量估算低于阈值时跳过 tiktoken）。重放场景下 `assemble` 38s → 122ms（310×）。
- **首次 assemble 慢**：为 `TAIL-ACCUMULATE` 增加 fast-token-estimate（基于字符，~51× 快于 tiktoken）前置短路，无边界且 fast estimate 明显高于阈值时跳过全量 tiktoken；同时为 `TAIL-ACCUMULATE` 增加向后 tool-pair 校正、user 消息保护、最少保留 10 条等安全检查。首次 assemble 29s → ~1.4s。
- **Token 计算精度**：`details` 字段加入 `INTERNAL_KEYS`（框架在送 LLM 前 strip 掉，不应计入 token）；新增 `_stripLargeFields()` 移除非内容大字段；就地截断后调用 `invalidateTokenCache(msg)` 修正 WeakMap 缓存陈旧问题；`bestTokens < 600` 时跳过截断防反向膨胀；stub 文本简化为纯英文避免触发模型内容过滤；`l3TiktokenEncoding` 默认从 `o200k_base` 改为 `cl100k_base`（匹配 DeepSeek / GLM / MiniMax 分词器）。
- **Offload 日志降级**：`AGGRESSIVE` / `EMERGENCY` 等多数日志降到 debug，仅当超过 10s 时才输出 `SLOW` warn；Opik tracer 初始化、`after_tool_call` 无 session 等"正常 fallback"场景日志同步降级，减少 `plugins list` 噪音。

#### Hermes / Docker 部署
- **`Dockerfile.hermes` 生成的 `config.yaml` 缺 `api_key`** ([#77](https://github.com/Tencent/TencentDB-Agent-Memory/issues/77) / [#81](https://github.com/Tencent/TencentDB-Agent-Memory/pull/81))：`provider: custom` 模式下 Hermes 从 `config.yaml.model.api_key` 读密钥，而原 CMD 脚本仅写入 `.env` 的 `OPENAI_API_KEY`，造成容器内首条对话即报 401。修复为同步写入 `model.api_key: "${MODEL_API_KEY}"`。
- **安装脚本多处问题** ([#18](https://github.com/Tencent/TencentDB-Agent-Memory/issues/18) / [#19](https://github.com/Tencent/TencentDB-Agent-Memory/issues/19) / [#20](https://github.com/Tencent/TencentDB-Agent-Memory/issues/20) / [#54](https://github.com/Tencent/TencentDB-Agent-Memory/pull/54) / [#55](https://github.com/Tencent/TencentDB-Agent-Memory/pull/55))：
  - 支持 `HERMES_AGENT_DIR` 环境变量覆盖，适配 FHS 布局（把 hermes-agent 装在 `/usr/local/lib/hermes-agent` 等）。
  - root 用户执行时不再 `su - root` 无限递归。
  - `MEMORY_TENCENTDB_GATEWAY_CMD` 在 systemd 环境下用 `command -v node` 解析的绝对路径，`npx tsx` 改为 `node --import tsx/esm`，避免 nvm PATH 不可见时找不到 node。

### ✨ 改进

- **L1.5 settle 60s 超时保护**（详见上）。
- **OpenAI-style standalone runner** 在 `enableTools=false` 时不再传任何工具（详见上）。
- **`l3TiktokenEncoding` 默认改为 `cl100k_base`**，匹配主流国产/开源模型分词器。
- **Docker 文档**：补充 `cd docker/opensource` 前置步骤；新增 question/consultation issue 模板。

### 🧪 测试 / 内部

- 新增 `src/utils/backup.test.ts`：`findLatestBackup` 4 个用例 + `restoreLatestDirectory` 5 个用例，真 fs 沙箱。
- 新增 `src/core/scene/scene-extractor.restore.integration.test.ts`：用真临时目录 + 真 BackupManager 验证 LLM 失败时 `scene_blocks/` 被恢复（含 step 日志，沙箱保留供人工 inspect）。
- 新增 `src/utils/openclaw-state-dir.test.ts` + `index.test.ts` cli-metadata 模式安全性用例。
- 新增 `__tests__/cleaner/verify-cleaner-safety.ts` E2E（SQLite + live VDB）。
- 新增 `src/offload/fast-token-estimate.ts` + `benchmark-token-estimate.ts` 基线脚本。

### ⚠️ 配置项变化（向后兼容）

| Key | 默认值 | 说明 |
|---|---|---|
| `recall.maxCharsPerMemory` | `0` | `0`/未设置 = 不裁剪 |
| `recall.maxTotalRecallChars` | `0` | `0`/未设置 = 不裁剪 |
| `embedding.sendDimensions` | `true` | `false` 时不在请求体携带 `dimensions`，适配 BGE-M3 等 |
| `l3TiktokenEncoding` | `cl100k_base`（原 `o200k_base`） | 仅在显式依赖 `o200k_base` 时需手动覆盖回去 |

---

## [0.3.5] - 2026-05-15

### 🐛 修复

- **兼容 OpenClaw v2026.5.7 zod v4 子路径**：显式声明 `zod@^4.4.3` 依赖，解决 `@ai-sdk/provider-utils@4.x` 需要 `zod/v4` 子路径导出但宿主环境可能 hoist zod@3.x 引发 `Cannot find module zod/v4` 的运行时错误。

### ✨ 改进

- **L1→L2 延迟从 90s 降至 10s**：`l2DelayAfterL1Seconds` 默认值 90→10，冷启动用户不再需要等待 ~90s 才能看到 L2 场景提取结果，体感更及时。

### 📖 文档

- README 新增 Docker Quick Start 章节，说明模型 URL/Name 环境变量配置方式。

---

## [0.3.4] - 2026-05-12

### 🐛 修复

- **兼容 OpenClaw v2026.4.7 以下版本 L1 抽取空输出**：旧宿主不支持 `systemPromptOverride`，通过 `extraSystemPrompt` 回退注入系统提示，确保 LLM 按数据提取助手身份工作。
- **TCVDB hybrid 召回冗余双重 HTTP 调用**：`auto-recall` 对 TCVDB 发两次相同的 `hybridSearch` 请求（且 keyword 路径将 FTS5 OR 表达式错误传入 BM25 编码器）。新增 `nativeHybridSearch` 短路，TCVDB 单次调用即可完成 dense + sparse + RRF，recall 耗时减半（~50-120ms）。
- **L2 parser 对齐 Go 后端**：增加 mermaid fallback，修复 `first{...last}` JSON 提取逻辑。

### ✨ 改进

- **VDB HTTP 请求级计时**：`tcvdb-client` 每次请求打一条 info 计时日志（`/document/hybridSearch 85ms`），retry/失败细节保持 debug 级别。
- **启动路径误导性日志降级为 DEBUG**：store manifest 不一致、sqlite schema migration、profile-sync MD5 mismatch 等正常场景不再打 warn/info，避免 AI 误判。
- **L1 提取调试日志**：新增 `[l1-debug]` 系列（RESOLVE / INVOKE / RESULT / EMPTY_DUMP / ENTRY / NO_JSON），方便定位 LLM 调用链问题。

### 🔧 兼容性适配

- **OC 2026.4.23 Zod schema 兼容 patch 脚本**（`scripts/bugfix-20260423/`）：一键修复 `allowConversationAccess` 被 `.strict()` 拒绝的问题，含轻量版脚本、全自动脚本、手动 SOP 文档。
- Offload 日志去掉 `Backend` 前缀，默认超时为 120s。

### 📦 新功能

- **Offload Local Mode**：支持本地模式运行 offload（不依赖远端后端）。
- **Docker 一体化镜像**（`Dockerfile.hermes`）：单容器捆绑 Hermes Agent + memory_tencentdb 插件 + TDAI Memory Gateway，统一 `MODEL_*` 环境变量驱动。

### ✅ 测试

- 修复 `fault-injection` FI-05 mock config 缺 `embedding` 字段
- 修复 `cli.test` dependencies 断言适配新增依赖
- 跳过 `patch-effectiveness` 已删除的 `install-plugin.sh` 测试

---

## [0.3.3] - 2026-05-08

### 🐛 修复

- **加固 hook-policy 版本决策逻辑**：仅当宿主版本为严格 `x.y.z` 语义化版本、且 `>= 2026.4.24` 时才自动写入 `hooks.allowConversationAccess`；无法解析（如 `unknown`、beta、snapshot 等非标准版本）时一律跳过，避免对旧版本或非预期版本误写配置导致启动失败。
- hook-policy 关键路径补充 debug 日志（原始版本串、解析后版本、最小要求版本、是否 patch 的决策），方便线上排查。

### ✅ 测试

- 新增 `src/utils/ensure-hook-policy.test.ts`，覆盖标准版本、预发布、`unknown`、边界值等决策 case。

## [0.3.2] - 2026-05-08

### 🐛 修复

- 兼容 OpenClaw v2026.4.23 前的版本，防止写入的 hook 配置导致无法启动
- 修改 allowConversationAccess 到 2026.4.24+ 添加。

## [0.3.1-beta.1] - 2026-05-07

### 🐛 修复

- **兼容 OpenClaw v2026.4.23+ hook 权限策略**：该版本引入 `allowConversationAccess` 安全门控（[openclaw#70786](https://github.com/openclaw/openclaw/pull/70786)），导致非 bundled 插件的 `agent_end` hook 被静默拦截，整个 capture pipeline 失效。新增 `ensurePluginHookPolicy()` 自动检测并补全配置，优先通过 SDK 触发 gateway 自动重启，fallback 手动写入配置文件。
- **兼容 OpenClaw 2026.5.3+ 安装校验**：新增 tsdown 构建配置生成 `dist/index.mjs`，满足新版安装时对编译产物的强制校验（不再允许纯 TypeScript 入口）。
- **声明 `activation.onStartup`**：确保 gateway 在启动时加载本插件。
- **声明 `contracts.tools`**：注册 `tdai_memory_search`、`tdai_conversation_search` 工具名，满足 tool registration contract 要求。

---

## [0.3.0] - 2026-05-06

### 🚀 新功能

**运维管理工具（CTL）**

- 新增 `memory-tencentdb-ctl` 命令行管理工具，支持 standalone 与 hermes 两种运行模式
- 新增 `install-memory-tencentdb` 一键安装脚本
- CTL 新增 `config vdb-off` 命令，支持将 Gateway 存储从 VDB 回退到 SQLite
- Gateway 安装脚本支持将环境变量写入 `~/.hermes/.env`（systemd 场景）

**Offload 增强**

- Offload 启动时自动应用 `after_tool_call` patch，patch 失败时自动禁用 offload
- 新增 `setup-offload.sh` 一键启用/禁用 offload 脚本，支持 `--backend-api-key` 参数
- L0 捕获过滤：排除 offload 注入的 MMD 上下文块，避免将压缩中间产物误存为记忆

**Gateway 自愈与稳定性**

- Hermes 插件新增 watchdog + lazy probe 机制，Gateway 异常时自动恢复
- Gateway YAML 配置解析支持任意深度嵌套

### ✨ 改进

- 数据目录与安装目录统一整合至 `~/.memory-tencentdb/`
- 引入 `$HERMES_HOME` 环境变量约定，移除硬编码 `~/.hermes` 路径
- CTL hermes 配置编辑改为缩进感知，保持原始文件格式
- 运维脚本保留在 tarball 中但不再注册为 bin 命令（减少全局命令污染）
- init/destroy 生命周期日志降级为 debug 级别
- patch 脚本兼容 pnpm 安装环境，使用 Node.js 动态解析 openclaw 安装路径

### 🐛 修复

**Core 稳定性**

- 修复 `ensureSchedulerStarted` 并发调用下的竞态问题
- 修复 `/session/end` 错误销毁全局 scheduler 的问题（改为按 session_key 作用域）
- 修复关闭 store 时未等待后台 fire-and-forget 任务完成的问题
- 修复 `disable_offload` 未正确删除 `slots.contextEngine` 配置的问题

**Offload**

- 修复 slot 占用检测逻辑：仅在 `ok=false`（slot 被占用）时拒绝，API 异常不再误判为冲突
- 修复 `registerContextEngine` 抛异常时未禁用 offload 的问题
- 修复 slot 被占用时未完全禁用所有 offload 功能的问题

**L3 压缩**

- 修复 aggressive/emergency 压缩在用户消息位于队首时卡死的问题
- 修复消息被大量 offload 后压缩停滞的问题

**迁移工具**

- 修复源数据目录或 SQLite 不存在时迁移脚本崩溃的问题（改为优雅跳过）
- 修复源数据为空时 config/manifest 未写入的问题

**脚本与运维**

- 修复 `set -e` 环境下 `((VAR++))` 在 VAR=0 时导致脚本退出的问题
- 修复 patch 脚本误报 FAILED 计数的问题（跳过无 after_tool_call 上下文的候选项）
- 修复 Hermes 退出时未终止 Gateway 子进程的问题

### ♻️ 重构

- 统一 patch 检测逻辑：始终委托给 patch 脚本并通过退出码判定结果

---

## [0.3.0-beta.1] - 2026-04-23

### 🚀 新功能

**短期记忆压缩（Context Offload）**

- 新增 Offload 模块，支持长对话场景下的上下文压缩与记忆卸载

**架构重构：Core + Gateway 多框架支持**

- 重构为 `TdaiCore` 宿主无关的核心层 + 适配器模式，解耦 OpenClaw 框架依赖
- 新增 `HostAdapter` / `LLMRunner` / `LLMRunnerFactory` 抽象接口，支持不同宿主的 LLM 调用
- 新增 Hermes Gateway 适配器（`memory_tencentdb` Hermes Plugin），支持通过 Hermes 框架独立运行
- `TdaiCore` 提供统一的 `handleBeforeRecall()` / `handleTurnCommitted()` / `searchMemories()` 等 API
- Gateway 零配置自动发现：Hermes 插件自动检测配置和数据目录
- 数据目录所有权从插件移至 Gateway 层管理

**Recall 注入优化（Cache 友好）**

- L1 召回记忆从 `appendSystemContext` 移到 `prependContext`（用户消息前缀），避免每轮系统提示词变化导致 prompt cache bust
- Persona / Scene Navigation / Tools Guide 保持在 `appendSystemContext`（稳定内容，连续多轮 cache 命中）
- 注册 `before_message_write` 钩子，在 user message 持久化到 JSONL 前 strip `<relevant-memories>` 标签，防止历史消息中累积旧的召回内容

**分场景 Embedding 超时**

- 新增 `embedding.recallTimeoutMs`（recall 路径）和 `embedding.captureTimeoutMs`（capture 路径）配置
- recall 超时时 hybrid 策略自动降级为纯关键词搜索；capture 超时时 L1 dedup 降级为 FTS
- 向前兼容：不配置时 fallback 到全局 `embedding.timeoutMs`

### ✨ 改进

- CleanContextRunner 通过 `systemPromptOverride` 替换 OpenClaw 默认系统提示词，每次 L1/L2/L3 调用节省 ~4500 input tokens
- L2（场景提取）和 L3（画像生成）prompt 拆分为 `systemPrompt` + `userPrompt`，角色划分更清晰
- Pipeline 默认参数调整：`l1IdleTimeoutSeconds` 60→600s，`l2MinIntervalSeconds` 300→900s，`l2MaxIntervalSeconds` 1800→3600s

### 🐛 修复

- 修复 `pullProfilesToLocal` 并发竞争导致 `ENOTEMPTY` 错误（乐观无锁修法：rename 竞争失败时静默使用对方结果）
- 修复 `originalUserMessageCount` 数据链路断裂导致 L0 recorder 无法定位被污染的 user message
- 修复 `RecallResult` 类型定义缺少 `prependContext` 字段（`types.ts` 与 `auto-recall.ts` 不一致）

---

## [0.2.2] - 2026-04-17

### 🐛 修复

- 修复因未声明 `undici` 依赖导致 TCVDB 客户端加载失败的问题（开发环境之前依赖 monorepo 根 `node_modules` 的传递解析）
- 将插件注册阶段的大量 INFO 日志降级为 DEBUG，避免 CLI 模式下输出过多无关日志

## [0.2.1] - 2026-04-16 (deprecated)

> NOTE: 此版本由于存在 undici 依赖导致插件启动失败的问题，已废弃
> 相关问题在 0.2.2 及以后版本中已修复

### 🚀 新功能

- TCVDB 新增 HTTPS 连接支持，可通过插件配置 `caPemPath` 或迁移脚本参数 `--tcvdb-ca-pem` 指定自定义 CA 证书 PEM 文件
- `read-local-memory` 脚本新增 L2 单文件查询，并将 L0 / L1 查询切换为直接从 `vectors.db` 读取，支持 SQL 层过滤、排序与分页

### ✨ 改进

- TCVDB 的 L0 / L1 向量索引默认调整为 `DISK_FLAT`，并在不支持该索引类型的实例上自动回退到 `HNSW`
- 默认服务端 embedding 模型调整为 `bge-large-zh`
- TCVDB 所有读接口统一启用 `readConsistency: "strongConsistency"`，消除 read-after-write 不一致
- 健康检测脚本 VDB 连接支持 HTTPS 自签证书

### 🐛 修复

- 修复 L3 persona sync 因未拉取远端 baseline 导致版本冲突跳过写入的问题
- 修复 `memories_since_last_persona` 被 L0 和 L1 双重计数导致 persona 触发阈值膨胀的问题
- 移除 `CheckpointManager` 中已被 `captureAtomically()` 替代的废弃方法

---

## [0.2.0] - 2026-04-15

### 🚀 新功能

**腾讯云向量数据库（TCVDB）存储后端**

- 新增腾讯云向量数据库存储后端，支持向量 + BM25 混合召回
- 支持 SQLite 与 TCVDB 之间的索引结构同步
- L2 场景 / L3 画像支持在本地缓存与向量数据库之间双向同步
- 插件配置（manifest）暴露 `storeBackend`、`tcvdb`、`bm25`、`embedding.timeoutMs` 等配置项

**本地 BM25 关键字检索**

- 使用本地 tcvdb-text 编码器替代原有的 BM25 HTTP sidecar 服务，消除外部依赖

**Seed 数据导入工具**

- 新增 CLI `seed` 命令，支持从外部数据批量导入记忆
- 提取共享的 pipeline-factory，供 seed 和正常运行时复用
- 支持 ISO 8601 时间戳格式（移除 JSONL 支持）

**数据迁移与运维工具**

- 新增 SQLite → 腾讯云向量数据库迁移脚本，支持 `--help` / `-h` 展示完整参数说明和使用示例
- 新增 VDB 数据导出脚本（含预编译 JS 和 CLI 启动器）
- 新增本地 Memory 数据查询脚本
- 注册全部 CLI bin 入口：`migrate-sqlite-to-tcvdb`、`export-tencent-vdb`、`read-local-memory`

**记忆搜索工具调用限制**

- `tdai_memory_search` + `tdai_conversation_search` 增加每轮合计最多 3 次的调用次数限制，通过 tool description 和召回引导提示词约束模型行为，防止陷入无效重复搜索

### 🐛 修复

- 修复 L2 场景合并（MERGE）无法删除旧文件的问题：OpenClaw 4.1+ 的 write 工具拒绝空白内容，改用 `[DELETED]` 标记实现软删除，SceneExtractor cleanup 阶段同步识别并清理
- 修复 L2 抽取产生孤立 BATCH/ARCHIVE 文件的问题，统一 maxScenes 上限为 15
- 修复 L3 启动时重复拉取 profile 的问题
- 过滤 skill wrapper 噪声标记（`¥¥[...]¥¥`）
- 处理 `createCollection` 并发竞态（错误码 15202）

### ♻️ 重构

- Pipeline checkpoint 游标语义从 timestamp 改为 update_at
- Runner 改用 `api.runtime.agent.runEmbeddedPiAgent`，避免跨环境导入失败
- 统一脚本构建流程：新增 `build:scripts` 一键编译命令，`prepack` 钩子确保 `npm pack` 前自动编译全部脚本产物

### 📚 文档

- 新增 AI Agent 长期记忆插件设计与实现技术文档
- 新增项目指南、研发系统分层架构文档
- 新增 VDB 存储设计文档及迁移指南

---

<details>
<summary>预发布版本</summary>

## [0.2.0-beta.1] - 2026-04-14

*此版本的内容已合并至 [0.2.0] 正式版。*

</details>

## [0.1.4] - 2026-04-10

### 🚀 Features

- *(auto-recall)* Add recall hint text before memories

## [0.1.3] - 2026-04-09

### 🚀 功能

- *(memory-tdai)* 用 reporter 抽象替换 emitMetric
- *(L3)* L3 使用读写工具，防止模型输出 CoT
- *(memory)* 添加 embedding 截断、召回超时，以及从 L0 捕获中剔除代码块
- *(config)* Embedding 超时支持配置
- *(report)* 在 schema 中暴露 report 配置项，默认值改为 false

### 🐛 修复

- *(capture)* 跳过心跳/定时任务/自动化/调度类消息
- *(recall)* 召回完成时清除超时定时器，避免误报超时警告

### 💼 Other

- 重命名包名为 memory-tencentdb
- *(deps)* 将 node-llama-cpp 改为可选依赖

### ⚡ 性能

- *(auto-capture)* 将 L0 向量嵌入移至后台以降低延迟

### 📚 文档

- 添加 allowPromptInjection 配置警告说明

## [0.1.2] — 2026-03-26

### 更新内容

1. 优化对话捕获与记忆抽取过滤机制

## [0.1.1] — 2026-03-25

### 更新内容

1. 兼容 openclaw 2026.3.23 更新

## [0.1.0] — 2026-03-25

> 首个正式发布版本。本地优先的四层记忆系统（L0→L1→L2→L3），基于 SQLite + LLM 实现对话捕获、记忆提取、场景归纳与用户画像。

### 更新内容

1. 关键字检索增加 FTS5 全文索引，采用 jieba 分词
2. 未配置远程 embedding 服务时，默认不开启 embedding 能力（不自动使用本地 embedding，且封禁主动使用本地 embedding 的配置入口）
3. 优化 L2、L3 生成 prompt 以控制生成内容大小（减少 token 开销）
4. Pipeline 调度器优化文件锁用法
5. 避免全量读取 L0、L1 数据
