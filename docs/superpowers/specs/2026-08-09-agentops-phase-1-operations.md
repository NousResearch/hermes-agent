# AgentOps Phase 1 运行、迁移与回滚设计

| 字段 | 内容 |
|---|---|
| 范围 | Phase 1 control-plane foundation |
| 状态 | Phase 1 实现已在隔离 worktree 完成，等待 G1 审阅；未安装服务、未执行生产迁移、未修改任何现有运行面 |
| 默认状态目录 | `~/.hermes/agentops/`，仅在 Owner 显式启动 daemon 后创建 |
| 默认权限 | `observe_only` |

## 1. Phase 1 可写边界

在本阶段，控制面只可在显式配置的专属 AgentOps state directory 内创建下列文件：`state.db`、SQLite WAL/SHM、`event-spool/`、`event-spool/quarantine/`、`backups/`、lock 和 UDS socket。该目录必须 canonicalize 后仍位于自身根中、由当前用户拥有、标记为 `.agentops-state`，且不是 symlink、Git worktree、Hermes 根状态目录或既有非-AgentOps 目录。`state.db`、spool 与 socket 必须分别是该根中的受控布局；否则 daemon 不启动 UDS，也不会打开 SQLite/WAL。

它不得写入 Hermes 的 session `state.db`、Cron jobs、配置、日志、Gateway、LaunchAgent、代码仓库或业务数据。配置/路径安全失败是 **fail closed**：不会以“degraded UDS”方式接触外部 socket 或不受控数据库。

插件 import 和 `agentops doctor` 均不得创建状态目录、数据库、socket 或后台线程。只有人工显式运行 `hermes agentops daemon --config <path>`（并在插件已被手动启用）才允许创建 AgentOps 自身状态。

## 2. 数据库迁移与备份

1. Store 以 `schema_migrations(singleton=1, version)` 的单调版本确认数据库版本；缺失、重复、未来版本或迁移缺口都会拒绝。
2. 新数据库从空状态创建 schema v1，并启用 WAL、foreign keys 和 `busy_timeout`。已有数据库先用只读连接完成 `integrity_check`、schema/version 和 audit-chain 预检，成功后才可切换 WAL。
3. 审计 head sequence/hash 作为 metadata 与 audit entry 在同一 SQLite transaction 中更新；检验首序号、连续性、行数、尾 hash 与 metadata 必须全部一致。
4. 备份仅可写入受控 `backups/`。恢复先在受控临时副本上做只读完整性/schema/version/audit 预检；成功后才原子替换。替换前会保留 pre-restore snapshot，重开失败必须以该快照回滚。
5. 发现未知的未来 schema、损坏数据库、审计异常或迁移异常时，daemon 不尝试修复 Target；它以 `observe_only` degraded health 记录安全原因，或在路径安全失败时不启动。
6. Phase 1 不承诺 downgrade migration。恢复动作只可针对配置中的 AgentOps SQLite 路径，不能引用 Hermes 现有 `state.db`。

## 3. Event spool 恢复

Event 先由 Producer 写入 AgentOps 自身 spool；文件名为 event ID，使用同目录临时文件和原子 replace。daemon 启动时按稳定顺序重放：

1. 用 schema-v1 与 secret gate 验证事件。
2. 通过 `event_id` 向 SQLite 幂等 append。
3. 成功或重复后删除 spool 文件。
4. 未知 schema、损坏 JSON、非法 UTF-8 或无效事件进入 metadata-only quarantine；一律只保存内容 hash、大小、理由和 redacted 标志，绝不保留原始字节。quarantine 使用唯一临时文件，启动时不读取地删除 orphan temp 并 fsync 父目录。

spool 只提供本地崩溃恢复，不能触发 Target 行为。spool 与 quarantine 分别受容量预算限制；超限时拒绝新事件或显式记录脱敏 drop，并将控制面标为 degraded/observe-only，不会静默转为写修复。任何 quarantine/orphan 清理失败都会被计入 replay `failed`，令 daemon health `ready=false`；若原始不可信文件无法脱敏或删除，绝不能报告健康。

## 4. daemon 启动、停止和未来 launchd

Phase 1 仅允许测试或人工前台 daemon。没有 plist、没有 `launchctl bootstrap`、没有自动启动。daemon 必须持有独立进程锁；锁以 `O_NOFOLLOW` 打开、校验当前用户和单一 hard link。已有 socket 会先探测 health，存活实例拒绝第二实例，只有当前用户、受控目录且无监听的 stale UDS 才可由持锁实例清理。目录必须回读为 `0700`、socket 必须回读为 `0600`，任一 chmod/owner/symlink 检查失败均不启动 API。

未来（不在本阶段）的 launchd 设计为：

- Label：`ai.hermes.agentops-control`；
- 独立固定 Python/venv 和已冻结的 release 目录；
- `KeepAlive` 只恢复 AgentOps 进程，不赋予重启任何 Target 的权限；
- ProgramArguments 只允许 `hermes agentops daemon --config <managed-config>`；
- 启动前验证 `0700` state directory、`0600` UDS、配置哈希和唯一进程/控制器锁；
- 安装、升级、卸载与现有 `ai.hermes.gateway*` 和 `com.molly.hermes-ai-native.*` 服务严格分离。

在 G4 前，未来 service 即使被安装也只能是 `observe_only`；它不能替换当前 `hermes_gateway_watchdog.py` 或其它 watchdog 的写职责。

## 5. 卸载与应急回滚

**正常卸载：** 先停止 AgentOps daemon；确认 UDS 不再监听；保留 `state.db`、spool、备份与审计导出；仅在 Owner 明确选择后移除 AgentOps 自身程序或 state directory。不会停止任何 Hermes Gateway 或删除任何既有服务。

**Phase 1 回滚：** 停止测试 daemon，删除其临时 socket，恢复该测试/AgentOps Store 的最近验证备份；如果没有 AgentOps 服务，则无需触碰 launchd。Gateway 继续由已有 launchd/watchdog 管理。

**安全事件：** 如果发现审计链异常、Secret 持久化、未知控制器或 UDS 权限不符合预期，停止 daemon（或保持无 store health）、保留脱敏证据，禁止进一步执行；本阶段不存在需要自动回滚的 Target 写动作。
