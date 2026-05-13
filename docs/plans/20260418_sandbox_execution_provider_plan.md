# Plan: Advanced Sandbox Execution Provider (Docker / Firecracker / gVisor)

**Date**: 2026-04-18  
**Role**: Plan Master (atomic breakdown)  
**Constitution**: `.cursorrules` Step 3 ledger/radar sniff completed; sync attempted from `Official_Hermes_Mirror` (merge succeeded; local default branch name may differ from `main`).

---

## Phase 1 — Digest (recon summary)

### Step 3 查账结论（`E:\MyPROJECT\NousPR\Master_Ledger.md` + `GitHub_Radar.md`）

| 检查项 | 结论 |
|--------|------|
| 账本索引中是否存在 **WIP / Open** 与「Sandbox Execution Provider / Firecracker / gVisor / 统一 skill 沙箱后端」**同名占坑** | **未发现**（索引未出现上述关键词专条） |
| 蓝海判定 | **可规划** — 方向与现有 MCP / Telegram / atomic-io 等 OPEN 项无明显标题重合 |
| **防撞避让（强）** | 账本 **#030**（OPEN）涉及 `container_runtime.py`、`tools/skill_manager_tool.py`、原子写与跨平台容器检测。**本特性若触碰 skill 落盘路径或复用 `container_runtime`，必须与 #030 / PR #11746 协调或等合并后再改同一文件**，避免并行冲突 |
| 雷达风向 | 近期 MERGED/OPEN 偏网关、MCP、代理、文件安全；**隔离运行时 / 多租户执行**仍为可叙事蓝海 |

### 工位技术事实（当前 `Alpha_03` 拉取后）

- **不存在** `src/sandbox/`；Python 包以仓库根目录的 `agent/`、`tools/`、`gateway/` 为主（`pyproject.toml` 的 `setuptools.packages.find` 当前 **未** 包含 `src`）。
- **已有** 终端执行栈：`tools/environments/base.py` 中 `BaseEnvironment`（`_run_bash` / `execute` / `cleanup`）+ `docker.py`、`modal.py`、`daytona.py`、`ssh.py`、`local.py` 等。
- **Skill / PTC 相关**：`tools/code_execution_tool.py` 本地 UDS + 远程 file-RPC；与 `terminal_tool` 选型的后端强耦合。
- **目标命名**：若坚持目录名 `src/sandbox/`，需在 `pyproject.toml` 中为 `setuptools.packages.find` 增加 `where`（例如 `[".", "src"]`）并 `include` `sandbox` / `sandbox.*`，否则包不可安装。

### 挂「免战牌」（执行前人工动作）

在**开始改业务代码前**，由账号持有者在 `Master_Ledger.md` 追加一条 **FEAT** 占位（或取得维护者确认），声明占用：`src/sandbox/`（或最终选定包路径）、`SandboxProvider` 接口、与 `hermes_cli/config.py` 的 `sandbox:` 配置块 — **避免与其他 Alpha 并行撞车**。规划阶段未自动改全局账本。

---

## Phase 2 — Atomic breakdown (file-level)

### [Step 1] ADR / 接口契约（冻结名词与边界）

- **File**: `docs/plans/20260418_sandbox_execution_provider_adr.md`（可选；若团队禁止额外 docs 可改为 PR 描述内嵌「Design」节）
- **Action**: Create
- **Details**: 写明：`SandboxProvider` 与现有 `BaseEnvironment` 的关系（组合优于继承 / 适配器二选一）；同步 vs 异步 API（建议 `asyncio` 子进程与线程池桥接）；**不负责** LLM 网关审批（仅消费 `config.sandbox.*`）。
- **Verification**: 评审可通过「接口清单 + 序列图」自检；无代码可跑 `pytest`。

### [Step 2] 包布局与构建注册

- **File**: `pyproject.toml`
- **Action**: Modify
- **Details**: 若采用 `src/sandbox/`：扩展 `[tool.setuptools.packages.find]` — `where = [".", "src"]`，`include` 增加 `sandbox`, `sandbox.*`；确认不与现有 `tools.*` 冲突。
- **Verification**: `python -c "import sandbox; import sandbox.providers"` 在 venv 内成功（`pip install -e .` 后）。

### [Step 3] 核心抽象与类型

- **File**: `src/sandbox/__init__.py`  
- **File**: `src/sandbox/types.py`（或拆分为多文件以遵守单文件行数纪律）
- **Action**: Create
- **Details**: 定义 `IsolationProfile`（`network_policy`, `fs_mounts`, `cpu_quota`, `mem_limit`, `cap_drop`, `seccomp_profile_ref` 等字段，Pydantic/dataclass）；`SandboxExecResult`（stdout, stderr, exit_code, artifacts）；`SandboxProvider` 抽象基类：`async def exec_cmd(...)`, `async def run_skill(...)`（签名与现有 skill 入口对齐占位）、`async def snapshot_fs()` / `diff_since()` / `rollback()` 钩子占位（可与 #030 原子写哲学对齐但**不修改** `atomic_io` 除非合并后）。
- **Verification**: `python -m pytest tests/sandbox/test_types.py -q`（新建最小单测：构造 `IsolationProfile` 默认值）。

### [Step 4] Provider registry + null/local provider

- **File**: `src/sandbox/registry.py`  
- **File**: `src/sandbox/providers/__init__.py`  
- **File**: `src/sandbox/providers/local.py`（no-op / pass-through 用于单测与回滚）
- **Action**: Create
- **Details**: `get_provider(name: str, config: dict) -> SandboxProvider`；从配置解析 `type`；默认 `local`。
- **Verification**: `pytest tests/sandbox/test_registry.py -q`。

### [Step 5] Docker provider（MVP）

- **File**: `src/sandbox/providers/docker.py`
- **Action**: Create
- **Details**: `asyncio.create_subprocess_exec` 调用 `docker run --rm`（或 `docker exec` 若长期会话）；映射 `IsolationProfile` → `--network`, `--cpus`, `--memory`, `--cap-drop`, `--security-opt seccomp=...`（**Linux-only**；Windows 分支显式 `NotImplementedError` 或降级文档）；镜像名来自 `config.sandbox.image`（默认占位 `hermes-sandbox` 仅文档，不硬编码敏感路径）。
- **Verification**: `pytest tests/sandbox/test_docker_provider.py` 使用 `pytest.importorskip` + mock subprocess（**不**强依赖本机 Docker，符合 CI dry-run 策略）。

### [Step 6] Firecracker provider（骨架 + API client 边界）

- **File**: `src/sandbox/providers/firecracker.py`
- **Action**: Create
- **Details**: 定义 `FirecrackerClient` Protocol（`start_vm`, `exec`, `stop`）；默认实现为 **stub**（连接配置来自 `config.sandbox.firecracker.*`）；文档说明需外部 socket/API。
- **Verification**: 单测 mock client：`test_firecracker_stub_lifecycle.py`。

### [Step 7] gVisor / runsc provider（可选运行时）

- **File**: `src/sandbox/providers/gvisor.py`
- **Action**: Create
- **Details**: 将 `docker` + `--runtime=runsc` 或独立 `runsc` 封装为单独 provider class，共享 Docker 参数构建器避免重复。
- **Verification**: 单元测试参数拼接快照（golden strings），无真实 runsc。

### [Step 8] 配置模型

- **File**: `hermes_cli/config.py`
- **Action**: Modify
- **Details**: `DEFAULT_CONFIG` 增加 `sandbox:` 块：`type`, `image`, `firecracker_socket`, `profiles.default`, `post_exec_diff_enabled` 等；`_config_version` 递增以触发迁移。
- **Verification**: `pytest tests/hermes_cli/test_config.py`（或现有 config 测试文件）新增键存在性与默认值断言。

### [Step 9] Selector 注入 — terminal 路径

- **File**: `tools/terminal_tool.py`（或环境工厂所在模块，需读现有 `TERMINAL_ENV` 分支）
- **Action**: Modify
- **Details**: 在创建 `*Environment` 之前读取 `sandbox.type`；当 `sandbox.type == docker` 且与 `TERMINAL_ENV=docker` 并存时，定义优先级（建议：**terminal 仍选 docker env，sandbox provider 只管 skill/code-exec 子路径**，避免双容器逻辑冲突 — 在 ADR 写死）。
- **Verification**: `pytest tests/tools/test_terminal_sandbox_selector.py`（新建，mock 配置）。

### [Step 10] Selector 注入 — code execution / skill 路径

- **File**: `tools/code_execution_tool.py`
- **Action**: Modify
- **Details**: 子进程 spawn 前选择 `SandboxProvider.exec_cmd` 承载用户脚本（若 provider 非 local）；保持 UDS 仅在「宿主 local + POSIX」启用；远程路径仍走 file-RPC，但 RPC **执行端**改为在 provider 隔离环境内轮询。
- **Verification**: `pytest tests/tools/test_code_execution_sandbox_provider.py`（mock provider，断言调用顺序）。

### [Step 11] Post-exec diff & rollback hook

- **File**: `src/sandbox/hooks.py`
- **Action**: Create
- **Details**: `post_exec_diff(before: Snapshot, after: Snapshot) -> DiffReport`；`rollback(snapshot_id)` 调用 provider；与 self-evolution 循环的挂载点以**回调注册表**形式暴露（不在此 PR 实现完整 diff 算法，可先 `filecmp` / `git diff --no-index` 占位并文档限制）。
- **Verification**: 临时目录集成测试 `tests/sandbox/test_hooks_diff.py`。

### [Step 12] seccomp + capability（硬核、跨平台声明）

- **File**: `src/sandbox/linux_hardening.py`
- **Action**: Create
- **Details**: 集中 Linux 分支：`prctl` / `seccomp` 加载 JSON 的 **可选** C 扩展或 `subprocess` 预置 `docker --security-opt`（优先后者以减少 native 编译）；非 Linux **不得**静默伪装成功 — 返回明确错误 JSON。
- **Verification**: 静态单测：非 Linux 平台跳过；Linux CI 可选 integration marker。

### [Step 13] 文档与网关叙事（英文）

- **File**: `AGENTS.md` 或 `website/docs/...`（仅当仓库惯例要求同步用户文档）
- **Action**: Modify（最小增量）
- **Details**: 英文说明 `sandbox:` 配置、provider 类型、与 gateway 审批流的关系（「gateway 仅审批，provider 执行」）。
- **Verification**: docs 构建若有 `docs-site-checks` workflow，本地可选 `npm`；否则人工链接检查。

### [Step 14] 全量回归（上游 CI 策略）

- **File**: N/A
- **Action**: Run
- **Details**: `python -m pytest tests/sandbox/ tests/tools/test_code_execution_sandbox_provider.py -q`；全量 `pytest` 视时间执行（`.cursorrules` 允许依赖 CI）。
- **Verification**: 0 failures。

---

## Phase 3 — Risk register

| 风险 | 缓解 |
|------|------|
| 与 #030 同文件冲突 | 首版避免修改 `skill_manager_tool.py` / `container_runtime.py`；仅新增包 + 注入点 |
| `src/` 与根包混用导致 import 混乱 | 单测强制 `import sandbox` 从已安装包入口 |
| Firecracker/gVisor 运维复杂 | MVP 以 Docker + runsc 参数为主；FC 为 stub + 文档 |
| Windows 宿主 | seccomp/capabilities 仅 Linux；provider 返回结构化 `unsupported_platform` |

---

## Phase 4 — Plan Master 确认提示

以上是精确到文件级别的原子化任务拆解。是否合理？如果无误，请回复「按计划执行」，实施代理将按步骤逐一击破并验证。
