# Hermes live runtime recovery audit — 2026-06-08

> 目的：把 QQ/gateway 救急 live 分支之外的已做 runtime 工作重新盘清楚，按小批次恢复，避免上下文丢失后误合、漏合或重复造轮子。

## 当前边界

- 当前 live repo：`/workspace/hermes-agent-runtime-codex`
- 当前 live branch：`runtime/live-f7877a505-20260608T063124Z`
- 当前 live HEAD：`72f5463ebc84bb29201e40023d8a87542688e232`（`72f5463eb feat(image-gen): restore custom Responses providers`）
- 当前相对 upstream：`0 behind / 10 ahead`（upstream：`fork/runtime/live-f7877a505-20260608T063124Z`，upstream HEAD：`a0a8171b6`）
- 当前工作区：tracked clean；仅保留 unrelated untracked `docs/codex-workflow-local-capability-and-external-recommendations-2026-06-08.md`。
- 本文档状态：第 1 批到第 6C 批均已恢复到 live 并创建本地 checkpoint；未 push，未重启 gateway/WebUI，未做运行态验证。

## 恢复总览

- 已完成本地 checkpoint：10 个。
- 已恢复范围：session_search scope、compression/gateway recovery、process/Codex output governance、terminal raw Codex policy、guarded Codex workflow tools、strong raw Codex block、browser runtime repair script、custom Codex proxy cache compat、custom Responses image providers。
- 仍需单独确认：push、gateway/WebUI restart、运行态验证（cache hit / image generation / browser repair check）。

## 恢复原则

1. **一批一批来**：每批只恢复一个小功能线，避免上下文变长后丢状态。
2. **先隔离后 live**：先在隔离 worktree 做恢复/测试，不直接改 live 运行分支。
3. **不全量 merge**：很多候选分支与 live 的 merge-base 很旧，直接 merge/cherry-pick 大串会带入几千文件差异。
4. **每批固定四件事**：候选 commit → 最小文件范围 → focused tests → 是否需要 gateway restart。
5. **Codex 输出只当候选**：Codex/分支说明不作为完成证据；以 `git diff`、文件内容、测试输出为准。
6. **重启单独审批**：代码恢复、测试通过不等于自动重启 gateway/WebUI。
7. **每批结束更新本文档**：记录已恢复/未恢复/跳过原因，作为下一批上下文锚点。

## 执行模式建议

采用 **半自动小批次推进**，不是全自动一锅端，也不是每个微步骤都等确认。

推荐授权边界：

1. **允许我自动做**：
   - 读取本文档和相关代码。
   - 创建/使用隔离 worktree。
   - 对当前小批次做最小移植。
   - 跑 focused tests / `py_compile` / `git diff --check`。
   - 更新本文档记录证据。
2. **必须停下来问用户确认**：
   - 把隔离 worktree 的改动落到 live 分支。
   - commit / push / PR / merge。
   - gateway/WebUI restart 或任何生产运行态变更。
   - reset / restore / stash / clean / 删除未知文件。
   - 一批范围变大、冲突严重、测试失败超过 2 次。
3. **每批报告格式固定**：
   - 本批目标。
   - touched files。
   - 测试命令和结果。
   - 是否已落 live：是/否。
   - 是否需要 restart：是/否。
   - 下一批建议。

默认执行节奏：

```text
用户授权“按文档自动推进”
→ 我执行一个小批次到隔离验证完成
→ 报告证据
→ 等用户确认是否落 live / 继续下一批
```

## 已证实不在当前 live 的候选功能线

> `in_live=no` 来自 `git merge-base --is-ancestor <commit> HEAD` 返回非 0。

| 优先级 | 功能线 | 候选 commit / branch | 主要文件 | 当前判断 |
|---|---|---|---|---|
| P0 | session_search 当前 QQ/chat scope handoff | `4f521e6cd` `fix: scope session search handoff recall` | `agent/tool_executor.py`, `gateway/session.py`, `hermes_state.py`, `run_agent.py`, `tools/session_search_tool.py`, tests | 需要恢复；直接影响 `/new` 后找回上下文是否乱搜全局 |
| P0 | compression provider-error retry | `430944988` `fix(compression): compress large provider errors before retry` | `agent/conversation_loop.py`, `agent/conversation_compression.py`, `agent/tool_executor.py`, `hermes_cli/config.py`, `hermes_state.py`, `run_agent.py`, `tools/session_search_tool.py`, tests | 需要恢复或复核；影响大请求/EOF/520/524 后是否先压缩再重试 |
| P0 | compression recovery halt | `09c894891` `fix(gateway): halt after compression recovery` | `agent/conversation_loop.py`, tests | 需要恢复或复核；防止压缩恢复后继续重跑重任务 |
| P0 | compression wall-clock cap | `502fb91caf` `fix: cap compression wall-clock latency` | `agent/context_compressor.py`, `agent/agent_init.py`, `hermes_cli/config.py`, tests | 需要恢复或复核；影响压缩卡 300s+ 的控制 |
| P0 | compression session split persistence | `92d1b7c43` `fix: preserve compression session splits on gateway errors` | `gateway/run.py`, tests | 需要恢复或复核；影响压缩后 session_id 切换和 gateway 持久化 |
| P1 | process long-output / Codex output summary | `fc5613430`, `1982dd9a1` | `tools/process_registry.py`, `gateway/run.py`, `agent/transports/codex_event_projector.py`, tests | 需要恢复；影响长输出/差异洪水进入上下文 |
| P1 | process wait timeout metadata / kill trust | `eeaf970aa` | `tools/process_registry.py`, tests | 需要恢复；影响 `timeout_kind=wait_window_expired`, `process_still_running`, `trusted_completion` |
| P1 | terminal raw Codex launch policy | `88cd13013` `fix(terminal): guard codex exec launch policy` | `tools/terminal_tool.py`, `tests/tools/test_terminal_tool.py` | 需要恢复；影响裸 `codex-yuna exec` / `codex exec` 是否被挡 |
| P2 | guarded Codex workflow orchestration | `f241c4352` / `77b7541dd` | `tools/codex_staged_implement_tool.py`, `tools/codex_workflow_run_tool.py`, `scripts/runtime/codex_*`, `toolsets.py`, tests | 需要恢复，但在 P1 后做；涉及新工具 schema |
| P3 | browser runtime repair script | `a8863e37f` | `scripts/runtime/repair_webui_browser_runtime.py`, tests | 可恢复；非当前稳定性第一优先 |
| P3 | custom Codex proxy cache compat | `38fc8b285` | `agent/chat_completion_helpers.py`, `agent/transports/codex.py`, `hermes_cli/runtime_provider.py`, `hermes_state.py`, tests | 可恢复；影响 custom `/codex` proxy cache |
| P3 | custom Responses image providers | `1110bc4e8` | `tools/image_generation_tool.py`, tests | 可恢复；影响 custom Responses 图片 route |
| P4 | skill provenance stats | `dc80f48f3` | skills usage/provenance相关 | 暂缓；不是当前救急恢复核心 |

## 小批次恢复计划

### 第 0 批：恢复清单锚点（当前批）

目标：只写本文档，固定后续恢复顺序和边界。

验收：

- `docs/runtime-recovery-audit-2026-06-08.md` 存在。
- `git status --short --branch --untracked-files=all` 只出现本文档新增/修改。
- 没有代码合入、没有测试误报、没有 gateway/WebUI restart。

### 第 1 批：session_search scope handoff

目标：优先修复“/new 后找回上下文是否乱搜”的风险。

候选：`4f521e6cd fix: scope session search handoff recall`

建议步骤：

1. 新建隔离 worktree，基于当前 live HEAD。
2. 只移植与 session_search scope 相关文件，不带无关大分支内容。
3. 跑 focused tests：
   - `python -m pytest tests/tools/test_session_search.py -q -o addopts=''`
   - `python -m pytest tests/test_hermes_state.py -q -o addopts=''`
   - 相关 gateway/session focused tests（按实际 diff 决定）。
4. Hermes 检查 diff，确认不会改变普通 CLI 全局搜索默认行为。
5. 通过后再询问是否落到 live。

### 第 2 批：compression / gateway recovery 基础

目标：恢复大上下文、压缩、gateway session split 的稳定性。

候选：`430944988`, `09c894891`, `502fb91caf`, `92d1b7c43`

建议分成 2A/2B，避免一次太大：

- 2A：provider-error compression retry + recovery halt。
- 2B：wall-clock cap + session split persistence。

建议测试：

- `python -m pytest tests/run_agent/test_413_compression.py -q -o addopts=''`
- `python -m pytest tests/agent/test_context_compressor.py -q -o addopts=''`
- `python -m pytest tests/test_compression_session_split_persistence.py -q -o addopts=''`（若存在/恢复）
- 相关 `gateway/run.py` focused tests。

### 第 3 批：process / Codex output governance

目标：恢复长输出 containment、wait timeout metadata、kill/trusted_completion 语义。

候选：`fc5613430`, `eeaf970aa`, `1982dd9a1`

建议测试：

- `python -m pytest tests/tools/test_process_registry.py -q -o addopts=''`
- `python -m pytest tests/gateway/test_background_process_notifications.py -q -o addopts=''`
- `python -m pytest tests/agent/transports/test_codex_event_projector.py -q -o addopts=''`

验收重点：

- `process(wait)` 超时是 wait-window，不是失败。
- kill/termination 后即使 exit 0，也必须 `trusted_completion=false`。
- Codex/source/diff flood 只进摘要，不污染主上下文。

### 第 4 批：terminal raw Codex policy

目标：挡住裸 `codex-yuna exec` / `codex exec` 实现路径，引导到 guarded workflow。

候选：`88cd13013`

建议测试：

- `python -m pytest tests/tools/test_terminal_tool.py -q -o addopts=''`
- 活体 entry probe 只在用户授权后做；避免真的启动写入型 Codex。

验收重点：

- 不误伤 `echo codex-yuna` / `python -c 'print("codex-yuna")'`。
- raw implementation 被挡。
- 低风险 `--version` / help 类 probe 不乱挡。

### 第 5 批：guarded Codex workflow tools

目标：恢复 `codex_workflow_run` / `codex_staged_implement` / guard scripts / review packet。

候选：`f241c4352` 或 `77b7541dd`，实施前必须判断哪个是更新且已验证版本。

建议测试：

- `python -m pytest tests/tools/test_codex_staged_implement_tool.py -q -o addopts=''`
- `python -m pytest tests/tools/test_codex_workflow_run_tool.py -q -o addopts=''`
- `python -m py_compile scripts/runtime/codex_impl_guard.py scripts/runtime/codex_review_guard.py scripts/runtime/codex_review_packet.py scripts/runtime/codex_stage_runner.py`

验收重点：

- tool schema 注册正确。
- allowed_files/allowed_globs 必填或安全默认。
- review packet bounded。
- implementation 输出只作为 candidate diff。

### 第 6 批：browser / image / custom provider 增强

目标：恢复非第一优先但用户环境需要的增强能力。

候选：

- `a8863e37f` browser repair script。
- `38fc8b285` custom codex proxy cache compat。
- `1110bc4e8` custom Responses image providers。

每个独立小批，不和 Codex workflow 混合。

## 暂不恢复 / 待确认

- `skill-provenance-stats`：不是当前救急恢复核心。
- 任何差异超过当前批目标的文件：先记录，不顺手合。
- 旧 worktree 中同名功能的多个版本：先比较 commit 时间、测试、最终 branch，再选一个。

## 每批固定报告格式

```text
阶段：第 N 批 / 功能线
已做：...
证据：commit / diff / tests
边界：未重启 / 未 push / 未部署
风险：...
下一步：是否落到 live / 是否重启 gateway
```

## 第 1 批执行记录：session_search scope handoff

状态：**已落到 live 工作区并完成 focused 验证，未 commit，未 push，未重启**。

- 隔离 worktree：`/workspace/.hermes-worktrees/hermes-agent-runtime-codex-session-search-b1-20260608095011`
- 隔离分支：`recover/session-search-scope-b1-20260608095011`
- 基线 HEAD：`a0a8171b6e54854a6c131ac03a3af2822cd6ccf5`
- 候选 commit：`4f521e6cd fix: scope session search handoff recall`
- 冲突处理：发生并已解决 4 个文件冲突：
  - `agent/tool_executor.py`：保留 live 的 tool middleware，增加 `profile/mode/scope/current_scope` 参数传递。
  - `hermes_state.py`：保留 live 的 `cwd`/declarative schema 结构，增加 session scope 字段写入，不引入旧 `lineage_id/prompt_cache` 插入结构。
  - `run_agent.py`：创建 session row 时写入 gateway user/chat/thread/session_key scope，同时保留 `cwd`。
  - `tools/session_search_tool.py`：保留 live 的 cross-profile `@session:<profile>/<id>` read 能力，同时加入 `previous/handoff/current scope` 搜索。
- 隔离 worktree touched files：
  - `agent/tool_executor.py`
  - `docs/plans/2026-05-31-session-search-scope-handoff-contract.md`
  - `gateway/session.py`
  - `hermes_state.py`
  - `run_agent.py`
  - `tests/test_hermes_state.py`
  - `tests/tools/test_session_search.py`
  - `tools/session_search_tool.py`
- 隔离验证命令与结果：
  - `python -m py_compile agent/tool_executor.py hermes_state.py run_agent.py tools/session_search_tool.py` ✅
  - `python -m pytest tests/tools/test_session_search.py -q -o addopts=''` ✅ `51 passed in 4.10s`
  - `python -m pytest tests/test_hermes_state.py -q -o addopts=''` ✅ `262 passed in 15.93s`
  - `python -m pytest tests/gateway/test_session.py tests/gateway/test_slack_channel_session_scope.py -q -o addopts=''` ✅ `85 passed, 6 warnings in 3.00s`（warnings 为既有 Slack AsyncMock coroutine warning，本批未处理）
  - `git diff --check HEAD` ✅
  - `*.pyc` 缓存复查 ✅ `total_count: 0`
- live 验证命令与结果：
  - `python -m py_compile agent/tool_executor.py hermes_state.py run_agent.py tools/session_search_tool.py` ✅
  - `python -m pytest tests/tools/test_session_search.py -q -o addopts=''` ✅ `51 passed in 4.38s`
  - `python -m pytest tests/test_hermes_state.py -q -o addopts=''` ✅ `262 passed in 15.07s`
  - `python -m pytest tests/gateway/test_session.py tests/gateway/test_slack_channel_session_scope.py -q -o addopts=''` ✅ `85 passed, 6 warnings in 2.72s`（同上，warnings 为既有 Slack AsyncMock coroutine warning）
  - `git diff --check` ✅
  - untracked docs whitespace check ✅
  - `*.pyc` 缓存复查 ✅ `0`
- 当前隔离 worktree 状态：保留 staged candidate diff，未 commit。
- live 状态：第 1 批代码已在 live 工作区，未 commit，未 push，未重启。

下一步选择：

1. 用户确认后，可 commit 第 1 批 live diff，或继续第 2 批。
2. 当前未重启 gateway/WebUI；如果需要让运行态立刻加载第 1 批代码，必须另行确认重启/热加载策略。

## 第 2 批执行记录：compression / gateway recovery 2A

状态：**2A 已落到 live 并创建本地稳定 commit，未 push，未重启**。

范围：2A 只恢复 `provider-error compression retry + recovery halt`；2B（wall-clock cap + session split persistence）未做。

- 隔离 worktree：`/workspace/.hermes-worktrees/hermes-agent-runtime-codex-compression-2a-20260608101914`
- 隔离分支：`recover/compression-2a-20260608101914`
- 基线 HEAD：`a015b1ca8 fix(session-search): restore scoped handoff recall`
- 候选来源：
  - `09c894891 fix(gateway): halt after compression recovery`
  - `9325d7e62 fix(compression): compress large provider errors before retry`
  - 注：文档原候选 `430944988` 与 `9325d7e62` tree-level diff 等价；本批采用 provider-error side worktree HEAD `9325d7e62` 以减少无关旧链路干扰。
- 冲突处理：
  - `agent/conversation_loop.py`：保留当前 live 的 `build_turn_context` / `TurnRetryState` 重构，不回退旧内联 prologue；加入 compression circuit breaker、provider-error large-request compression retry、gateway halt 返回。
  - `agent/turn_context.py`：用轻量 marker 记录 preflight compression 已降低上下文，由 `conversation_loop.py` 返回后决定是否 halt，避免 `turn_context.py` 反向 import `conversation_loop.py`。
  - `agent/turn_retry_state.py`：把旧 commit 的 `provider_error_compression_attempted` 裸局部变量迁移为 `TurnRetryState.provider_error_compression_attempted`。
  - `tests/run_agent/test_413_compression.py`：保留当前 live multi-pass preflight 语义；gateway 路径验证首次压缩后 halt，`cli/webui/local` 与 env-disabled breaker 允许继续。
- 隔离 worktree touched files：
  - `agent/agent_init.py`
  - `agent/conversation_loop.py`
  - `agent/turn_context.py`
  - `agent/turn_retry_state.py`
  - `hermes_cli/config.py`
  - `tests/run_agent/test_413_compression.py`
- 隔离验证命令与结果：
  - `python -m py_compile agent/context_compressor.py agent/conversation_compression.py agent/conversation_loop.py agent/turn_context.py agent/turn_retry_state.py run_agent.py agent/agent_init.py hermes_cli/config.py` ✅
  - `python -m pytest tests/run_agent/test_413_compression.py -q -o addopts=''` ✅ `47 passed, 1 warning in 27.14s`
  - `python -m pytest tests/agent/test_context_compressor.py -q -o addopts=''` ✅ `94 passed, 1 warning in 5.95s`
  - `python -m pytest tests/run_agent/test_compression_persistence.py tests/run_agent/test_compression_feasibility.py -q -o addopts=''` ✅ `22 passed, 1 warning in 23.82s`
  - `git diff --check HEAD` ✅
  - `*.pyc` 缓存清理与复查 ✅ `deleted_pyc=8`，最终 `0`
- live 应用方式：从隔离 worktree 生成 staged binary patch `/tmp/hermes-b2a-compression-recovery.patch`，`git apply --check` 通过后应用到 live。
- live 验证命令与结果：
  - `python -m py_compile agent/context_compressor.py agent/conversation_compression.py agent/conversation_loop.py agent/turn_context.py agent/turn_retry_state.py run_agent.py agent/agent_init.py hermes_cli/config.py` ✅
  - `python -m pytest tests/run_agent/test_413_compression.py -q -o addopts=''` ✅ `47 passed, 1 warning in 29.16s`
  - `python -m pytest tests/agent/test_context_compressor.py -q -o addopts=''` ✅ `94 passed, 1 warning in 6.41s`
  - `python -m pytest tests/run_agent/test_compression_persistence.py tests/run_agent/test_compression_feasibility.py -q -o addopts=''` ✅ `22 passed, 1 warning in 28.53s`
  - `git diff --check HEAD` ✅
  - `*.pyc` 缓存清理与复查 ✅ `deleted_pyc=8`，最终 `0`
- 当前隔离 worktree 状态：保留 staged candidate diff，未 commit。
- live 状态：2A 代码已应用到 live 工作区并创建本地 commit `1a85a7009 fix(compression): restore provider-error recovery halt`；未 push，未重启。另发现一个非本轮未跟踪文档 `docs/codex-workflow-local-capability-and-external-recommendations-2026-06-08.md`，本批未触碰。

## 第 2 批执行记录：compression / gateway recovery 2B

状态：**2B 已应用到 live 工作区并完成 focused 验证；本地 checkpoint commit 在本批收尾创建；未 push，未重启**。

范围：2B 恢复 `compression wall-clock cap + gateway session split persistence`。

- 隔离 worktree：`/workspace/.hermes-worktrees/hermes-agent-runtime-codex-compression-2b-20260608115841`
- 隔离分支：`recover/compression-2b-20260608115841`
- 基线 HEAD：`1a85a7009 fix(compression): restore provider-error recovery halt`
- 候选来源：
  - `502fb91ca fix: cap compression wall-clock latency`
  - `92d1b7c43 fix: preserve compression session splits on gateway errors`
- 冲突处理：
  - `gateway/run.py`：保留当前 live 的 `_dispose_unused_adapter` fd-leak 修复，同时加入 2B 的 `_messages_to_persist_after_agent_run`，没有整文件覆盖旧版本。
- 隔离 worktree touched files：
  - `agent/agent_init.py`
  - `agent/context_compressor.py`
  - `gateway/run.py`
  - `hermes_cli/config.py`
  - `tests/agent/test_context_compressor.py`
  - `tests/run_agent/test_413_compression.py`
  - `tests/test_compression_session_split_persistence.py`
- 隔离验证命令与结果：
  - `python -m py_compile agent/context_compressor.py agent/agent_init.py gateway/run.py hermes_cli/config.py tests/agent/test_context_compressor.py tests/run_agent/test_413_compression.py tests/test_compression_session_split_persistence.py` ✅
  - `python -m pytest tests/agent/test_context_compressor.py -q -o addopts=''` ✅ `116 passed, 1 warning in 7.35s`
  - `python -m pytest tests/run_agent/test_413_compression.py -q -o addopts=''` ✅ `49 passed, 1 warning in 29.84s`
  - `python -m pytest tests/test_compression_session_split_persistence.py -q -o addopts=''` ✅ `8 passed, 1 warning in 4.22s`
  - `python -m pytest tests/run_agent/test_compression_persistence.py tests/run_agent/test_compression_feasibility.py -q -o addopts=''` ✅ `22 passed, 1 warning in 33.05s`
  - `git diff --cached --check` ✅
  - `*.pyc` 缓存清理与复查 ✅ `deleted_pyc=7`，最终 `0`
- 当前隔离 worktree 状态：保留 staged candidate diff，未 commit。
- live 应用方式：从隔离 worktree 生成 staged binary patch `/tmp/hermes-compression-2b-live.patch`，`git apply --check` 通过后应用到 live。
- live 验证命令与结果：
  - `python -m py_compile agent/context_compressor.py agent/agent_init.py gateway/run.py hermes_cli/config.py tests/agent/test_context_compressor.py tests/run_agent/test_413_compression.py tests/test_compression_session_split_persistence.py` ✅
  - `python -m pytest tests/agent/test_context_compressor.py -q -o addopts=''` ✅ `116 passed, 1 warning in 7.72s`
  - `python -m pytest tests/run_agent/test_413_compression.py -q -o addopts=''` ✅ `49 passed, 1 warning in 26.94s`
  - `python -m pytest tests/test_compression_session_split_persistence.py -q -o addopts=''` ✅ `8 passed, 1 warning in 4.43s`
  - `python -m pytest tests/run_agent/test_compression_persistence.py tests/run_agent/test_compression_feasibility.py -q -o addopts=''` ✅ `22 passed, 1 warning in 8.78s`
  - `git diff --check HEAD -- <2B files + recovery audit doc>` ✅
  - `*.pyc` 缓存清理与复查 ✅ `deleted_pyc=7`，最终 `0`
- live 状态：2B 代码已应用到 live 工作区并完成同组 focused verification；未 push，未重启。另有非本轮未跟踪文档 `docs/codex-workflow-local-capability-and-external-recommendations-2026-06-08.md`，本批未触碰。

下一步选择：

1. 第 2 批 2B 完成本地 checkpoint commit 后，第 2 批代码恢复线可视为 live 工作区本地稳定点。
2. 当前未重启 gateway/WebUI；如果需要运行态加载第 2 批，必须另行确认 restart/reload。

## 第 3 批执行记录：process / Codex output governance

状态：**第 3 批已应用到 live 工作区并完成 focused 验证；本地 checkpoint commit 在本批收尾创建；未 push，未重启**。

范围：恢复 `process long-output / Codex output summary`、`wait timeout metadata`、`kill/trusted_completion` 语义。

- 隔离 worktree：`/workspace/.hermes-worktrees/hermes-agent-runtime-codex-process-b3-20260608122441`
- 隔离分支：`recover/process-b3-20260608122441`
- 基线 HEAD：`3a5e6d4df fix(compression): restore wall-clock cap and gateway split persistence`
- 候选来源按顺序移植：
  - `eeaf970aa fix(process): address codex wait guard review`
  - `fc5613430 fix: harden background process long-output metadata`
  - `1982dd9a1 fix(process): summarize codex output in context paths`
- 冲突处理：
  - `tools/process_registry.py`：合并输出统计 / diff flood / Codex context-safe summary 与 kill/trusted_completion / wait-window metadata；补齐 `kill_process(force/reason)`、`kill_all(force/reason)`、`_terminate_host_pid(...)->dict`。
  - `tests/tools/test_process_registry.py`：保留既有 process tests，同时加入 Codex output summary、wait timeout、kill trust 相关测试。
- 隔离 worktree touched files：
  - `tools/process_registry.py`
  - `gateway/run.py`
  - `agent/transports/codex_event_projector.py`
  - `tests/tools/test_process_registry.py`
  - `tests/gateway/test_background_process_notifications.py`
  - `tests/agent/transports/test_codex_event_projector.py`
- 隔离验证命令与结果（live 复跑同组命令结果一致）：
  - `python -m py_compile tools/process_registry.py gateway/run.py agent/transports/codex_event_projector.py tests/tools/test_process_registry.py tests/gateway/test_background_process_notifications.py tests/agent/transports/test_codex_event_projector.py` ✅
  - `python -m pytest tests/tools/test_process_registry.py -q -o addopts='' -k '<第3批新增/合并缺口相关 12 项>'` ✅ `12 passed, 85 deselected in 2.09s`
  - `python -m pytest tests/gateway/test_background_process_notifications.py -q -o addopts=''` ✅ `32 passed in 3.08s`
  - `python -m pytest tests/agent/transports/test_codex_event_projector.py -q -o addopts=''` ✅ `24 passed in 0.53s`
  - `git diff --check HEAD` ✅
  - `*.pyc` 缓存清理与复查 ✅ `deleted_pyc=6`，最终 `0`
- 全量 `tests/tools/test_process_registry.py` 说明：本环境缺 `psutil` / `ptyprocess`，且 live-system guard 会拦 `os.kill(pid, 0)` / `proc.kill()`；其中抽样 baseline 在当前 live HEAD 也复现 `4 failed`，因此未把这些环境/既有失败算作第 3 批候选失败。
- 当前隔离 worktree 状态：保留 staged candidate diff 作为来源证据。
- live 状态：已应用第 3 批代码并完成 focused 验证；未 push，未重启。live 仍有非本轮未跟踪文档 `docs/codex-workflow-local-capability-and-external-recommendations-2026-06-08.md`，本批未触碰。

下一步选择：

1. 本批收尾创建本地 checkpoint commit，作为第 3 批 stable point。
2. 当前未重启 gateway/WebUI；如果需要运行态加载第 3 批，必须另行确认 restart/reload。
3. 第 3 批 checkpoint 后，可继续第 4 批 `terminal raw Codex policy`。

## 第 4 批执行记录：terminal raw Codex policy

状态：**第 4 批已应用到 live 工作区并完成 focused 验证；本地 checkpoint commit 在本批收尾创建；未 push，未重启**。

范围：恢复 `codex-yuna exec` / `codex exec` 通过 `terminal` 工具启动时的基础安全策略：阻断 foreground / non-PTY 的 raw Codex exec，引导使用 `background=true`、`notify_on_complete=true`、`pty=true`，并默认把 background Codex exec 转为 completion 通知。

- 隔离 worktree：`/workspace/.hermes-worktrees/hermes-agent-runtime-codex-terminal-b4-20260608130313`
- 隔离分支：`recover/terminal-b4-20260608130313`
- 基线 HEAD：`12b2d6771 fix(process): restore codex output governance`
- 候选来源：
  - `88cd13013 fix(terminal): guard codex exec launch policy`
- 候选备注：
  - 另发现后续相关 commit `8b5118fae fix: block unguarded codex implementation launches`，但它涉及更强的 unguarded implementation block，并 touching `tools/process_registry.py`；可能依赖第 5 批 guarded workflow tools，因此本批未混入，避免扩大第 4 批范围。
- 冲突处理：
  - `tools/terminal_tool.py`：候选新增 Codex executable-position detection / heredoc 过滤 / env prefix 处理 / foreground 与 PTY policy helper；live 原位置无等价 helper，保留候选新增 helper，没有整文件覆盖。
- 隔离 worktree touched files：
  - `tools/terminal_tool.py`
  - `tests/tools/test_terminal_tool.py`
- 隔离验证命令与结果（live 复跑同组命令结果一致）：
  - `PYTHONDONTWRITEBYTECODE=1 python -m py_compile tools/terminal_tool.py tests/tools/test_terminal_tool.py` ✅
  - `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/tools/test_terminal_tool.py -q -o addopts=''` ✅ `24 passed in 1.40s`
  - `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/tools/test_terminal_tool_pty_fallback.py tests/tools/test_terminal_none_command_guard.py -q -o addopts=''` ✅ `5 passed in 1.29s`
  - `git diff --check HEAD` ✅
  - `*.pyc` 缓存清理与复查 ✅ `deleted_pyc=2`，最终 `0`
- 当前隔离 worktree 状态：保留 staged candidate diff 作为来源证据。
- live 状态：已应用第 4 批代码并完成 focused 验证；未 push，未重启。live 仍有非本轮未跟踪文档 `docs/codex-workflow-local-capability-and-external-recommendations-2026-06-08.md`，本批未触碰。

下一步选择：

1. 本批收尾创建本地 checkpoint commit，作为第 4 批 stable point。
2. 当前未重启 gateway/WebUI；如果需要运行态加载第 4 批，必须另行确认 restart/reload。
3. 第 4 批 checkpoint 后，可继续第 5 批 `guarded Codex workflow tools`。

## 第 5 批执行记录：guarded Codex workflow tools

状态：**第 5 批已应用到 live 工作区并完成 focused 验证；本地 checkpoint commit 在本批收尾创建；未 push，未重启**。

范围：恢复 `codex_workflow_run` / `codex_staged_implement` 工具、guard scripts、review packet、stage runner 与 toolset/config 注册。

- 隔离 worktree：`/workspace/.hermes-worktrees/hermes-agent-runtime-codex-guarded-b5-20260608132143`
- 隔离分支：`recover/guarded-b5-20260608132143`
- 基线 HEAD：`c585a07aa fix(terminal): restore codex exec launch policy`
- 候选判断：
  - `f241c4352 fix: add guarded codex workflow orchestration`
  - `77b7541dd fix: add guarded codex workflow orchestration`
  - 两者 `git patch-id --stable` 相同，功能 diff 等价；`77b7541dd` 是 fork/main 上 later cherry-pick，因此本批选 `77b7541dd`。
  - `8b5118fae fix: block unguarded codex implementation launches` 暂未混入；它是更强 raw implementation block，touches `process_registry.py` / `terminal_tool.py`，建议作为 5B 或后续强化，在 core guarded tools 落稳后处理。
- 冲突处理：
  - `77b7541dd` 在隔离 worktree 无冲突应用。
- 隔离 worktree touched files：
  - `tools/codex_staged_implement_tool.py`
  - `tools/codex_workflow_run_tool.py`
  - `scripts/runtime/codex_impl_guard.py`
  - `scripts/runtime/codex_review_guard.py`
  - `scripts/runtime/codex_review_packet.py`
  - `scripts/runtime/codex_stage_runner.py`
  - `tests/tools/test_codex_staged_implement_tool.py`
  - `tests/tools/test_codex_workflow_run_tool.py`
  - `hermes_cli/tools_config.py`
  - `toolsets.py`
- 隔离验证命令与结果（live 复跑同组命令结果一致）：
  - `PYTHONDONTWRITEBYTECODE=1 python -m py_compile tools/codex_staged_implement_tool.py tools/codex_workflow_run_tool.py scripts/runtime/codex_impl_guard.py scripts/runtime/codex_review_guard.py scripts/runtime/codex_review_packet.py scripts/runtime/codex_stage_runner.py tests/tools/test_codex_staged_implement_tool.py tests/tools/test_codex_workflow_run_tool.py hermes_cli/tools_config.py toolsets.py` ✅
  - `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/tools/test_codex_staged_implement_tool.py -q -o addopts=''` ✅ `92 passed, 1 warning in 17.13s`
  - `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/tools/test_codex_workflow_run_tool.py -q -o addopts=''` ✅ `49 passed in 3.14s`
  - `git diff --check HEAD` ✅
  - `*.pyc` 缓存清理与复查 ✅ `deleted_pyc=10`，最终 `0`
- 当前隔离 worktree 状态：保留 staged candidate diff 作为来源证据。
- live 状态：已应用第 5 批代码并完成 focused 验证；未 push，未重启。live 仍有非本轮未跟踪文档 `docs/codex-workflow-local-capability-and-external-recommendations-2026-06-08.md`，本批未触碰。

下一步选择：

1. 本批收尾创建本地 checkpoint commit，作为第 5 批 stable point。
2. 当前未重启 gateway/WebUI；如果需要运行态加载第 5 批，必须另行确认 restart/reload。
3. 第 5 批落 live 后，当前会话工具 schema 仍不一定立刻暴露 `codex_staged_implement` / `codex_workflow_run`；需要 gateway/WebUI reload/restart 或新运行进程加载后才生效。
4. 第 5 批 checkpoint 后，建议单独考虑 5B：`8b5118fae` 更强 raw implementation block。

## 第 5B 批执行记录：guarded workflow 后续 raw implementation block

状态：**5B 已应用到 live 工作区并完成 focused 验证；本地 checkpoint commit 在本批收尾创建；未 push，未重启**。

范围：恢复更强的 raw `codex-yuna exec` / `codex exec` implementation block：即使 background/PTY 也阻断裸实现调用，引导到 `codex_staged_implement` / `codex_stage_runner.py` / `codex_impl_guard.py` / `codex_review_guard.py`。

- 隔离 worktree：`/workspace/.hermes-worktrees/hermes-agent-runtime-codex-guarded-5b-20260608133358`
- 隔离分支：`recover/guarded-5b-20260608133358`
- 基线 HEAD：`ee777a7c5 fix(codex): restore guarded workflow tools`
- 候选来源：
  - `8b5118fae fix: block unguarded codex implementation launches`
- 冲突处理：
  - `tests/tools/test_terminal_tool.py`：合并第 4 批 raw Codex launch policy tests 与 5B 新 block/review tests；删除/调整 4 个与 5B 新策略冲突的旧 background/escape-hatch 期望。
  - `tools/terminal_tool.py`：保留第 4 批 executable-position detection，同时补入 `8b5118fae` 依赖的 `_shell_tokens` / `_codex_review_launch_error` / review-output-control helper，并加入 stronger raw implementation block。
- 隔离 worktree touched files：
  - `tools/terminal_tool.py`
  - `tools/process_registry.py`
  - `tests/tools/test_terminal_tool.py`
  - `tests/tools/test_process_registry.py`
- 隔离验证命令与结果（live 复跑同组命令结果一致）：
  - `PYTHONDONTWRITEBYTECODE=1 python -m py_compile tools/terminal_tool.py tools/process_registry.py tests/tools/test_terminal_tool.py tests/tools/test_process_registry.py` ✅
  - `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/tools/test_terminal_tool.py -q -o addopts=''` ✅ `29 passed in 1.42s`
  - `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/tools/test_process_registry.py -q -o addopts='' -k 'codex_unguarded or unguarded_codex or codex_staged or trusted_completion or timeout_metadata'` ✅ `1 passed, 97 deselected in 1.03s`
  - `git diff --check HEAD` ✅
  - `*.pyc` 缓存清理与复查 ✅ `deleted_pyc=4`，最终 `0`
- 当前隔离 worktree 状态：保留 staged candidate diff 作为来源证据。
- live 状态：已应用 5B 代码并完成 focused 验证；未 push，未重启。live 仍有非本轮未跟踪文档 `docs/codex-workflow-local-capability-and-external-recommendations-2026-06-08.md`，本批未触碰。
- 审批记录：5B 解冲突/测试调整期间，多次写入仅限隔离 worktree，并在用户通过审批后继续；未绕过审批改 live。

下一步选择：

1. 本批收尾创建本地 checkpoint commit，作为 5B stable point。
2. 当前未重启 gateway/WebUI；如果需要运行态加载 5B，必须另行确认 restart/reload。

## 第 6A 批执行记录：browser runtime repair script

状态：**6A 已应用到 live 工作区并完成 focused 验证；本地 checkpoint commit 在本批收尾创建；未 push，未重启**。

范围：恢复 WebUI/browser 本地运行时修复脚本 `scripts/runtime/repair_webui_browser_runtime.py` 及其 focused tests。第 6 批其余候选（custom Codex proxy cache compat、custom Responses image providers）不混入本批。

- 隔离 worktree：`/workspace/.hermes-worktrees/hermes-agent-runtime-codex-browser-6a-20260608143723`
- 隔离分支：`recover/browser-6a-20260608143723`
- 基线 HEAD：`803062c48 fix(codex): block unguarded implementation launches`
- 候选来源：
  - `a8863e37f fix(browser): add WebUI runtime repair script`
- 冲突处理：
  - 无冲突应用。
- 隔离 worktree touched files：
  - `scripts/runtime/repair_webui_browser_runtime.py`
  - `tests/scripts/test_repair_webui_browser_runtime.py`
- 隔离验证命令与结果（live 复跑同组命令结果一致）：
  - `PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/runtime/repair_webui_browser_runtime.py tests/scripts/test_repair_webui_browser_runtime.py` ✅
  - `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/scripts/test_repair_webui_browser_runtime.py -q -o addopts=''` ✅ `7 passed in 0.49s`
  - `PYTHONDONTWRITEBYTECODE=1 python scripts/runtime/repair_webui_browser_runtime.py --help` ✅ `1153 bytes`
  - `git diff --check HEAD -- scripts/runtime/repair_webui_browser_runtime.py tests/scripts/test_repair_webui_browser_runtime.py docs/runtime-recovery-audit-2026-06-08.md` ✅
  - `*.pyc` 缓存清理与复查 ✅ `deleted_pyc=2`，最终 `0`
- 当前隔离 worktree 状态：保留 staged candidate diff 作为来源证据。
- live 状态：已应用 6A 代码并完成 focused 验证；未 push，未重启。live 仍有非本轮未跟踪文档 `docs/codex-workflow-local-capability-and-external-recommendations-2026-06-08.md`，本批未触碰。

下一步选择：

1. 本批收尾创建本地 checkpoint commit，作为 6A stable point。
2. 当前未重启 gateway/WebUI；该脚本落 live 也不等于运行脚本或修复当前 browser runtime，实际执行 `--check` / `--repair` 需要另行确认。
3. 6A checkpoint 后，再分别处理 6B `custom Codex proxy cache compat` 与 6C `custom Responses image providers`。

## 第 6B 批执行记录：custom Codex proxy cache compat

状态：**6B 已应用到 live 工作区并完成 focused 验证；本地 checkpoint commit 在本批收尾创建；未 push，未重启**。

范围：恢复 custom `/codex` Responses 网关的 cache compatibility：custom `/codex` 路由识别为 `codex_responses`、Codex request 注入 transport/cache metadata、SessionDB 记录 `prompt_cache_key` / `prompt_cache_supported`。第 6 批其余候选（custom Responses image providers）不混入本批。

- 适用 skill：`codex-proxy-cache-compat`
- 隔离 worktree：`/workspace/.hermes-worktrees/hermes-agent-runtime-codex-codex-proxy-6b-20260608145311`
- 隔离分支：`recover/codex-proxy-6b-20260608145311`
- 基线 HEAD：`b67a0a29b fix(browser): restore WebUI runtime repair script`
- 候选来源：
  - `38fc8b285 fix: support custom codex proxy cache compat`
- 冲突处理：
  - `hermes_state.py`：保留 live 的 `SCHEMA_VERSION = 15`、QQ scope 字段（`user_id_alt/chat_type/chat_id/thread_id/session_key`）与 `cwd`，同时合入 `prompt_cache_key` / `prompt_cache_supported` 列、insert 参数与 `update_prompt_cache_state` / `get_prompt_cache_state`。
- 隔离 worktree touched files：
  - `agent/agent_init.py`
  - `agent/chat_completion_helpers.py`
  - `agent/transports/codex.py`
  - `hermes_cli/runtime_provider.py`
  - `hermes_state.py`
  - `tests/agent/transports/test_codex_transport.py`
  - `tests/hermes_cli/test_runtime_provider_resolution.py`
  - `tests/run_agent/test_run_agent_codex_responses.py`
  - `tests/test_hermes_state.py`
- 隔离验证命令与结果（live 复跑同组命令结果一致）：
  - `PYTHONDONTWRITEBYTECODE=1 python -m py_compile agent/agent_init.py agent/chat_completion_helpers.py agent/transports/codex.py hermes_cli/runtime_provider.py hermes_state.py tests/agent/transports/test_codex_transport.py tests/hermes_cli/test_runtime_provider_resolution.py tests/run_agent/test_run_agent_codex_responses.py tests/test_hermes_state.py` ✅
  - `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/hermes_cli/test_runtime_provider_resolution.py -q -o addopts=''` ✅ `128 passed in 3.02s`
  - `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/agent/transports/test_codex_transport.py -q -o addopts=''` ✅ `53 passed, 1 warning in 3.98s`
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/hermes_test_stubs:${PYTHONPATH:-} python -m pytest -p no:cacheprovider tests/run_agent/test_run_agent_codex_responses.py -q -o addopts=''` ✅ `72 passed, 1 warning in 37.93s`
  - `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_hermes_state.py -q -o addopts=''` ✅ `263 passed in 15.08s`
  - `git diff --check HEAD -- <6B files + recovery audit doc>` ✅
  - `*.pyc` 缓存清理与复查 ✅ `deleted_pyc=9`，最终 `0`
- 当前隔离 worktree 状态：保留 staged candidate diff 作为来源证据。
- live 状态：已应用 6B 代码并完成 focused 验证；未 push，未重启。live 仍有非本轮未跟踪文档 `docs/codex-workflow-local-capability-and-external-recommendations-2026-06-08.md`，本批未触碰。

下一步选择：

1. 本批收尾创建本地 checkpoint commit，作为 6B stable point。
2. 当前未重启 gateway/WebUI；cache compat 的运行态验证需要 reload/restart 后用同一 stable thread / endpoint / model 检查 `cached_tokens` / `cache_read_tokens`，另行确认。
3. 6B checkpoint 后，再处理 6C `custom Responses image providers`。

## 第 6C 批执行记录：custom Responses image providers

状态：**6C 已应用到 live 工作区并完成 focused 验证；本地 checkpoint commit 在本批收尾创建；未 push，未重启**。

范围：恢复 `image_gen.provider: custom:<name>` + custom Responses-style gateway 的图片生成路径，让 registered image generation dispatch 能识别并调用 custom provider 的 Responses image model。第 6 批其余候选已拆分完成，不混入本批。

- 适用参考：`hermes-agent` / `references/custom-image-provider-routes.md`、`references/image-gen-provider-routing.md`
- 隔离 worktree：`/workspace/.hermes-worktrees/hermes-agent-runtime-codex-image-6c-20260608150913`
- 隔离分支：`recover/image-6c-20260608150913`
- 基线 HEAD：`179ab747c fix(codex): restore custom proxy cache compat`
- 候选来源：
  - `1110bc4e8 feat(image-gen): support custom Responses image providers`
- 冲突处理：
  - 无冲突应用。
- 隔离 worktree touched files：
  - `tools/image_generation_tool.py`
  - `tests/tools/test_image_generation_plugin_dispatch.py`
- 隔离验证命令与结果（live 复跑同组命令结果一致）：
  - `PYTHONDONTWRITEBYTECODE=1 python -m py_compile tools/image_generation_tool.py tests/tools/test_image_generation_plugin_dispatch.py` ✅
  - `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/tools/test_image_generation_plugin_dispatch.py -q -o addopts=''` ✅ `6 passed in 0.55s`
  - `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/tools/test_image_generation.py tests/tools/test_image_generation_env.py tests/tools/test_image_generation_artifacts.py -q -o addopts=''` ✅ `66 passed in 1.55s`
  - `git diff --check HEAD -- tools/image_generation_tool.py tests/tools/test_image_generation_plugin_dispatch.py docs/runtime-recovery-audit-2026-06-08.md` ✅
  - `*.pyc` 缓存清理与复查 ✅ `deleted_pyc=2`，最终 `0`
- 当前隔离 worktree 状态：保留 staged candidate diff 作为来源证据。
- live 状态：已应用 6C 代码并完成 focused 验证；未 push，未重启。live 仍有非本轮未跟踪文档 `docs/codex-workflow-local-capability-and-external-recommendations-2026-06-08.md`，本批未触碰。

下一步选择：

1. 本批收尾创建本地 checkpoint commit，作为 6C stable point。
2. 当前未重启 gateway/WebUI；custom image provider 的运行态验证需要 reload/restart 或新进程加载代码后，通过 `_handle_image_generate` / 实际 `image_generate` 工具路径验证，另行确认。

## 当前待办

- [x] 第 0 批：初版恢复清单。
- [x] 第 1 批：session_search scope handoff（已落 live 并本地 commit，未 push/未重启）。
- [x] 第 2 批：compression / gateway recovery（2A 已落 live 并本地 commit；2B 已落 live 并完成 focused 验证；未 push/未重启）。
- [x] 第 3 批：process / Codex output governance（已落 live 并完成 focused 验证；未 push/未重启）。
- [x] 第 4 批：terminal raw Codex policy（已落 live 并完成 focused 验证；未 push/未重启）。
- [x] 第 5 批：guarded Codex workflow tools（已落 live 并完成 focused 验证；未 push/未重启）。
- [x] 第 5B 批：strong raw Codex implementation block（已落 live 并完成 focused 验证；未 push/未重启）。
- [x] 第 6A 批：browser runtime repair script（已落 live 并完成 focused 验证；未 push/未重启）。
- [x] 第 6B 批：custom Codex proxy cache compat（已落 live 并完成 focused 验证；未 push/未重启）。
- [x] 第 6C 批：custom Responses image providers（已落 live 并完成 focused 验证；未 push/未重启）。
