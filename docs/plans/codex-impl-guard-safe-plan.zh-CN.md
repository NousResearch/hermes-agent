# Codex Implementation Guard 安全边界版开发计划

> 状态：计划已进入 Phase 11 候选实现与 review 修复。
> 范围：规划并分轮实现 `codex_impl_guard.py` 与后续 `codex_stage_runner`。
> 原则：Codex 只产候选 patch；Hermes 负责边界、归因、验证、最终判断。

## 背景

当前运行时已经有多层 Codex 保护：

- `tools/process_registry.py`
  - Codex 输出上下文安全摘要。
  - `trusted_completion`。
  - `diff_flood_detected` / `source_flood_detected`。
  - `wait_window_expired` 不等于失败。
  - Codex kill guard 与 takeover 推荐语义。
- `tools/terminal_tool.py`
  - 裸 `codex-yuna exec` / `codex exec` 前台运行提示改用 background + `notify_on_complete=true`。
  - Codex review 运行必须带 structured output flags。
- `scripts/runtime/codex_review_guard.py`
  - `--sandbox read-only`。
  - `--json` / `--output-schema` / `--output-last-message` / `--color never`。
  - raw log 落盘，stdout 只输出 bounded JSON。
  - source/diff/JSONL field flood 检测与终止。
  - schema-valid final/recovered review 判定。
- `scripts/runtime/codex_review_packet.py`
  - 构造 bounded review packet。
  - scoped diff/untracked preview。

本计划不重做这些能力。新功能只补“开发阶段编排层”。

## 目标

让 Codex implementation 从“一口吞大任务”变成：

```text
一个小 slice
→ Codex 受控写入候选 diff
→ Hermes 收集 git 证据
→ Hermes 检查 allowlist / dirty baseline / flood / exit reason
→ Hermes 决定 passed / failed / takeover_candidate / review_needed
```

## 非目标

本计划第一阶段不做：

- 不做完整多阶段自动 runner。
- 不做新的 review guard。
- 不替换 `process_registry.py` 的 flood/trusted_completion 语义。
- 不让 Codex commit/push/PR/deploy/restart。
- 不自动清理、reset、revert 用户已有改动。
- 不处理跨 repo 多 agent 协作。

## 设计总览

分两层落地：

1. `scripts/runtime/codex_impl_guard.py`
   - 单次 implementation wrapper。
   - 负责受控启动 Codex、采集 git evidence、输出 bounded JSON。
   - 第一阶段必须可独立测试。
2. `scripts/runtime/codex_stage_runner.py`（后续阶段）
   - 多 slice 编排器。
   - 只在 `codex_impl_guard.py` 稳定后实现。

## Phase 1：实现前的边界文档与测试约定

### 1.1 复用边界

新增 `codex_impl_guard.py` 时，必须明确：

- 复用或抽取 `codex_review_guard.py` 中的输出 flood 检测 helper：
  - ANSI strip。
  - source-like line 检测。
  - diff-like line 检测。
  - JSONL field flood 检测。
  - bounded JSON value 裁剪。
- 复用 `process_registry.py` 已有语义，不另起同名但含义不同的字段：
  - `trusted_completion`。
  - `diff_flood_detected`。
  - `source_flood_detected`。
  - `terminated_by_guard`。
  - `recommended_next_action`。
- `codex_impl_guard.py` 自己新增的职责只限：
  - git baseline。
  - changed/untracked/deleted/renamed/submodule evidence。
  - allowlist 判定。
  - slice status 判定。
  - verification command 结果收集。

### 1.2 状态枚举

`codex_impl_guard.py` stdout 只输出一个 JSON object，包含：

```json
{
  "status": "passed | failed | takeover_candidate | blocked_by_allowlist | review_needed | unusable",
  "reason": "短机器可读原因",
  "trusted_completion": true,
  "codex_exit_code": 0,
  "terminated_by_guard": false,
  "diff_flood_detected": false,
  "source_flood_detected": false,
  "changed_files": [],
  "untracked_files": [],
  "deleted_files": [],
  "renamed_files": [],
  "submodule_changes": [],
  "allowlist_violations": [],
  "dirty_baseline": [],
  "dirty_baseline_policy": "require-clean | allow-listed-owned | fail-on-overlap",
  "verification": [],
  "diff_stat": "bounded string",
  "raw_log_path": "/tmp/...",
  "final_file": "/tmp/...",
  "recommended_next_action": "..."
}
```

Exit code 约定：

- `0`：`passed` 或 `review_needed`，没有安全阻塞。
- `1`：`failed` 或 `blocked_by_allowlist`。
- `2`：`takeover_candidate`，有候选 diff，但不可信完成，需要 Hermes 接管。
- `3`：`unusable`，Codex 未能产生可判定结果或环境不可用。

状态表固定规则：

| 条件 | 状态 | 原因 |
|---|---|---|
| Codex exit 0，allowlist 通过，verification 通过或未请求 | `passed` / `review_needed` | 正常结束，但仍由 Hermes 决定是否进入 review |
| Codex exit nonzero 且没有任何安全候选 diff | `failed` | 没有可接管成果 |
| Codex exit nonzero 但存在安全候选 diff，且 allowlist / dirty baseline / workdir escape / untracked 检查全部通过 | `takeover_candidate` | 只代表可人工接管，不代表成功 |
| Codex timeout/flood/被 guard kill，且存在安全候选 diff | `takeover_candidate` | 不可信完成，Hermes 接管 |
| 任何 allowlist、workdir escape、artifact violation | `blocked_by_allowlist` | 安全阻塞，不能继续 runner |
| workdir/Codex/sandbox/配置不可用，或 dirty baseline 策略不通过且未启动 Codex | `unusable` | 环境/前置状态不可用 |

### 1.3 allowed glob 匹配口径

`allowed_file` 与 `allowed_glob` 都只接受 repo-relative path pattern：

- 禁止绝对路径。
- 禁止以 `../` 开头或包含 path escape。
- 禁止空 pattern。
- glob 匹配对象是 git/repo-relative POSIX path，例如 `tools/foo.py`。
- 所有候选路径先转成 repo-relative POSIX path，再做 glob 匹配。
- `allowed_file` 只做精确 repo-relative path 匹配。
- `allowed_glob` 使用 Python `fnmatch.fnmatchcase` 语义，大小写敏感。
- symlink 目标仍必须在 workdir 内，glob 命中不等于放行 symlink escape。

## Phase 2：`codex_impl_guard.py` 单次 wrapper

### 2.1 输入参数

建议参数：

```bash
python scripts/runtime/codex_impl_guard.py \
  --workdir /path/to/repo \
  --prompt-file /tmp/slice-prompt.md \
  --allowed-file path/to/file.py \
  --allowed-glob 'tests/foo/*.py' \
  --dirty-baseline-policy require-clean \
  --verify-cmd-id focused-tests \
  --timeout-seconds 900 \
  --raw-log /tmp/codex-impl.raw.log \
  --final-file /tmp/codex-impl.final.json
```

第一阶段只允许从一个内置 registry 中选择 `--verify-cmd-id`，不要允许任意 shell 字符串直接传入。registry 可以先很小：

```text
none
focused-pytest:<path>::<test>
diff-check
targeted-unittest:<module-or-discovery-pattern>
```

如果验证命令需要更复杂，先让 wrapper 输出 `review_needed`，由 Hermes 手动跑。

### 2.2 Codex 启动方式

`codex_impl_guard.py` 是受控 wrapper，可以前台运行，因为它自己保证 stdout bounded。内部启动 Codex 时：

- 默认使用 `codex-yuna exec`。
- 使用 writable sandbox 模式，不使用 `--sandbox read-only`。
- 必须禁止 Codex commit/push/deploy/restart。
- prompt 内必须写明：
  - 只做当前 slice。
  - 只能改 allowlist 内文件。
  - 不要 broad grep。
  - 不要 full diff。
  - 不要读/写 secrets。
  - 不要清理/reset/revert 既有改动。
  - Hermes 会最终验证。

当前最小实现使用 `--sandbox workspace-write`，不使用 `--full-auto`。若后续改为 `--full-auto`，必须先确认当前 Codex CLI sandbox 语义，并继续依赖后置 git evidence + allowlist fail-closed；若 sandbox 不可用，需要明确拒绝或要求外层隔离，不默认 bypass。

### 2.3 stdout / raw log

- raw Codex stdout/stderr 写 `raw_log_path`。
- `codex_impl_guard.py` 自己 stdout 只输出 bounded JSON。
- 即使 Codex 输出 flood，也不能把 flood 内容进主上下文。
- flood 检测触发时：
  - 终止 Codex。
  - 标记 `trusted_completion=false`。
  - 继续做 git evidence 收集。
  - 如果 evidence 通过安全检查，返回 `takeover_candidate`。
  - 如果 evidence 越界，返回 `blocked_by_allowlist`。

## Phase 3：git evidence 与 allowlist

### 3.1 执行前 baseline

启动 Codex 前收集：

```bash
git status --porcelain=v1 -z --untracked-files=all
git diff --name-status -z
git diff --stat
```

如果 `dirty_baseline_policy=require-clean` 且有 dirty 文件：

- 直接返回 `unusable` 或 `failed`。
- 不启动 Codex。
- 报告 dirty baseline。

如果使用 `allow-listed-owned`：

- 只允许 dirty baseline 全部在 allowlist 内。
- Codex 结束后必须区分 preexisting 与 slice delta。

默认第一版建议：`require-clean`。在当前 dirty repo 中测试时，用临时 worktree 或专门 fixture repo。

### 3.2 执行后 evidence

Codex 结束或被 guard 终止后收集：

```bash
git status --porcelain=v1 -z --untracked-files=all
git diff --name-status -z
git diff --stat -- <scoped files>
git ls-files --others --exclude-standard -z
```

需要分类：

- modified。
- added。
- deleted。
- renamed。
- copied。
- untracked。
- submodule change。
- ignored 不纳入，除非 verify 产生显式 artifact residue 检查。

### 3.3 路径规范化

所有 allowlist 检查必须：

- 以 `workdir.resolve()` 为根。
- 对每个候选路径做 `resolve(strict=False)`。
- 拒绝绝对路径在 workdir 外的文件。
- 拒绝 `..` escape。
- 拒绝 symlink 指向 workdir 外。
- 对 untracked 文件也同样检查。
- 对 deleted 路径用 git path，不依赖文件存在。
- nested repo / submodule 变化默认 blocked，除非 allowlist 显式允许该 submodule path。

### 3.4 allowlist violation 策略

任何越界改动：

```text
status=blocked_by_allowlist
exit_code=1
trusted_completion=false
recommended_next_action=inspect changed files manually; do not continue runner
```

wrapper 不自动 revert。是否保留/回滚由 Hermes 和用户决定。

## Phase 4：takeover 规则

只有满足以下条件，才能返回 `takeover_candidate`：

- Codex 被 timeout/flood/guard kill，或 `trusted_completion=false`。
- 有 git diff 或 untracked candidate。
- 所有 changed/untracked/deleted/renamed/submodule evidence 已收集。
- 无 allowlist violation。
- 无 workdir escape。
- dirty baseline 策略通过。
- raw log/final file 路径已记录。
- verify 未运行或未通过时，明确标记 `verification` 状态，不宣称通过。

`takeover_candidate` 不是成功。含义是：

```text
有可检查候选 diff；Hermes 应停止等待 Codex，直接读 touched files，必要时补小漏并跑验证。
```

## Phase 5：terminal policy 接入

第一阶段不要放宽 `terminal_tool.py` 现有规则。

需要新增/确认规则：

- 裸 `codex-yuna exec` / `codex exec` 前台运行继续提示改 background + notify。
- 裸 Codex review 继续要求 structured output flags。
- `python scripts/runtime/codex_review_guard.py ...` 是 review 受控入口。
- `python scripts/runtime/codex_impl_guard.py ...` 是 implementation 受控入口，可以前台跑，因为 stdout bounded。
- `terminal_tool.py` 不应把 `codex_impl_guard.py` 误判成普通长任务，也不应把它当裸 Codex 放行的后门。

建议测试：

- foreground `codex-yuna exec '...'` 仍有 guidance。
- foreground `python scripts/runtime/codex_impl_guard.py --help` 不被拦。
- foreground `python scripts/runtime/codex_impl_guard.py ...` 在测试 fixture 中可返回 bounded JSON。
- review guard 规则不退化。

## Phase 6：verification command 限制

第一版不要让计划文件传任意 shell 字符串。使用 registry：

```json
{
  "none": [],
  "diff-check": ["git", "diff", "--check"],
  "pytest-file": ["python", "-m", "pytest", "<path>", "-q", "-o", "addopts="],
  "unittest-discover": ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "<pattern>", "-v"]
}
```

约束：

- argv list，不走 shell。
- timeout 必填。
- cwd 固定为 workdir。
- stdout/stderr bounded。
- exit code 进入 `verification` 数组。
- 失败不自动重试。
- 有 cache/artifact residue 时只报告，不自动删除。

### 6.1 验证命令写入边界

验证命令本身也可能写文件，例如 `pytest` / `unittest` 可能生成 `.pytest_cache`、`__pycache__`、coverage、本地 DB 或临时报告。因此 verification 必须是独立 evidence 阶段，不能绕过 Phase 3 的 git 边界。

每个 `verify_cmd_id` 必须声明：

```json
{
  "id": "pytest-file",
  "argv_template": ["python", "-m", "pytest", "<path>", "-q", "-o", "addopts="],
  "cwd": "workdir",
  "timeout_seconds": 120,
  "writes_allowed": false,
  "allowed_artifact_globs": [".pytest_cache/**", "**/__pycache__/**", "**/*.pyc"],
  "env_policy": "minimal-safe"
}
```

默认策略：

- `writes_allowed=false`。
- `diff-check` 不允许任何写入。
- `pytest-file` / `unittest-discover` 如果会写缓存，第一版优先通过环境变量或参数隔离/禁用缓存；如果仍产生允许的 cache artifact，只能记录为 `allowed_artifacts`，不得自动删除。
- 非 `allowed_artifact_globs` 的新增/修改/删除，一律返回 `blocked_by_allowlist` 或 `failed`，并写入 `verification[*].artifact_violations`。
- verification 失败或产生 artifact violation 时，不得返回 `passed`。

### 6.2 验证前后 evidence

每个 verification command 前后都要采集：

```bash
git status --porcelain=v1 -z --untracked-files=all
git diff --name-status -z
git ls-files --others --exclude-standard -z
```

然后计算 verification delta：

```json
{
  "id": "pytest-file",
  "exit_code": 0,
  "stdout_preview": "bounded",
  "stderr_preview": "bounded",
  "duration_seconds": 1.23,
  "pre_status_count": 0,
  "post_status_count": 0,
  "new_artifacts": [],
  "allowed_artifacts": [],
  "artifact_violations": [],
  "status": "passed | failed | blocked_by_artifact_violation | timed_out"
}
```

如果 verification 产生允许缓存，wrapper 只报告，不自动清理；最终是否清理由 Hermes 在完成前验证阶段处理。这样避免 wrapper 静默删除用户文件或掩盖测试副作用。

### 6.3 第一版 registry 限制

第一版 registry 只启用最小低风险命令：

- `none`。
- `diff-check`。
- `pytest-file` / `unittest-discover` 作为后续扩展保留在设计里；在当前最小实现中必须 fail-closed，不得悄悄执行或当作通过。

不允许：

- 任意 shell 字符串。
- `npm install` / `pip install` / package manager。
- build 命令。
- deploy/restart 命令。
- 会写外部路径的命令。

## Phase 7：测试计划

新增测试文件建议：

- `tests/scripts/test_codex_impl_guard.py`
- 后续：`tests/scripts/test_codex_stage_runner.py`

### 7.1 impl guard 必测

- `--help` 可运行。
- workdir 不存在 -> `unusable`。
- dirty baseline + `require-clean` -> 不启动 Codex，返回 `unusable/failed`。
- clean fixture repo + Codex fake binary 修改 allowlist 内文件 -> `passed` 或 `review_needed`。
- fake binary 修改 allowlist 外文件 -> `blocked_by_allowlist`。
- fake binary 新增 untracked allowlist 内文件 -> included in `untracked_files`。
- fake binary 新增 untracked allowlist 外文件 -> `blocked_by_allowlist`。
- symlink 指向 workdir 外 -> blocked。
- deleted / renamed 文件进入 evidence。
- submodule-like gitlink change 默认 blocked。
- flood 后有安全 diff -> `takeover_candidate`。
- flood 后越界 diff -> `blocked_by_allowlist`。
- Codex exit nonzero + no diff -> `failed`。
- Codex exit nonzero + safe diff -> `takeover_candidate` 或 `failed`，按状态表固定。
- raw log 不进入 stdout。
- stdout JSON 长度 bounded。

### 7.2 terminal policy 必测

- 裸 Codex foreground guidance 保持。
- review structured flags guard 保持。
- impl guard wrapper 不触发裸 Codex foreground guidance。
- impl guard wrapper stdout bounded。

### 7.3 stage runner 后续必测

- slice 1 failed 时不跑 slice 2。
- `blocked_by_allowlist` 立即停止。
- `takeover_candidate` 停止 runner，等待 Hermes 接管。
- 每个 slice 单独 baseline。
- verification 失败进入 slice JSON。
- untracked 文件纳入 scope。

## Phase 8：`codex_stage_runner.py` 详细设计

只有在 `codex_impl_guard.py` 通过测试和 Codex read-only review 后再做。

### 8.1 设计定位

`codex_stage_runner.py` 不是让 Codex 自己规划，也不是让 Codex 连续自由开发。它只做编排：

```text
读取 slices JSON
→ 按顺序执行一个 slice
→ 每个 slice 调 codex_impl_guard.py
→ 根据 impl guard JSON 决定继续/停止/接管
→ 输出 bounded runner summary JSON
```

分工固定：

```text
Hermes：拆 slice、定义 allowed files/globs、review、验证、接管
Codex：只实现当前 slice
codex_impl_guard.py：单 slice 安全笼子
codex_stage_runner.py：顺序编排器
```

### 8.2 输入格式

第一版只接受 JSON 文件，不从 Markdown 自动解析。入口建议：

```bash
python scripts/runtime/codex_stage_runner.py \
  --plan-file /tmp/codex-stage-plan.json \
  --raw-dir /tmp/codex-stage-raw \
  --timeout-seconds 900
```

JSON schema 草案：

```json
{
  "repo": "/path/to/repo",
  "dirty_baseline_policy": "require-clean",
  "continue_policy": "stop-on-review-needed",
  "slices": [
    {
      "id": "short-id",
      "goal": "只做一个明确小目标",
      "prompt_file": "/tmp/slice.md",
      "allowed_files": ["path/to/file.py"],
      "allowed_globs": ["tests/foo/*.py"],
      "verify_cmd_ids": ["diff-check", "pytest-file:tests/foo/test_bar.py"]
    }
  ]
}
```

输入约束：

- `repo` 必须是 git repo root 或可解析到 git repo root。
- `slices` 必须非空。
- `id` 必须唯一，只允许 `[a-zA-Z0-9_.-]`。
- `prompt_file` 必须存在，路径不能从 runner JSON 中携带 secrets。
- `allowed_files` / `allowed_globs` 继续沿用 `codex_impl_guard.py` 的 repo-relative 规则。
- `verify_cmd_ids` 只允许 impl guard registry 支持的低风险 id。
- 第一版不支持内联 prompt，避免 JSON 过大或混入指令注入内容。

### 8.3 runner 状态

runner 自己也只输出一个 JSON object：

```json
{
  "status": "completed | stopped | failed | unusable",
  "reason": "machine-readable reason",
  "repo": "/path/to/repo",
  "continue_policy": "stop-on-review-needed",
  "completed_slices": [],
  "stopped_slice": "slice-id",
  "slice_results": [],
  "recommended_next_action": "..."
}
```

状态含义：

| impl guard status | runner 默认动作 | runner reason |
|---|---|---|
| `passed` | 默认停止；仅 `continue-on-passed` 可继续 | `slice_passed` |
| `review_needed` | 停止，等 Hermes 双 review | `slice_review_needed` |
| `takeover_candidate` | 停止，等 Hermes 接管 | `slice_takeover_candidate` |
| `blocked_by_allowlist` | 停止，安全阻塞 | `slice_blocked_by_allowlist` |
| `failed` | 停止 | `slice_failed` |
| `unusable` | 停止 | `slice_unusable` |
| 未知状态 | 停止 | `unknown_slice_status` |

默认：

```text
review_needed = stop
```

原因：避免多个 slice 的 diff 混在一起。只有 Hermes review + 测试确认后，才进入下一个 slice。

### 8.4 continue policy

第一版支持两个策略，但默认只用安全策略：

#### `stop-on-review-needed`（默认）

```text
passed 停（默认，避免多个 slice diff 混在一起）
review_needed 停
takeover_candidate 停
blocked/failed/unusable 停
```

#### `continue-on-passed`（预留）

只有当 impl guard 返回 `passed` 且 verification 已通过时，才继续下一个 slice。

第一版不建议让 `review_needed` 自动继续。

### 8.5 runner 禁止行为

runner 不做：

- 不自动修复。
- 不自动 revert。
- 不自动 review。
- 不自动 commit/push/PR。
- 不自动部署/重启。
- 不并发执行 slice。
- 不跨 repo。
- 不解析 Markdown 生成 slice。
- 不让 Codex 自己改 stage plan。

### 8.6 runner 测试清单

新增：`tests/scripts/test_codex_stage_runner.py`

必测：

- plan file 不存在 -> `unusable`。
- invalid JSON -> `unusable`。
- duplicate slice id -> `unusable`。
- 空 slices -> `unusable`。
- slice1 `review_needed` -> runner 停，不跑 slice2。
- slice1 `failed` -> runner 停，不跑 slice2。
- slice1 `blocked_by_allowlist` -> runner 停，不跑 slice2。
- slice1 `takeover_candidate` -> runner 停，不跑 slice2。
- slice1 `passed` + `continue-on-passed` -> 跑 slice2。
- unknown impl guard status -> runner 停。
- runner stdout bounded JSON。
- raw/final 路径进入 slice result。

## Phase 9：terminal 工具层复用确认

这个阶段不重新设计 terminal policy。当前 `tools/terminal_tool.py` 已有：

- 裸 `codex-yuna exec` / `codex exec` foreground guidance。
- Codex review structured flags 检查。
- shell token / command position 解析。
- help/version 例外。

所以 Phase 9 只做复用确认和回归测试，不重复造新 policy。

### 9.1 需要确认的行为

- 裸 `codex-yuna exec ...` foreground 仍提示 background + `notify_on_complete=true`。
- 裸 `codex exec ...` foreground 仍提示 background + `notify_on_complete=true`。
- 裸 Codex review 缺 structured flags 仍报错。
- `python scripts/runtime/codex_review_guard.py ...` 是合法 review wrapper。
- `python scripts/runtime/codex_impl_guard.py ...` 不被误判成裸 Codex。
- `python scripts/runtime/codex_stage_runner.py ...` 不被误判成裸 Codex。
- 同一 shell command 里如果混入裸 Codex，例如 `python scripts/runtime/codex_impl_guard.py ... && codex-yuna exec ...`，仍触发裸 Codex foreground guidance。

### 9.2 测试策略

只补 regression tests，不重写工具层：

- `tests/tools/test_terminal_tool.py` 中补 wrapper 不误拦测试。
- 如果现有裸 Codex foreground 和 review structured flags 已有覆盖，不重复写同义测试。
- 若发现覆盖缺口，只补最小测试。

## Phase 10：端到端 smoke 计划

目标：证明 stage runner 确实按 slice 顺序工作，并在 review/失败/接管时停住。

### 10.1 使用 fake repo + fake Codex

不要用真实项目做 smoke。结构：

```text
/tmp/codex-stage-runner-smoke/
  repo/
    demo.py
    tests/
  fake_codex.py
  slices.json
  slice-1.md
  slice-2.md
```

fake Codex 根据 prompt/slice id 模拟：

- 正常改 allowed file。
- 返回 nonzero。
- 写 allowlist 外文件。
- 写 allowed file 后 flood/timeout。
- 正常结束但需要 review。

### 10.2 必测 smoke 场景

#### review_needed 停

```text
slice1 -> review_needed
slice2 -> 不执行
```

预期：

```json
{
  "status": "stopped",
  "reason": "slice_review_needed",
  "stopped_slice": "slice1"
}
```

#### failed 停

```text
slice1 -> failed
slice2 -> 不执行
```

#### blocked_by_allowlist 停

```text
slice1 -> blocked_by_allowlist
slice2 -> 不执行
```

#### takeover_candidate 停

```text
slice1 -> takeover_candidate
slice2 -> 不执行
```

#### passed 才可继续

仅在 `continue_policy=continue-on-passed` 时：

```text
slice1 -> passed
slice2 -> 执行
```

### 10.3 smoke 非目标

不测：

- 真实 Codex 模型质量。
- 并发。
- 自动 review。
- 自动修复。
- commit/push/PR。
- 部署/重启。

## Phase 11：最终执行顺序

实现时分轮推进，不一口气做完。

### 11.1 实现轮 1：`codex_impl_guard.py` 小补强

只处理当前 review 的非阻塞建议：

- 补 `src/../*.py` allowed-glob escape 测试。
- 补 JSON field/source flood 直接测试。
- 清理 `_classify_records` 中维护性小瑕疵。

验收：

```bash
python -m pytest tests/scripts/test_codex_impl_guard.py -q -o addopts=
```

### 11.2 实现轮 2：`codex_stage_runner.py` 最小版

只实现：

- 读 `--plan-file` JSON。
- 校验 schema。
- 顺序调用 `codex_impl_guard.py`。
- 默认 `review_needed` 停。
- 输出 bounded JSON。

验收：

```bash
python -m pytest tests/scripts/test_codex_stage_runner.py -q -o addopts=
```

### 11.3 实现轮 3：端到端 smoke tests

只用 fake Codex / fake repo。

验收：

```bash
python -m pytest tests/scripts/test_codex_stage_runner.py -q -o addopts=
```

### 11.4 实现轮 4：terminal policy 回归测试

只补缺口，不重写 terminal policy。

验收：

```bash
python -m pytest tests/tools/test_terminal_tool.py -q -o addopts=
```

### 11.5 每轮固定质量门

每轮都必须：

```text
focused tests
→ git diff --check
→ Codex packet-only review
→ Hermes review Codex findings
→ 清理 cache/artifact
→ 再决定是否进入下一轮
```

不 commit、不 push，除非用户另行授权。

## Phase 12：普通 Codex 入口强制路由

目标：解决“guard / runner 已存在，但其它会话仍裸跑 Codex 导致 diff flood”的接入缺口。

### 12.1 范围

只做最小强制路由：

- `tools/terminal_tool.py` 检测裸 `codex-yuna exec` / `codex exec` implementation 调用。
- 不论 foreground/background，只要不是受控 wrapper，就返回 `blocked`。
- 允许以下受控入口：
  - `python scripts/runtime/codex_stage_runner.py --plan-file <JSON>`
  - `python scripts/runtime/codex_impl_guard.py ...`
  - `python scripts/runtime/codex_review_guard.py ...`
- `codex-yuna exec --help` / `codex --version` 继续放行。
- read-only review 继续走已有 `codex_review_guard.py` / structured-output 检查。

### 12.2 非目标

第一版不做透明 prompt 改写，因为裸 prompt 缺少安全 contract：

- `allowed_files` / `allowed_globs`
- `dirty_baseline_policy`
- `verify_cmd_ids`
- `continue_policy`
- review-needed stop 边界

因此不能把任意 `codex-yuna exec "prompt"` 自动转换成 staged runner；必须先由 Hermes 生成 JSON stage plan。

### 12.3 验收

```bash
python -m pytest tests/tools/test_terminal_tool.py -q -o addopts=
python -m pytest tests/scripts/test_codex_stage_runner.py -q -o addopts=
git diff --check -- docs/plans/codex-impl-guard-safe-plan.zh-CN.md tools/terminal_tool.py tests/tools/test_terminal_tool.py
```

### 12.4 预期效果

- 其它会话如果继续裸跑写入型 Codex，会被 terminal tool 明确拦截。
- agent 必须改用 staged runner / impl guard，才允许 Codex 写代码。
- diff flood 从“事后收拾”前移到“入口拦截”。
- QQ/WebUI 能收到更短的中文报错提示，并带机器字段：
  - `code: codex_unguarded_impl_blocked`
  - `user_message_zh`
  - `technical_detail`
  - `recommended_action: use_codex_staged_implement`

## Phase 13：`codex_staged_implement` 正式 tool 入口（显式 scope）

目标：提供唯一正式的写入型 Codex staged implementation 入口，让 agent 不再手写 shell / JSON plan，也不再裸跑 `codex-yuna exec`。

### 13.1 输入 schema（第一版必须显式 scope）

第一版不做自动推断，调用方必须提供明确范围：

```json
{
  "workdir": "/workspace/repo",
  "task": "实现一个小阶段",
  "allowed_files": ["tools/terminal_tool.py"],
  "allowed_globs": ["tests/tools/test_terminal_tool.py"],
  "verify_cmd_ids": ["diff-check"],
  "continue_policy": "stop-on-review-needed",
  "dirty_baseline_policy": "require-clean"
}
```

约束：

- `workdir` 必须解析为 git repo root；返回 JSON 中记录 `resolved_workdir` 和 `git_head`。
- `allowed_files` / `allowed_globs` 至少一个非空；空 scope 返回 `rejected_scope`。
- `verify_cmd_ids` 只能来自预定义 registry，第一版只允许 `diff-check` / `none`；不接受任意 shell 字符串。
- `continue_policy` 第一版只允许 `stop-on-review-needed`，不自动跨 review 继续。
- `dirty_baseline_policy` 第一版只允许 `require-clean`；其它值返回 `unsupported_dirty_policy`。

### 13.2 allowlist canonicalization / containment

`codex_staged_implement` 必须在调用 runner 前做和 runner/impl guard 一致的 fail-closed 校验：

- 所有 `allowed_files` / `allowed_globs` 都按 repo root 解析为 repo-relative pattern。
- 拒绝绝对路径、空字符串、`.`、`/`、`*`、`**/*`、`**/*.py` 这类宽 scope。
- 拒绝包含 `..` 的路径或 glob。
- 拒绝 symlink escape：如果 allowed file 已存在且解析后不在 repo root 内，返回 `rejected_scope`。
- 拒绝 `.git`、`.hg`、`.svn`、`.venv`、`venv`、`node_modules`、cache/build/output 目录。
- `allowed_globs` 只能使用白名单模板或明确文件级 glob；不能作为全仓库兜底。
- Codex 结束后必须重新读取 git evidence，并把实际 changed files 与 resolved allowlist 比对；越界则返回 `blocked_by_allowlist`，不能把 runner 退出当成功。

### 13.3 dirty worktree policy

Phase 13 必须简单、硬：

- 任何 pre-existing tracked dirty、untracked、staged、conflicted、deleted、renamed、submodule/gitlink evidence 都返回 `dirty_worktree`。
- 不自动 `stash`、`reset`、`clean`、`checkout`、`restore`。
- 不因为 dirty 文件与 allowlist 不重叠就继续写；最多在 `recommended_next_action` 建议 clean worktree / isolated worktree。
- 返回 JSON 记录 `dirty_paths`、`dirty_count`、`dirty_baseline_policy`。

### 13.4 runner invocation boundary

正式 tool 只能通过 argv list 调用：

```text
python scripts/runtime/codex_stage_runner.py --plan-file <tmp-json> --raw-dir <tmp-dir>
```

禁止：

- shell 拼接裸 `codex-yuna exec` / `codex exec`。
- shell-level background / detached execution。
- 把 raw Codex stdout/stderr 直接放入 tool result。
- 在 repo 内写 prompt、plan、raw log、final file。

临时文件要求：

- 使用 collision-resistant temp dir，例如 `/tmp/codex-staged-implement-<random>/`。
- 权限尽量私有；raw log / prompt / final JSON 都在 repo 外。
- 返回路径可用于审计，但内容不自动注入主上下文。

### 13.5 output schema / 状态机

tool 不能把 runner 退出等同于成功。返回状态必须可机读：

```text
rejected_scope
unsupported_dirty_policy
dirty_worktree
runner_unusable
implementation_failed
blocked_by_allowlist
takeover_candidate
review_needed
verification_failed
ready_for_review
```

状态语义：

- `review_needed` 是硬停，表示 Codex 留下 allowlist 内候选 diff，需要 read-only review；不是完成。
- `ready_for_review` 只表示 tool 层证据完整、下一步应 review；不代表测试全过或可提交。
- `takeover_candidate` 表示有安全候选 diff 但 Codex completion 不可信，必须 Hermes 接管检查。
- `verification_failed` 表示 registry verification 没过或产生未允许 artifact。
- 所有非通过路径都必须带 `reason`、`recommended_next_action`。

最小返回字段：

```json
{
  "status": "review_needed",
  "reason": "slice_review_needed",
  "resolved_workdir": "/workspace/repo",
  "git_head": "...",
  "resolved_allowlist": {"files": [], "globs": []},
  "dirty_check": {"status": "clean", "paths": []},
  "stage_plan_path": "/tmp/.../plan.json",
  "raw_dir": "/tmp/.../raw",
  "runner_exit_code": 1,
  "changed_files": [],
  "stopped_slice": "slice-1",
  "next_required_action": "run_read_only_review"
}
```

### 13.6 非阻塞建议细化为实现约束

Codex 复审给出的非阻塞建议，在 Phase 13 实现时按以下口径落地，避免实现者自由发挥：

1. `verify_cmd_ids: ["none"]` 只允许低风险场景：
   - `none` 表示 tool 不执行额外验证命令，不代表无需验证。
   - 返回 JSON 必须写明 `verification_policy: "deferred_to_hermes"`。
   - `next_required_action` 不能是完成态，只能是 `run_read_only_review` / `run_hermes_verification`。
   - 若 task/scope 属于安全边界、权限、路径、删除、部署、依赖安装、持久化迁移等高风险类别，第一版应拒绝 `none` 或返回 `rejected_verify_policy`。

2. `ready_for_review` 与 `review_needed` 必须区分：
   - `review_needed`：runner/impl guard 的 slice 硬停信号，表示已有候选 diff，必须停下来做 read-only review。
   - `ready_for_review`：tool 自己完成证据收集和状态映射后的归一化状态，表示下一步可以进入 review；仍不是完成、不是测试通过、不是可提交。
   - 二者都不得被 agent 或 runner 解释为 success/completed。

3. 返回 JSON 字段级截断策略：
   - 任意 string 最大 4,000 chars。
   - 任意 list 最大 200 items。
   - 任意 dict 最大 80 keys。
   - runner stdout/stderr 预览最大 8,000 chars。
   - raw Codex/stdout/stderr 不进入 tool result，只返回 repo 外路径。
   - 截断时保留 `...[truncated]` 或计数字段，避免静默丢证据。

4. dry-run / metadata 读取必须只读：
   - 允许读取 `git rev-parse --show-toplevel`、`git rev-parse HEAD`、`git status --porcelain=v1 -z --untracked-files=all`、文件存在性和 `realpath`。
   - 禁止执行 hooks、安装、构建、测试、format、compile、package manager、或任何会写 repo/cache/temp 的命令。
   - 禁止在 repo 内写 prompt/plan/raw/final；所有临时产物使用 repo 外 collision-resistant temp dir。

### 13.7 tests

Phase 13 测试必须覆盖：

- tool schema：缺 `allowed_files`/`allowed_globs` 返回 `rejected_scope`。
- path validation：绝对路径、`..`、symlink escape、宽 glob、`.git` / cache 目录被拒。
- dirty policy：tracked/untracked/staged/conflict fixture 均返回 `dirty_worktree`，且不修改工作区。
- runner invocation：使用 fake runner，验证 argv list 包含 `--plan-file` / `--raw-dir`，没有裸 `codex-yuna exec`。
- output bounded：fake runner 大 stdout 不进入 result；只返回 bounded JSON。
- status mapping：runner `review_needed`、`blocked_by_allowlist`、`takeover_candidate`、malformed JSON、timeout 都映射正确。
- terminal integration：`terminal_tool.py` 继续拦裸 Codex；新增 tool 是唯一写入型入口。

验收命令候选：

```bash
python -m pytest tests/tools/test_codex_staged_implement_tool.py tests/tools/test_terminal_tool.py tests/scripts/test_codex_stage_runner.py -q -o addopts=
git diff --check -- tools/codex_staged_implement_tool.py tests/tools/test_codex_staged_implement_tool.py tools/terminal_tool.py docs/plans/codex-impl-guard-safe-plan.zh-CN.md
```

## Phase 14：`dry_run_plan` 只生成计划，不执行 Codex

目标：先提供 plan proposal 能力，但不写 repo、不跑 Codex。

### 14.1 行为

```json
{
  "mode": "dry_run_plan",
  "workdir": "/workspace/repo",
  "task": "..."
}
```

返回：

- proposed stage plan JSON。
- risk classification：`low` / `needs_scope` / `unsupported`。
- proposed allowlist 和理由。
- `needs_user_confirmation: true`。

硬规则：

- 不调用 `codex_stage_runner.py`。
- 不调用 `codex_impl_guard.py`。
- 不调用 `codex-yuna` / `codex`。
- 不在 repo 内写文件。
- 只可读取必要 metadata，例如 repo root、HEAD、status、已知文件是否存在。
- scope 不确定时返回 `needs_scope`，不能猜。

### 14.2 tests

- `dry_run_plan` 不产生 repo diff。
- 不调用 runner / Codex（fake spy 断言）。
- uncertain task 返回 `needs_scope`。
- 返回 plan 可被 `codex_stage_runner.py` schema 校验接受（可用 fake repo/plan validator）。

## Phase 15：保守自动 stage plan 推断

目标：只对低风险、已知模板生成 allowlist；第一版仍不能静默执行 inferred plan。

### 15.1 推断硬规则

- 自动推断默认只用于 `dry_run_plan`。
- inferred plan 进入执行必须有显式 `mode: execute_inferred` 或用户/agent 确认信号；不能在同一次默认调用中直接执行。
- 禁止宽 glob：`**/*`、`**/*.py`、`tools/**`、`tests/**`、repo root glob 全部拒绝。
- docs-only 必须来自用户/任务明确点名的 docs path；不能自动放开 `docs/plans/*.md`。
- 一个 slice 只覆盖一个组件组；跨组件任务返回 `needs_split`。
- dirty worktree 仍按 Phase 13 默认 fail-closed。

### 15.2 第一批允许模板

仅允许以下小模板：

| task signal | allowed_files |
|---|---|
| terminal tool / terminal policy | `tools/terminal_tool.py`, `tests/tools/test_terminal_tool.py` |
| process registry Codex metadata | `tools/process_registry.py`, `tests/tools/test_process_registry.py` |
| stage runner | `scripts/runtime/codex_stage_runner.py`, `tests/scripts/test_codex_stage_runner.py` |
| impl guard | `scripts/runtime/codex_impl_guard.py`, `tests/scripts/test_codex_impl_guard.py` |
| review guard / packet | 明确二选一：`scripts/runtime/codex_review_guard.py` + 对应 test，或 `scripts/runtime/codex_review_packet.py` + 对应 test |
| docs-only | 用户明确点名的单个 docs path |

所有模板都必须验证文件存在；不存在时返回 `needs_scope` 或 `unsupported_template`。

### 15.3 tests

- 每个模板 positive case 生成最小 allowlist。
- 模糊任务返回 `needs_scope`。
- 跨组件任务返回 `needs_split`。
- 宽 glob 永远不出现在 output。
- inferred plan 默认不可执行；必须显式确认后才允许进入 Phase 13 execution path。

## Phase 13–15 共同非目标

- 不透明转换任意裸 `codex-yuna exec "prompt"`。
- 不自动扩大 allowlist。
- 不自动 commit / push / PR / restart / deploy。
- 不自动 stash / reset / clean / revert。
- 不把 raw Codex output 注入主上下文。
- 不替代 read-only Codex review 和 Hermes verification。

## 验收标准

第一阶段完成时，必须能证明：

- 单次 impl guard 输出 bounded JSON。
- raw Codex 输出不进主上下文。
- dirty baseline 默认 fail-closed。
- allowlist 越界 fail-closed。
- untracked/deleted/renamed/submodule evidence 不遗漏。
- flood/timeout/kill 不被当作成功。
- safe candidate diff 返回 `takeover_candidate`，不是 `passed`。
- 现有 review guard 和 terminal guard 不退化。

## 当前仓库注意事项

当前工作区已有多处未提交改动。实现前不要在当前 dirty tree 中直接跑 Codex implementation。建议：

- 使用 clean worktree / 临时 fixture repo 测试 wrapper。
- 或先完成当前 review guard 相关改动的归档/提交，再进入 impl guard 阶段。

本计划本身只新增文档，不代表允许开始实现。
