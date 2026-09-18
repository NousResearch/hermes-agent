# Skill Precipitation Validator — 变更日志

## 分支
`feat/skill-precipitation-validator`

## 日期
2026-08-07（新增：自愈修复循环）
2026-08-11（逃逸扫描加固：`/dev/null` 白名单 · cp/mv 按目标判定 · rm 旗标容错）
2026-08-11（响应解析修复：三层降级解析，消除「执行成功却报 failed」误报）
2026-08-11（分支恢复：`hermes update` autostash 导致工作滞留于 `stash@{0}`，已恢复）

## 概述
在 Background Review 系统沉淀 skill 之后，自动 fork 一个**沙箱验证子 agent**，在一次性临时工作区内**真实执行**新沉淀的 skill 的步骤，验证其可用性。**若执行失败，验证器会通过 `skill_manage` 就地修复该 skill（这是唯一允许越出沙箱的写入），并重新执行，最多迭代 3 次**，直到可用或确认失败。除此之外，验证全程不修改真实环境状态。

## 改动文件

### `agent/background_review.py`

#### 新增常量
- `_VERIFY_ACTIONS` — `{"create", "patch", "edit"}`，需要验证的 skill_manage 操作集合
- `_VERIFY_MAX_ITERATIONS` — `32`，启用修复时验证 fork 的工具迭代上限（容纳多次「诊断→修复→重执行」）
- `_VERIFY_READONLY_MAX_ITERATIONS` — `12`，关闭修复时的迭代上限（保持原只读验证成本）
- `_VERIFY_MAX_FIX_ATTEMPTS` — `3`，诊断→修复→重执行的最大循环次数
- `_VERIFY_PROMPT` — 真实执行验证提示词模板（占位符 `{skill_name}`/`{action_label}`/`{scratch}`/`{max_fix_attempts}`），含 REPAIR 修复段落
- `_VERIFY_VERDICTS` — `("VERIFIED", "FAILED", "UNABLE")`，响应解析的裁决令牌（2026-08-11 新增）
- `_VERIFY_DEFAULT_DETAIL` — 裸令牌（模型只回 `VERIFIED:` 不带理由）时的默认详情文案（2026-08-11 新增）
- `_VERIFY_FAIL_WORDS` / `_VERIFY_PASS_WORDS` — 无令牌时的措辞判据；失败词**优先**判定（2026-08-11 新增）
- `_VERIFY_NEGATED_FAIL` — `without error`/`no error`/`error-free` 等含失败词的**肯定**短语，扫描前剔除（2026-08-11 新增）

#### 新增函数

| 函数 | 说明 |
|------|------|
| `_extract_precipitated_skill_names(review_messages, prior_snapshot)` | 从 review fork 的消息中提取成功创建/修改的 skill 名称和操作 |
| `_path_escapes_sandbox(path, scratch)` | 判断路径是否逃逸沙箱（`~`/绝对路径/`..` 逃逸） |
| `_is_discard_target(target)` | 判断写目标是否为 `/dev/null` 丢弃设备（写入不产生状态，逃逸扫描豁免） |
| `_command_escapes_sandbox(command, scratch)` | 后置扫描：判断终端命令是否逃逸沙箱（cd 逃逸、写/删绝对路径、系统目录、`git config --global`） |
| `_scan_executed_commands_for_escape(session_messages, scratch)` | 遍历验证 agent 执行的命令，收集逃逸项 |
| `_analyze_skill_manage_activity(session_messages, expected_name)` | 后置扫描：检查验证 agent 的 `skill_manage` 调用，标记对**非目标 skill** 的写入或失败写入（违规），并判断是否发生过对目标 skill 的成功修复 |
| `_run_skill_verification(agent, skill_name, action, allow_repair=True)` | Fork 沙箱验证 agent，真实执行 skill；失败时按需修复并重执行，返回 `(status, detail)`（status ∈ verified/repaired/failed/unable） |
| `_verify_precipitated_skills(agent, review_messages, prior_snapshot)` | 编排器：提取 → 逐一沙箱验证（含修复）→ 返回可读结果（✅/🔧/❌/⚠️） |
| `_bg_review_aux_bool(key, default)` | 统一的 `auxiliary.background_review.*` 布尔配置读取（字符串 `false/off/0/no` 解析为 False） |
| `_skill_verification_enabled(agent)` | 配置开关：`auxiliary.background_review.verify_skills`（默认 true），基于 `_bg_review_aux_bool` |
| `_skill_repair_enabled(agent)` | 配置开关：`auxiliary.background_review.repair_skills`（默认 true） |
| `_verification_texts(session_messages)` | 收集验证 fork 的**全部** assistant 文本（最新在前），供解析使用（2026-08-11 新增） |
| `_parse_verification_result(session_messages)` | 三层降级解析裁决：严格前缀 → 任意位置令牌 → 措辞判定，返回 `(status, detail)`（2026-08-11 新增） |

#### 修改函数

| 函数 | 改动 |
|------|------|
| `_run_review_in_thread` | 在 action summary 输出之后，新增 skill verification 代码块 |
| `_run_skill_verification` | 新增 `allow_repair` 参数；条件迭代上限；关闭修复时白名单剔除 `skill_manage` 并追加禁用提示；注入 `max_fix_attempts`；`_analyze_skill_manage_activity` 结果参与判定；违规/逃逸优先于 agent 自报结果；返回 `repaired` 状态 |
| `_skill_verification_enabled` | 重构为基于 `_bg_review_aux_bool` 的薄封装（行为不变） |
| `_command_escapes_sandbox` | 逃逸扫描加固（2026-08-11）：`/dev/null` 豁免；`cp`/`install` 按目标（最后一个参数）判定、`mv` 同时查源与目标、`rm` 检查所有目标；`rm` 旗标容错（`-fr`/`-rfv`/`--force`）；系统目录检测并入 op 循环 |
| `_run_skill_verification`（响应解析） | 2026-08-11：删除内联的严格前缀解析块，改用 `_parse_verification_result()`——原实现只读**最后一条** assistant 消息且强制要求 `VERIFIED:` 前缀，导致「真实执行成功但用自然语言总结」被误报为 failed |

#### 其他改动
- 导入 `Tuple`、`re`、`shutil`、`tempfile` 类型/模块
- 更新 `__all__` 导出列表（新增 `_analyze_skill_manage_activity`/`_skill_repair_enabled`/`_VERIFY_MAX_FIX_ATTEMPTS`；2026-08-11 追加 `_is_discard_target`）

## 实现细节

### 验证流程（沙箱真实执行 + 自愈修复）
1. Background Review fork 完成后，`summarize_background_review_actions()` 输出 action summary
2. `_extract_precipitated_skill_names()` 扫描 review 消息，找到所有 `skill_manage(action="create"/"patch"/"edit")` 调用
3. 比对 tool result，排除失败的写操作和已存在于 prior_snapshot 中的旧消息；按 skill name 去重
4. 对每个 skill 调用 `_run_skill_verification()`：
   - `tempfile.mkdtemp()` 创建一次性 scratch 目录
   - fork `AIAgent`（迭代上限 32，继承主 agent 模型）
   - `set_session_cwd(scratch)` 钉死工作目录 → terminal 命令默认在 scratch 内运行
   - 工具白名单：`["skills", "terminal", "file"]`（skill_view + skill_manage + 真实执行能力）
   - 危险命令 auto-deny（复用 `_subagent_auto_deny`：rm -rf /、git push --force 等）
   - 验证 agent 按 skill 步骤在 scratch 内真实执行（git skill → `git init` 临时仓库）
   - **失败 → 修复循环**：诊断根因 → `skill_manage` 就地 patch/edit/write_file/remove_file 真实 skill → 沙箱内重执行 → 重复至多 3 次
   - 后置扫描 ×2：
     - `_scan_executed_commands_for_escape()` — 终端命令逃逸（`cd ~`、写 `/etc`、`git config --global` 等）
     - `_analyze_skill_manage_activity()` — skill_manage 是否只写了目标 skill；写入其他 skill 或写入失败 → 违规
   - 违规/逃逸 **优先于** agent 自报结果，一律判定失败
   - 解析 `VERIFIED:`/`FAILED:`/`UNABLE:` 响应
   - 删除 scratch 目录
5. 输出结果：`✅` 验证通过 / `🔧` 修复后通过（曾失败，已修复）/ `⚠️` 无法沙箱验证 / `❌` 验证失败（含逃逸、违规、修复次数用尽）

### 关键设计决策
- **真实执行**：验证 agent 真的运行 skill 的命令，而非仅目测 SKILL.md
- **自愈修复**：验证失败的 skill 由验证器就地修复并重执行，目标状态是「可用」，而非只报告
- **修复 = 唯一的越沙箱写入**：除目标 skill 文件（经 `skill_manage`）外，沙箱内一切照旧，不改真实环境状态
- **修复写入的护栏**：`_analyze_skill_manage_activity` 后置验证只写了目标 skill；写入其他 skill → 违规 → 判定失败
- **沙箱隔离**：一次性 scratch 目录 + cwd 钉死 + 危险命令 auto-deny + 双重后置扫描，真实环境零修改（除被修复的目标 skill）
- **同一模型**：验证 agent 使用主 agent 的 model/provider，而非 review fork 可能被路由到的 aux 模型
- **git skill 特例**：在 scratch 内 `git init` 全新临时仓库，所有 branch/tag/commit 都发生在那里；禁止 `git config --global`
- **最佳努力**：验证失败仅报告，不回滚；修复不改变 skill 的归属/provenance
- **可关闭**：`auxiliary.background_review.verify_skills: false`（整个验证）/ `repair_skills: false`（只验证不修复）

### 诚实的安全边界（软沙箱）
- 终端逃逸扫描是启发式的（正则匹配 cd/写/删/重定向/系统目录/global git config），不是防弹容器
- 修复写入通过 `skill_manage` 的既有护栏（pinned/external/bundled/hub skill 的写保护），叠加我们自己的目标扫描；但 `skill_manage` 语义正确性（例如 patch 改错了地方但语法正确）无法被机械判定，只能靠修复后重执行来兜底
- 对「agent 自己刚写的 skill」这一高信任场景足够；将来可升级 Docker 硬沙箱

### 逃逸扫描加固（2026-08-11 增量改进）

对启发式扫描做了三类修正，消除误报与漏报（全部经实证用例核对）：

1. **`/dev/null` 白名单**（消除误报）：`2>/dev/null`、`tee /dev/null`、`> /dev/null` 不再被判逃逸——写丢弃设备不产生状态。但豁免是**按目标**的：`rm -rf /tmp/leak.bin 2>/dev/null` 里真正的逃逸目标照常拦截。
2. **cp/mv 按目标判定**（消除漏报）：旧实现只查动词后第一个参数（=源），`mv out.log /var/tmp/x.log`、`cp out.log /etc/passwd` 这类**目的地逃逸**全部漏网；同时 `cp /etc/passwd .` 又因源在系统目录被误报。现按语义区分——`cp`/`install`/`ln -s` 只查目的地（最后一个参数，源只读），`mv` 同时查源（会被删除）与目的地，`rm`/`rmdir`/`touch`/`tee` 检查所有目标。
3. **rm 旗标容错**（消除漏报）：旧正则只认 `rm -rf`，`rm -fr`、`rm -rfv`、`rm -rf --force /etc/passwd` 全部漏网；现以 `rm(?:\s+-{1,2}[A-Za-z0-9]+)*` 兼容任意旗标写法。

影响权衡：误报（`/dev/null` 清理 idiom 被误杀 → 合格 skill 验证失败）与漏报（mv/cp/rm 旗标逃逸被放行 → 真实状态修改静默穿透安全网）都修了；其中漏报危害更大，是优先方向。

### 响应解析修复（2026-08-11 增量修复）

**缺陷（E2E 实测复现）**：验证 fork 在沙箱内**真实执行成功**，但回复用自然语言总结：

```
All four steps executed successfully: 1. `git status` — showed pending changes
(modified README.md, untracked feature.txt), 2. `git add -A`, 3. `git commit …`
```

旧解析**只读最后一条 assistant 消息**且要求严格以 `VERIFIED:` 开头 → 落入 `else` 分支 → 报
`❌ failed — unexpected response format: All four steps executed successfully: …`。
即「技能明明跑通了，成功结果却被丢弃」。次生缺陷：verdict 出现在**更早**的消息里时同样丢失。

**修复**：新增 `_parse_verification_result()`，扫描**全部** assistant 消息（最新在前），三层降级：

1. **严格**：消息以 `VERIFIED:` / `FAILED:` / `UNABLE:` 开头 → 直接取该状态与理由
2. **宽容**：令牌出现在消息**任意位置**（模型常先写一段说明再给 verdict）
3. **措辞**：无令牌时按提示词措辞判定——`successfully` / `completed` / `all steps` / `ran clean` 等 → verified；`fail` / `missing` / `not found` / `could not` / `cannot` 等 → failed
4. 仅当措辞亦无信号时，才回退 `unexpected response format`

**两个细节**：
- **否定短语豁免**：`ran without errors` 含失败词 `error`，故扫描前先剔除 `without error` / `no error` / `error-free`，避免成功表述被读成失败信号
- **失败词优先**：歧义时判 failed——把坏技能误报为 verified 是代价更高的方向

**兼容性**：外层判定顺序不变（违规/逃逸 → unable → verified/repaired → failed），`_analyze_skill_manage_activity` 的越权降级仍优先于 agent 自报结果。

**验证**：单元测试 40/40（新增 10 个解析用例）；同一个 git skill 的 E2E 结果由 `failed` 转为 `verified`，detail 为真实执行结果。

## 测试方法

### 单元测试
```bash
python tests/manual/test_skill_verification.py
```
40 个场景全部通过：提取（create/patch/delete/failed/stale/dedup/多 skill）、沙箱逃逸检测（cd ~、cd /etc、cd ..、写 ~、rm 系统目录、git config --global、安全命令不误报、命令扫描；2026-08-11 追加：`/dev/null` 不误报、`/dev/null` 豁免不掩盖真实逃逸、mv 目的地逃逸、cp 目的地逃逸、cp 源只读不误报、mv 源删除判逃逸、rm 旗标变体命中）、修复循环（`_analyze_skill_manage_activity`：无写入/修复成功/写入他 skill/写入失败/remove_file 修复、提示词格式化含 REPAIR 段）、**响应解析**（2026-08-11 追加 10 例：自然语言成功总结的回归用例、严格 VERIFIED、令牌居中、严格 FAILED、UNABLE、无语义令牌的措辞失败、`ran without errors` 否定豁免、verdict 位于更早消息、无任何信号回退、空消息）。

### 现有回归测试
```bash
uv run --with pytest pytest tests/run_agent/test_background_review.py \
  tests/run_agent/test_background_review_summary.py \
  tests/test_background_review_list_shapes.py \
  tests/test_background_review_session_isolation.py \
  tests/run_agent/test_background_review_toolset_restriction.py \
  tests/run_agent/test_background_review_cost_controls.py \
  tests/run_agent/test_background_review_cache_parity.py -q
```
39 个测试全部通过（按 `pytest --collect-only` 实际收集数；早期记录误写为 35，一并纠正）。

### 语法/导入检查
```bash
python -c "import ast; ast.parse(open('agent/background_review.py').read())"
uv run python -c "import agent.background_review"
```

### 端到端手动测试
1. `experimental setup` 配置 API Key
2. `experimental chat` 启动
3. 触发 skill 沉淀（多次工具调用 + 学习信号）
4. 预期输出：
   ```
   💾 Self-improvement review: Skill 'xxx' created
   🔍 Skill verification: ✅ xxx: <真实执行结果>
   ```
5. 若 skill 有缺陷，预期看到 `🔧 xxx: repaired — <修复内容>`（验证器自动修复后通过）

### 端到端实测（`tests/manual/e2e_skill_verification.py`）

不依赖完整的 review 触发流程，以最小 parent agent 直接调用 `_run_skill_verification()`，真实发起 LLM 调用：

```bash
cd ~/.hermes/hermes-agent
PYTHONPATH="" venv/bin/python tests/manual/e2e_skill_verification.py <skill_name> create [true|false]
```

实测结果：

| 场景 | 模式 | 结果 |
|------|------|------|
| `git-quick-commit`（健康 skill） | 只读（`repair=false`） | `verified` — 沙箱内 `git init` 并执行 4 步，产出提交 |
| `broken-skill-test`（步骤引用不存在的 `scripts/setup.sh`） | 修复（`repair=true`） | `repaired` — 诊断缺失 → `skill_manage(write_file)` 创建脚本 → 沙箱内重执行通过 |

隔离性经核实：`/tmp/hermes-skill-verify-*` 无残留；验证过程中非目标 skill 未被写入。

- 重放原始触发消息
- 验证失败自动回滚（连同修复的改动一起回滚）
- 变更 manifest / audit ledger（记录验证器对 skill 的每一次修复写入）
- 修复写入的语义级 diff 审查
- Docker 硬沙箱升级
- 并行验证多个 skill

## 未实现（后续 PR）
