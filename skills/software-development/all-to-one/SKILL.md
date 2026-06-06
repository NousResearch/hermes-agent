---
name: all-to-one
version: 2.0.0
description: |
  All To One（总整理）是项目记忆压缩协议：在一次 AI 协作的修复、开发、部署、迁移、排错或阶段完工后，把混乱过程沉淀成未来 5-10 分钟可接回的项目记忆。它不是普通总结，而是生成可验证、可复用、可学习、可交接的文档：what changed, why it worked, what broke, how it was verified, what risks remain, and how to resume.
triggers:
  - "总整理"
  - "All To One"
  - "A2O"
  - "复盘一下这个项目"
  - "完工后整理"
  - "阶段做完了整理"
  - "帮我梳理这次修了什么"
  - "保存这次项目理解"
  - "下次重开不用重新读"
  - "生成项目记忆"
  - "Codex 总整理"
  - "在 Codex 里用 All To One"
  - "读取全部上下文整理"
mutating: true
---

# All To One（总整理）

## One-Line Promise

把混乱的 AI 编程/排错过程，变成下次人或 agent 5-10 分钟能接回来的项目记忆。

All To One turns chaotic AI coding sessions into durable project memory: what changed, why it worked, what broke, how it was verified, what risks remain, and how to resume in 5 minutes.

## Contract

使用本 skill 时，必须产出一份能让未来用户、未来 agent、团队新人快速接手的“项目总整理”。它保证：

- **可接回**：未来不需要重新扫完整聊天记录或全仓库，也能知道从哪里继续。
- **可验证**：清楚区分工具验证、截图确认、观察到的事实、推断和未验证项。
- **可学习**：解释关键原理，让用户不只是知道“怎么点”，还知道“为什么”。
- **可复用**：保留错误路线、根因、判断标准和恢复路线，避免下次重踩。
- **可交接**：文件、命令、配置、服务、数据状态、风险、下一步都有明确位置。

## Core Identity

All To One 不是：

- 聊天摘要
- 项目流水账
- TODO 清单
- 漂亮周报
- “AI 做了很多事”的包装文案

All To One 是：

```text
项目记忆压缩协议 = 下一轮 AI 启动上下文 + 用户学习笔记 + 团队交接文档 + 未来排错地图
```

它要回答 7 个问题：

1. 我们原来到底要解决什么？
2. 真实过程中发生了什么？
3. 哪些判断错了，为什么后来改方向？
4. 最后系统变成了什么状态？
5. 为什么这个修法/架构/配置有效？
6. 什么已经验证，什么只是推测？
7. 下次重开，怎样 5-10 分钟接回来？

## Codex / External Agent Usage

All To One 必须能脱离 Hermes，在 Codex、Claude Code、Cursor、GitHub Copilot CLI 等 coding agent 中使用。

### What Codex Should Receive

在 Codex 里使用时，把本 skill 当成一份“项目总整理协议”。Codex 应优先读取：

1. 当前对话上下文：用户目标、约束、报错、关键判断、已做操作。
2. 当前仓库状态：`git status`、`git diff`、最近提交、关键配置。
3. 实际验证结果：测试、构建、lint、启动、健康检查。
4. 项目文档：`README.md`、`AGENTS.md`、`CLAUDE.md`、`docs/`、部署说明。
5. 运行环境信息：OS、包管理器、服务、Docker、数据库、端口、外部依赖。

### Read-All-Context Principle

用户说“总整理 / All To One”时，默认允许 agent **读取所有必要上下文** 来整理项目记忆。

这里的“所有上下文”不是无限制吞全仓库，而是：

- 可以读取当前对话中可见的全部内容。
- 可以读取仓库中与本次任务相关的文件、diff、日志、测试输出、文档。
- 可以读取项目已有记忆文件，例如 `docs/{project-name}-a2o.md`、`docs/all-to-one.md`、`docs/handoff.md`、`docs/PROJECT_MEMORY.md`。
- 可以检查 git 状态和关键入口，避免漏掉真实改动。
- 不应该读取无关大文件、依赖目录、构建产物、隐私文件、二进制文件来凑完整。

如果上下文太大，先做“上下文索引”：列出已读取来源、未读取但可能相关来源、为什么跳过。

### Codex Execution Steps

在 Codex 中执行 All To One 时，按这个顺序：

```text
1. Identify scope
   - Determine project/task name, mode, current completion state.

2. Inventory context
   - Read current conversation if available.
   - Run git status/diff/log.
   - Inspect README/AGENTS/CLAUDE/docs when present.
   - Identify changed files and config entry points.

3. Verify current state
   - Run the cheapest meaningful checks first.
   - Prefer existing project scripts: test/lint/build/smoke.
   - If unable to verify, mark [blocked] or [unverified].

4. Reconstruct timeline
   - Goal → attempts → errors → decisions → final state.

5. Extract root causes and principles
   - Explain why the issue happened and why the fix works.

6. Produce All To One document
   - Use the selected mode template.
   - Add project-type checklist sections when relevant.

7. Store or print
   - Before writing files, determine output basename.
   - Default basename = current project/repo/folder name, e.g. `talent-system` → `docs/talent-system-a2o.md` + `docs/talent-system-a2o.docx`.
   - If project name cannot be determined or multiple scopes exist, ask user for filename before output.
   - If user explicitly requested a path/name, use that path/name.
   - Otherwise print in chat only.
```

### Portable Codex Prompt

Use this prompt inside Codex when the skill file is not installed:

```text
You are using the All To One project memory compression protocol.
Your job is not to summarize chat. Your job is to create durable project memory for future humans and agents.

Read all necessary context: conversation, git status, git diff, recent commits, README/AGENTS/CLAUDE/docs, key config, tests/build output, and relevant logs. Do not blindly read dependency folders, build artifacts, secrets, binaries, or unrelated large files.

Produce a document that explains:
1. what the original goal was,
2. what actually happened,
3. what changed in files/config/system state,
4. what bugs/errors appeared,
5. the root causes,
6. why the final solution works,
7. what was verified and what was not,
8. what risks remain,
9. how a future agent can resume in 5-10 minutes.

Use evidence tags:
[verified] command/test/tool output proves it
[screenshot] user screenshot proves it
[observed] observed in conversation or session
[inferred] reasoned but not directly verified
[unverified] not checked yet
[blocked] could not check due to blocker

Never turn inferred/unverified claims into verified facts.
Prefer project-named outputs for persistent files: `docs/{project-name}-a2o.md` and, when user wants a readable handoff, `docs/{project-name}-a2o.docx`. Example: `talent-system` → `docs/talent-system-a2o.md`. If project name is unclear, ask before writing. If the project already has a memory/handoff document, update it only when user confirms it is the right canonical file; otherwise create project-named output to avoid generic `All To One.docx` clutter.
```

### Recommended Repository Files For Codex

For a public or reusable Codex setup, include:

```text
.all-to-one/SKILL.md          # full protocol
.all-to-one/PROMPT.md         # portable prompt
.all-to-one/templates/quick.md
.all-to-one/templates/standard.md
.all-to-one/templates/handoff.md
.all-to-one/templates/deep.md
docs/all-to-one.md            # generated project memory, if project-specific
```

If the repository uses `AGENTS.md`, add a short instruction:

```md
## All To One
When the user says "总整理", "All To One", "A2O", or asks for handoff/project memory, read `.all-to-one/SKILL.md` and generate/update `docs/all-to-one.md`. You may inspect all task-relevant context: conversation, git diff/status/log, docs, configs, test outputs, and logs. Mark every key claim with evidence tags: [verified], [observed], [inferred], [unverified], [blocked].
```

## Modes

根据任务规模选择模式。用户没有指定时，默认 `standard`。

### quick

适合小修复、小配置、单一 bug。目标是 10-25 行内说清楚。

必须包含：

- 一句话结论
- 改了什么（代码层要有路径+行号/diff hunk）
- 根因
- 验证结果
- 下次怎么查

### standard

默认模式。适合一次完整修复、部署、安装、迁移、功能开发。

必须包含完整输出模板中的 10 个章节，但每节尽量短；优先表格、列表、路径、命令，少写散文。

### deep

适合用户想学习原理、大型项目阶段复盘、反复踩坑任务。

在 `standard` 基础上追加学习内容，但默认仍保持 **caveman-lite 压缩风格**：短句、少形容词、少铺垫，只保留能指导判断的原理。

追加：

- 原理教学区
- 错误心智模型 vs 正确心智模型
- 关键概念解释
- 如果从零再做一遍，最佳路线是什么

### handoff

适合交给下一位 agent、同事、外包、未来自己。

在 `standard` 基础上更强调：

- 当前状态快照
- 最短启动路径
- 禁止触碰区域
- 风险和回滚
- 接手人第一小时行动清单

### archive

适合项目已结束，只做长期存档。

重点压缩：

- 最终状态
- 关键决策
- 关键路径
- 历史坑点
- 未来恢复入口

## Evidence Levels

所有关键事实，尤其是“已完成”“已修复”“已验证”，必须标记证据等级。

使用这些标签：

```text
[verified]   工具/命令/测试真实跑过，输出支持结论
[screenshot] 用户截图或界面现象确认
[observed]   当前对话/操作过程中明确观察到
[inferred]   基于现象推断，但未直接验证
[unverified] 尚未验证，不能当成事实
[blocked]    因权限、网络、缺设备、缺信息等无法验证
```

规则：

- 不能把 `[inferred]` 写成 `[verified]`。
- 没有实际验证，就必须写 `[unverified]` 或 `[blocked]`。
- 如果用户截图证明了结果，标 `[screenshot]`，不要假装是 agent 亲自验证。
- 验证失败也要记录，它是未来排错的重要线索。

## When To Use

必须触发：

- 用户说“总整理”“All To One”“A2O”“复盘”“完工整理”“阶段总结”。
- 一个技术项目经过多轮排错、安装、配置、编码、迁移、部署后已经能跑或阶段收敛。
- 用户担心记忆体丢失、项目重开、重新理解浪费 token。
- 本轮任务包含大量截图、命令、错误、配置、文件改动，未来很可能复用。
- 用户想通过复盘学习背后原理。
- 准备交接给未来 agent、同事、团队新人。

不要用于：

- 普通一句话摘要。
- 只记录 TODO。
- 还没形成稳定阶段结果的混乱进行中任务，除非用户明确要求“阶段性总整理”。
- 纯代码审查；用 `review`。
- 单纯保存上下文；用 `context-save`。
- 单纯同步项目文档和记忆；用 `neat-freak`。

## Workflow

### 1. Scope：先定范围

确认或推断：

- 项目/系统名。
- 时间范围：当前会话、本阶段、整个项目。
- 模式：quick / standard / deep / handoff / archive。
- 目标读者：未来自己、未来 agent、团队同事、外包、新人。
- 输出位置：聊天内 / 项目文档 / 用户指定路径。

低风险默认：如果用户只说“总整理”，默认整理当前会话或当前项目的已完成阶段，并先在聊天内输出；写入仓库文件前按用户意图执行，若不明确则先问。

### 2. Evidence：收集事实证据

优先用真实上下文和工具输出，不靠记忆编造。

应收集：

- 原始目标和约束。
- 关键截图、报错、日志。
- 实际执行过的命令。
- 修改过的文件、配置、服务、分区、依赖、环境变量。
- 测试、健康检查、启动验证、用户确认。
- 错误路线、误判、回滚、删除项。

若在代码仓库中，按需检查：

- `git status`
- `git diff`
- `git log --oneline -n 5`
- 关键配置文件
- 测试/构建/健康检查输出

不要为了“看起来完整”无限扫全仓库。All To One 的目标是压缩认知，不是重新吞一遍上下文。

### 3. Timeline：还原真实过程

用“真实过程”，不要写理想流程。

必须保留：

- 起点状态。
- 第一轮尝试。
- 报错/异常现象。
- 哪些判断被推翻。
- 哪个证据改变了方向。
- 最终收敛路径。

写法：

```text
步骤 → 现象 → 当时判断 → 后来发现 → 最终处理
```

### 4. Changes：列出关键改动

按类别写清楚，尤其代码层必须具体：

- 文件/代码：路径、函数/组件、改动前后、行号或 diff hunk、原因、影响。
- 配置/系统：服务、配置项、权限、端口、启动项。
- 数据/外部状态：数据库、Docker volume、EFI/NVRAM、对象存储、第三方平台。
- 删除/弃用：临时文件、旧配置、错误入口。

代码改动最低标准：

```text
- `path/to/file.ts:120-145`：`functionName()` 从 A 改成 B；原因：C；影响：D。
```

如果有 git diff，必须优先用 diff hunk：

```text
@@ path/to/file.ts:120 @@
- old important line
+ new important line
说明：为什么这几行关键。
```

如果没有 git diff 或项目未纳入 git，必须明确写：`[blocked] 无可用 git diff，无法证明行级代码改动`，不要用“改了一些代码”糊弄。

禁止空话：

- “优化了代码”
- “修复了一些问题”
- “调整了配置”

必须具体到对象、行号、函数或 diff。

### 5. Root Cause：写根因，不只写现象

每个关键 bug/坑点要拆成：

- 表现：用户看到什么。
- 直接原因：哪个文件/配置/命令导致。
- 底层机制：系统为什么会这样。
- 修复动作：具体怎么改。
- 识别方式：下次看到什么就知道是同类问题。

### 6. Principles：解释背后原理

用大白话解释：

- 这个系统本来怎么工作。
- 为什么原来的状态不对。
- 为什么这个方案有效。
- 换系统/版本/机器后，哪些地方可能不同。

deep 模式必须补充：

- 错误心智模型：用户/agent 一开始可能怎么误解。
- 正确心智模型：应该怎么理解。
- 可迁移原则：以后类似项目怎么判断。

### 7. Verification：记录验证和未验证

必须列出：

- 已验证：命令/截图/现象/测试结果。
- 未验证：原因、风险、后续怎么验证。
- blocker：权限、网络、设备、时间、依赖问题。

没有验证就明确写：

```text
[unverified] 未验证：原因是……
```

### 8. Resume：生成未来重开路径

这是 All To One 最重要的交付物。

必须提供：

- 未来 agent 第一步读什么。
- 不要重复读什么。
- 最短检查命令。
- 正常标志。
- 异常分支。
- 下一步优化入口。

格式：

```text
下次重开 5-10 分钟路径：
1. 先读……
2. 跑……确认状态
3. 如果看到……说明正常
4. 如果看到……走修复路线 A
5. 不要再……，因为……
```

### 9. Store：存档策略

优先产出 **项目命名 + 可直接打开** 的交付物，不要默认叫 `All To One.md/.docx` 或只叫 `all-to-one.md`。

文件命名规则：

1. 先确定项目名：优先 `package.json name`、git repo 名、当前目录名。
2. 默认 basename：`{project-name}-a2o`。
3. 默认输出：`docs/{project-name}-a2o.md` + `docs/{project-name}-a2o.docx`。
4. 如果项目名不清楚、多个项目混在一起、或用户要交给别人看：先问文件名/标题，不要自作主张。
5. 如果用户显式指定路径/文件名：用用户指定。
6. 如果已有旧文件 `docs/all-to-one.md`：只在确认它是 canonical 项目记忆时更新；否则新建项目命名文件，并可注明旧文件位置。

示例：

- `talent-system` → `docs/talent-system-a2o.md` + `docs/talent-system-a2o.docx`
- `sub2api` → `docs/sub2api-a2o.md` + `docs/sub2api-a2o.docx`
- 项目名不明 → 询问：`这份 A2O 文件名用什么？`

转换规则：

- 用户要“总结看问题 / 交给人看 / 及时打开”：同时生成 `.docx`。
- 如果当前环境有 `pandoc` / `python-docx` / LibreOffice / macOS `textutil`，必须实际转换并回读/检查文件存在；不能只说“可转换”。
- 如果无法生成 `.docx`，明确标 `[blocked]` 并给最短替代：`.md` + 转换命令。

同一项目优先维护同一个项目命名 A2O 文件，不要制造碎片。

如果项目有 README/CLAUDE.md/AGENTS.md，需要写“建议同步项”，但不要自动到处改，除非用户要求。

## Output Templates

### Quick Template

```md
# A2O Quick：{任务名}

## 结论
- {一句话说明当前状态} {证据等级}

## 改了什么
- `{path/config}`：{具体改动}，原因：{原因}

## 根因
- {问题表现} → {真实根因}

## 验证
- {验证动作} → {结果} {证据等级}

## 下次重开
1. 先看 `{path}`
2. 跑 `{command}`
3. 正常看到 `{signal}`
```

### Standard Template

```md
# All To One：{项目/任务名}

## 1. 一句话结论
{现在完成了什么，当前状态是什么。必须带证据等级。}

## 2. 背景和目标
- 原始目标：
- 关键约束：
- 最终采用路线：
- 未采用路线：

## 3. 最终系统状态
- 环境：
- 关键组件：
- 当前能工作的功能：
- 当前不能/未验证的功能：
- 禁止触碰区域：

## 4. 真实流程时间线
| 阶段 | 现象/动作 | 当时判断 | 后来确认 | 结果 |
|---|---|---|---|---|

## 5. 关键改动清单
### 文件/代码
- `{path}`：{改了什么}；原因：{为什么}

### 配置/系统
- `{service/config}`：{改了什么}；原因：{为什么}

### 数据/外部状态
- `{database/EFI/docker/etc}`：{状态变化}

### 删除/弃用
- `{old thing}`：{为什么不用了}

## 6. Bug、坑点和根因
| 问题 | 表现 | 根因 | 修复 | 下次怎么识别 | 证据 |
|---|---|---|---|---|---|

## 7. 背后原理大白话
{用用户能复述的语言解释系统机制。}

## 8. 验证记录
### 已验证
- {命令/截图/测试} → {结果} [verified/screenshot/observed]

### 未验证 / Blocker
- {事项} → {原因} [unverified/blocked]

## 9. 未来风险和优化方向
- 风险：
- 优化：
- 不建议做：

## 10. 下次重开 5-10 分钟路径
1. 先读：
2. 再跑：
3. 正常标志：
4. 异常分支：
5. 不要重复：
```

### Handoff Add-On

```md
## 接手人第一小时
1. {第一步}
2. {第二步}
3. {第三步}

## 红线
- 不要动：
- 不要删：
- 不要重装/重建：

## 回滚路线
- 如果失败，先：
- 再：
```

### Deep Add-On

```md
## 学习区：这次真正要理解的 3-5 个原理
1. {原理}：{大白话解释}；以后怎么迁移使用

## 错误心智模型 vs 正确心智模型
| 错误理解 | 为什么错 | 正确理解 |
|---|---|---|

## 如果从零再做一遍
1. {最佳路线}
2. {避坑点}
```

## Project-Type Checklists

根据项目类型追加专用检查项。

### Linux / Multiboot / GRUB

必须记录：

- 分区表：EFI、各系统根分区、数据分区、危险分区。
- 启动链路：BIOS/UEFI → NVRAM/启动项 → EFI 文件 → GRUB/systemd-boot → 系统。
- 修改过的文件：`/etc/grub.d/40_custom`、`/etc/default/grub`、EFI 路径。
- 关键命令：`lsblk -f`、`efibootmgr -v`、`find /boot/efi/EFI`、`update-grub`。
- 红线：不要格式化 EFI；不要误动 Windows BitLocker；不要把安装器入口当已安装系统入口。
- 恢复路线：从哪个系统进、怎么 chainload、怎么重新生成 GRUB。

### Docker / NAS Deployment

必须记录：

- 部署方式：Compose/UI/SSH/CI。
- `docker-compose.yml` 路径和关键服务。
- 镜像来源、构建方式、拉取限制。
- 端口、volume、env、网络、代理。
- 健康检查命令和日志位置。
- 回滚方式。
- NAS 特殊限制：是否只能单 yml、是否无 SSH、是否 Docker Hub 拉取失败。

### Web App / Next.js

必须记录：

- 包管理器和版本：npm/pnpm/yarn/bun。
- 入口：路由、API、数据库、鉴权。
- 改动文件和影响页面。
- 环境变量。
- 数据迁移。
- 验证：lint、test、build、smoke。
- 如果 smoke/login/protected pages 依赖数据库：先用最小 `select 1` 探测数据库连接，再跑 smoke；lint/build 通过只能证明代码层，不等于业务运行层已验收。细节见 `references/nextjs-db-smoke-checks.md`。
- 部署平台和回滚。

### Agent / Hermes Config

必须记录：

- Hermes profile。
- 修改的 config、skills、plugins、cron、gateway。
- 触发方式和工具链。
- 验证命令：`hermes tools`、相关日志、实际消息收发。
- 跨 profile 风险。
- 长期记忆 vs skill vs 项目文档的边界。

### Database / Migration

必须记录：

- 数据库类型、连接位置、schema 文件。
- 迁移命令和版本。
- 备份位置。
- 读写影响范围。
- 回滚策略。
- 验证查询。

### Bug Fix / Refactor

必须记录：

- 原 bug 复现方式。
- 根因文件/函数。
- 修复策略。
- 为什么没有选其他方案。
- 新增/修改测试。
- 回归风险。

## Before / After Example

Bad summary:

```text
修好了 openSUSE 启动问题，更新了 GRUB，现在能进系统。
```

Good All To One fragment:

```md
## 一句话结论
openSUSE 已安装在 `/dev/nvme0n1p5`，但 BIOS NVRAM 没写入启动项；最终通过 Ubuntu GRUB chainload `/EFI/systemd/shim.efi` 进入 openSUSE。 [screenshot]

## 根因
主板 NVRAM 写入时报 `No space left on device`，这不是硬盘满，而是 UEFI 启动项存储空间满，导致 openSUSE 安装器没能注册启动项。 [observed]

## 关键改动
- `/etc/grub.d/40_custom`：新增 `openSUSE Tumbleweed` menuentry，使用 EFI 分区 UUID `B5BB-052E` 定位 `/EFI/systemd/shim.efi`。
- 删除/弃用：DVD ISO 安装项已经不再需要，保留会误导用户再次进入安装器。

## 下次重开
1. 先确认 `/boot/efi/EFI/systemd/shim.efi` 是否存在。
2. 再看 `/etc/grub.d/40_custom` 是否有 openSUSE chainloader。
3. 跑 `sudo update-grub` 后重启。
4. 如果 shim 失败，改 chainloader 到 `/EFI/systemd/grub.efi`。
```

## Quality Bar

一份合格的 All To One 必须满足：

- 未来 agent 读完不用重新扫完整聊天和全仓库。
- 用户看完能讲清楚“为什么这样修”。
- 关键命令、路径、文件、服务、配置都能定位。
- 错误路线和误判被记录，避免下次重踩。
- 每个“已完成/已验证”都有证据等级。
- 有明确的 5-10 分钟重开路径。
- 有“不建议做/不要碰”的红线。

优秀标准：

- 可以直接作为项目 `docs/all-to-one.md` 长期维护。
- 可以直接贴给下一个 agent 当启动上下文。
- 可以作为用户自己的学习笔记。
- 可以交给同事/新人理解项目状态。

不合格表现：

- 只有“做了 A、做了 B、成功了”的流水账。
- 没有根因和原理。
- 没有文件路径/命令/验证证据。
- 把推测写成事实。
- 没有未来重开路径。
- 为了好看隐藏失败路线和未验证项。

## Anti-Patterns

- 不要写漂亮但空泛的项目总结。
- 不要只总结最终方案，删除中间踩坑信息。
- 不要机械压缩聊天记录；要提炼结构、证据和决策。
- 不要把 `[inferred]` 包装成 `[verified]`。
- 不要把临时任务进度写入长期 memory；项目内文档更合适。
- 不要每次都新建文档导致碎片化；同一项目优先更新现有总整理。
- 不要无限制读取全仓库来“补安全感”；只读取能支撑结论和重开路径的关键证据。
- 不要省略用户亲自完成的关键步骤；这些往往是未来恢复的关键。

## Relationship To Other Skills

- `context-save`：保存当前工作状态，偏“下次继续干”。
- `neat-freak`：同步项目文档和记忆，偏“清理一致性”。
- `all-to-one`：复盘完整过程、根因和原理，偏“让人真正理解并能重开”。
- `review`：检查代码问题，不负责项目学习型总整理。
- `testing`：验证测试健康；All To One 只记录验证结果和解释意义。

## Maintenance Rule

如果使用本 skill 时发现某类项目经常需要固定字段，应优先 patch 本 skill 增加项目类型 checklist，而不是新建一堆重复整理 skill。

如果某次 All To One 输出被用户评价为空、假、长、难用，必须反向 patch 本 skill：补充缺失的证据规则、模板、模式或反模式。
