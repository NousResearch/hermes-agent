# Harness / Agenting Engineering for Hermes

Status: initial project working agreement. This document translates the ideas from two WeChat articles about Agentic Engineering and Harness Engineering into Hermes-specific engineering rules.

Source articles reviewed:

- `https://mp.weixin.qq.com/s/FB2v0kQHJDNBF4EmqPkXIQ` — argues that Vibe Coding breaks down as projects grow, and that Spec-Driven Development, Context Engineering, token-budget discipline, and reusable skills turn AI coding from a craft into an engineering practice.
- `https://mp.weixin.qq.com/s/dhN-zkSDNntG_U-0TQQQVA` — defines Harness Engineering as encoding engineering discipline into AI-assistant-executable behavior through Hooks, Subagents, Skills, Rules, MCP/LSP, Plugins, commands, monitors, and state/memory.

This document is not a philosophy note. It is the common contract for Hermes contributors and AI coding assistants: **do not rely on “try until it runs.” Encode what “correct” means, route context explicitly, verify with evidence, and preserve reusable knowledge.**

## Definition

**Harness / Agenting Engineering** is the practice of turning engineering discipline into executable assistant behavior:

- safety checks become hooks, gates, review checklists, and negative tests;
- code conventions become short always-on rules;
- complex workflows become skills or commands;
- subsystem knowledge becomes contract docs and context indexes;
- external systems become MCP/LSP/plugin tools with explicit budgets;
- large tasks become delegated subagent work with review boundaries;
- cross-session lessons become durable skills or narrowly scoped memory.

The goal is to move Hermes from “a helpful CLI/WebUI agent that can code” to **a team-grade engineering substrate** whose quality practices can be reused in any project.

## Core rule: Spec → Context → Implementation → Evidence → Retention

Every non-trivial Hermes engineering task should follow this lifecycle.

### 1. Spec before code

Before editing, define the smallest useful specification:

- problem statement and non-goals;
- public contract or invariant being changed;
- acceptance criteria;
- failure modes and security/privacy boundaries;
- verification evidence needed before claiming done.

For small fixes this can be a short task note or PR body section. For larger work use an RFC, plan, issue, or markdown design. The important rule is that the assistant knows what counts as correct **before** it writes code.

### 2. Context before edits

Route context explicitly instead of trusting the model to infer the repository:

- start from `AGENTS.md`, `CONTRIBUTING.md`, and the relevant contract/RFC docs;
- inspect current code and tests for the touched subsystem;
- load relevant Hermes skills before acting;
- prefer targeted searches over broad context dumps;
- keep active tool/MCP surfaces bounded and task-relevant.

Context engineering is not “stuff the prompt.” It is choosing the minimum correct context that prevents local fixes from violating system contracts.

### 3. Implementation inside the harness

Implementation must stay inside the repository’s extension and contract model:

- prefer existing public extension points over hardcoded special cases;
- if a plugin needs a missing capability, extend the generic plugin surface instead of patching plugin-specific logic into core;
- keep one logical change per PR;
- avoid new dependencies, build tools, frameworks, or long-lived processes unless the benefit and rollback story are explicit;
- update docs and tests with behavior changes.

### 4. Evidence before “done”

A coding assistant may not claim a fix is complete without evidence appropriate to the touched subsystem.

At minimum, include:

- commands run and results;
- tests added/updated or reason not applicable;
- manual verification for UI, setup, runtime-state, or integration behavior;
- negative/security checks where the change touches auth, tokens, secrets, approvals, shell execution, filesystem writes, webhooks, plugins, or network calls;
- screenshots or browser notes for UI/UX changes;
- state-layer and invariant proof for streaming, sessions, replay, compression, cancellation, approval, sidebar, or workspace changes.

Green CI is necessary but not always sufficient. The evidence must match the risk surface.

### 5. Retention after completion

After a difficult or repeated workflow:

- patch or create a skill for reusable procedures;
- update contract docs when the public rule changed;
- save durable memory only for stable user/project preferences or environment facts;
- never store credentials, raw tokens, full private paths, or stale task progress in memory or tracked docs.

Retention is what makes the next session better instead of forcing the same explanation again.

## How to use this harness in Hermes

This section is the operator-facing usage guide for the current Hermes Harness / Agenting Engineering setup.

### 1. Normal WebUI / chat usage

For non-trivial engineering work, just ask normally in WebUI, CLI, or connected gateway platforms:

```text
请修复这个 bug
请实现一个新功能并加测试
请重构这段逻辑并保证不破坏现有行为
```

When `HERMES_HARNESS_PREFLIGHT` is unset or set to `advisory`, Hermes will keep the visible user message unchanged, but will prepend a model-facing Harness reminder for engineering-like tasks. The reminder asks the assistant to define scope, acceptance criteria, risk surface, rollback, context routing, and verification evidence before implementation.

Plain questions should not trigger the harness:

```text
解释一下什么是 MCP
总结一下这篇文章
翻译这段文字
```

### 2. Explicit intake flow for larger tasks

For larger, risky, or multi-session work, create an intake form before implementation:

```bash
hermes harness new \
  --title "Fix gateway preflight regression" \
  --workspace /path/to/repo \
  --mode "Implement changes" \
  --output /tmp/harness-intake.md

hermes harness check /tmp/harness-intake.md
hermes harness prompt /tmp/harness-intake.md --output /tmp/harness-prompt.txt --force
```

If `hermes harness ...` is unavailable in a particular shell, use the profile-local helper directly:

```bash
~/.hermes/bin/hermes-harness new --title "My task" --workspace /path/to/repo --mode "Implement changes"
~/.hermes/bin/hermes-harness check /path/to/intake.md
~/.hermes/bin/hermes-harness prompt /path/to/intake.md
```

Blank generated forms are expected to fail `check` until the required fields are filled.

### 3. In-session helper

Use this in Hermes sessions when you want the reminder without creating a file:

```text
/intake
```

It prints the Harness / Agenting Engineering intake instructions. It is advisory only.

### 4. Preflight modes

The soft preflight default is controlled by `harness_engineering.preflight_mode` in `~/.hermes/config.yaml`:

```yaml
harness_engineering:
  preflight_mode: advisory  # advisory | strict | off
```

`HERMES_HARNESS_PREFLIGHT` takes precedence when set, which is useful for per-process overrides:

| Mode | Behavior | Use when |
|---|---|---|
| unset / `advisory` | Soft rewrite for engineering-like requests | Default daily development |
| `strict` | Prepends an intake-required instruction | Risky work: auth, secrets, shell, filesystem, deploys, data deletion, cross-platform behavior |
| `off` | No rewrite | Debugging false positives or temporarily disabling the harness |

Examples:

```bash
hermes config set harness_engineering.preflight_mode advisory
hermes config set harness_engineering.preflight_mode strict
hermes config set harness_engineering.preflight_mode off

export HERMES_HARNESS_PREFLIGHT=advisory
export HERMES_HARNESS_PREFLIGHT=strict
export HERMES_HARNESS_PREFLIGHT=off
```

Restart the relevant process after changing the variable:

- WebUI/API server: restart the WebUI/API process.
- Gateway/WeChat/Telegram/etc.: restart the gateway.
- CLI: start a new `hermes` session.

### 5. Verification commands

Run these from the Hermes Agent repository when changing this harness:

```bash
cd /home/kevin/tools/hermes/hermes-agent

venv/bin/python -m pytest \
  tests/gateway/test_pre_gateway_dispatch.py \
  tests/hermes_cli/test_plugins.py -q

scripts/run_tests.sh \
  tests/gateway/test_pre_gateway_dispatch.py \
  tests/hermes_cli/test_plugins.py

venv/bin/python -m py_compile \
  hermes_cli/plugins.py \
  gateway/run.py \
  hermes-webui/api/streaming.py \
  ~/.hermes/plugins/harness_engineering/__init__.py
```

If pytest fails before collection with `unrecognized arguments: --timeout=30 --timeout-method=signal`, install `pytest-timeout` into the active venv or run a focused check with `-o addopts=''`.

### 6. Acceptance checks for the current setup

Use these behavioral checks after changes:

- Engineering request such as `请修复这个 bug` triggers `[Harness / Agenting Engineering preflight]`.
- Plain explanation such as `解释一下什么是 MCP` is allowed unchanged.
- `HERMES_HARNESS_PREFLIGHT=strict` emits the `intake required` variant.
- `HERMES_HARNESS_PREFLIGHT=off` disables rewrites.
- WebUI must not mutate the visible/persisted user message; only the model-facing current turn is rewritten.
- The current Harness plugin should use `allow` / `rewrite`, not `skip`, so it remains a soft guard rather than a hard blocker.

### 7. Notes and pitfalls

- The harness is not a replacement for tests. It is a mechanism to make the assistant ask for and produce the right evidence.
- Do not put long procedures into always-on rules; keep long workflows in skills or docs.
- Do not turn every advisory into a hard gate. Promote only low-noise checks after observing false positives.
- Do not store task progress, PR numbers, secrets, tokens, cookies, or raw private config in memory or docs.
- For WebUI changes, keep using `hermes-webui/scripts/harness_quality_gate.py` where applicable to generate changed-file routing and suggested checks.
- For plugins/MCP/providers, prefer generic extension points over core special cases and document credentials as environment variables only.

## Practices imported from “Vibe Coding died, Agentic Engineering arrived”

The first source article is not imported as named-tool fan fiction. Its examples are translated into Hermes-native practices:

| Article example / practice | Engineering meaning | Hermes adoption |
|---|---|---|
| OpenSpec / Spec-Driven Development | Tell the assistant what “correct” means before code exists | Non-trivial work must start with a lightweight spec: problem, acceptance criteria, non-goals, risk surface, and required evidence. Larger work should use a plan, RFC, issue, or design doc. |
| Archon / harness builder | Make AI coding deterministic and repeatable with checks around generation | Use Harness Evidence, focused tests, negative checks, contract routing, and advisory/CI quality gates. WebUI already has `scripts/harness_quality_gate.py`; core Hermes should grow similar project commands. |
| Context Engineering | AI fails when it lacks repository/subsystem context | Route context explicitly through `AGENTS.md`, `CONTRIBUTING.md`, contract docs, RFCs, touched tests, and relevant skills before editing. Prefer targeted searches over broad prompt stuffing. |
| graphify-style code knowledge graph | Convert codebase structure into queryable context | Hermes should treat architecture docs, contract docs, tests, symbol-aware search, LSP/static analysis, and future code-index tools as context indexes. Do not rely on filename guesses for cross-subsystem edits. |
| context7-style live documentation via MCP | Give the assistant current docs/API context instead of stale memory | Prefer task-scoped MCP/docs tools for external APIs and dependencies, with bounded tool counts and explicit security/config boundaries. |
| caveman / token saving | Token budget is an engineering constraint, not an afterthought | Use skills, summaries, targeted reads, compressed handoffs, and narrow context routing. Avoid dumping whole repos, whole logs, or many MCP tools into context. |
| agent-skills / production prompt templates | Reusable procedures should become versioned skills | Difficult or repeated workflows must patch/create skills with exact commands, pitfalls, and verification. Small always-on rules stay in `AGENTS.md`, not in bulky prompts. |
| rsync-style AI PR failure cautionary tale | AI-generated code without human-spec security boundaries can introduce subtle defects | Any change touching auth, secrets, shell, filesystem, network, approvals, plugins, or data deletion needs negative/security verification before “done.” |

These practices are intentionally expressed as Hermes rules and gates rather than instructions to install the referenced projects. If a referenced tool becomes useful later, add it through a plugin/MCP/command with bounded context and security review.

## Mapping the nine extension mechanisms to Hermes

| Mechanism | Hermes surface | Use it for | Do not use it for |
|---|---|---|---|
| Rules | `AGENTS.md`, `CONTRIBUTING.md`, contract docs, system/developer instructions | Short always-on constraints: secrets redaction, one logical PR, run evidence gates | Long workflows or rare procedures |
| Skills | `skills/`, `optional-skills/`, user/profile skills | Complex task procedures that should load on demand | Tiny universal rules that should be in `AGENTS.md` |
| Commands | CLI subcommands, scripts, project commands | Repeatable operator actions and review/gate helpers | One-off shell snippets hidden in chat history |
| Hooks | plugin lifecycle hooks, gateway hooks, future CI/preflight hooks | Automatic checks at tool, LLM, session, approval, or dispatch boundaries | Business logic that belongs in core services |
| Monitors | observability plugins, logs, cron/webhook monitors | Drift detection, cost/token alerts, runtime health, quality telemetry | Blocking policy without a clear escape hatch |
| Subagents | Hermes subagents, Kanban worker/orchestrator, external coding agents | Parallel research/implementation/review with bounded context | Replacing final integration ownership |
| MCP | native MCP / mcporter integrations | External APIs, databases, docs, code intelligence, search | Loading many irrelevant tools into context |
| LSP / code intelligence | language servers, static analyzers, pyright/ruff/eslint-like gates | Symbol-aware edits, diagnostics, refactor safety | Guessing code structure from filenames only |
| Plugins | `plugins/`, model-provider plugins, memory-provider plugins | Packaged extension surfaces and shareable team capabilities | Hardcoded provider/tool special cases in core |

## Placement rules: Rule vs Skill vs Hook vs Plugin

Use this decision table before adding new process.

| Need | Put it in |
|---|---|
| Must apply every time, one sentence, low nuance | Rule in `AGENTS.md` / `CONTRIBUTING.md` |
| Multi-step procedure, task-specific, too long for always-on context | Skill |
| Must run automatically at a lifecycle point | Hook or CI/preflight script |
| Adds a reusable tool, provider, platform, memory backend, command, or integration | Plugin / MCP server |
| Captures public behavior, invariant, or review expectation | Contract doc / RFC / tests |
| Captures stable local/user preference | Durable memory, narrowly scoped |
| Captures temporary task progress | Session transcript / issue / PR, not memory |

## Hermes quality gates by change type

### Python core / CLI / gateway

- Read the relevant architecture and plugin surface docs before editing.
- Run focused tests first, then broader pytest where practical.
- Run lint gates that catch runtime bugs, not style churn.
- For tool, shell, approval, filesystem, credential, or platform changes, include at least one negative/abuse-path check.

### WebUI

- Start from `hermes-webui/AGENTS.md`, `hermes-webui/docs/CONTRACTS.md`, and touched RFC/design docs.
- Use `python3 scripts/harness_quality_gate.py` to generate the advisory Harness Evidence block for changed files.
- For frontend changes, run `npm run lint:runtime` and browser smoke/manual browser evidence where applicable.
- For runtime state changes, name the mutated state layer and prove replay/recovery/sidebar/session invariant.

### Plugins / MCP / providers

- Prefer extension surfaces over core special cases.
- Keep tool count and prompt footprint bounded; load only task-relevant tools.
- Document installation, configuration, security boundaries, and failure behavior.
- Never commit API keys, cookies, OAuth tokens, or full local config files.

### Skills

- Follow the skill authoring rules in `AGENTS.md` and the `hermes-agent-skill-authoring` skill.
- Keep descriptions short and task-triggered.
- Include exact commands, pitfalls, verification, and when not to use the skill.
- Patch stale skills immediately when a real task proves them incomplete.

### Docs / contracts

- Docs that change public behavior must move with tests or explicit verification.
- Contract-changing PRs need `Contract Routing`; intentional contract changes need `Contract Change`.
- Do not let tests silently redefine product behavior without updating public docs.

## Standard Harness Evidence block

Use this block in PRs, review notes, or AI handoffs when a change is non-trivial.

```markdown
## Harness Evidence

Spec:
- Problem:
- Acceptance criteria:
- Non-goals:
- Risk surface:

Context Routing:
- Read:
- Skills loaded:
- Relevant contracts/RFCs:

Implementation:
- Touched areas:
- Extension point used:
- Docs updated:

Verification:
- Automated:
- Manual:
- Negative/security:
- UI/state evidence:

Retention:
- Skill/memory/doc updated:
- Follow-ups:
```

Keep this lightweight for small fixes. The goal is to make engineering evidence explicit, not to create bureaucracy.

## Operating principles transferable to any project

1. **Define correctness before generation.** AI coding becomes reliable when specs, invariants, and tests precede code.
2. **Engineer context as an artifact.** Context should be routed, indexed, and budgeted, not improvised in every session.
3. **Automate judgment points.** Repeated review questions should become hooks, gates, commands, or checklists.
4. **Delegate with boundaries.** Subagents are useful when their task, context, and review criteria are explicit.
5. **Retain only reusable knowledge.** Skills and contract docs compound; stale task notes and secrets create risk.
6. **Evidence beats assertion.** An assistant’s “done” means commands, tests, logs, screenshots, or invariant proof — not confidence.

## Adoption path

### Phase 1 — advisory, now

- Link this document from AI entry points.
- Require non-trivial work to include Harness Evidence.
- Use existing WebUI `scripts/harness_quality_gate.py` for changed-file routing.
- Create/patch skills after difficult workflows.

### Phase 2 — semi-automated

- Add lightweight project commands that generate evidence blocks for core Hermes, not only WebUI.
- Add optional pre-PR checks for secrets, missing tests, and contract-routing drift.
- Add review-agent prompts/subagents for security, runtime-state, docs, and UI evidence.

### Phase 3 — enforceable harness

- Promote low-noise checks to CI/hooks.
- Add observability for token/tool budget, flaky tests, and repeated failure modes.
- Package reusable quality gates as plugins or MCP tools that can be installed into other projects.

The adoption sequence matters: start advisory, measure false positives, then enforce only the checks that reliably catch real defects.
