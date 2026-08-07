---
name: harness-agenting-engineering
description: Spec-first AI engineering with evidence gates.
version: 0.1.0
author: Kevin Lin (@wumujushi) + Hermes Agent
license: MIT
platforms:
  - linux
  - macos
metadata:
  hermes:
    category: software-development
    tags:
      - ai-engineering
      - harness
      - quality
      - agents
    related_skills:
      - engineering-workflow
      - test-driven-development
      - systematic-debugging
      - hermes-agent-skill-authoring
---

# Harness / Agenting Engineering

Use this skill when a task asks for sustainable engineering quality, AI-assisted coding discipline, project rules, PR readiness, workflow standardization, hooks/plugins/subagents/skills/rules/MCP, or moving from “vibe coding” to evidence-backed engineering.

## Principle

Do not rely on “try until it runs.” Turn engineering discipline into assistant-executable behavior:

1. **Spec** — define correctness before code.
2. **Context** — route repository/subsystem context explicitly.
3. **Implementation** — use the right extension surface and keep scope narrow.
4. **Evidence** — verify before claiming done.
5. **Retention** — preserve reusable lessons as skills, rules, contracts, or narrow durable memory.

## Imported practices from “Vibe Coding died, Agentic Engineering arrived”

| Article example / practice | Engineering meaning | Hermes behavior |
|---|---|---|
| OpenSpec / Spec-Driven Development | Define correctness before code | Start non-trivial work with problem, acceptance criteria, non-goals, risk surface, and required evidence. |
| Archon / harness builder | Make AI coding deterministic and repeatable | Use Harness Evidence, focused tests, negative checks, contract routing, and quality gates. |
| Context Engineering | Provide repository/subsystem context before edits | Read entry rules, contracts, touched code/tests, and relevant skills; use targeted search, not context dumps. |
| graphify-style code knowledge graph | Convert code structure into queryable context | Prefer architecture docs, contracts, tests, symbol-aware search, LSP/static analysis, and future code indexes over filename guessing. |
| context7-style live documentation via MCP | Use current external docs/API context | Add task-scoped MCP/docs tools with bounded tool counts and explicit security/config boundaries. |
| caveman / token saving | Treat token budget as engineering constraint | Use skills, summaries, targeted reads, compressed handoffs, and narrow context routing. |
| agent-skills / production templates | Reuse procedures as versioned skills | Patch/create skills for difficult or repeated workflows; keep small universal rules in `AGENTS.md`. |
| rsync-style AI PR failure cautionary tale | Missing security spec creates subtle defects | Require negative/security verification for auth, secrets, shell, filesystem, network, approvals, plugins, or data deletion changes. |

These are imported as Hermes-native rules and gates, not as a requirement to install those named projects.

## Required workflow

### 1. Spec before code

For non-trivial work, write or identify:

- problem statement;
- acceptance criteria;
- non-goals;
- affected contract/invariant;
- risk surface: secrets, auth, shell, filesystem, network, plugins, approvals, state, UI, data loss;
- evidence needed before “done.”

Keep it lightweight for small fixes, but never skip defining what correct means.

### 2. Context routing

Before editing:

- read project entry rules such as `AGENTS.md`, `CONTRIBUTING.md`, `docs/CONTRACTS.md`, RFCs, design docs, or TESTING docs;
- inspect current code and tests for the touched subsystem;
- load relevant Hermes skills;
- use targeted searches instead of dumping the entire repository;
- keep MCP/tool surfaces task-relevant and bounded.

### 3. Extension placement

Choose the smallest correct mechanism:

| Need | Put it in |
|---|---|
| Always-on short rule | `AGENTS.md`, `CONTRIBUTING.md`, rules docs |
| Multi-step reusable workflow | Skill |
| Automatic lifecycle enforcement | Hook, CI gate, preflight script |
| External system/tool/data integration | MCP server or plugin |
| Packaged reusable capability | Plugin |
| Public invariant or review expectation | Contract doc/RFC/test |
| Stable user/project preference | Durable memory |
| Temporary task status | Issue/PR/session, not memory |

### 4. Implementation discipline

- One logical change per PR/task.
- Prefer existing extension points over core special cases.
- If a plugin needs a missing capability, extend the generic plugin surface rather than hardcoding plugin-specific logic into core.
- Avoid new dependencies/frameworks/long-lived processes unless benefit and rollback are explicit.
- Update docs and tests alongside behavior changes.

### 5. Evidence gates

Do not claim completion without evidence matched to the risk:

- commands run and pass/fail results;
- focused tests and, when practical, broader tests;
- lint/static checks that catch runtime defects;
- manual verification for UI/setup/runtime/integration behavior;
- screenshots or browser notes for UI/UX;
- state-layer and invariant proof for streaming/session/replay/compression/sidebar/workspace changes;
- negative/security checks for secrets, auth, shell, approvals, plugins, filesystem, network, webhooks, or data deletion.

### 6. Retention

After a hard or repeated workflow:

- patch or create a skill if the procedure is reusable;
- update contract docs if the public rule changed;
- save memory only for stable preferences/environment facts;
- never store credentials, raw tokens, full private config, or stale task progress.

## Harness Evidence template

Use this in PRs, handoffs, or final reports for non-trivial work:

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

## Unified task intake template

For non-trivial work, use `templates/task-intake-form.md` before sending the task to Hermes / an LLM / a coding agent. The form collects:

- spec and acceptance criteria;
- context routing;
- risk surface and negative checks;
- extension choice: Plugin / MCP / Command / Skill / Rule / Contract / built-in change;
- verification evidence;
- retention decision.

The condensed final prompt at the bottom of the template can be pasted directly into Hermes after the form is filled.

## How to use this harness

### Normal chat usage

For non-trivial engineering work, ask normally in Hermes. Current in-repo soft
preflight integration is gateway-scoped; CLI and WebUI users can invoke the
explicit `hermes harness classify`, `hermes harness new`, and `/intake` surfaces
until those entrypoints grow their own model-facing bridge.

```text
请修复这个 bug
请实现一个新功能并加测试
请重构这段逻辑并保证不破坏现有行为
```

When `harness_engineering.preflight_mode` is unset or set to `advisory`, gateway
messages keep the visible/persisted user message unchanged, but the hook returns
a model-facing Harness reminder for engineering-like tasks. The reminder asks
for scope, acceptance criteria, risk surface, rollback, context routing, and
verification evidence before implementation.

Plain explanation/summarization/translation requests should not trigger the harness.

### Explicit intake for larger tasks

Use `classify` when a caller needs a stable advisory route before deciding
whether to create a full intake. Classification does not dispatch work, write
state, or replace the caller's own permission checks.

```bash
hermes harness classify --text "Refactor auth token storage and add tests" --format json
```

Core routes:

| Route | Meaning |
|---|---|
| `answer_directly` | Plain chat; no Harness intake needed. |
| `research_then_report` | Gather evidence and report assumptions/gaps. |
| `bounded_engineering` | Small scoped code work with focused tests. |
| `harness_advisory` | Larger engineering work; define scope and verification before editing. |
| `intake_required` | High-risk, scheduled/ops, or multi-agent work; create/fill an intake before implementation. |

```bash
hermes harness new \
  --title "Fix gateway preflight regression" \
  --workspace /path/to/repo \
  --mode "Implement changes" \
  --output /tmp/harness-intake.md

hermes harness check /tmp/harness-intake.md
hermes harness prompt /tmp/harness-intake.md --output /tmp/harness-prompt.txt --force
```

Fallback helper:

```bash
${HERMES_HOME:-$HOME/.hermes}/bin/hermes-harness new --title "My task" --workspace /path/to/repo --mode "Implement changes"
${HERMES_HOME:-$HOME/.hermes}/bin/hermes-harness check /path/to/intake.md
${HERMES_HOME:-$HOME/.hermes}/bin/hermes-harness prompt /path/to/intake.md
```

Blank generated forms are expected to fail `check` until problem, acceptance criteria, risk surface, and verification fields are filled.

### In-session helper

```text
/intake
```

This prints the Harness / Agenting Engineering intake instructions. It is advisory only.

### Preflight modes

`harness_engineering.preflight_mode` in `config.yaml` controls the gateway soft
preflight:

| Mode | Behavior | Use when |
|---|---|---|
| unset / `advisory` | Soft gateway rewrite for engineering-like requests | Default daily development |
| `strict` | Prepends an intake-required instruction | Risky work: auth, secrets, shell, filesystem, deploys, data deletion, cross-platform behavior |
| `off` | No rewrite | Debugging false positives or temporarily disabling the harness |

Examples:

```bash
hermes config set harness_engineering.preflight_mode advisory
hermes config set harness_engineering.preflight_mode strict
hermes config set harness_engineering.preflight_mode off
```

Restart the gateway after changing the setting.

### Current acceptance checks

- Engineering request such as `请修复这个 bug` triggers `[Harness / Agenting Engineering preflight]`.
- Plain explanation such as `解释一下什么是 MCP` is allowed unchanged.
- `harness_engineering.preflight_mode: strict` emits the `intake required` variant.
- `harness_engineering.preflight_mode: off` disables gateway rewrites.
- WebUI/CLI coverage claims stay limited to explicit `hermes harness ...` commands until their entrypoints wire an equivalent model-facing bridge.
- The current Harness plugin should use `allow` / `rewrite`, not `skip`, so it remains a soft guard rather than a hard blocker.

## Local command helper

This skill includes a local helper command for Level 2 soft enforcement:

```bash
# Create a new intake form, prefilled with title/workspace/mode
${HERMES_HOME:-$HOME/.hermes}/bin/hermes-harness new \
  --title "My task" \
  --workspace /path/to/repo \
  --mode "Implement changes"

# Validate required fields in a filled form
${HERMES_HOME:-$HOME/.hermes}/bin/hermes-harness check /path/to/intake.md

# Render the condensed prompt to paste into Hermes / an LLM / a coding agent
${HERMES_HOME:-$HOME/.hermes}/bin/hermes-harness prompt /path/to/intake.md

# Print the blank template
${HERMES_HOME:-$HOME/.hermes}/bin/hermes-harness template
```

The executable wrapper lives at `bin/hermes-harness` under the active Hermes profile home; the source script lives at `scripts/harness_intake.py` inside this skill. It has no third-party dependencies and intentionally avoids importing Hermes internals, so it can run from CLI, WebUI terminals, WSL, cron scripts, or other agents.

If the active profile's `bin` directory is on `PATH`, use `hermes-harness ...` directly.

## Hermes plugin entrypoints

This skill can be paired with the `harness_engineering` standalone plugin for
Level 3 soft integration. Install or enable it in the active profile before
expecting `/intake`, `hermes harness ...`, or gateway preflight hooks to load:

```bash
mkdir -p "${HERMES_HOME:-$HOME/.hermes}/plugins"
cp -R plugins/harness_engineering "${HERMES_HOME:-$HOME/.hermes}/plugins/harness_engineering"
hermes plugins enable harness_engineering
```

Bundled `standalone` plugins are opt-in via `plugins.enabled`; this plugin does
not auto-load until enabled. Once enabled, it registers:

```bash
# Hermes-native CLI wrapper around the same helper
hermes harness template
hermes harness template --output /path/to/template.md
hermes harness new --title "My task" --workspace /path/to/repo --mode "Implement changes" --output /path/to/intake.md
hermes harness check /path/to/intake.md
hermes harness prompt /path/to/intake.md
hermes harness prompt /path/to/intake.md --allow-incomplete --output /path/to/prompt.txt --force

# Lifecycle bridge: intake -> Kanban triage -> worker/reviewer plan -> evidence -> GC
hermes harness kanban create /path/to/intake.md --triage --json
hermes harness kanban create /path/to/intake.md --dry-run --json
hermes harness kanban decompose <task-id> --workspace worktree --branch wt/<task-id>
hermes harness kanban decompose <task-id> --execute  # explicit only; default is dry-run
hermes harness evidence <task-id> --workspace /path/to/repo --output /path/to/evidence.md
hermes harness gc-template --output /path/to/weekly-harness-gc.md
hermes harness migration-pack --output-dir /path/to/repo --json
```

Lifecycle command safety boundaries:

- `kanban create` turns a filled intake into a Kanban card and defaults to `--triage`, with an idempotency key derived from the intake path.
- `kanban decompose` defaults to dry-run and prints implementation/review child-card commands; it only creates cards when `--execute` is explicit.
- Review child cards are created blocked by default so a reviewer can check spec compliance and evidence without silently starting implementation.
- `evidence` captures Kanban task JSON when available plus Git HEAD/status/diff-stat and completion note placeholders.
- `gc-template` writes a weekly drift review checklist; it must not auto-repair, auto-restart services, delete state, modify credentials, or dispatch workers.

It also registers the in-session plugin slash command:

```text
/intake
```

`/intake` is intentionally advisory: it returns the Harness / Agenting Engineering intake instructions and does not intercept every message. Use this before non-trivial engineering tasks when a full WebUI/gateway preflight gate is not yet enabled.

Important implementation notes:

- `hermes harness ...` delegates first to the bundled helper, then to the active profile's `bin/hermes-harness`; keep the helper and plugin argument surfaces in sync.
- The plugin handler raises `SystemExit(code)` so failed `check` results propagate as non-zero process exit codes for scripts/CI.
- Blank/generated forms are expected to fail `check` until problem, acceptance criteria, risk surface, and verification evidence are filled.

## Plugin / MCP / Command integration choice

Use this quick decision path:

1. **MCP** — use when an external tool/server/API already exists, or the integration should remain outside Hermes core. Add via `hermes mcp add NAME --command ...` or `hermes mcp add NAME --url ...`, then `hermes mcp test NAME` and `hermes mcp configure NAME` to filter tools.
2. **Plugin** — use when the capability should be a reusable Hermes extension: custom tools, hooks, slash commands, provider backends, memory/context/image/video/search providers, or bundled skills. Create `~/.hermes/plugins/<name>/plugin.yaml` plus schemas/handlers/hooks, or an in-repo plugin when it should ship with Hermes.
3. **Command / script** — use when the capability is a repeatable operator action, quality gate, evidence generator, migration, or setup helper. Prefer a script or CLI subcommand over asking the model to remember shell snippets.
4. **Skill** — use when the capability is a reusable multi-step procedure rather than executable code.
5. **Rule / contract** — use when the behavior is an always-on constraint or public invariant.

Security default: start with the smallest useful surface, whitelist tools where possible, document credentials as env vars, and never pass broad secrets or full filesystem access to an untrusted MCP/plugin.

## Hermes repository notes

For Hermes Agent work, also read the repository-level `docs/harness-agenting-engineering.md` when present. For WebUI work, run or reference `hermes-webui/scripts/harness_quality_gate.py` to produce advisory changed-file routing and recommended checks.

## Level 4 soft preflight integration

The user-local `harness_engineering` plugin registers a `pre_gateway_dispatch` hook for soft enforcement:

- `harness_engineering.preflight_mode: advisory` (default): rewrite non-trivial engineering prompts by prepending a Harness / Agenting Engineering reminder.
- `harness_engineering.preflight_mode: strict`: prepend an intake-required instruction for high-risk work.
- `harness_engineering.preflight_mode: off`: disable the preflight.

Gateway messages pass through `gateway/run.py` and can honor `skip`, `rewrite`, and `allow` hook actions. Current main does not route CLI or WebUI/browser-originated chats through this hook; those surfaces are covered by explicit `hermes harness ...` and `/intake` commands only.

Verification commands used for this path:

```bash
# In the Hermes Agent repo; use venv if .venv is absent.
venv/bin/python -m pytest -o addopts='' \
  tests/gateway/test_pre_gateway_dispatch.py \
  tests/hermes_cli/test_plugins.py -q

venv/bin/python -m py_compile \
  hermes_cli/plugins.py gateway/run.py plugins/harness_engineering/__init__.py
```

Pitfall: if the active test environment lacks `pytest-timeout`, project `pyproject.toml` addopts (`--timeout=30 --timeout-method=signal`) make pytest fail before collection. For focused validation, use `-o addopts=''` or install the dev extras into the active venv.

## Pitfalls

- Do not turn every rule into a skill; always-on constraints belong in rules.
- Do not turn rare workflows into always-on prompt bloat; make them skills.
- Do not load many MCP tools “just in case”; tool count consumes context.
- Do not delegate to subagents without explicit scope, context, and review criteria.
- Do not store task progress or PR numbers in memory.
- Do not write secrets into docs, skills, examples, or logs; redact as `[REDACTED]`.
