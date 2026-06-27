---
name: harness-agenting-engineering
description: Spec-first AI engineering with evidence gates.
version: 0.1.0
author: Hermes Agent
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

### Normal WebUI / chat usage

For non-trivial engineering work, ask normally in WebUI, CLI, or connected gateway platforms:

```text
请修复这个 bug
请实现一个新功能并加测试
请重构这段逻辑并保证不破坏现有行为
```

When `HERMES_HARNESS_PREFLIGHT` is unset or set to `advisory`, Hermes keeps the visible/persisted user message unchanged, but prepends a model-facing Harness reminder for engineering-like tasks. The reminder asks for scope, acceptance criteria, risk surface, rollback, context routing, and verification evidence before implementation.

Plain explanation/summarization/translation requests should not trigger the harness.

### Explicit intake for larger tasks

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
~/.hermes/bin/hermes-harness new --title "My task" --workspace /path/to/repo --mode "Implement changes"
~/.hermes/bin/hermes-harness check /path/to/intake.md
~/.hermes/bin/hermes-harness prompt /path/to/intake.md
```

Blank generated forms are expected to fail `check` until problem, acceptance criteria, risk surface, and verification fields are filled.

### In-session helper

```text
/intake
```

This prints the Harness / Agenting Engineering intake instructions. It is advisory only.

### Preflight modes

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

Restart the relevant process after changing the variable: WebUI/API server, gateway, or CLI session.

### Current acceptance checks

- Engineering request such as `请修复这个 bug` triggers `[Harness / Agenting Engineering preflight]`.
- Plain explanation such as `解释一下什么是 MCP` is allowed unchanged.
- `HERMES_HARNESS_PREFLIGHT=strict` emits the `intake required` variant.
- `HERMES_HARNESS_PREFLIGHT=off` disables rewrites.
- WebUI must not mutate the visible/persisted user message; only the model-facing current turn is rewritten.
- The current Harness plugin should use `allow` / `rewrite`, not `skip`, so it remains a soft guard rather than a hard blocker.

## Local command helper

This skill includes a local helper command for Level 2 soft enforcement:

```bash
# Create a new intake form, prefilled with title/workspace/mode
~/.hermes/bin/hermes-harness new \
  --title "My task" \
  --workspace /path/to/repo \
  --mode "Implement changes"

# Validate required fields in a filled form
~/.hermes/bin/hermes-harness check /path/to/intake.md

# Render the condensed prompt to paste into Hermes / an LLM / a coding agent
~/.hermes/bin/hermes-harness prompt /path/to/intake.md

# Print the blank template
~/.hermes/bin/hermes-harness template
```

The executable wrapper lives at `~/.hermes/bin/hermes-harness`; the source script lives at `scripts/harness_intake.py` inside this skill. It has no third-party dependencies and intentionally avoids importing Hermes internals, so it can run from CLI, WebUI terminals, WSL, cron scripts, or other agents.

If `~/.hermes/bin` is on `PATH`, use `hermes-harness ...` directly.

## Hermes plugin entrypoints

This profile also has a user-local plugin for Level 3 soft integration:

```text
~/.hermes/plugins/harness_engineering/
```

It is enabled in the active `default` profile and registers:

```bash
# Hermes-native CLI wrapper around the same helper
hermes harness template
hermes harness template --output /path/to/template.md
hermes harness new --title "My task" --workspace /path/to/repo --mode "Implement changes" --out /path/to/intake.md
hermes harness check /path/to/intake.md
hermes harness prompt /path/to/intake.md
hermes harness prompt /path/to/intake.md --allow-incomplete --output /path/to/prompt.txt --force
```

It also registers the in-session plugin slash command:

```text
/intake
```

`/intake` is intentionally advisory: it returns the Harness / Agenting Engineering intake instructions and does not intercept every message. Use this before non-trivial engineering tasks when a full WebUI/gateway preflight gate is not yet enabled.

Important implementation notes:

- `hermes harness ...` delegates to `~/.hermes/bin/hermes-harness`; keep the helper and plugin argument surfaces in sync.
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

- `HERMES_HARNESS_PREFLIGHT=advisory` (default): rewrite non-trivial engineering prompts by prepending a Harness / Agenting Engineering reminder.
- `HERMES_HARNESS_PREFLIGHT=strict`: prepend an intake-required instruction for high-risk work.
- `HERMES_HARNESS_PREFLIGHT=off`: disable the preflight.

Gateway messages pass through `gateway/run.py` and can honor `skip`, `rewrite`, and `allow` hook actions. WebUI/browser-originated chats do not pass through gateway dispatch, so `hermes-webui/api/streaming.py` bridges only `rewrite` actions into the model-facing current turn; it must not mutate the visible/persisted user message and must not block requests.

Verification commands used for this path:

```bash
# In the Hermes Agent repo; use venv if .venv is absent.
venv/bin/python -m pytest -o addopts='' \
  tests/gateway/test_pre_gateway_dispatch.py \
  tests/hermes_cli/test_plugins.py -q

PYTHONPATH=/path/to/hermes-agent:/path/to/hermes-agent/hermes-webui \
venv/bin/python - <<'PY'
import os
from api.streaming import _apply_webui_pre_gateway_dispatch_preflight
os.environ['HERMES_HARNESS_PREFLIGHT'] = 'advisory'
rewritten = _apply_webui_pre_gateway_dispatch_preflight('请修复这个 bug', session_id='sid', workspace='/tmp/ws', profile='default')
assert rewritten.startswith('[Harness / Agenting Engineering preflight]')
assert 'acceptance criteria' in rewritten.lower()
assert 'risk surface' in rewritten.lower()
assert _apply_webui_pre_gateway_dispatch_preflight('解释一下什么是 MCP', session_id='sid', workspace='/tmp/ws', profile='default') == '解释一下什么是 MCP'
os.environ['HERMES_HARNESS_PREFLIGHT'] = 'off'
assert _apply_webui_pre_gateway_dispatch_preflight('请修复这个 bug', session_id='sid', workspace='/tmp/ws', profile='default') == '请修复这个 bug'
PY

venv/bin/python -m py_compile \
  hermes_cli/plugins.py gateway/run.py hermes-webui/api/streaming.py \
  ~/.hermes/plugins/harness_engineering/__init__.py
```

Pitfall: if the active test environment lacks `pytest-timeout`, project `pyproject.toml` addopts (`--timeout=30 --timeout-method=signal`) make pytest fail before collection. For focused validation, use `-o addopts=''` or install the dev extras into the active venv.

## Pitfalls

- Do not turn every rule into a skill; always-on constraints belong in rules.
- Do not turn rare workflows into always-on prompt bloat; make them skills.
- Do not load many MCP tools “just in case”; tool count consumes context.
- Do not delegate to subagents without explicit scope, context, and review criteria.
- Do not store task progress or PR numbers in memory.
- Do not write secrets into docs, skills, examples, or logs; redact as `[REDACTED]`.
