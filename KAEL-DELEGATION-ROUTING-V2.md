# Kael Delegation Routing v2

This fork carries a hardcoded Kael routing helper at `scripts/kael_delegation_router.py`.

It is not generic theory. It encodes the live lane split Kael is supposed to use in this environment:
- parent on `gpt-5.4`
- bounded specialist leaves on `gpt-5.5`
- long-context clean-room probes via Codex CLI
- multi-domain local synthesis via `gpt-5.4` orchestrator CLI children

## Live doctrine pointers
- Live skill: `~/.hermes/skills/delegation-routing-v2/SKILL.md`
- Live config: `~/.hermes/config.yaml` under `delegation` and `kael_delegation`
- Live spec: `/home/ubuntu/business/reports/delegation-system-hardcoded-2026-04-25.md`
- Prior architecture grounding: `/home/ubuntu/business/reports/subagent-kael-delegation-architecture-2026-04-24.md`
- Fork workflow: `/home/ubuntu/business/reports/hermes-fork-workflow-2026-04-25.md`

## Lane summary

### 1. Parent
- lane: `parent`
- model: `gpt-5.4`
- provider: `openai-codex`
- keeps: `STATE.md` writes, final synthesis, ship/no-ship judgments, irreversible actions
- rule: never delegate work that fits in 1-2 tool calls

### 2. gpt-5.5 specialist leaf
- lane: `gpt55_specialist`
- model: `gpt-5.5`
- provider: `openai-codex`
- role: `leaf`
- best for: bounded reasoning, code review, structured JSON/markdown output, classification, summarization, inspection
- default toolsets:
  - code → `['file', 'terminal']`
  - research → `['file', 'web']`
  - inspection → `['file']`
- timeout policy: retry once, then mark partial

### 3. Codex CLI long-context lane
- lane: `codex_cli_long_context`
- role: `standalone_process`
- best for: 5+ files, >300k tokens, clean-room probes, large corpus analysis, write-allowed generation in isolation
- commands:
  - read-only → `codex exec --skip-git-repo-check --sandbox read-only --ephemeral '<prompt>'`
  - workspace-write → `codex exec --skip-git-repo-check --sandbox workspace-write --ephemeral '<prompt>'`
  - json → `codex exec --skip-git-repo-check --sandbox read-only --ephemeral --json '<prompt>'`
- failure policy: retry with alternate sandbox mode if appropriate

### 4. gpt-5.4 orchestrator CLI child
- lane: `gpt54_orchestrator_cli`
- model: `gpt-5.4`
- provider: `openai-codex`
- role: `orchestrator`
- best for: 2+ broad domains that each need local synthesis before parent judgment
- command:
  - `hermes chat --provider openai-codex --model gpt-5.4 -s delegation-routing-v2 -Q -q '<self-contained orchestrator prompt>'`

## Hardcoded routing rules
Apply in this order:

1. **A — shared state / irreversible action** → `parent`
2. **B — single tool call or pure reasoning under 50k** → `parent`
3. **F — 2+ broad domains needing local synthesis** → `gpt54_orchestrator_cli`
4. **E — N independent subtasks** → `parallel_fanout`, with child lane chosen by subtask size
5. **D — 5+ files, >300k tokens, or clean-room** → `codex_cli_long_context`
6. **C — bounded structured work under 200k** → `gpt55_specialist`
7. Fallback → `parent`

Why `F` and `E` are checked before `D`: once decomposition is explicit, Kael should choose the correct child topology rather than collapsing everything into one oversized lane.

## Concurrency and spawn limits
- `max_concurrent_children=10`
- `max_spawn_depth=3`
- research bursts: spawn all immediately and supervise in parallel
- code review across many independent files: 1 `gpt-5.5` leaf per file when each file is bounded
- architecture decisions spanning multiple domains: 1 `gpt-5.4` orchestrator CLI child per domain, with leaves beneath it

## Failure recovery
- `gpt-5.5` timeout → retry once, then mark partial and continue
- Codex CLI failure → inspect sandbox mode and retry with alternate sandbox when appropriate
- native `delegate_task` failure → fall back to `terminal + hermes chat`
- bad child output → do not trust silently; add a follow-up lane and annotate the gap

## Smoke-test commands

### Router examples
```bash
cd ~/.hermes/hermes-agent
./scripts/kael_delegation_router.py --examples
```

### One-shot gpt-5.5 leaf
```bash
# from a live Kael session
# delegate_task(goal="Reply exactly CHILD_OK", model={provider:"openai-codex", model:"gpt-5.5"}, toolsets=[])
```

### Codex CLI long-context lane
```bash
cd ~/.hermes/hermes-agent
codex exec --skip-git-repo-check --sandbox read-only --ephemeral "Reply exactly CODEX_OK"
```

### gpt-5.4 orchestrator CLI lane
```bash
cd ~/.hermes/hermes-agent
hermes chat --provider openai-codex --model gpt-5.4 -s delegation-routing-v2 -Q -q "Reply exactly ORCHESTRATOR_OK"
```

## Worked examples
1. **Review one module diff and return JSON findings**
   - route: `gpt55_specialist`
   - toolsets: `['file', 'terminal']`
2. **Read 6 docs and summarize system drift**
   - route: `codex_cli_long_context`
3. **Audit 6 small files independently for style issues**
   - route: `parallel_fanout` → `gpt55_specialist` children
4. **Update `STATE.md` after reading child reports**
   - route: `parent`
5. **Compare config, git workflow, and doctrine docs, then decide final policy**
   - route: `gpt54_orchestrator_cli`

## Testable contract
The router emits machine-readable JSON so shell checks and future wrappers can assert:
- lane choice
- model/provider
- default toolsets
- command template
- timeout/failure recovery
- fanout metadata

That makes the routing policy harder to skip than a narrative note in a report.
