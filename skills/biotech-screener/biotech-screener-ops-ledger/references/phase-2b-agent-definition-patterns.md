# Phase 2B — Agent Definition Patterns

Concrete examples from the Phase 2B prep (Spec 089, 2026-05-19).
Use as templates when defining new LLM-driven observation agents.

---

## Directory structure

```
agents/<agent-name>/
  IDENTITY.md    — one-line role card
  SOUL.md        — principles, constraints, trigger, format
  HEARTBEAT.md   — execution checklist
```

---

## IDENTITY.md pattern

```
# I am <Name>, the <role>

I operate in observe-only mode. I do not make changes.
```

Real example:
```
# I am Keeper, the Hermes Held Spec Ledger
# I operate in observe-only mode. I do not make changes.
```

---

## SOUL.md structure

Sections in order:
1. **Core Identity** — one paragraph, purpose + boundary
2. **Operating Principles** — 3-5 numbered rules (think like governance, not code)
3. **Mandatory Bans** — no-go zones as hard rules
4. **Output Format** — exact structure the LLM should emit
5. **Autonomy Level** — "observe-only" or similar
6. **Boundaries** — MAY / MAY NOT lists
7. **Edge Cases** — what to do when data is missing, dates mismatch, artifacts absent

DO NOT include:
- Production scoring logic
- Agent hub deployment config
- Cron schedule details (those go in the runbook)

---

## HEARTBEAT.md checklist

Execution checklist, operator-facing. Sections:
1. **Preflight Checks** — what must be true before starting
2. **Inputs** — exact file paths to read
3. **Reasoning** — numbered analytical steps
4. **Output** — what to emit and where
5. **Verification** — sanity checks after output
6. **Failure Recovery** — error handling per failure mode
7. **Operational Notes** — scheduling constraints, known quirks

---

## AGENT_REGISTRY.json entries (shadow phase)

```json
{
  "name": "hermes-held-spec-ledger",
  "status": "shadow",
  "llm_policy": "manual_only",
  "authority_level": "observe_only",
  "requires_preflight": true
}
```

Field semantics:
- `status: shadow` — agent defined but not wired to any cron or scheduler
- `llm_policy: manual_only` — will only run with explicit `--skip-preflight` flag
- `authority_level: observe_only` — cannot write production data or trigger actions
- `requires_preflight: true` — preflight checklist file must be read before LLM prompt

At activation (prep -> live): flip `status` to `active`. Keep other fields.

---

## Runbook structure

Location: `references/hermes-<agent-name>-jobs.md`

Sections:
1. **Purpose** — what the agent evaluates and why
2. **Inputs** — exact ledger/artifact paths the agent reads
3. **Invocation** — exact commands (preflight mode + dry-run mode)
4. **Output** — stdout format, verification
5. **Failure Modes** — table of what can go wrong, symptoms, diagnostic commands
6. **Rollback** — agent shutdown, registry revert, cron removal
7. **Activation criteria** — the checklist from the ops-ledger skill

---

## Example HEARTBEAT patterns

Read artifact, analyze ledgers, write analysis:
```
1. Read <input_path>
2. For each item in the ledger:
   - Check status
   - Evaluate whether conditions are still current
   - Flag any drift from expected state
3. Write analysis to stdout with header "=== <agent-name> ==="
4. Verify: analysis mentions each input item by name
```

Contradiction scan pattern:
```
1. Read latest_state.json
2. Read latest.json (held spec) and latest.json (first fire)
3. For each check ID (C01, C02, ...):
   - Gather evidence
   - Classify: HARD_CONTRADICTION / POSSIBLE_DRIFT / OK
4. If any HARD_CONTRADICTION found: surface with evidence and recommended action
5. Output markdown report
```

---

## Pitfalls

- Agent definition files are NOT the place for cron schedules. That goes in the runbook.
- `run_agent_direct.py` takes `--message` as the user prompt — use `--message DRY_RUN` for testing.
- **`run_agent_direct.py` provides NO tool access** — the LLM cannot read files, run shell commands, or access the filesystem. It only sees what's in the `--message` string and the system prompt (constructed from IDENTITY.md + SOUL.md + HEARTBEAT.md). If the agent needs to analyze ledger contents (e.g. held_spec_ledger/latest.json), the data must be inlined into `--message` via a thin wrapper script. A dry-run against an agent that expects to read files will produce unreadable output (e.g. "LEDGER_UNREADABLE").
- The runner reads IDENTITY.md -> SOUL.md -> HEARTBEAT.md sequentially and concatenates them into the system prompt. Keep each file focused.
- Preflight checks that reference ledger files will fail on first dry-run if ledgers haven't been generated yet. Use `--skip-preflight` for development, then write the preflight to match actual post-builder state.
- Agent files at `agents/<name>/` are tracked in git. The runbook at `references/` is also tracked. Both need proper commits on their own branch.
