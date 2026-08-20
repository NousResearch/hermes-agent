# Codex-as-Hostile-Auditor Pattern

Dispatch Codex CLI as a read-only hostile reviewer against a specific codebase, collect findings, then fix in the controller. This is distinct from the "fix everything" multi-batch pattern — the auditor only reads and reports.

## When to use

- You need a fresh pair of eyes on a codebase you've been working in
- The user says "hostile audit X" or "audit this repo"
- You want findings from a different model family than the controller
- The audit is the safety net, not the agent loop

## The pattern

### 1. Write the audit prompt

Write a strict, self-contained prompt to a temp file. Key elements:

```
You are a read-only hostile reviewer of the CURRENT local source at <path>.
Do not edit any file, do not commit, and do not run broad/long test suites.
Verify each alleged issue against current source and tests — not the public snapshot.
For each ID return: CONFIRMED / ALREADY_FIXED / PARTIAL / NOT_A_DEFECT;
exact file:line evidence; concrete impact; smallest correction and RED regression.
Also identify any current P0/P1 issue the audit missed.
Write only a concise final report to stdout.
```

### 2. Dispatch Codex

```bash
codex exec --dangerously-bypass-approvals-and-sandbox \
  -m gpt-5.6-terra \
  -C /path/to/repo \
  "$(cat /tmp/audit-prompt.md)" \
  2>&1 | tee /tmp/audit-output.log
```

- `--dangerously-bypass-approvals-and-sandbox` is appropriate because the audit IS the safety net — Codex only reads
- `-m gpt-5.6-terra` or `gpt-5.6-sol` — frontier models produce better audits
- Pipe to `tee` so you get the output in the controller session

### 3. Parse findings

Codex will grep, read files, and produce a structured report. Key output sections:
- Per-ID verdict (CONFIRMED/ALREADY_FIXED/PARTIAL/NOT_A_DEFECT)
- File:line evidence
- Impact assessment
- Smallest correction
- RED regression test
- New P0/P1 findings the original audit missed

### 4. Fix in controller

Do NOT dispatch Codex to fix. The controller:
- Verifies each finding against live source (Codex summaries are self-reports)
- Fixes P0/P1 items directly
- Runs the verification bar
- Reports claim-bound results

## Pitfalls

- **Codex may hit auth errors** (MCP transport channel closed) — this is normal for read-only audits, the findings are still valid
- **Codex may re-read the same files multiple times** — the prompt should say "do not run broad/long test suites" to keep it focused
- **The audit log can be large** (5K+ lines) — use `tail` or `grep` to extract the final report section
- **Codex findings need controller verification** — a CONFIRMED finding with file:line evidence is strong, but still verify the cited line before acting
- **Codex can produce false positives** — in the 2026-07-13 agent-memory-kits audit, Codex claimed `doctor.py` used `~/.hermes/semantic-memory.db` while the actual code used `~/.local/share/semantic-memory`. The finding was wrong. Always verify Codex audit findings against live source before acting on them. A `grep` for the claimed string is a 2-second sanity check that prevents acting on fabricated evidence.

## Worked examples

### AiDENs hostile audit (2026-07-13)
- Dispatched Codex with 7 pre-identified KIT issues to verify
- Codex confirmed 6, found 1 PARTIAL, discovered 1 new P1
- Controller fixed all in-scope items, ran `verify_current.sh`, produced claim-bound report
- 90,702 tokens spent

### agent-memory-kits audit (2026-07-13)
- Dispatched Codex with 7 KIT issues + "find anything missed"
- Codex confirmed all 7, found 1 new P1 (inconsistent default stores)
- Controller received findings, no fixes dispatched (audit-only pass)

### semantic-memory-mcp audit (2026-07-13)
- Dispatched Codex with 10 MCP issues + "find anything missed"
- Codex confirmed/partialed all 10, found 1 new P1 (unbounded HTTP cardinality)
- Controller received findings, no fixes dispatched (audit-only pass)
