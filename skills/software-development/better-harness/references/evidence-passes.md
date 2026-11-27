# Evidence Passes — Three Independent Lanes (Hermes adaptation)

The lead launches exactly three fresh, read-only leaf subagents in parallel via
`delegate_task(tasks=[...])`. Each child receives ONLY its own brief plus the
relevant slice of the Step-1 evidence bundle. **Children must not delegate,
must not read another brief, and must not assign final severity or scores.**

Common output contract (each lane, user's language):

- scope, provider/window coverage, material omissions, confidence;
- up to three representative Task Episodes (Session lane) or the strongest
  capability evidence (Project/Agent lanes);
- normally **three to five potential findings**, up to three in quick mode;
  fewer when evidence is weak — **never fill a quota**;
- natural-language candidates; no fixed JSON schema; each must make its
  consequence, evidence, owner boundary, and uncertainty understandable;
- end with: **"The claims the lead must not make from this evidence:"** — an
  explicit list of conclusions this lane's evidence cannot support.

---

## Lane 1 — Session Evidence

**Input:** the lead's `session_search(query=..., limit=...)` results within the
window (7 days quick / 30 days normal) and episode limit. Do not pass raw
session dumps.

**Source facts (Hermes):** `session_search` discovery shape returns sessions
with titles, bookends, and FTS5 match windows. Read no more than a few windows
around important claims (`around_message_id`); never page entire transcripts.

**Reconstruct Task Episodes** — one goal + one acceptance boundary:

- request summaries describe user intent; repeated turns inside one Episode are
  not repeated-work evidence;
- a change, check, handoff, completion marker, or assistant statement alone
  does not prove acceptance or delivery;
- only a reviewed check relevant to the final change closes validation;
- direct feedback, provider-confirmed delivery/recovery, or relevant validation
  can support an outcome; otherwise keep it a lead or `Unobserved`.

**Friction attribution** — assign to exactly one of: `Harness`, `Repository`,
`Model`, `Requirement`, `External`, `Task complexity`, `Unknown`. Use
`Harness` only when an observed mechanism leads through behavior to a
consequence.

**Look for:** ① a stable repeated workflow across ≥2 distinct comparable Task
Episodes — separate *procedure demand* (repeatable steps + validation gates)
from *knowledge demand* (short decisions, corrections, traps, preferences);
② repeated validation behavior (tests/lint/build/review/regression/
failure-rerun/delivery/rollback) — repetition does not prove the loop is
effective without a later accepted outcome; ③ repeated bug work needing
correlated diagnosis; ④ consequential one-offs where a control, permission,
missing diagnostic, correction, or failed recovery materially changed the task;
⑤ blind spots that prevent a named project decision.

**Boundary:** do not inspect project files, configured assets, memory bodies,
or other briefs. Missing facts are unavailable evidence, not zero activity.

---

## Lane 2 — Project Harness Evidence

**Input:** target repo path, scoped git history slice, current diff, and the
compact asset counts (to notice zero-skills situations). Open at most **3
owners in quick mode, 5 in normal mode**.

**Hermes commands (leads, not conclusions):**

```bash
git log --oneline -30 --since="<window>"      # history slice
git diff --stat HEAD~1 HEAD                   # current change
git status --short
```

`search_files` / `read_file` for: README, AGENTS.md, CONTRIBUTING.md, docs,
CI config, tests, package manifests + lockfiles, scripts (setup/doctor/health/
reset), hook or gate config. Churn and method length are risk leads — use them
to choose where to exercise the five capabilities, not as dimensions.

**Judge (map into the five Agent Work Loop dimensions):**

| Capability | Evidence to inspect |
| --- | --- |
| Context Map | entry docs as navigation maps, not encyclopedias; which dirs/commands to use and avoid; separation of source/generated/schema/migration |
| Environment Readiness | runtime pins, lockfiles, deterministic seeds/fixtures, reset flow, health checks, example env files, mock services; Docker/devcontainer is positive but not proof of reproducibility |
| Fast Feedback | low-cost smoke/lint/typecheck/focused tests; affected-check routing; failure artifacts with locations; slow full-suite-only feedback is a defect |
| Quality Gates | lint/type/architecture/schema/generated-drift/security checks that mechanically enforce rules; rules only in prose are not gates |
| Change Safety | hooks, permission/sandbox rules, approval flows, dry-run, rollback docs, audit trails; workflow files do not prove required checks — mark `UNVERIFIED` |

**Verify core risk first:** for the most material core/failure path (from
history + diff), check whether the real route is discoverable, runnable,
readable, correlatable, verifiable, and safe/reversible. Map each observability
result into the affected capability; observability is cross-cutting evidence,
not a sixth dimension.

**Boundary:** no session facts, no user memory, no prior reports, no other
briefs. Static declarations prove intent or presence — not runnable behavior,
enforcement, delivery, or ownership.

---

## Lane 3 — Agent Customize Evidence (Hermes layer)

**Input:** the lead's inventory envelopes — `skills_list` (categories +
descriptions), `memory` targets, `cronjob list`, active profile config. The
lead supplies these; **do not rerun scanners** — consume the envelopes.

**Interpret the baseline (counts route inspection only):**

- zero project skills makes repeated-work discovery important, but may be
  correct when built-ins or simple instructions already own the work;
- many skills → review trigger quality, procedural content, routing, overlap,
  and use evidence;
- many memories → review exact/near duplicates, conflict, staleness,
  retrieval value — count never earns Learning Capture credit;
- many cron jobs → review trigger, state, validation path, safety boundary,
  stop rule;
- a short AGENTS.md is not defective by length; check whether it routes to
  current owners and states the few always-needed constraints.

**Asset four axes — keep separate:**

- **Presence:** configured / resolved-active / absent-in-scope / unavailable
- **Content:** relevant / current / discoverable / executable / maintainable
- **Use:** routed and applied in the same Task Episode; a file read, invocation
  name, or count does not prove use
- **Outcome:** effective only after a comparable later outcome without
  guardrail regression

**Build the asset coverage map:** for each inspected owner chain (skill,
memory, cron, plugin), summarize only what evidence can support — scope,
canonical owner, trigger/routing surface, provenance/freshness, and the four
axis states without collapsing them. For memory, distinguish exact-title
collision, possible semantic overlap, conflict/staleness, demonstrated
coverage, and unknown coverage. Never return user-home memory paths or titles;
provider, scope, count, and scan state are sufficient.

**Boundary:** no application code, no session facts, no memory bodies beyond
authority, no raw transcripts. A failed inventory envelope stays unavailable.

---

## Reconciliation (lead only)

After the three lanes return: retain every candidate; merge only candidates
with the same target, observed consequence, owner, and repair route; keep a
working reason for every deferral; then validate consequence, cause chain,
smallest owner, evidence boundary, confidence, and verifier; assign final
severity and one primary check; derive dimension scores from the evidence
ceiling ladder in `agent-work-loop.md`; freeze before drafting. Do not launch a
fourth evidence agent.
