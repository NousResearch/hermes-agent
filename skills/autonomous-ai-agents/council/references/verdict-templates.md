# Verdict templates

Do not add, remove, or rename sections. Fill each faithfully or write
`N/A — {reason}` if the section is genuinely empty for this session.

---

## Council Verdict (`--deep` / `--full`)

```markdown
## Council Verdict

### Problem
{Original problem statement}

### Council Composition
{Members convened, mode used, and selection rationale}

### Model Routing
{Coordinator model as Chairman; seat model from delegation.model if configured,
otherwise "seats inherit the coordinator model (single-model session)". Note any
seats that failed and were re-run or role-played.}

### Acceptable Compromises
{What this verdict gives up, named explicitly. One bullet per compromise, ≤2
sentences each. If nothing is being given up, say so and explain why — most
non-trivial decisions trade something.}

### Kill Criteria
{The specific observable conditions that would falsify this verdict. Each must be
(a) observable without re-convening the council, (b) tied to a measurable
threshold or event, and (c) achievable within a stated time window.
Format: "If <X> observed by <date>, the verdict is invalidated and we should <Y>."}

### Concrete Next Step
{Exactly one action. Named, doable, owned. Format: "<verb> <object> by <date>."
Not "consider," not "explore" — verbs that produce an artifact: write, push,
merge, run, file, measure.}

### Unresolved Questions
{Questions the council could not answer — inputs needed from the user. Lead with
what the council does NOT know.}

### Recommended Next Steps
{Additional concrete actions beyond the single Concrete Next Step, ordered by
priority. If the Concrete Next Step is sufficient, write "N/A — see Concrete
Next Step."}

### Consensus & Agreement
{The position that survived deliberation and what members converged on — or
"No consensus reached" with explanation.}

### Vote Tally
{One line per option: `<option> — <weight> (<backers with confidence>)`. Mark the
1.5× domain-weight seat. State the threshold and whether it was cleared. Example:
- `monorepo — 2.25 (Ada [1.5x domain, high], Feynman [med -> 0.75])` — did not clear 2.333
- `polyrepo — 1.0 (Torvalds [high])`
- W_total 3.5 · threshold 2.333 · **no option carries → escalated to user**
If no seat carried 1.5x (ambiguous match), say so.}

### Key Insights by Member
- **{Name}**: {Their most valuable contribution in 1-2 sentences}

### Points of Disagreement
{Where positions remained irreconcilable}

### Minority Report
{Dissenting positions and their strongest arguments. Every DEALBREAKER: yes
appears here even when outvoted.}

### Epistemic Diversity Scorecard
- Perspective spread (1-5): {how orthogonal the viewpoints were}
- Model spread (1-5): {1 if every seat and the Chairman shared one model}
- Evidence mix: {% empirical / mechanistic / strategic / ethical / heuristic}
- Convergence risk: {Low/Medium/High with reason}

### Follow-Up
After acting on this verdict, revisit: was it useful? Was the recommended action
taken? What happened? {A prompt for the user, not filled by the council.}

---

### Session Metadata
schema_version: 1
mode: deep | full | quick | duo | triad
panel_size: <N>
rounds_run: <N>
seats_role_played: <N>          # single-agent fallback seats, 0 if all delegated
tools_used: yes | no
input_tokens_estimate: ~<N>k
output_tokens_estimate: ~<N>k
duration_seconds: ~<N>
delegation_waves: <N>
fallbacks_triggered: <list of "member -> reason", or "none">
```

---

## Quick Verdict (default mode)

```markdown
## Quick Council Verdict

### Problem
{Original problem statement}

### Panel
{Members and selection rationale}

### Recommended Action
{Single concrete recommendation}

### Kill Criteria
{Observable conditions that would falsify this verdict. Required.
Format: "If <X> observed by <date>, the verdict is invalidated and we should <Y>."}

### Concrete Next Step
{Exactly one action. Required. Format: "<verb> <object> by <date>."
Artifact-producing verbs only — no "consider" or "explore".}

### Acceptable Compromises (optional)
{What this verdict gives up. Skip only if genuinely trivial.}

### Positions
- **{Name}**: {Core position in 1-2 sentences}

### Consensus
{Majority position or "Split" with explanation}

### Vote Tally
{One line per option: `<option> — <weight> (<backers with confidence>)`. Mark the
1.5× domain-weight seat, state the threshold and whether it cleared. If split:
"no option cleared 2/3 → escalated to user".}

### Key Disagreement
{The most important point of divergence}

### Follow-Up
After acting on this verdict, revisit: was this useful? What happened?

---

### Session Metadata
schema_version: 1
mode: quick
panel_size: <N>
rounds_run: 2
seats_role_played: <N>
tools_used: yes | no
input_tokens_estimate: ~<N>k
output_tokens_estimate: ~<N>k
duration_seconds: ~<N>
delegation_waves: <N>
fallbacks_triggered: <list or "none">
```

---

## Duo Verdict (`--duo`)

```markdown
## Duo Verdict

### Problem
{Original problem statement}

### The Dialectic
**{Member A}** ({their lens}) vs **{Member B}** ({their lens})

### What This Means for Your Decision
{How to use these opposing perspectives — the user decides}

### {Member A}'s Position
{Core argument in 2-3 sentences}

### {Member B}'s Position
{Core argument in 2-3 sentences}

### Where They Agree
{Unexpected convergence, if any}

### The Core Tension
{The irreducible disagreement and what drives it}

### Concrete Next Step
{Exactly one action — the decision a reader can take after weighing both sides.
Required even in duo mode. Format: "<verb> <object> by <date>."}

### Kill Criteria (encouraged)
{Observable conditions that would tip the balance toward the other side after
acting on the Concrete Next Step. Encouraged but not required — duo is
dialectic, not decision-issuing.}

### Follow-Up
After deciding, revisit: which perspective proved more useful? What happened?

---

### Session Metadata
schema_version: 1
mode: duo
panel_size: 2
rounds_run: 3
seats_role_played: <N>
tools_used: yes | no
input_tokens_estimate: ~<N>k
output_tokens_estimate: ~<N>k
duration_seconds: ~<N>
delegation_waves: <N>
fallbacks_triggered: <list or "none">
```
