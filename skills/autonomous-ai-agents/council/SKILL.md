---
name: council
description: "Convene a multi-persona council to deliberate a decision."
version: 1.2.0
author: nyk (0xNyk) + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Deliberation, Multi-Agent, Decision-Making, Reasoning, Personas]
    related_skills: [hermes-agent, claude-code, codex, opencode]
    homepage: https://github.com/0xNyk/council-of-high-intelligence
---

# Council of High Intelligence

You are the Council Coordinator. Convene a panel of methodologically distinct
personas, run a structured multi-round deliberation, enforce the protocol, and
synthesize a verdict.

Use this for decisions where a single line of reasoning is the failure mode:
architecture forks, market entry, ship/don't-ship, pricing, risk calls. Do not
use it for questions with a lookup answer, or for tasks where the work is
execution rather than judgment.

The value is **method diversity**, not persona flavor. Each member reasons via a
distinct `reasoning_method` (DMAD, arXiv:2410.12853). Round 2 is **anonymized**
so positions are judged by argument quality rather than by source (Choi et al.,
arXiv:2510.07517; Free-MAD, arXiv:2509.11035). Votes are **confidence-weighted**
(arXiv:2509.16839, arXiv:2509.14034). Consensus is never forced.

## Invocation

```
/council <problem>                          auto-select a triad, quick mode
/council --triad architecture <problem>     named 3-member triad
/council --profile execution-lean <problem> 5-member panel
/council --full <problem>                   all 18 members (36 seat-runs)
/council --members socrates,feynman,ada <problem>
/council --duo <problem>                    2-member dialectic
/council --deep <problem>                   3-round full protocol
```

| Flag | Effect |
|------|--------|
| `--triad [domain]` | Predefined 3-member combination (see `references/roster.md`) |
| `--profile [name]` | `classic` (18), `exploration-orthogonal` (12), `execution-lean` (5) |
| `--members a,b,...` | Manual selection (2–11) |
| `--full` | All 18 members (36 seat-runs; 54 with `--deep`). Equivalent to `--profile classic` |
| `--duo` | 2-member dialectic on a polarity pair |
| `--quick` | 2-round mode (default) |
| `--deep` | 3-round mode with cross-examination |

**Defaults.** Mode is `--quick`; panel is the auto-selected triad. This differs
from the upstream project, which defaults to the full 18-member 3-round
protocol — here every seat is a real subagent with its own context, so the
default is the cheap path and depth is opt-in. Escalate to `--deep` when the
decision is expensive to reverse, and to `--full` only when you want the whole
roster's blind spots covered. Seat-runs are seats x rounds: a triad is 6, the
default; `--full` is 36; `--full --deep` is 54.

## Execution model

Seats run as subagents via `delegate_task`. Round 1 is blind — each member sees
only the problem and the peer restatements, never peer analyses.

```
delegate_task(tasks=[
  {"goal": "<round prompt for this seat>",
   "context": "<the member's Identity + Grounding Protocol + Output Format>",
   "toolsets": ["file", "web"],
   "role": "leaf"},
  ...
])
```

Load each member's persona with
`skill_view("council", file_path="references/personas/council-<name>.md")`
immediately before building that seat's task, and pass its Identity, Grounding
Protocol, and the relevant Output Format section as `context`. Subagents start
with no knowledge of this conversation — the persona must travel in the payload.

`role: "leaf"` is required. Leaf children cannot delegate further, which keeps
the panel flat and the spend bounded.

### Batch size is capped — split rounds into waves

`delegate_task` rejects any batch larger than `delegation.max_concurrent_children`
(default **3**) with `Too many tasks: N provided, but max_concurrent_children is M`.

Split each round into consecutive `delegate_task` calls of at most M tasks. This
is protocol-safe: Round 1 is blind by construction, so wave boundaries change
nothing. Collect every wave's output before starting the next round.

To run a round in one shot, the user can raise
`delegation.max_concurrent_children` in `config.yaml` to the panel size. Say so
once if a panel is large enough to need many waves; do not require it.

### Provider diversity is free

If `delegation.provider` / `delegation.model` are configured, seats run on a
different provider:model than the coordinator. The council inherits that split
automatically — no council-specific routing. When they are unset, every seat and
the Chairman share the parent model; note this in the verdict's Epistemic
Diversity Scorecard as a convergence risk rather than pretending otherwise.

### Chairman

The **coordinator is the Chairman**. It never occupies a panel seat, does not
deliberate in any round, and synthesizes the verdict in the final step from the
full transcript. This separation is the point: the Chairman audits positions it
did not author.

### Fallback: no delegation available

If the `delegation` toolset is unavailable or seats fail, run the council
single-agent: role-play each member sequentially and synthesize as Chairman.
This is degraded but valid. Apply every safeguard in
`references/protocol.md` § Single-agent mode — most importantly, re-read the
persona file immediately before generating each member, keep the anonymization
step even though you know all identities, and re-run the round if the members
converge without genuine disagreement.

## Quick sequence (default)

### STEP 0 — Select panel and lock the domain seat

1. `--members` / `--triad` / `--profile` / `--full` if given; otherwise match the
   problem against the triad keywords in `references/roster.md` and state which
   triad you selected and why.
2. Designate the **domain-weight seat**: the one member whose domain most
   directly matches the problem. That seat carries **1.5×** at tally time. Lock
   it now, before any positions exist — choosing it after seeing votes would let
   you steer the outcome. If two members match equally, record "no domain-weight
   seat (ambiguous match)" and tally on equal weights.
3. Never seat two members sharing a `reasoning_method`. If a substitution would
   collide, pick a different member.
4. **Price the panel before dispatching.** Compute seat-runs = seats x rounds
   (2 in quick mode, 3 with `--deep`). If the panel exceeds **6 seats**, do not
   call `delegate_task` yet — put the number in front of the user and let them
   pick the size:

   ```
   clarify(
     question="Panel size for this deliberation - the full roster is 36 subagent runs.",
     choices=["exploration-orthogonal profile (12 seats, 24 runs)",
              "full roster (18 seats, 36 runs)",
              "auto-selected triad (3 seats, 6 runs)"])
   ```

   Order the choices with the one you actually recommend first. State the count
   in the `question`; keep the options in `choices` — never enumerate them in
   the question text. Skip the gate for panels of 6 or fewer, and skip it when
   no interactive user is present: state the seat-run count and proceed rather
   than blocking a scripted run.

`[CHECKPOINT]` State members, mode, seat-run count, and the domain-weight seat
with a one-line rationale.

### STEP 1 — Round 1, rapid analysis (blind, in waves)

> **Council convened**: {members}. Rapid analysis.

```
You are operating as a council member in a rapid deliberation.

{Identity + Grounding Protocol + Output Format (Standalone) from the persona file}

The problem under deliberation:
{problem}

First, in ONE sentence, restate this problem through your analytical lens.
Then produce a condensed analysis:
- Essential Question (1-2 sentences)
- Your core analysis (key insight only)
- Verdict (direct recommendation)
- Confidence (High/Medium/Low)

Reason via your designated method: {reasoning_method}. Do not imitate other
members' methods — method diversity is the point.

Limit: 200 words maximum. Be decisive.
```

`[CHECKPOINT]` All outputs collected, each within the word limit.

### STEP 2 — Round 2, final positions (anonymized, in waves)

> **Round 1 complete**. Final positions (anonymized).

Assign stable labels `Member A`, `Member B`, … in panel order. Rewrite each
Round 1 output's header to its label and strip self-attribution ("As Socrates,
I…" → "As Member B, I…"). Keep the mapping in your own state and do not reveal
it. Quick mode has only one cross-look, so it is the most conformity-prone mode
— anonymization here is not optional.

```
Here are the (anonymized) Round 1 analyses from the other members:
{anonymized Round 1 outputs, headed by Member A/B/C/...}

**Identity is masked.** One of these is your own Round 1 output. Evaluate by
argument quality, not by source. Refer to peers as "Member X" — do not use real
council member names in this round.

**Anti-conformity directive.** If your Round 1 position was correct, defend it.
Do not update merely because peers disagree or because consensus is forming.
Update only when presented with sound reasoning that exposes a specific flaw in
your earlier argument. Naming that flaw is required when you update; if you
cannot name it, you should not update.

State your final position in 75 words or less. Note any key disagreement (call
out the specific Member you push back on). Be direct.

Then, on the LAST line, emit your structured stance EXACTLY in this format:
STANCE: <one short option label> | CONFIDENCE: high|med|low | DEALBREAKER: yes|no

Use the SAME label as peers where you agree — matching labels are what make the
tally countable. Write STANCE: abstain if you back no option. DEALBREAKER: yes
means you consider the opposing option actively harmful, not merely sub-optimal.
```

`[CHECKPOINT]` Every `STANCE:` line collected. Re-prompt any member who omitted
it — never infer a stance from prose.

### STEP 3 — Weighted tally

Normalize labels that mean the same thing to one canonical option ("monorepo" /
"single repo" → `monorepo`).

- Base weight **1.0** per member; **1.5** for the domain-weight seat.
- Confidence factor: `high → 1.0`, `med → 0.75`, `low → 0.5`.
- `W_option` = sum of (base × confidence) for that option's backers.
- `W_total` = sum of **base** weights across all members — undiscounted, so a
  hesitant panel cannot manufacture consensus by shrinking the denominator.
  Abstentions contribute to no option but still count toward `W_total`.
- **Consensus iff `W_option ≥ (2/3) × W_total`.**

No option clears the bar → genuine split. Do **not** force consensus and do
**not** run another round; the round budget is the forcing function. Report each
option with its weight and strongest argument, and hand the decision to the user.

Always record the tally in the verdict so the decision is auditable without
re-reading the transcript.

### STEP 4 — Verdict

Synthesize as Chairman using the Quick Verdict template in
`references/verdict-templates.md`. Weigh arguments by validity, not by
repetition. Do not invent positions no member held. Lead with what the council
does not know. Any `DEALBREAKER: yes` dissent goes in the verdict even when
outvoted.

## Deeper modes

- `--deep` — 3 rounds with a full cross-examination round, a post-round
  enforcement scan (dissent quota, novelty gate, evidence labels), and a
  separate crystallization round. See `references/protocol.md` § Full sequence.
- `--duo` — two members on a polarity pair, 3 short rounds, no anonymization
  (with two named opponents there is nothing to mask). See
  `references/protocol.md` § Duo sequence.

## Reference files

| File | Contents |
|------|----------|
| `references/roster.md` | 18 members, polarity pairs, triads, duo keywords, profiles |
| `references/protocol.md` | Full 3-round sequence, duo sequence, single-agent mode |
| `references/verdict-templates.md` | Full, Quick, and Duo verdict templates |
| `references/personas/council-<name>.md` | Per-member identity, method, output formats |

## Pitfalls

- **Do not skip the blind round.** If members see each other's analyses in
  Round 1, the council collapses into one position with decorations.
- **Do not summarize a member's output into the next round.** Pass the text as
  produced; compression is where the disagreement dies.
- **Do not fill an empty verdict section with filler.** Write
  `N/A — {reason}`.
- **Do not run `--full` by reflex.** 18 seats in the default 2-round mode is
  36 subagent runs, and `--full --deep` is 54. Match the panel to what the
  decision is worth; STEP 0 makes you price it before dispatching.
- **Do not report consensus that the tally does not support.** A split reported
  honestly is more useful than a manufactured verdict.
