# Council protocol — full, duo, and single-agent sequences

Read `SKILL.md` first. This file covers the modes it defers: the 3-round `--deep`
sequence, the `--duo` dialectic, and single-agent mode when `delegation` is
unavailable.

All seat dispatch follows the SKILL.md execution model: `delegate_task` with
`role: "leaf"`, personas passed in `context`, and rounds split into waves of at
most `delegation.max_concurrent_children`.

---

## Full sequence (`--deep`)

Follow the steps in order. Do not skip steps or merge rounds.

### STEP 0 — Select panel, lock the domain seat

As in SKILL.md STEP 0: resolve the panel, designate the 1.5× domain-weight seat
*before* any positions exist, and reject any panel with duplicate
`reasoning_method` values. The panel-size gate applies here too, and bites
harder — this sequence is 3 rounds, so a full roster is 54 seat-runs rather
than 36. Price it and run the `clarify` step for any panel above 6 seats.

`[CHECKPOINT]` Members, mode, seat-run count, domain-weight seat + rationale.

### STEP 1 — Problem restate gate

Before any analysis, every member restates the problem. This catches
wrong-question failures before three rounds are spent on them.

```
{Identity + Grounding Protocol from the persona file}

The problem under deliberation:
{problem}

Before you begin analysis, restate this problem in TWO parts:
1. **Your restatement**: One sentence capturing the core question through your
   analytical lens.
2. **Alternative framing**: One sentence reframing the problem in a way the
   original statement may have missed.

Do NOT begin your analysis yet. 50 words maximum total.
```

`[CHECKPOINT]` If any restatement diverges sharply from the original problem,
surface it to the user — it may be a framing bug worth fixing before
deliberating. Include all restatements in the Round 1 prompt so members see each
other's framings.

### STEP 2 — Round 1, independent analysis (blind)

> **Council convened**: {members}. Beginning Round 1 — independent analysis.

Each member sees only the problem and the restatements. No peer analyses.

```
You are operating as a council member in a structured deliberation.

{Identity + Grounding Protocol + Output Format (Standalone) from the persona file}

The problem under deliberation:
{problem}

Here is how each member reframed the problem:
{all restatements from STEP 1}

Reason via your designated method: {reasoning_method}. Do not imitate other
members' methods — method diversity is the point (DMAD, arXiv:2410.12853).
Produce your independent analysis using your Output Format (Standalone).
Do NOT try to anticipate what other members will say.

Limit: 400 words maximum.
```

`[CHECKPOINT]` All Round 1 outputs collected, each within the word limit and
following the member's Output Format.

### STEP 3 — Round 2, cross-examination (anonymized)

> **Round 1 complete** ({N} analyses). Beginning Round 2 — cross-examination.

**Anonymization.** Round 2 masks identities to prevent conformity from social
signal (Choi et al., arXiv:2510.07517; Free-MAD, arXiv:2509.11035;
arXiv:2511.07784).

1. Build a stable label map for the session: `Member A` → first panel member,
   `Member B` → second, and so on. Labels stay fixed for the whole round.
2. Rewrite each Round 1 output's header to its label and strip in-body
   self-references ("As Socrates, I…" → "As Member B, I…"). Change nothing else.
3. Keep the mapping in coordinator state only. Restore it for Round 3, the
   tally, and the verdict transcript.

**Execution.** Panel ≤ 4: run sequentially, each member seeing prior Round 2
responses under the same labels. Panel ≥ 5: run in parallel waves, each member
seeing all anonymized Round 1 outputs.

```
You are council-{name} in Round 2 of a structured deliberation.

{Identity + Grounding Protocol + Output Format (Council Round 2) from the persona file}

**Identity is masked in this round.** The Round 1 analyses below are labeled
Member A, Member B, … — you do not know which colleague produced which. One of
them is your own Round 1 output, anonymized along with the rest. Evaluate by
argument quality, not by source. Do not try to guess identities and do not
reference any council member by their real name in this round.

Here are the (anonymized) Round 1 analyses:

{anonymized Round 1 outputs, headed by Member A/B/C/...}

**Anti-conformity directive.** If your Round 1 position was correct, defend it.
Do not update merely because peers disagree, because consensus is forming, or
because a position is repeated by multiple members. Update only when presented
with sound, validity-aligned reasoning that exposes a specific flaw in your
earlier argument. Naming that flaw is required when you update; if you cannot
name it, you should not update.

Now respond using your Output Format (Council Round 2):
1. Which member's position do you most disagree with, and why? Engage their
   specific claims. Refer to them as "Member X".
2. Which member's insight strengthens your position? How? Refer to "Member Y".
3. Restate your position in light of this exchange, noting any changes.
4. Label your key claims: empirical | mechanistic | strategic | ethical | heuristic

Limit: 300 words maximum. You MUST engage at least 2 other members by label.
```

`[CHECKPOINT]` Restore the label → name mapping. Keep the transcript in both
forms: anonymized (what members saw) and de-anonymized (for the STEP 6 audit).

### STEP 4 — Enforcement scan

Run all checks on Round 2 outputs in one pass.

**`[VERIFY]` Dissent quota** — at least 2 members must raise non-overlapping
objections. If fewer:

```
Your Round 2 response agreed with the emerging consensus. The council requires
dissent for quality. State your strongest objection to the majority position in
150 words. What are they getting wrong?
```

**`[VERIFY]` Novelty gate** — each response must contain at least one new claim,
test, risk, or reframing absent from that member's Round 1. If missing:

```
Your Round 2 response restated your Round 1 position without engaging the
challenges raised. Address {member}'s challenge to your position directly.
What changes?
```

**`[VERIFY]` Agreement check** — if more than 70% converge on one position, send
the counterfactual prompt to the 2 most likely dissenters:

```
Assume the current consensus is wrong. What is the strongest alternative, and
what evidence would flip the decision?
```

**`[VERIFY]` Evidence labels** — confirm claims are tagged
`empirical | mechanistic | strategic | ethical | heuristic`. Note reasoning
monoculture if more than 80% share one type.

**`[VERIFY]` Anti-recursion** — Socrates re-asking an answered question triggers
the hemlock rule (force a 50-word position). Any pair exchanging more than 2
messages gets cut off.

### STEP 5 — Round 3, crystallization

> **Cross-examination complete**. Round 3 — final positions.

```
Final round. State your position declaratively in 100 words or less.
Socrates: you get exactly ONE question. Make it count. Then state your position.
No new arguments — only crystallization of your stance.

Then, on the LAST line, emit your structured stance EXACTLY in this format so
the council can tally it:
STANCE: <one short option label> | CONFIDENCE: high|med|low | DEALBREAKER: yes|no

- STANCE must be a terse label for the option you back (e.g. "monorepo",
  "ship now", "do not ship"). Use the SAME wording as peers where you agree —
  matching labels are what make the tally countable. If you genuinely back no
  option, write STANCE: abstain.
- DEALBREAKER: yes means you consider the opposing option actively harmful, not
  merely sub-optimal — it is surfaced in the Minority Report even if outvoted.
```

`[CHECKPOINT]` Collect every `STANCE:` line. Normalize synonymous labels to one
canonical option. Re-prompt any member who omitted the line — never infer a
stance from prose.

### STEP 6 — Weighted tally

Identical to SKILL.md STEP 3: base weight 1.0 (1.5 for the domain seat),
confidence factor high 1.0 / med 0.75 / low 0.5, consensus iff
`W_option ≥ (2/3) × W_total` where `W_total` sums **base** weights.

No option clears the bar → genuine split. Do not force consensus and do not run
another round. Present each option with its weighted tally and strongest
argument, and hand the decision to the user. An exact tie between two options
below threshold is reported as a live split — the domain seat has already been
applied, so there is deliberately no further mechanical tiebreaker.

### STEP 7 — Verdict (Chairman)

Synthesize as Chairman from the full de-anonymized transcript, using the Council
Verdict template in `verdict-templates.md`.

- Weigh arguments by validity, not by repetition or seniority.
- Surface genuine disagreement; never invent a position no member held.
- Lead with what the council does not know.
- Fill each section faithfully or write `N/A — {reason}`.

### STEP 8 — Session metadata

Append the metadata block from `verdict-templates.md` below a separator. Fill
every field knowable from coordinator state; write `~unknown` for anything the
runtime does not expose. `schema_version: 1` is fixed so sessions stay
aggregatable.

---

## Duo sequence (`--duo`)

Two members on a polarity pair. Dialectic, not decision-issuing.

### DUO STEP 0 — Select the pair

Use `--members a,b` if given; otherwise match the problem against the duo
keywords in `roster.md`. State the pair and the tension it represents.

### DUO STEP 1 — Opening positions (parallel)

> **Duo convened**: {A} vs {B} — {tension}.

```
You are operating as one half of a structured dialectic with one opponent.

{Identity + Grounding Protocol + Output Format (Standalone) from the persona file}

The problem under deliberation:
{problem}

First, in ONE sentence, restate this problem through your analytical lens.
Then state your position using your Output Format (Standalone).

Limit: 300 words maximum.
```

### DUO STEP 2 — Direct response (parallel)

**Anonymization does not apply in duo mode.** With two named opponents, identity
cannot be meaningfully masked — each side knows the other by elimination — and
the dialectic depends on each knowing the other's analytical lens. The
conformity failure mode that motivates Round 2 anonymization does not arise in a
2-member exchange.

```
Your opponent ({other member}) argued:

{other member's Round 1 output}

**Anti-conformity directive.** If your Round 1 position was correct, defend it.
Concede only what is specifically and validly disproved — not what merely sounds
forceful. Name the flaw in your earlier argument when conceding; if you cannot
name it, the concession is not warranted.

Respond directly:
1. Where are they wrong? Engage their specific claims.
2. Where are they right? Concede what deserves conceding.
3. Restate your position, strengthened by this exchange.

Limit: 200 words maximum.
```

### DUO STEP 3 — Final statements (parallel)

```
Final statement. 50 words maximum. State your position. No new arguments.
```

### DUO STEP 4 — Verdict

Synthesize as Chairman using the Duo Verdict template. The Chairman must not be
either duo member — as coordinator you never hold a seat, so this holds by
construction.

---

## Single-agent mode

When the `delegation` toolset is unavailable or seats fail, role-play each
member sequentially and synthesize as Chairman. Single-agent is the degraded
path, not a broken one — but the failure mode is real: every member sounds the
same because one model generates them all. All six safeguards are mandatory.

1. **Read before writing.** Load the persona file with
   `skill_view("council", file_path="references/personas/council-<name>.md")`
   immediately before generating that member's analysis — not in a batch at the
   start. The identity and analytical-method sections prime the persona, and
   that priming decays across intervening generations.
   A repeat view of an unchanged persona file returns a short "content
   unchanged since it was loaded earlier" stub instead of the text. That is
   correct token behaviour but it does not re-prime anything, so when you get
   the stub — which is what rounds 2 and 3 will get — restate that member's
   Identity and Grounding Protocol inline from the earlier result before
   generating. The safeguard is the re-priming, not the tool call.
2. **Enforce distinct output formats.** Each member has a different Standalone
   Output Format (Sun Tzu: Terrain Map → Position Assessment → Decisive Point;
   Machiavelli: Incentive Map → Stated vs Revealed → Uncomfortable Truth;
   Aurelius: Control Boundary → Clear-Eyed Assessment → Duty → Resilient Path).
   Following them forces structural diversity that prose alone will not produce.
3. **Active disagreement.** The anti-conformity directive is load-bearing here.
   If all members converge, something is wrong — regenerate with an explicit
   instruction to find the strongest objection.
4. **Word limits.** Self-generated outputs run long. Enforce the limits strictly.
5. **Evidence labels.** Require every member to tag claims. This is what stops
   the coordinator from generating unsupported assertions across all seats.
6. **Anonymization still applies.** Even though you know all identities, perform
   the Round 2 anonymization step. It is part of the audit record, and it forces
   evaluation by argument quality rather than by which member you assigned a
   position to.

Record `provider_count: 1` and note in the verdict's Epistemic Diversity
Scorecard that provider spread is 1 and convergence risk is correspondingly
elevated.
