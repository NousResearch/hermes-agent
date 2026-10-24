# Skill Lifecycle Design Inputs: SkillOps and Skill-V

**Sources** (full text read, not secondary summaries):
- Pu, Song, Zhao. *SkillOps: Managing LLM Agent Skill Libraries as Self-Maintaining Software
  Ecosystems*. arXiv:2605.13716v1 [cs.SE], 13 May 2026. ("SkillOps")
- Ma, Qian, Ma, Liang, Ji, Cai, Chen, Jiang, Sun. *Skill-V: Verifiable Self-Evolving Skill Library
  for Interactive Agents*. arXiv:2610.11781v1 [cs.CV], 8 Oct 2026. ("Skill-V")

Both are preprints (SkillOps is submitted to NeurIPS 2026; Skill-V cites SkillOps as "concurrent
work" and positions itself as a maintenance-focused follow-on). Neither has been peer-reviewed.
This note is the shared input for the Hermes skill-ops design work; every claim below is tied to a
specific section/table/figure so it can be checked against the source text directly.

---

## (a) Skill contract / record definition

**SkillOps** (§2, Fig. 1): a skill is a 5-tuple `s = (P, O, A, V, F)`:
- `P` precondition for calling the skill
- `O` the executable operation
- `A` the typed artifact produced
- `V` a validator over `A` (empty validator ⇒ a "validation gap", one of the five health
  dimensions)
- `F` the set of known failure modes

Skills are not stored in isolation — they sit in a **Hierarchical Skill Ecosystem Graph (HSEG)**,
`L = (S, R)`, with four *typed edge* relations between skills (§2): `dep` (dependency,
`A_si ⊆ P_sj`), `comp` (compatibility, output/input type match), `red` (redundancy, equivalent
interfaces), `alt` (alternative, same goal/different operation). The contract is deliberately
*executable*, not descriptive text — this is what lets SkillOps check applicability and
composability before invocation (§2).

**Skill-V** (§3.2, Table 7 in App. B.2): a skill is a 4-tuple `s_i = (φ_i, p_i, ρ_i, ν_i)`:
- `φ_i` semantic intent / applicability scope (includes "protected semantic constraints" that
  revisions may never alter)
- `p_i` the reusable natural-language procedure
- `ρ_i ∈ {∅, (A_i, C_i)}` an **executable rubric**: applicability conditions `A_i` plus a set of
  weighted behavioral criteria `C_i = {c_ij}` (required steps, forbidden behaviors, pre/post
  conditions, termination conditions; each criterion carries severity HARD/SOFT, weight, match
  mode, and an evidence matcher — App. B.2, Table 7)
- `ν_i` version record: "skill version, rubric version, traceability, update reason, and evidence
  case identifiers" (Table 7)

A skill can exist with `ρ_i = ∅` ("pending compilation") — it is usable for prompting/retrieval
immediately on creation but cannot yet produce a rubric judgment (§3.2 "Contract initialization";
App. B.4).

**Contrast**: SkillOps' contract is unversioned and graph-centric (cross-skill typed edges are
first-class); Skill-V's contract is versioned and evidence-centric (the rubric is explicitly a
falsifiable prediction of task outcome, graded against replay evidence). Skill-V cites SkillOps
(§2 Related Work, "Reliable skill maintenance") as doing *typed contracts + library-level health*
but not addressing "whether a stored skill remains applicable and behaviorally well-specified as
new evidence arrives" — i.e., Skill-V's own framing is that SkillOps contracts are not falsifiable
/ versioned in the way Skill-V's are. This difference is real and verified in both source texts,
not an invented contrast.

---

## (b) Usage/outcome metrics and confounding by task difficulty

**SkillOps** (§3.2, Eq. 4): five per-skill health dimensions, each in `[0,1]`:
`Utility U(s)` (fraction of recent task calls that successfully used `s`), `Redundancy R(s)`
(normalized size of largest `red`-cluster), `Compatibility C(s)` (fraction of incident `dep` edges
that are also `comp` edges), `Failure-Risk F(s)` (empirical failure rate), `Validation-Gap G(s)`
(`1[V_s = ∅]`). Library health `H(L)` is the uniformly-weighted mean of these five across all
skills (Eq. 4). **Confounding by task difficulty is not addressed.** `Utility` and `Failure-Risk`
are raw success/failure rates over recent calls with no normalization for task difficulty,
retrieval-pool composition, or sample size — a skill called only on hard tasks will show high
`Failure-Risk` regardless of its own quality. The paper's main robustness check (H2, Table 4/8,
§5.4) stress-tests *library noise density* (15%→90% synthetic degradation), not task difficulty,
so difficulty-confounding in the per-skill metrics is an unverified/unaddressed gap, not something
the paper claims to solve. **Flag: this is a real limitation of SkillOps, not something I failed
to find** — Limitations (§7) discusses gold-PDDL-argument dependence and the rule-based
maintenance loop's blindness to "semantic redundancy or complex skill conflicts," but does not
mention difficulty confounding in `U(s)`/`F(s)` at all.

**Skill-V** (§3.3 Eq. 3–4; App. B.6 Eq. 6): the executable rubric produces a per-trajectory
judgment `ŷ_rub(τ) = 1[q(τ) ≥ η ∧ H(τ)=0]` (aggregate weighted-criterion score vs. threshold
η=0.7, hard-constraint gate). Comparing `ŷ_rub` against the real environment outcome `y(τ)`
yields the maintenance signal — not raw success rate, but **rubric-outcome agreement**:
`Disc` (disagreement rate), `FNR` (false-negative rate on successful trajectories), `BAcc`
(balanced accuracy) — App. B.6 Eq. 6. This is explicitly confound-aware in one respect: by
conditioning on `A(x)=1` (only trajectories where the skill's own applicability conditions match
the task), Skill-V restricts each skill's evidence set `E_i` to tasks it claims to apply to, so a
skill is only judged in the territory it declares relevant — it does not pool outcomes across
tasks of different type/difficulty indiscriminately. **However, it does not control for difficulty
*within* the applicable-task population** — e.g. two "applicable" tasks of very different
difficulty are weighted identically in `Disc`/`FNR`/`BAcc`. The paper does not claim to solve this
either; it is a real residual gap, not a contradiction.

**Cross-paper note**: neither paper normalizes outcome metrics by task difficulty. SkillOps
confounds difficulty into `Utility`/`Failure-Risk` directly; Skill-V confounds it inside the
rubric-outcome evidence pool but at least scopes evidence to applicable tasks first. For the
Hermes design this means: **a difficulty-normalization step for per-skill success/failure metrics
is novel work, not something to port from either paper.**

---

## (c) Validating revisions over time — replay and regression definition

**SkillOps**: there is **no replay-based regression test for maintenance actions**. The
Library-Time Loop (Algorithm 2 / 4, §3.2) triggers purely on a library-health *delta* threshold
(`ΔH < Θ_maint` ⇒ skip maintenance, §3.2/Alg.4 line 1–4) and then applies rule-based actions
(`merge`, `repair`, `retire`, `add_validator`, `add_adapter`) directly from observable signals
(body-hash collisions, utility logs, failure logs, type mismatches — App. G). There is no held-out
replay set and no "non-regression" criterion comparing old vs. new skill behavior before a change
is committed — actions are applied once triggered, with ablations (Table 5, H3) only measuring
*downstream task success* after the fact, not a pre-commit gate. **This is the single starkest
difference from Skill-V and is explicitly confirmed by Skill-V's own related-work framing** (§2:
SkillOps "maintains library-level utility, compatibility, risk, and validation" but Skill-V's
contribution is specifically the evidence-preserving, non-regressive revision gate that SkillOps
lacks).

**Skill-V** (§3.3 "Skill revision and validation"; App. B.6): revisions go through an explicit
**Evidence Gate**. For skill `s_i`, maintain a bounded replay buffer `E_i` of applicable
trajectories. A candidate rubric `ρ'_i` is **evidence-preserving** relative to current `ρ_i` iff,
replayed over the *same* `E_i` (Eq. 7):
`Disc(ρ'_i) ≤ Disc(ρ_i)`, `FNR(ρ'_i) ≤ FNR(ρ_i)`, `BAcc(ρ'_i) ≥ BAcc(ρ_i)`,
plus non-empty executable support, schema validity, and exact preservation of protected semantic
constraints `φ_i`; if a *hard* criterion changes, at least one of the three metrics must *strictly*
improve. Accepted revisions replace the contract in place (version bump, §3.2 "versioning");
rejected ones leave the current version untouched (§3.3). The ablation in Table 4 (§4.3,
"Evidence Gate filters regressive skill revisions") quantifies the effect directly:
without the gate, 9/46 committed revisions regress on ≥1 replay metric and only 27/46 (58.7%)
would pass the full gate retrospectively; with the gate, 10/10 committed revisions are
non-regressive, and task success rises from 75.0% → 95.3%. Table 8 (App. B.2) gives two concrete
worked examples of accepted revisions with before/after `Disc`/`FNR`/`BAcc` on fixed replay
support (10→10 and 12→12 trajectories).

**Verification note**: Skill-V's Table 4 "w/o Evidence Gate" row reports 46 committed revisions
vs. "Full Skill-V" reporting only 10 accepted revisions elsewhere (App. D.1, Table 11: "10
contract revisions were successfully accepted by step 150"). These are consistent, not
contradictory — Table 4 is an ablation condition where the gate is *disabled*, so more (lower
quality) proposals get committed (46), vs. the real run where the gate filters down to 10. Flagged
here only because it is easy to misread as a discrepancy on first pass.

---

## (d) Retirement / forgetting — archiving vs. deletion, minimum sample sizes

**SkillOps**: `retire(s)` is **hard deletion** — "removes an obsolete or consistently failing
skill *and its incident edges*" (§2, action definitions). Trigger condition (Algorithm 4, line
33–35): `h_u(s) < θ_u AND DuplicateExists(s, S)` — i.e., retirement requires *both* low utility
*and* an existing duplicate; a uniquely low-utility skill with no duplicate is not retired (this
guards against deleting low-traffic-but-irreplaceable skills). **No explicit minimum sample size
is given anywhere in the text** for how many calls must be observed before `U(s)` is trusted
enough to retire on — this is an unaddressed gap, not a documented design choice. The numeric
value of `θ_u` itself is also not given in Appendix D's hyperparameter list (only `θ=0.5` for the
library-level health-change trigger, `k=10`/`top-5` for retrieval, and the adapter Jaccard
threshold 0.3 are specified) — **flag: `θ_u`'s value is unverifiable from the paper text as
provided.** The ablation `NoRetire` (Table 5) shows retire contributes a modest +6.3pp at
200-skill scale (73.2%→79.5%), the smallest of the five maintenance actions.

**Skill-V**: **explicitly does not retire or delete skills.** This is stated as a named limitation
(Appendix A, "Skill deletion"): "Our framework only updates operational boundaries to prevent
unintended behavior changes. If an agent initially acquires a fundamentally flawed strategy, the
system will tightly restrict its use rather than discarding it entirely. Removing these
inefficient strategies will therefore require additional mechanisms for explicit skill deletion.
Future systems could implement automated skill deprecation to permanently retire skills that
consistently fail applicability checks." In-place revision never changes library size (App. D.1:
"revisions execute as in-place updates ... they fundamentally do not contribute to size growth").
Compactness in Skill-V (≈200 entries vs. Skill1's 5,000-skill cap, §4.2 "Competitive performance
does not require aggressive library growth") comes from *not adding* redundant skills and from
tight applicability filtering at use-time (§3.4), not from pruning/retiring existing ones. There is
therefore **no minimum-sample-size policy to extract** from Skill-V for retirement, because
retirement doesn't exist in the mechanism — only a stated future-work gap.

**Design implication**: Hermes cannot copy a retirement policy wholesale from either paper.
SkillOps retires via (duplicate-exists AND low-utility) with an unspecified/unverifiable
threshold and no minimum-sample guard; Skill-V has no retirement at all. A Hermes skill-retirement
design needs its own minimum-sample and archiving (vs. hard-delete) policy — this is original
design work for the next task, not extractable from either source.

---

## (e) Lineage tracking

**SkillOps**: **no per-skill version history is tracked.** Skills are contracts `(P,O,A,V,F)`
with no `ν` (version) field; the paper's provenance mechanism operates at the *library* level, not
per-change — `L' = run_maintenance(L)` is a pure transformation with no diff/changelog object
described in the text, and I find no section describing audit logging of individual merges,
repairs, or retirements beyond the aggregate action counters in Table 15 (counts of
merge/retire/repair/add_validator/add_adapter per library-size run). There is no per-skill "this
contract changed because of trace X on date Y" record in the paper as written.

**Skill-V**: lineage is a first-class field. `ν_i` = "skill version, rubric version, traceability,
update reason, and evidence case identifiers" (Table 7, App. B.2). Table 8 demonstrates this
concretely for two real revisions from the step-150 ALFWorld checkpoint: each row shows the
specific counterexample trajectory (rubric judgment vs. actual outcome), the exact before/after
rubric text, the replay metrics that justified acceptance, and the explicit version transition
(e.g. "Skill v1 → v2; Rubric v1 → v2" and "Skill v1 → v2; Rubric v2 → v3" — note the rubric
version increments independently of skill version, since a skill can be revised multiple times
before a fundamentally new skill version). App. D.1/Table 11 additionally reports aggregate
lineage statistics per checkpoint (net growth count, accepted-revision count, and the specific
update steps at which each revision landed: "5, 60, 65, 85, 120, 125, and 150" for ALFWorld).

**Design implication**: Skill-V's `ν_i` (version + evidence-case-id + update-reason) is the closer
template for a Hermes skill-change audit trail; SkillOps offers no comparable mechanism to port.

---

## (f) Human-in-the-loop review and rubber-stamping avoidance

**Neither paper describes any human-in-the-loop review step.** This was checked carefully because
it is directly relevant to the design task's framing (avoiding rubber-stamping), and it is a
negative result worth stating plainly rather than a contrastive finding:

- **SkillOps**: fully automated. The central claim of the paper is that the "V4" maintenance
  implementation is rule-based and uses "nearly zero library-time LLM calls" (Abstract; §7
  Limitations; Table 15 — at most 9 LLM calls per full maintenance pass at N=2000, cost
  $0.0003–$0.0006). There is no mention of a human reviewer, approval gate, or audit UI anywhere in
  the main text or appendices I read (§1–K, full document).
- **Skill-V**: also fully automated. Skill additions come from an LLM "updater" analyzing failed
  trajectories (§3.3 "Skill addition"; App. B.4); revisions come from the same updater proposing
  changes that are then checked by a **deterministic** (not human) validation gate (App. B.6
  "Deterministic Revision Validation" — schema checks, traceability checks, and the replay-based
  Evidence Gate). The paper is explicit that "Skill-V never executes model-generated code" and
  constrains evidence matchers to "a predefined declarative operator set" (App. B.2) as a safety
  measure, but this is a code-execution safety constraint, not a human review step. No section
  mentions a human approving, auditing, or overriding a skill addition or revision.

**Flag for the design task**: the "how does each paper avoid rubber-stamping" question has **no
positive answer in either source** — both papers substitute *automated, criteria-based gates*
(SkillOps: observable-signal thresholds; Skill-V: the non-regression Evidence Gate) for human
review entirely, and treat that substitution as a feature (low cost, determinism) rather than a
gap. If Hermes's skill-ops design wants human-in-the-loop review, that mechanism — and specifically
how to avoid it becoming a rubber stamp — is original design work with no template in either
paper. The nearest usable primitive from Skill-V is the **non-regression statistical gate**
(Disc/FNR/BAcc thresholds vs. replay evidence), which could be surfaced to a human reviewer as a
structured decision aid (e.g. "this revision passes/fails the gate, here is the replay diff")
rather than run fully automated — but that combination is not what either paper describes or
evaluates.

---

## Summary table

| Dimension | SkillOps | Skill-V |
|---|---|---|
| Contract shape | `(P,O,A,V,F)`, unversioned, graph-embedded | `(φ,p,ρ,ν)`, versioned, rubric-falsifiable |
| Outcome metric | Utility/Failure-Risk (raw rates) | Disc/FNR/BAcc (rubric-vs-outcome agreement) |
| Difficulty confounding | Not addressed | Partially mitigated (scoped to applicable tasks only) |
| Revision validation | None (rule-trigger only, no replay gate) | Explicit non-regression Evidence Gate (Eq. 7) |
| Retirement | Hard delete; duplicate-exists + low-utility; no min-sample given; `θ_u` value unverifiable | None implemented (named future work) |
| Lineage | Aggregate action counts only, no per-skill version history | Per-skill version + rubric version + evidence-case IDs |
| Human review | None (explicitly automated, near-zero LLM cost is a selling point) | None (deterministic gate substitutes for review) |

## Open items for downstream design tasks
1. Difficulty-normalized outcome metrics: novel work, no template in either paper.
2. Retirement policy with minimum sample size + archive-vs-delete: novel work; SkillOps's
   duplicate-exists+low-utility rule is a partial precedent but its threshold is unverifiable from
   the text and it has no sample-size guard.
3. Human-in-the-loop review that isn't a rubber stamp: novel work. Recommend adapting Skill-V's
   Evidence Gate (Eq. 7) as a *decision aid* surfaced to a human rather than a fully automated
   commit gate.
4. Lineage/versioning: port Skill-V's `ν_i` shape (skill version, rubric version, update reason,
   evidence case IDs) rather than inventing one from scratch.
