# CLAUDE.md — Motion Granted Citation Database

Session-orientation file for agents working in this repo. Read at session start along with `binding/v7.2.md` (architectural) and `AGENT-DISCIPLINE.md` (operational).

---

## 1. Authoritative spec

The Citation Database is MG's Authority Layer (AL). Its architecture binds via a layered hierarchy; later layers never override earlier ones.

1. **`binding/v7.2.md`** — the ratified architectural baseline. 2,214 lines across §0-§29 + ADRs 0001-0007. Covers: schema, 10 canonical invariants I-1..I-10 (§1) plus I-6a/I-6b/I-6c at §6 line 407-409, 8-dimension ordering function (§6 line 380-391), Cardinal Sin enforcement (§12 line 893-917), STOP gate (§9), 16-UPPERCASE `al_treatment_type` taxonomy (§12 line 960-966), **12 access-control roles** (§13 line 1015-1028), 11 event classes (§14 line 1071-1075), MCP read surface (§15), 4 readiness tiers (§18), migration order 001-015 (§20 line 1408-1422), six §24 blockers (N-1, N-2, R-1, R-2, R-3, S-5), §26 retention (counsel-gated), §29 acknowledged gaps.
2. **`AGENT-DISCIPLINE.md`** — operational doctrine. Zero-inference rules (§1), authority hierarchy (§2), the A-L category table identifying high-hallucination-risk surfaces (§3), pending-resolution protocol for §24 blockers (§4), session scope rules (§5), file-header conventions (§6), self-audit rule (§7). Extends `AGENTS.md` for MG-specific failure modes.
3. **`docs/operator/`** — Chen-verified reference for operator execution: `OPERATOR-REVIEW-CHECKLIST.md`, `ROADMAP-TIER-A-TO-D.md`, `EMPIRICAL-REMEDIATION-NOTES.md`, `CLAY-MEMOS.md`.
4. **`BACKLOG.md`** — ratified open items. As of 2026-04-23: R-1 Option B, R-3 option (a), N-2 acknowledgment — all ratified by Clay email 2026-04-23 1:48 PM; in-repo checkbox sync pending. Tier B carry-overs from Chen remediation; §29 gap list with tier tags.
5. **Everything under `archive/`** — historical and working material. Informational only; never authoritative. Read-only.

When sources disagree: v7.2 wins. Working material conflicting with v7.2 is reconciled in favor of v7.2 silently. **Silence in v7.2 is a gap, not a default** — log the silence inline (`-- v7.2 silent; chose <X> per minimal-assumption default`) and surface it for operator review. See AGENT-DISCIPLINE.md §1.3.

**Intra-v7.2 drift to be aware of.** §13 line 1015-1028 enumerates **12** roles (canonical); §18 line 1330 and §29 line 2170 still say "11." Canonical count is 12; the "11" references are stale. BACKLOG.md line 71 logs the reconciliation.

---

## 2. Core discipline

### 2.1 Zero-inference

The repo's single most load-bearing discipline. Defined in AGENT-DISCIPLINE.md §1 — **do not duplicate; reference**.

Summary for orientation:

- Every value emitted in any output file (SQL, Markdown, JSON, code) is **quoted verbatim** from an authoritative source with a section + line cite. Never inferred.
- **Counts are reporting facts.** A delta between v1 (N) and v7.2 (N+K) documents that K extra values exist; it does not license inventing the extra K. The K come from reading v7.2's enumeration.
- **Names are grep-verified.** Every identifier (table name, column, ENUM value, ERRCODE, migration number, section number) is grep-confirmed against v7.2 before emission.
- Schema/contract output runs **triple-checkpoint verification** (AGENT-DISCIPLINE.md §1.2 rule 4): DISCOVER, per-file write, pre-commit self-audit.
- **"Claude's recommendation" in v7.2 is advisory.** When v7.2 surfaces a recommendation next to a Clay ruling required (R-1, R-3), draft per the recommendation and tag `<X>_PENDING`; emit the reverse option commented adjacent so a flip is a single-line edit.

**The failure mode this prevents.** A prior session hallucinated four `al_treatment_type` values from a count-delta (v1's 15 lowercase vs. v7.2's 16 UPPERCASE). The correct list lives at §12 line 960-967, verbatim. The invented values were not in v7.2. See AGENT-DISCIPLINE.md §1.1.

The A-L category table in AGENT-DISCIPLINE.md §3 enumerates the 12 highest-hallucination-risk surfaces with their v7.2 cites (e.g., row G = roles → §13 line 1015-1028, **12 roles including `platform_admin`**; row H = event classes → §14; row I = invariants → §1 + §6 I-6a/b/c).

### 2.2 Operator-only boundaries

Agents never perform the following actions; operator does:

- No `git push` to `main` or any protected branch.
- No `gh pr merge`, no force-push.
- No direct commits to `main` from a session. Work on named feature branches (e.g., `overnight/<task>-<date>`, worktree branches).
- No Clay-binding rulings. Drafts tagged `<X>_PENDING` per AGENT-DISCIPLINE.md §1.2 rule 5.
- No DB apply (no `psql`, no Supabase `apply_migration`) unless the session mission explicitly authorizes staging-DB write.
- No edits to `producer/**`, `binding/**`, `reports/**`, `archive/**`, `al/**` unless the session mission explicitly authorizes.
- No modifications under `archive/`. If stale content needs correction, correct in a new file outside `archive/`; archive stands as provenance.

Per AGENT-DISCIPLINE.md §5 scope rules.

### 2.3 Triple safety-net

Zero-inference is enforced in practice through three stages:

1. **Read-before-replace.** The `Edit` tool requires a prior `Read` of the target file — tool-contract rule, applied by discipline.
2. **`grep-verifier` sub-agent.** For textual/pattern claims about file contents, verification is outsourced. Every load-bearing identifier grep-confirmed against authoritative source before emission. See `.claude/agents/grep-verifier.md`. Historical AI-audit false-positive rate: 38-54%.
3. **`chen` sub-agent adversarial review.** Four modes (deep subsystem, finding expansion, spec-to-code delta, pre-launch failure); labels CONFIRMED / HYPOTHESIS / UNVERIFIED / DISPROVEN; BLOCKER/MAJOR/MINOR/NIT severity (rubric observed in session commits `5d2e4c8`, `2e3fd63`, `dec9f3e`). Chen's track record in this repo: 3 BLOCKER + 3 MAJOR caught in scaffold-rebuild PR #4 (commit `2e3fd63`), 10 MAJOR + 6 MINOR caught in operator-review-package PR #3 (commit `dec9f3e`), 8 MAJOR caught in reports PR (commit `5d2e4c8`). See `.claude/agents/chen.md`.

`code-reviewer` is a single-pass diff reviewer for post-change checks (HYBRID grep + GitNexus where available). `architect` is the sole sub-agent with swarm-invocation authority (`.claude/agents/architect.md`).

---

## 3. Workflow

### 3.1 Overnight session pattern

Long-horizon work (scaffold rebuild, producer port, doc final) is staged in phases, each logged:

- `PHASE-1` probe — branch creation, tooling inventory, path-sanity checks.
- `PHASE-2-WAVE-N` — parallel swarm dispatch through `architect`.
- `PHASE-3` synthesis — main thread merges swarm outputs, resolves conflicts.
- `PHASE-4+` commit + push (operator owns push) + PR open.

Each phase appends to `worktree-session/SESSION-LOG-*.md`. Wave outputs land under `worktree-session/swarm-raw/`. Empirical examples: SESSION-LOG-REBUILD.md (six phases, two waves, six swarm outputs).

**Architect owns decomposition.** Other sub-agents route swarm requests through Architect; they do not swarm directly. Swarm decompose rules: by file boundary, no cross-file dependencies, max 5 sub-agents per swarm (`AGENTS.md`), read-only sub-agents investigate and report, main thread synthesizes.

**Parallel cap.** 2-3 parallel sub-agents is the empirical sweet spot; 5 is the ceiling.

**Worktree isolation.** Feature work lands on `overnight/*` branches in sibling worktrees (`git worktree add ../Case-Database-<name>`). Unmerged branches stay unmerged until Chen-reviewed. Session working material lives at `worktree-session/` and merges with the PR.

### 3.2 Commit conventions

Format: `<type>(<scope>): <summary> per <authority>` plus a Co-Authored-By trailer when an agent contributed.

- `type` ∈ `{feat, fix, docs, chore, reorganize, refactor}`.
- `scope` names a subsystem slice (`scaffold`, `producer`, `reports`, `review-pkg`, `findings`, `session`, `phase-1a`, `reorg`, `rebuild-doc`).
- `per <authority>` cites the ratifying reference: `per v7.2 §24 R-2`, `per Chen REVIEW-PR-3`, `per §13 line 1015-1028`, etc.

Empirical examples in `git log`: `fix(scaffold): resolve 3 BLOCKERs + 3 MAJORs per Chen REVIEW-PR-3`, `reorganize(pr-3): promote 4 operator docs to docs/operator/, archive session artifacts`.

### 3.3 Session artifact conventions

| Path | Role | Retention |
|---|---|---|
| `worktree-session/` | Current session's logs + swarm-raw outputs + pre-flight reports. | Ephemeral within the session; merges with the PR. |
| `archive/session-artifacts/<date>-<name>/` | Historical session evidence post-archive. | Read-only after archive. |
| `docs/operator/` | Chen-verified reference material. | Ratified; maintain against v7.2 cites. |
| `drafts/scaffold/` | 15 SQL migrations + tests + REBUILD-MANIFEST. | On `main`; execution material for staging apply. |
| `drafts/runbook/`, `drafts/remediation/` | Phase-1A execution material. | **Not on `main`** — on `overnight/phase-1a-ready-2026-04-23` (local-only); lands at Phase-1A ratification. |
| `reports/` | Session investigation outputs. | On `main`; historical. |

---

## 4. Key architectural reminders

Cite v7.2 §X line N when asserting any of the following. Do not paraphrase from training data.

### 4.1 Interpretation A (§3 line 268-287)

AL owns every canonical table under `al.*`. Porter's Data Engine is a derivation producer only — runs L0→L3, writes to `al.al_treatment_derivation`, never writes to the sink or history. The sink is promoted by AL's Sink Promotion Service. Adapter window permitted during Phase 1A-1B; retired at end of Phase 1D.

### 4.2 Identity tuple (I-5, §1 line 171-173; §15 line 1127)

Effective treatment identity = `(cited_authority_id, citing_authority_id, jurisdiction_scope)`. Every tuple-scoped MCP query MUST include jurisdiction. A case overruled in one jurisdiction remains good law in another until proven otherwise.

### 4.3 Cardinal Sin (I-6, §1 line 175-177; §12 line 893-917)

No row in `al_effective_treatment` with `verification_status='VERIFIED'` AND `confidence >= 0.99` unless produced by a non-error derivation path. Enforced by DB-level trigger `al_enforce_cardinal_sin_trigger` (migration 012; ERRCODE `CS001`). `verification_status` is assigned by `promote_derivation_to_sink` at sink-write time per the §12 line 905-916 projection table; body belongs in migration 014.

### 4.4 Ordering function (§6 line 380-409)

8 dimensions, highest priority first:

```
override_present DESC, blessed DESC, stop_gate_passed DESC,
severity_rank DESC, confidence DESC,
pipeline_version DESC, derivation_version DESC,
derivation_content_hash ASC
```

**NO wall-clock** (I-6b, §6 line 397-399). `created_at`, `promoted_at`, `transitioned_at`, `evaluated_at` are observational metadata only — never authoritative tiebreakers. Any added dimension must be replay-invariant on identical inputs. I-6a (totality), I-6b (determinism), I-6c (monotonicity-under-override) at §6 line 407-409 (NOT at §1).

### 4.5 Two-person STOP override (§9 + §24 N-2)

`al_override` rows with `severity_rank=4` require `second_approver_id IS NOT NULL` AND `second_approver_id != created_by_user_id`. Enforced by trigger in migration 012 (ERRCODE `OV001`). Application-layer feature flag for STOP-severity override creation stays OFF until T-OV-1 passes. See `drafts/scaffold/012_functions_triggers.sql:157-164` + BACKLOG.md line 13.

### 4.6 Prompt immutability

Producer pipeline prompts under `producer/pipeline/prompts/` are IMMUTABLE until the four strict xfails (F3, F5, F7, F8) close. Doctrine lives in BACKLOG.md line 14 + line 69 + `producer/prompts/README.md` line 63-66 + v7.2 I-10 + §10/§12 per-run + per-derivation attestation. The integrity hash is **SHA-256 of prompt-file content**, not Git-SHA. (v7.2 has no §3.7; don't cite §3.7 — the doctrine is distributed across the cites above.)

---

## 5. PENDING-EXPECTED for reviews

Items listed below are **intentional** draft states or deferrals. Reviewers and sub-agents MUST NOT flag these as bugs. This is a **pattern**, not a literal file; the current instance for this workstream is `worktree-session/CHEN-FALSE-POSITIVES.md` (git `549a3ef`, 12 items enumerated).

- **R-1 Option B active** at `drafts/scaffold/005_authority.sql:52`; Option A commented adjacent. Clay ratified by email 2026-04-23 1:48 PM; BACKLOG.md line 11 checkbox sync pending (see DOC-TODOS.md).
- **R-3 option (a) active** at `drafts/scaffold/008_derivation.sql:70`. Clay ratified by email 2026-04-23 1:48 PM; BACKLOG.md line 12 checkbox sync pending.
- **N-2 trigger active** at `drafts/scaffold/012_functions_triggers.sql:157, 162, 174`; application feature flag OFF until T-OV-1 passes. Clay acknowledged by email 2026-04-23 1:48 PM; BACKLOG.md line 13 sync pending.
- **Migration 013 DRAFT ONLY** — `drafts/scaffold/013_roles_grants.sql` outlines the 12 role grants; not applied; migration 013 GRANT/REVOKE wiring deferred per Chen A-BLOCKER-2 workflow.
- **Migration 014 OUTLINE ONLY** — stub bodies in 012 raise NI001-NI005; `drafts/scaffold/014_promote_full.sql` has a Tier B outline. Full bodies (`promote_derivation_to_sink`, `al_merge_authority`, `recompute_sink_for_tuple`, `recompute_sink_for_authority`, `compute_verification_status` body per §12 line 917) package together at Tier B.
- **Migration 015 NI-stub + TRIGGER COMMENTED OUT** — body raises NI005; CREATE TRIGGER at `drafts/scaffold/015_scope_lineage_trigger.sql:83-88` commented per Chen A-BLOCKER-2. Wiring ships alongside the `al_populate_scope_lineage` body at Tier B.
- **`al_merge_authority`, `recompute_sink_*`** all raise `NI002`/`NI003`/`NI004` stubs. Expected per §24 post-blockers list.
- **4 strict xfails** in producer V1 tests: F3 (prompt-side jurisdiction request), F5 (prompt-side STOP signal removal), F7 (dicta vs holding split), F8 (injection fence). Prompts IMMUTABLE until these close (BACKLOG.md line 14). Producer test counts: 194 collected → 190 passing + 4 strict xfails per commit `0293796`.
- **§26 retention counsel-gated** — best-estimate defaults stand until counsel ratifies (§26 line 1759; BACKLOG.md line 44).

**ERRCODE families** (chosen per discipline audit; v7.2 silent on specific SQLSTATEs): `CS001` Cardinal Sin, `SG001` STOP gate, `OV001` two-person override, `NI001-NI005` not_implemented stubs, `PP001`/`PR001` preconditions, `VS001` verification status fallthrough. See `drafts/scaffold/012_functions_triggers.sql:17-21` + `drafts/scaffold/REBUILD-MANIFEST.md` §5.

---

## 6. What's in the repo

```
binding/v7.2.md                architectural baseline (2,214 lines)
AGENT-DISCIPLINE.md            operational doctrine
AGENTS.md                      tanner-stack mode-selection + parallel rules
BACKLOG.md                     ratified open items + Tier B carryovers
CHANGELOG.md                   chronological repo history
CLAUDE.md                      this file
CONTRIBUTING.md                session conventions + how to contribute
LEARNINGS.md                   session-derived patterns
LICENSE                        MIT
NOTICE.md                      third-party attributions + disclaimer
README.md                      repo overview
SECURITY.md                    security + schema guardrails
VERIFICATION.md                harness verification report
.claude/agents/                architect, chen, code-reviewer, grep-verifier
.claude/commands/              slash-command definitions
.claude/rules/                 per-subsystem path-scoped rules (example-rule.md)
.claude/workflows/             multi-step workflow definitions
.github/                       repo CI config (genericized from tanner-stack)
archive/                       read-only historical source material
docs/                          tanner-stack harness docs + navigation README
docs/operator/                 Chen-verified operator reference (4 docs)
drafts/scaffold/               15 SQL migrations + 11 test files + REBUILD-MANIFEST
personas/                      tanner-stack persona source material
producer/                      V1 pipeline port (L0-L3, 190 pass + 4 xfail)
prompts/                       tanner-stack superprompts + style guides
reports/                       session investigation outputs
skills/                        tanner-stack skills library
workflows/                     tanner-stack workflow library
worktree-session/              current session working material
```

---

## 7. Tier gates

v7.2 §18 line 1294-1299 readiness tiers:

- **Tier A (scaffold safe to apply) — scaffold-ready as of 2026-04-23.** Six §24 blockers resolved (R-1 / R-3 / N-2 ratified per Clay email 2026-04-23 1:48 PM). Migrations 001-012 apply cleanly on paper; **staging apply has not yet happened** (BACKLOG.md line 19 unchecked). 013/014/015 remain DRAFT / OUTLINE / NI-stub per §20.
- **Tier B (scaffold functionally operational) — PENDING.** Stage 9 full implementations (§19): migration 013 roles + GRANT/REVOKE for 12 roles; migration 014 `promote_derivation_to_sink` + `al_merge_authority` + `recompute_sink_*` + `compute_verification_status` body; migration 015 `al_scope_lineage` trigger wiring + body; `al_recompute_queue` DDL.
- **Tier C (staging validated) — PENDING.** Phase 1A staging + Phase 1B shadow mode; stress S-1/S-2/S-3; chaos C-1/C-2/C-3.
- **Tier D (Phase 1D cutover authorized) — PENDING.** §28 cutover criteria hold 7 consecutive days + Porter and Clay sign-off + counsel §26 ratification.

**Language discipline.** Do not call any work "production" or "production-ready" until Tier D. "Production-grade" applied to drafts refers to document quality, not tier attainment. Chen flagged overclaim language in PR #4 remediation (commit `dec9f3e`).

---

## 8. Operator notes

- Repo lives inside a OneDrive-synced tree. Exclude `.git/` from OneDrive sync to avoid sync churn and two-machine corruption risk.
- Do not modify files under `archive/`. If stale content needs correction, correct it in a new file outside `archive/` and let the archived original stand as provenance.
- `overnight/*` branches stay unmerged until Chen-reviewed; operator owns the merge.
- Clay-binding artifacts (BACKLOG checkboxes, `docs/operator/CLAY-MEMOS.md` signatures) are synced post-session by the operator; agents should cite the ratifying email/source rather than waiting for checkbox state.

---

## 9. References

- `binding/v7.2.md` — architectural spec, 2,214 lines, §0-§29 + ADRs 0001-0007.
- `AGENT-DISCIPLINE.md` — operational doctrine (zero-inference, A-L table, scope rules).
- `AGENTS.md` — tanner-stack mode + parallel rules.
- `BACKLOG.md` — ratified open items, Tier B carryovers, §29 gap list.
- `CONTRIBUTING.md` — session conventions, commit format, worktree isolation pattern.
- `CHANGELOG.md` — chronological repo history.
- `SECURITY.md` — security model, schema guardrails, vulnerability reporting.
- `docs/operator/README.md` + the four operator-reference docs.
- `drafts/scaffold/REBUILD-MANIFEST.md` — scaffold provenance per file.
- `.claude/agents/{architect,chen,code-reviewer,grep-verifier}.md` — sub-agent contracts.
- `.claude/rules/example-rule.md` — per-subsystem rule template.

Authority: `binding/v7.2.md`. Discipline: `AGENT-DISCIPLINE.md`. Every load-bearing schema / invariant / taxonomy / role / tier claim cites a v7.2 section + line.

## 9.5 Project: Hermes Agent

This fork layers Tanner-stack methodology onto Hermes Agent. Hermes remains a coding agent for legal-tech infrastructure; it is not itself a legal-tech agent.

| Subsystem | Key paths | Notes |
|---|---|---|
| Agent core loop | `run_agent.py`, `model_tools.py` | Tool loop, provider transport, cache-sensitive context. |
| CLI | `cli.py`, `hermes_cli/` | Interactive command surface and setup/config flows. |
| Web dashboard backend/frontend | `hermes_cli/web_server.py`, `web/` | Dashboard backend on `:9119`, frontend on `:5173`. |
| Gateway/platforms | `gateway/`, `gateway/platforms/` | Messaging adapters and session routing. |
| Plugins | `plugins/` | Upstream-vendored extension surface. |
| Runtime skills | `skills/`, `optional-skills/` | Hermes runtime skills, separate from methodology `.claude/skills/`. |
| Tool registry | `tools/registry.py`, `tools/`, `toolsets.py` | Auto-discovered tool schema and dispatch. |
| TUI | `ui-tui/`, `tui_gateway/` | Ink frontend plus Python JSON-RPC backend. |
| MCP server | `mcp_serve.py`, `acp_adapter/` | Editor/MCP integration surfaces. |
| Cron | `cron/` | Scheduler and job persistence. |
| RL/training | `environments/`, `batch_runner.py`, `rl_cli.py` | Optional training/eval surfaces. |
| Tests | `tests/` | Baseline accepted with documented upstream failures in `BACKLOG.md`. |

Inviolable rules:
1. Always run `scripts/run_tests.sh` before declaring change complete.
2. Always run `ruff check .` before commit.
3. Never break a `gateway/platforms/<name>.py` adapter without smoke-testing every enabled platform.
4. Plugin and skill schema is contract; no upstream-uncoordinated changes.
5. WSL2-only on Windows; never test native Windows.
6. Never edit `~/.hermes/` from a session.
7. Bundled-skill edits require `hermes setup` to re-seed `~/.hermes/skills/`.
8. Source of truth: CDB is canonical for methodology. Edit in CDB and re-sync; do not edit methodology files in this fork directly.
9. Success metric: within two weeks of Stage 9 merge, `/escalate` runs once on a real bug overnight, produces a fix the operator approves, and saves at least three hours debugging time. If not hit, schedule operator review.

Do not edit zones: `web/dist/`, `__pycache__/`, `node_modules/`, `.venv/`, `venv/`, `*.pyc`, `~/.hermes/`, `*.bak`, `*.old`, `*_backup.*`, `vendor/`, `third_party/`, `.plans/`.

## 9.6 Role registry (model-agnostic methodology)

Methodology references roles, not model names. Per-machine mappings live in `config/models.yml` and are gitignored. Apply them with `scripts/apply-models-yml.sh`.

Roles: `primary_reasoning`, `fast_iteration`, `adversarial_review`, `legal_tech_review`, `escalate_head`, `escalate_subagent_<n>`, `cheap_routine`.

Family taxonomy: `anthropic | openai | google | meta | xai | deepseek | local | other`.

Provider migration target: swap providers in under two hours by editing `config/models.yml`, running `scripts/apply-models-yml.sh`, smoke-testing Hermes, and leaving committed methodology untouched. See `docs/provider-migration.md`.
