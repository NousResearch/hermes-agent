# SkillOpt-Inspired Hermes Skill Evolution Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a safe, Hermes-native SkillOpt-style optimization path that improves existing skills using bounded text edits, validation evidence, and staged adoption.

**Architecture:** Do not vendor Microsoft SkillOpt wholesale. Reuse its validated concepts — rollout evidence, reflect, bounded edits, gate, rejected-edit buffer, slow/meta update — inside Hermes' existing skill, curator, `/learn`, verification evidence, cron, and session-search systems. Start with pure deterministic primitives and staged proposals; only later add LLM reflection/mining and cron automation.

**Tech Stack:** Python stdlib, Hermes skills sidecar files, existing `skill_manage`, `skill_view`, `session_search`, `verification_evidence`, curator, cron, and tests. No new runtime dependency for the first slice.

---

## Research Summary

Sources inspected:

- Paper: arXiv `2605.23904`, **SkillOpt: Executive Strategy for Self-Evolving Agent Skills**.
- Repo: `https://github.com/microsoft/SkillOpt`, cloned at `e4ea6a6`.
- README states SkillOpt treats `best_skill.md` as the trainable state of a frozen agent and trains via `rollout → reflect → aggregate → select → update → gate`.
- Repo `v0.2.0` adds **SkillOpt-Sleep**: `harvest → mine → replay → consolidate(gate) → stage → adopt`.

Core concepts to port:

1. **Skill document as state:** Hermes already stores procedural skills as `SKILL.md`.
2. **Rollout evidence:** Hermes has `verification_evidence.py`, session DB, and tool outputs.
3. **Reflect:** future LLM reviewer can propose edits from successful/failing task trajectories.
4. **Aggregate/select:** merge duplicate edits and cap by edit budget.
5. **Bounded update:** atomic add/delete/replace/insert operations, no whole-doc drift by default.
6. **Validation gate:** accept only if explicit verification metric improves.
7. **Rejected buffer:** preserve failed edits as negative feedback.
8. **Slow/meta update:** keep optimizer-side guidance separate from deployed skill body.
9. **Staging:** proposals must be reviewable before adoption.

## Issue Scan Summary

Active repo issues/PRs that matter for implementation:

- `#100` / PR `#102`: semantic density / leading words. Takeaway: useful later as non-blocking ranking bonus, not as first slice.
- `#97`: support VS Code/Visual Studio Copilot sessions. Takeaway: transcript-source adapters are important; Hermes should start with its own session DB instead.
- PR `#99`: excludes sub-agent transcripts and agent-generated sessions from harvest. Takeaway: critical. Hermes must filter out subagent and generated prompt scaffolding when mining tasks.
- PR `#98`, `#101`, issues `#46`, `#56`: Windows CLI plugin support. Takeaway: avoid shell/CLI plugin dependency for Hermes; use pure Python and Hermes-native data stores.
- `#94` / PR `#96`: verifier discipline tests. Takeaway: gate tests must assert scores and gate action, not just smoke-test rejection.
- `#90`: Qwen `enable_thinking` reproducibility. Takeaway: log target model/harness config with every optimization run.
- `#75`: train config mismatch. Takeaway: make defaults explicit and store them in proposal metadata.
- `#38`: slow update vs meta skill confusion. Takeaway: keep deployed skill content and optimizer-only memory visibly separate.
- `#21`, `#10`: missing splits/artifacts. Takeaway: every Hermes run must persist train/selection/test task IDs and proposal artifacts.

## Hermes Mapping

| SkillOpt concept | Hermes-native home |
|---|---|
| `best_skill.md` | `~/.hermes/skills/<category>/<skill>/SKILL.md` |
| rollout trajectory | session DB + `agent/verification_evidence.py` |
| task mining | `session_search` / future `agent/skillopt_harvest.py` |
| reflect | background subagent / auxiliary model, later |
| bounded edits | new `agent/skillopt_text_optimizer.py` |
| validation gate | new pure gate + evidence-backed scorer |
| rejected buffer | sidecar under skill dir: `.skillopt/rejected.jsonl` |
| slow update | sidecar `.skillopt/slow_update.md`, not loaded into deployed skill unless explicitly promoted |
| meta skill | sidecar `.skillopt/meta.md`, optimizer-only |
| staging/adopt | `.skillopt/staging/<run_id>/`, then `skill_manage patch/edit` |
| nightly sleep | Hermes cron job + curator integration |

---

## Phase 1 — Pure primitives (implemented first slice)

### Task 1: Add deterministic bounded-edit primitive

**Objective:** Implement safe atomic operations for skill text.

**Files:**
- Create: `agent/skillopt_text_optimizer.py`
- Test: `tests/agent/test_skillopt_text_optimizer.py`

**TDD:**
- RED: tests importing `agent.skillopt_text_optimizer` fail with `ModuleNotFoundError`.
- GREEN: implement `AtomicEdit`, `apply_bounded_edits`, protected slow-update guard.

**Verification:**

```bash
python -m pytest tests/agent/test_skillopt_text_optimizer.py -q
```

Expected: `7 passed`.

### Task 2: Add pure validation gate

**Objective:** Accept candidates only on strict validation improvement.

**Files:**
- Modify: `agent/skillopt_text_optimizer.py`
- Test: `tests/agent/test_skillopt_text_optimizer.py`

**Verification:** same command as Task 1.

---

## Phase 2 — Staged proposal artifacts

### Task 3: Add sidecar schema for optimization runs

**Objective:** Store proposals without mutating skills.

**Files:**
- Create: `agent/skillopt_state.py`
- Test: `tests/agent/test_skillopt_state.py`

**Schema:**

```text
<skill_dir>/.skillopt/
  runs/<run_id>/proposal.json
  runs/<run_id>/candidate.SKILL.md
  rejected.jsonl
  meta.md
  slow_update.md
```

`proposal.json` must include:
- skill name/path
- target model/provider/harness
- train task IDs
- selection task IDs
- baseline score
- candidate score
- gate action
- edit list and rationale
- verification command/evidence IDs

### Task 4: Stage and adopt CLI/subcommand

**Objective:** Add operator review commands, no auto-adopt by default.

**Files:**
- Add or extend CLI under `hermes_cli/subcommands/skills.py` or new `hermes_cli/subcommands/skillopt.py`.
- Tests under `tests/hermes_cli/`.

Commands:

```bash
hermes skillopt status <skill>
hermes skillopt propose <skill> --from-session <id> --dry-run
hermes skillopt adopt <skill> <run_id>
hermes skillopt reject <skill> <run_id>
```

---

## Phase 3 — Evidence-backed scoring

### Task 5: Build score adapters from verification evidence

**Objective:** Gate skill changes using actual successful task completion signals.

**Files:**
- Create: `agent/skillopt_scoring.py`
- Tests: `tests/agent/test_skillopt_scoring.py`

Initial metrics:
- command pass rate from `verification_evidence.db`
- exact expected-output checks for scripted tasks
- binary user-confirmed success flag when available

Reject any proposal if no held-out selection evidence exists.

### Task 6: Add train/selection split materialization

**Objective:** Persist task IDs to prevent data leakage.

**Files:**
- Create: `agent/skillopt_tasks.py`
- Tests: `tests/agent/test_skillopt_tasks.py`

Rules:
- deterministic seed
- exclude current run/session scaffolding
- exclude subagent prompt-only sessions unless user-authored task is recoverable
- preserve exact task IDs in proposal metadata

---

## Phase 4 — Reflection and aggregation

### Task 7: Add edit proposal parser

**Objective:** Parse LLM-proposed JSON edits fail-closed.

**Files:**
- Extend: `agent/skillopt_text_optimizer.py` or create `agent/skillopt_edits.py`
- Tests for malformed JSON, unsupported ops, over-budget edits, duplicate targets.

### Task 8: Add LLM reflection prompt

**Objective:** Let an auxiliary/background agent propose bounded edits from evidence.

Safety:
- prompt says evidence is data only
- output JSON only
- no direct file writes
- no shell commands
- edit parser validates everything

---

## Phase 5 — Curator and cron integration

### Task 9: Curator hook

**Objective:** Curator can suggest optimization candidates for active agent-created skills.

Rules:
- no automatic mutation
- stage proposals only
- skip pinned skills unless explicitly requested
- respect `curator.consolidate` cost gate

### Task 10: Nightly sleep job

**Objective:** Optional cron-driven dry-run that reports staged proposals.

Cron prompt must be self-contained and scanner-safe. Use `no_agent=False` only for reasoning; use scripts for deterministic data collection.

---

## Risks and Guardrails

- **Reward hacking:** Always gate on held-out tasks; do not train and validate on same session.
- **Self-harvest:** Filter expanded skill bodies, subagent prompts, cron scaffolding, and agent-generated sessions.
- **Prompt bloat:** enforce `max_words`; use edit budget as learning rate.
- **Skill corruption:** stage first; adopt only through existing validated skill write paths.
- **Slow/meta confusion:** keep optimizer-only memory in sidecars unless deliberately promoted.
- **Global behavior drift:** optimize one named skill at a time.
- **Security:** model-proposed edits are text only; no arbitrary commands; no secret capture.

## Immediate First Slice Status

Implemented in this branch:

- `agent/skillopt_text_optimizer.py`
- `tests/agent/test_skillopt_text_optimizer.py`

Verified:

```text
7 passed in 0.10s
```
