# Hermes Skills Consolidation Plan

> **For Hermes:** This is a review/migration plan only. Do not delete or archive any skill until MJ explicitly approves a phase.

**Goal:** Consolidate Hermes skills from narrow/duplicated one-offs into a smaller, class-level, update-first skill system without losing local operational knowledge.

**Architecture:** Treat active runtime skills (`~/.hermes/skills`) as the live user profile layer, bundled repo skills (`skills/`) as shared source, and optional skills (`optional-skills/`) as installable inventory. Consolidation should update or promote umbrella skills first, then retire duplicates only after link/reference/cron/config rewrites are verified.

**Tech Stack:** Markdown `SKILL.md`, YAML frontmatter, `agent.skill_utils`, `tools/skill_manager_tool.py`, Hermes curator, cron skill reference rewriting, Linear/Kanban proof comments.

---

## Audit Snapshot

Audit command source: Python filesystem scan over:
- Active runtime skills: `/Users/alfred/.hermes/skills`
- Bundled repo skills: `/Users/alfred/.hermes/hermes-agent/skills`
- Optional repo skills: `/Users/alfred/.hermes/hermes-agent/optional-skills`

Runtime source-of-truth check:
- `agent.skill_utils.get_all_skills_dirs()` returned only `/Users/alfred/.hermes/skills` for this profile.
- `agent.skill_utils.get_disabled_skill_names()` returned `[]`.
- Therefore, the active profile is using the user-local skill tree, not the repo `skills/` tree directly.

Inventory counts:
- Active user-local skills: 114
- Bundled repo skills: 87
- Optional repo skills: 81
- Total scanned `SKILL.md` files: 282
- Active frontmatter/body validation failures: 0
- All scanned frontmatter/body validation failures: 0

Active skills by top-level category:
- apple: 5
- autonomous-ai-agents: 11
- creative: 20
- data-science: 1
- devops: 6
- dogfood: 1
- email: 1
- gaming: 2
- github: 6
- mcp: 1
- media: 5
- mlops: 13
- note-taking: 1
- openclaw: 6
- productivity: 11
- red-teaming: 1
- research: 5
- smart-home: 1
- social-media: 1
- software-development: 15
- yuanbao: 1

Source duplication counts:
- Active skills with same name as a bundled/optional source skill: 91
- Exact active/source duplicate content: 76
- Divergent active/source duplicate content: 15
- Active duplicate names within `/Users/alfred/.hermes/skills`: 0

Divergent active/source duplicates to protect before any sync:
- `axolotl`: active copy differs from optional only by removed `platforms` line.
- `fine-tuning-with-trl`: active copy differs from optional only by removed `platforms` line.
- `outlines`: active copy differs from optional only by removed `platforms` line.
- `unsloth`: active copy differs from optional only by removed `platforms` line.
- `google-workspace`: active copy is older in places but has `python3` setup wording; repo has version/platform updates. Needs manual merge.
- `claude-code`: active copy adds MJ Max-sub operating strategy and May 2026 Claude Code notes. Must not overwrite.
- `codex`: active copy adds OpenAI Codex OAuth/provider gotchas and broader task triggers. Must not overwrite.
- `hermes-agent`: active copy adds multiple MJ/Hermes operational references and auth/profile/gateway lessons. Must not overwrite.
- `hermes-agent-skill-authoring`: active copy diverges from source. Needs manual hunk classification before any sync.
- `kanban-worker`: active copy adds worktree branch default and notification routing. Promote if still relevant.
- `linear`: active copy adds Hermes/Kanban bridge pattern. Promote if broadly useful.
- `obsidian`: active copy adds wikilink graph audit reference. Likely MJ-local; keep local unless generalized.
- `spike`: active copy adds hard kill criteria. Promote/update umbrella skill if general.
- `systematic-debugging`: active copy adds no-implementation applicability audit trigger. Promote/update if general.
- `test-driven-development`: active copy adds Claude Code / Max subagent notes. Likely MJ-local or autonomous-agent integration.

Raw data artifact:
- `/Users/alfred/.hermes/hermes-agent/docs/plans/2026-05-26-skills-audit-data.json`

---

## Duplicate / Divergence Report

### 1. Profile-local mirror duplication

Pattern:
- Most active skills are copies of bundled repo skills under `~/.hermes/skills`.
- 76 are exact duplicates and 15 diverge.

Risk:
- Exact duplicates inflate inventory and confuse audits, but are low semantic risk.
- Divergent duplicates are high risk because a naive reinstall/update from repo could erase local operational lessons.

Rule:
- Never bulk-sync repo -> user-local by filename.
- For exact duplicates, prefer a loader-level or installer-level mechanism that can use bundled skills without physical user-local copies.
- For divergent duplicates, merge intentionally: promote generic lessons upstream, keep MJ-specific rules local, then only remove local copy if active behavior remains identical.

### 2. Autonomous coding agents overlap

Observed active cluster:
- `autonomous-ai-agents/claude-code`
- `autonomous-ai-agents/codex`
- `autonomous-ai-agents/opencode`
- `autonomous-ai-agents/codex-host-agent`
- `software-development/subagent-driven-development`
- `software-development/requesting-code-review`
- `software-development/test-driven-development`

Specific similarity signal:
- `codex` vs `opencode`: Jaccard-like metadata/headings score 0.301.

Overlap:
- One-shot/background/interactive coding-agent operation repeats across Claude Code, Codex, and OpenCode.
- Review/TDD/delegation guidance crosses from software-development into autonomous-agent skills.

Consolidation target:
- Keep provider-specific skills for exact CLI flags and auth gotchas.
- Add/strengthen one umbrella skill: `autonomous-coding-agents` or use existing `subagent-driven-development` as umbrella for routing, review gates, billing/subscription rules, and when to choose Claude Code vs Codex vs OpenCode.
- Provider-specific skills should point to the umbrella and only contain deltas.

### 3. Development discipline overlap

Observed active cluster:
- `software-development/writing-plans`
- `software-development/plan`
- `software-development/spike`
- `software-development/test-driven-development`
- `software-development/systematic-debugging`
- `software-development/requesting-code-review`
- `software-development/subagent-driven-development`
- `software-development/exceptional-engineering-discipline`
- `software-development/mj-shipping-workflow`
- `software-development/llm-rails-development-process`

Overlap:
- Multiple skills instruct when to plan, test, review, and ship.
- MJ-specific shipping workflow and generic software-development workflow are interleaved.

Consolidation target:
- Keep `mj-shipping-workflow` as MJ-local preference and `exceptional-engineering-discipline` as generic quality gate.
- Treat `writing-plans`, `spike`, `systematic-debugging`, `test-driven-development`, and `requesting-code-review` as stages in one lifecycle, not independent competing triggers.
- Add a lightweight decision table in one umbrella: “plan vs spike vs debug vs TDD vs review.”

### 4. OpenClaw/Hermes operations overlap

Observed active cluster:
- `openclaw/*` six active local skills
- `devops/openclaw-lifecycle-daemon`
- `autonomous-ai-agents/openclaw-*`
- `autonomous-ai-agents/hermes-agent`
- `devops/kanban-worker`
- `devops/kanban-orchestrator`
- `productivity/linear`

Overlap:
- Hermes runtime, Kanban, Linear bridge, OpenClaw lifecycle, and OpenClaw shipping instructions all reference agent ops and task coordination.

Consolidation target:
- Do not merge Hermes and OpenClaw domain skills; they are different systems.
- Create clearer taxonomy boundaries:
  - Hermes runtime/profile/gateway/Kanban -> Hermes skills.
  - OpenClaw lifecycle/EMA/rails -> OpenClaw skills.
  - Linear/Kanban bridge -> Linear/productivity or a dedicated work-tracking reference used by both.
- Cross-link instead of merging across system boundaries.

### 5. Creative generation overlap

Observed active cluster:
- `creative/baoyu-article-illustrator`
- `creative/baoyu-comic`
- `creative/baoyu-infographic`
- `creative/ascii-art`
- `creative/ascii-video`
- `creative/manim-video`
- `creative/p5js`
- `creative/sketch`
- `creative/pretext`
- `creative/popular-web-designs`
- `creative/claude-design`

Specific similarity signals:
- `baoyu-article-illustrator` vs `baoyu-infographic`: score 0.269.
- `ascii-video` vs `manim-video`: score 0.235.

Overlap:
- Many skills cover prompt structuring, layout/design style selection, visual output verification, and asset delivery.

Consolidation target:
- Keep output-medium skills separate where tools differ materially.
- Introduce or strengthen a creative “visual generation routing” umbrella with a decision table: infographic vs comic vs article illustration vs sketch/HTML vs animation/video vs pixel/ascii.
- Move repeated prompt-quality/style guidance into references shared by sibling skills.

### 6. MLOps/training/inference overlap

Observed active cluster:
- `mlops/training/axolotl`
- `mlops/training/unsloth`
- `mlops/training/trl-fine-tuning`
- `mlops/inference/vllm`
- `mlops/inference/llama-cpp`
- `mlops/inference/outlines`
- `mlops/evaluation/lm-evaluation-harness`
- `mlops/huggingface-hub`
- `mlops/research/dspy`

Specific similarity signal:
- `axolotl` vs `unsloth`: score 0.375.

Overlap:
- Training framework skills share LoRA/QLoRA, hardware sizing, dataset prep, and HuggingFace workflows.

Consolidation target:
- Keep tool-specific skills because commands/configs differ.
- Add shared references under an umbrella `llm-training-lifecycle` or one `mlops/training/README-like` skill: dataset prep, model choice, hardware planning, evaluation gates, upload/publish.
- Provider/tool skills should link to the shared lifecycle and focus on exact commands.

### 7. Productivity/document/knowledge IO overlap

Observed active cluster:
- `productivity/google-workspace`
- `productivity/notion`
- `productivity/airtable`
- `productivity/ocr-and-documents`
- `productivity/nano-pdf`
- `productivity/powerpoint`
- `note-taking/obsidian`
- `email/himalaya`
- `productivity/teams-meeting-pipeline`

Overlap:
- Read/search/create/edit document workflows repeat across tools.
- Authentication/setup gotchas are tool-specific.

Consolidation target:
- Keep tool-specific skills.
- Add shared “document IO routing” guidance: when to use OCR/PDF extraction vs Google Docs vs Notion vs Obsidian vs PowerPoint.

---

## Proposed Taxonomy

Use four explicit skill layers:

1. **System skills**
   - Purpose: operating Hermes, OpenClaw, gateways, daemon lifecycles, Kanban/Linear bridges.
   - Examples: `hermes-agent`, `kanban-worker`, `openclaw-lifecycle-daemon`, `linear`.
   - Rule: system boundary beats category similarity. Cross-link instead of merging Hermes and OpenClaw.

2. **Workflow umbrella skills**
   - Purpose: choose approach, sequence phases, define gates, route among narrower skills.
   - Examples/targets: `mj-shipping-workflow`, `exceptional-engineering-discipline`, future `autonomous-coding-agents`, future `visual-generation-routing`, future `llm-training-lifecycle`, future `document-io-routing`.
   - Rule: if 3+ skills repeat the same decision logic, put the decision table here.

3. **Tool/provider skills**
   - Purpose: exact commands, auth, flags, API quirks, environment setup.
   - Examples: `claude-code`, `codex`, `opencode`, `google-workspace`, `vllm`, `axolotl`, `himalaya`.
   - Rule: keep when the commands differ materially. Remove generic process prose that belongs to an umbrella.

4. **Local preference/tenant skills**
   - Purpose: MJ-specific operating preferences, OpenClaw brand topology, local host quirks, single-user assumptions.
   - Examples: active local additions in `hermes-agent`, `claude-code`, `codex`, OpenClaw skills.
   - Rule: do not promote private/local facts into bundled repo skills unless generalized and safe.

---

## Update-First Consolidation Rules

1. **No deletion before explicit approval.** This plan is not approval to delete, archive, or uninstall anything.
2. **Umbrella before removal.** If multiple skills overlap, first update or create an umbrella that preserves the shared workflow.
3. **Promote generic, preserve local.** Split divergent local additions into:
   - generic lessons suitable for repo skills,
   - MJ-local/host-specific notes that stay in `~/.hermes/skills` or memory/reference files.
4. **One authoritative decision table per domain.** Avoid repeated “when to use X vs Y” sections across siblings.
5. **Tool skills stay narrow.** Keep CLI/API commands, setup, auth, and troubleshooting in the tool-specific skill.
6. **References over bloat.** Move long background docs into `references/*.md`, then link them from umbrella/tool skills.
7. **Rewrite references before archive.** Before retiring any skill, search and rewrite:
   - `metadata.hermes.related_skills`
   - cron job `skills` fields
   - gateway/channel auto-skill config
   - profile startup instructions
   - docs links and plan references
8. **Validate loader behavior after each batch.** `skills_list`, `skill_view`, and frontmatter validation must all pass in a fresh session or script-level scan.
9. **Rollback is file-level and config-level.** Every phase must have a tar/git backup before moving files.

---

## Phased Migration Plan

### Phase 0: Freeze destructive operations

**Objective:** Preserve current behavior while planning.

**Actions:**
1. Record this audit and plan in Linear/Kanban.
2. Do not delete, archive, uninstall, or rename any skill.
3. Treat `/Users/alfred/.hermes/skills` as production runtime state.

**Proof:**
- Inventory counts above.
- No skill files removed.

**Rollback:**
- None needed; this phase is read-only except plan artifacts.

### Phase 1: Create backups and machine-readable inventory

**Objective:** Make rollback and diff review safe.

**Actions:**
1. Create timestamped tarballs of:
   - `/Users/alfred/.hermes/skills`
   - `/Users/alfred/.hermes/hermes-agent/skills`
   - `/Users/alfred/.hermes/hermes-agent/optional-skills`
2. Store a JSON inventory with name, root, path, sha, desc, tags, related skills, and validation result.
3. Add a script or documented command that regenerates the inventory.

**Proof gate:**
- Backup files exist and can be listed.
- Regenerated inventory count equals current audit or expected delta.

**Rollback:**
- Restore the tarball for the affected tree.

**Phase 1 execution proof (2026-05-26):**
- Backup directory: `/Users/alfred/.hermes/backups/skills-consolidation-20260526-161030/`
- Backup files created and tar-list smoke checked:
  - `user-local-skills.tgz`
  - `repo-bundled-skills.tgz`
  - `repo-optional-skills.tgz`
- Repeatable inventory script: `scripts/skills_inventory_audit.py`
- Regenerated inventory artifact: `docs/plans/2026-05-26-skills-audit-data.json`
- Readback counts: active 114, bundled 87, optional 81, active validation failures 0.

### Phase 1.5: Owner-map validation harness

**Objective:** Prove the new structure changes future agent behavior before consolidating or deleting any skill.

**Actions:**
1. Add a machine-readable owner registry: `docs/plans/skills-owner-map.yaml`.
2. Add synthetic routing cases: `docs/plans/skills-routing-cases.yaml`.
3. Add a validation harness: `scripts/skills_owner_routing_check.py`.
4. Gate future consolidation on routing accuracy: new lessons should map to an existing owner skill before a new sibling skill is created.

**Proof gate:**
- Owner map has no structural validation errors.
- Synthetic routing pass rate is at least 90%.
- Any failed case either updates the owner map or records why a new skill is truly needed.

**Execution proof (2026-05-26):**
- Owner registry entries: 15.
- Synthetic cases: 21.
- Validation result: 21/21 passed, 100% pass rate.
- Validation artifact: `docs/plans/2026-05-26-skills-routing-validation.json`.
- No skill deletion/archive/rename performed.

**Rollback:**
- Remove the three Phase 1.5 artifacts and continue using the previous review-only plan.

### Phase 2: Resolve exact active/source duplicates without changing semantics

**Objective:** Reduce mirror duplication safely.

**Candidate set:**
- 76 exact active/source duplicates.

**Actions:**
1. Decide mechanism first:
   - Option A: keep copies, but mark as installed-from-source; no runtime change.
   - Option B: configure loader/install flow so bundled skills can be loaded without physical copies in `~/.hermes/skills`.
   - Option C: remove exact local copies only after loader is verified to still expose them from repo source.
2. Prefer A or B initially. C requires explicit MJ approval.
3. After each candidate batch, verify `skills_list` still shows expected skills and `skill_view(<name>)` resolves.

**Proof gate:**
- Skill count changes are intentional and documented.
- No missing skill names in a before/after name diff.

**Rollback:**
- Restore removed exact copies from backup.
- Revert loader/config change.

### Phase 3: Merge divergent active/source duplicates one by one

**Objective:** Preserve local lessons while reducing fork drift.

**Candidate set:**
- `axolotl`, `claude-code`, `codex`, `fine-tuning-with-trl`, `google-workspace`, `hermes-agent`, `hermes-agent-skill-authoring`, `kanban-worker`, `linear`, `obsidian`, `outlines`, `spike`, `systematic-debugging`, `test-driven-development`, `unsloth`.

**Actions:**
1. For each candidate, classify each local change as:
   - generic repo improvement,
   - MJ-local rule/reference,
   - stale/outdated delta,
   - platform compatibility override.
2. Promote generic changes to repo skill or reference file.
3. Keep MJ-local changes in user-local skill or a clearly local reference.
4. Only after source and local layers are reconciled, decide whether local copy should remain.

**Proof gate:**
- Per-skill diff review note with decision for every local hunk.
- Fresh validation: frontmatter, description length, body non-empty.
- `skill_view` confirms expected final content source.

**Rollback:**
- Revert repo commit for promoted changes.
- Restore original local copy from backup.

### Phase 4: Add/update umbrella workflow skills

**Objective:** Stop overlap from reappearing.

**Priority umbrellas:**
1. Autonomous coding agents routing:
   - existing candidates: `subagent-driven-development`, `claude-code`, `codex`, `opencode`.
   - decision needed: update existing umbrella vs create `autonomous-coding-agents`.
2. Software shipping lifecycle:
   - clarify relationship among `writing-plans`, `spike`, `systematic-debugging`, `test-driven-development`, `requesting-code-review`, `mj-shipping-workflow`.
3. Creative visual routing:
   - decision table for infographic/comic/article illustration/sketch/design/ascii/video/manim/p5.
4. LLM training lifecycle:
   - shared LoRA/QLoRA/dataset/eval/HF gates for Axolotl/Unsloth/TRL.
5. Document IO routing:
   - PDF/OCR/Google/Notion/Obsidian/PowerPoint/meeting pipeline.

**Proof gate:**
- Each umbrella has a trigger, decision table, related skills, and “do not use for” boundaries.
- Sibling skills link to umbrella and remove duplicated decision prose.

**Rollback:**
- Revert umbrella skill file and sibling link edits.

### Phase 5: Reference rewrites and safe retirements

**Objective:** Remove or archive only after no references depend on old names.

**Actions:**
1. Search code/config/docs/session-independent state for each retirement candidate:
   - `metadata.hermes.related_skills`
   - cron jobs via `cron.jobs.rewrite_skill_refs` or equivalent CLI/tool flow
   - profile startup files
   - gateway/channel skill bindings
   - documentation and plans
2. Use forwarding/absorbed-into metadata where available.
3. Archive rather than delete first.

**Proof gate:**
- Reference scan returns no stale references.
- Fresh session can load replacement skills.
- Any cron jobs that referenced old names are rewritten.

**Rollback:**
- Unarchive/restore skill directory.
- Restore previous cron/config references from backup.

### Phase 6: Curator policy hardening

**Objective:** Make the new structure durable.

**Actions:**
1. Update curator prompt/rules if needed to enforce class-level umbrella preference.
2. Add a periodic inventory report that highlights:
   - duplicate names across roots,
   - divergent active/source copies,
   - narrow sibling proliferation,
   - broken related skill links.
3. Add tests if the consolidation touches loader/curator behavior.

**Proof gate:**
- Curator dry-run proposes update-first consolidation, not deletion-first cleanup.
- Tests pass for any modified loader/curator code.

**Rollback:**
- Revert curator/rule changes.

---

## Recommended First Approval Request

Suggestion: approve Phase 1 only.

Reasoning: Backups and a repeatable inventory are pure safety work and do not change skill behavior. After that, review the 14 divergent duplicates before any exact-copy cleanup or umbrella edits.

OK?

---

## Verification Checklist

- [x] Active inventory count captured: 114.
- [x] Repo/optional inventory counts captured: 87 bundled, 81 optional.
- [x] Duplicate report captured: 91 active/source name duplicates, 76 exact, 15 divergent.
- [x] Active validation failures checked: 0.
- [x] Proposed taxonomy documented.
- [x] Update-first rules documented.
- [x] Phased migration and rollback plan documented.
- [x] No skill deletion performed.
- [ ] Explicit MJ approval received for any deletion/archive/rename.
