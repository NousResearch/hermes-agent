---
title: Hermes Agent 3-Tier Knowledge Center — Implementation Plan
tags:
  - hermes-agent
  - knowledge-center
  - implementation-plan
  - phases
status: active
updated: 2026-05-23
---

# 3-Tier Knowledge Center Implementation Plan

## Compliance Tracking Legend

| Field | Meaning |
|-------|---------|
| **Done %** | Numeric completion of the issue (0–100) |
| **Remaining %** | 100 − Done % |
| **Evidence** | Command, file path, or test output proving completion |
| **Localhost/VPS** | Verification that the feature works on localhost (or VPS if applicable) |

---

## Phase 0: Safety Baseline

**Goal:** Verify existing system state before making any changes. Create test harness.

| Issue | Description | Verification Gate | Done % | Remaining % | Evidence |
|-------|-------------|-------------------|--------|-------------|----------|
| 0.1 | Verify `hermes dashboard` runs on localhost | `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9119/` returns 200 | 0 | 100 | HTTP status code |
| 0.2 | Verify `/chat` endpoint loads | `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9119/chat` returns 200 | 0 | 100 | HTTP status code |
| 0.3 | Verify existing Obsidian vault structure exists | `[ -d ~/ObsidianVault/HermesAgent/projects ] && [ -d ~/ObsidianVault/HermesAgent/playbooks ]` | 0 | 100 | Directory existence |
| 0.4 | Verify MOC.md exists and is valid markdown | `test -f ~/ObsidianVault/HermesAgent/MOC.md && head -5 ~/ObsidianVault/HermesAgent/MOC.md` | 0 | 100 | File exists + frontmatter |
| 0.5 | Verify context packs exist (count ≥ 30) | `ls docs/hermes-agent-standalone/context-packs/*.md \| wc -l` | 0 | 100 | Count ≥ 30 |
| 0.6 | Verify project vault notes exist (count ≥ 30) | `ls ~/ObsidianVault/HermesAgent/projects/*.md \| wc -l` | 0 | 100 | Count ≥ 30 |
| 0.7 | Verify existing skills load without error | `hermes --help` exits 0, no import errors in logs | 0 | 100 | Exit code 0 |
| 0.8 | Verify no forbidden runtime URLs in existing docs | `rg -n "7421\|7422" docs/hermes-agent-standalone/ ~/ObsidianVault/HermesAgent/` returns 0 matches | 0 | 100 | rg output = empty |
| 0.9 | Verify no symlinks in Obsidian vault | `find ~/ObsidianVault/HermesAgent -type l -print` returns 0 results | 0 | 100 | find output = empty |
| 0.10 | Verify no secret values in docs or vault | `rg -n "(sk-[A-Za-z0-9]{20,}\|ghp_[A-Za-z0-9]{20,}\|AIza[A-Za-z0-9_-]{20,})" docs/ ~/ObsidianVault/` returns 0 matches | 0 | 100 | rg output = empty |
| 0.11 | Create test harness directory | `mkdir -p tests/knowledge_center/` exists | 0 | 100 | Directory exists |
| 0.12 | Write Phase 0 compliance report | File `docs/hermes-agent-standalone/reports/phase-0-safety-baseline.md` exists with all issue rows | 0 | 100 | File exists |

**Phase 0 Completion Rule:** All 12 issues must be 100/0. Any failure blocks Phase 1.

---

## Phase 1: Domain Knowledge Layer (Tier 2 Foundation)

**Goal:** Create the `domains/` structure in Obsidian vault, domain index, and domain-aware context loader.

### Issue Breakdown

| Issue | Description | Detailed Requirements | Verification Gate | Done % | Remaining % | Evidence |
|-------|-------------|----------------------|-------------------|--------|-------------|----------|
| 1.1 | Create `domains/` directory structure in Obsidian vault | Create: `~/ObsidianVault/HermesAgent/domains/` with subdirectories: `frontend/`, `backend/`, `devops/`, `security/`, `testing/`, `data/`, `mobile/`, `infrastructure/`. Each subdirectory must have a `README.md` with frontmatter (`title`, `tags`, `status`, `updated`, `description`). | `ls ~/ObsidianVault/HermesAgent/domains/*/README.md` shows 8 files. Each has valid YAML frontmatter. | 0 | 100 | File count = 8 + frontmatter valid |
| 1.2 | Create domain index (`domains/index.md`) | File must contain: table of all 8 domains, each with slug, description, project count (initially 0), note count (initially 0), last updated timestamp. Must use wikilinks to each domain README. Must have frontmatter with `title: Domain Knowledge Index`. | `test -f ~/ObsidianVault/HermesAgent/domains/index.md`. File contains all 8 domain slugs. Wikilinks resolve (target files exist). | 0 | 100 | File exists + all 8 slugs present + links valid |
| 1.3 | Create domain README templates | Each domain README must contain: description, when-to-use rules, example patterns placeholder, project mapping table (empty initially), verification checklist. Must follow the same frontmatter standard as existing vault notes. | Each of 8 README.md files contains: `---` frontmatter, `## Description`, `## When to Use`, `## Patterns`, `## Projects`, `## Verification Checklist`. | 0 | 100 | All 8 files have required sections |
| 1.4 | Update vault MOC.md to include domains entry | Add `- [[domains/index\|Domain Knowledge]]` to MOC.md entry points section. Must not break existing links. | `grep "domains/index" ~/ObsidianVault/HermesAgent/MOC.md` finds the line. MOC.md still has all original entries. | 0 | 100 | grep finds new line + original entries intact |
| 1.5 | Update project vault notes with domain tags | For each existing project note in `projects/*.md`, add a `domain:` frontmatter field based on project stack: `node/next` → `["frontend","backend"]`, `python` → `["backend","data"]`, `docker/mixed` → `["devops","backend"]`, `unknown/mixed` → `[]`. Must not remove existing frontmatter fields. | `rg "^domain:" ~/ObsidianVault/HermesAgent/projects/*.md` returns count ≥ 30. No project note lost its original `title`, `tags`, `status` fields. | 0 | 100 | domain field present in all notes + original fields intact |
| 1.6 | Create domain-project mapping file | Create `~/ObsidianVault/HermesAgent/domains/mapping.md` — a table mapping each project slug to its domains. Must be auto-generatable from project frontmatter. | File exists. Contains table with all project slugs. Each row has ≥1 domain. | 0 | 100 | File exists + table complete |
| 1.7 | Create `agent/knowledge_domains.py` — domain relevance matcher | Module with: `DomainRelevanceMatcher` class. Methods: `classify(project_slug) → list[str]` (returns domain slugs for a project), `match_knowledge(content: str) → list[str]` (returns relevant domain slugs for arbitrary text using keyword + heuristics), `get_domain_notes(domains: list[str]) → list[Path]` (returns paths to domain KB notes). Must use filesystem only — no network calls. | `python -c "from agent.knowledge_domains import DomainRelevanceMatcher; m = DomainRelevanceMatcher(); print(m.classify('tech-tools-hermes-agent'))"` exits 0 and returns list. | 0 | 100 | Import succeeds + classify returns domains |
| 1.8 | Write unit tests for `DomainRelevanceMatcher` | Test file: `tests/knowledge_center/test_knowledge_domains.py`. Must test: classify with known project slugs, match_knowledge with sample text, get_domain_notes with valid/invalid domains, edge cases (empty project, unknown stack). Must use only stdlib + pytest + unittest.mock. | `scripts/run_tests.sh tests/knowledge_center/test_knowledge_domains.py -q` passes all tests. | 0 | 100 | Test run passes |
| 1.9 | Create domain loader tool (`tools/domain_knowledge_loader.py`) | Tool that: accepts `domains` parameter (list of domain slugs), reads matching domain notes from vault, returns combined markdown. Must use `read_file` tool internally (not shell). Must have `check_requirements()` that verifies vault path exists. Must return JSON string. Tool name: `load_domain_knowledge`. | Tool registered: `python -c "from tools.registry import registry; print('load_domain_knowledge' in registry._handlers)"` returns True. | 0 | 100 | Tool registered + returns valid JSON |
| 1.10 | Write unit tests for domain loader tool | Test file: `tests/knowledge_center/test_domain_knowledge_loader.py`. Must test: loading valid domains, loading nonexistent domains (graceful error), vault path missing (check_requirements returns False), content returned as valid JSON. | `scripts/run_tests.sh tests/knowledge_center/test_domain_knowledge_loader.py -q` passes. | 0 | 100 | Test run passes |
| 1.11 | Integrate domain loader into context assembly | Modify `agent/system_prompt.py` (or appropriate context assembly file) to: after loading project context pack, call `DomainRelevanceMatcher.classify()` to get domains, call `load_domain_knowledge` tool to get domain notes, inject into system prompt as `<domain-knowledge>` block (distinct from `<memory-context>` block). Must be cache-aware — invalidation only when domains change. | Start a new session, verify system prompt contains `<domain-knowledge>` block with correct domain notes for the project. | 0 | 100 | System prompt contains domain block |
| 1.12 | Verify token budget for domain loading | Measure: load Tier 1 (project context pack) + Tier 2 (domain notes) + Tier 3 (playbooks). Total must be ≤ 3000 tokens. If exceeds, implement truncation (keep most recent note per domain, truncate old notes). | Run measurement script, output shows total ≤ 3000 tokens. | 0 | 100 | Measurement output ≤ 3000 |
| 1.13 | Write Phase 1 compliance report | File `docs/hermes-agent-standalone/reports/phase-1-domain-layer.md` with all issue rows, evidence, localhost status. | File exists with all 13 issues listed. Each has Done % and Remaining %. | 0 | 100 | File exists + complete |
| 1.14 | Localhost verification — domain loading works end-to-end | Start `hermes dashboard`, open `/chat`, send message that triggers a project session. Verify in agent logs that domain knowledge was loaded. Verify response uses domain-relevant knowledge. | `rg "domain-knowledge" ~/.hermes/logs/agent.log` shows domain loading entries after a chat turn. | 0 | 100 | Log entries confirm domain loading |

**Phase 1 Completion Rule:** All 14 issues must be 100/0. Issues 1.8, 1.10 must pass tests. Issue 1.12 must show ≤ 3000 tokens. Issue 1.14 must show localhost verification.

---

## Phase 2: Knowledge Promote Tool + Relevance Matcher

**Goal:** Create the tool and logic for promoting knowledge from project-local to domain-shared.

### Issue Breakdown

| Issue | Description | Detailed Requirements | Verification Gate | Done % | Remaining % | Evidence |
|-------|-------------|----------------------|-------------------|--------|-------------|----------|
| 2.1 | Create `agent/knowledge_relevance.py` — cross-project relevance engine | Module with: `KnowledgeRelevanceEngine` class. Methods: `is_cross_project_relevant(content: str, source_project: str) → bool` (checks if content pattern exists in ≥2 projects), `find_matching_projects(content: str, source_project: str) → list[str]` (returns project slugs that would benefit from this knowledge), `get_relevance_score(content: str, target_project: str) → float` (0.0–1.0 score). Must use keyword matching + stack similarity + domain overlap heuristics. No network calls. | `python -c "from agent.knowledge_relevance import KnowledgeRelevanceEngine; e = KnowledgeRelevanceEngine(); print(e.is_cross_project_relevant('Docker compose healthcheck pattern', 'tech-tools-hermes-agent'))"` exits 0. | 0 | 100 | Import succeeds + returns bool |
| 2.2 | Write unit tests for relevance engine | Test file: `tests/knowledge_center/test_knowledge_relevance.py`. Must test: cross-project detection with matching content, no-match case, relevance scoring edge cases, projects with same stack get higher scores. | `scripts/run_tests.sh tests/knowledge_center/test_knowledge_relevance.py -q` passes. | 0 | 100 | Test run passes |
| 2.3 | Create `tools/knowledge_promote.py` — promote tool | Tool that: accepts `title`, `content`, `source_project`, `target_domain`, `summary` parameters. Writes knowledge note to `~/ObsidianVault/HermesAgent/domains/{domain}/{slug}.md` with frontmatter (`title`, `tags`, `status: approved`, `origin_project`, `promoted_at`, `updated`). Updates domain index note count. Updates source project note with backlink. Returns JSON with success status and note path. Tool name: `promote_knowledge`. | Tool registered. Writing a test note creates file with correct frontmatter, updates index, creates backlink. | 0 | 100 | File created + frontmatter valid + index updated |
| 2.4 | Write unit tests for promote tool | Test file: `tests/knowledge_center/test_knowledge_promote.py`. Must test: successful promotion (file created, index updated, backlink created), invalid domain (error returned), missing vault path (check_requirements fails), duplicate title (append version number). | `scripts/run_tests.sh tests/knowledge_center/test_knowledge_promote.py -q` passes. | 0 | 100 | Test run passes |
| 2.5 | Create `tools/knowledge_review.py` — review queue tool | Tool that: accepts `action` (list/approve/reject/defer), `knowledge_id`. Maintains review queue in `~/ObsidianVault/HermesAgent/domains/.review_queue.json`. Statuses: pending, approved, rejected, deferred. On approve → calls promote_knowledge logic. On reject → archives to `domains/.rejected/`. Tool name: `review_knowledge`. | Tool registered. Adding to queue, approving, and rejecting all work correctly. Queue file is valid JSON. | 0 | 100 | Queue operations work + JSON valid |
| 2.6 | Write unit tests for review tool | Test file: `tests/knowledge_center/test_knowledge_review.py`. Must test: list queue, approve (moves to domain KB), reject (moves to rejected), defer (keeps pending), empty queue. | `scripts/run_tests.sh tests/knowledge_center/test_knowledge_review.py -q` passes. | 0 | 100 | Test run passes |
| 2.7 | Integrate relevance engine into agent post-turn review | Modify `agent/background_review.py` (or appropriate post-turn hook) to: after each turn, check if any knowledge was created/modified. If yes, run `KnowledgeRelevanceEngine.is_cross_project_relevant()`. If true AND no existing deny preference → add to review queue. Must NOT interrupt the main conversation — runs in background thread. | Start session, create knowledge, verify it appears in `.review_queue.json` if cross-project relevant. | 0 | 100 | Queue entry created for cross-project knowledge |
| 2.8 | Create review queue slash command | Add `/knowledge-review` slash command (or extend existing `/knowledge` command) that shows pending items and allows approve/reject/defer. Must work in CLI and gateway. CommandDef added to `COMMAND_REGISTRY`. Handler added to `HermesCLI.process_command()`. | `hermes knowledge-review` (or `/knowledge-review` in gateway) shows queue and accepts actions. | 0 | 100 | Command works in CLI |
| 2.9 | Write Phase 2 compliance report | File `docs/hermes-agent-standalone/reports/phase-2-promote-tool.md` with all issue rows, evidence, localhost status. | File exists with all 9 issues listed. | 0 | 100 | File exists + complete |
| 2.10 | Localhost verification — promote flow works end-to-end | Start `hermes dashboard`, open `/chat`. Send message that creates knowledge. Verify: (a) knowledge appears in project-local KB, (b) relevance engine detects cross-project match, (c) entry appears in review queue, (d) approve via slash command → knowledge promoted to domain KB. | All 4 steps verified via log entries + file existence + queue state changes. | 0 | 100 | End-to-end flow verified |

**Phase 2 Completion Rule:** All 10 issues must be 100/0. Issues 2.2, 2.4, 2.6 must pass tests. Issue 2.10 must show localhost end-to-end verification.

---

## Phase 3: Preference Memory + Ask-Before-Promote Flow

**Goal:** Implement user preference system so agent remembers promote/deny decisions and reduces friction.

### Issue Breakdown

| Issue | Description | Detailed Requirements | Verification Gate | Done % | Remaining % | Evidence |
|-------|-------------|----------------------|-------------------|--------|-------------|----------|
| 3.1 | Create preference storage | File: `~/.hermes/knowledge_preferences.json`. Schema: array of `{id, domain, project, pattern, allow: bool, reason, created_at}`. Must be profile-aware (use `get_hermes_home()`). Must be created on first write, not on startup. | File does not exist initially. After first preference save, file exists with valid JSON array. | 0 | 100 | File created on demand + valid JSON |
| 3.2 | Create `agent/knowledge_preferences.py` — preference manager | Module with: `KnowledgePreferenceManager` class. Methods: `save_preference(domain, project, pattern, allow, reason) → str` (returns preference ID), `check_preference(domain, project, content) → Optional[dict]` (returns matching preference or None), `list_preferences() → list[dict]`, `delete_preference(id) → bool`. Must use file locking for concurrent access. Must use `get_hermes_home()` for path. | `python -c "from agent.knowledge_preferences import KnowledgePreferenceManager; m = KnowledgePreferenceManager(); m.save_preference('frontend', 'proj-a', 'react.*pattern', True, 'useful'); print(m.check_preference('frontend', 'proj-a', 'react pattern'))"` exits 0 and returns preference dict. | 0 | 100 | Import succeeds + save + check works |
| 3.3 | Write unit tests for preference manager | Test file: `tests/knowledge_center/test_knowledge_preferences.py`. Must test: save preference, check match (exact domain+project), check no match, check partial match (domain only), list all, delete, concurrent access (file locking), profile isolation (uses get_hermes_home). | `scripts/run_tests.sh tests/knowledge_center/test_knowledge_preferences.py -q` passes. | 0 | 100 | Test run passes |
| 3.4 | Create `tools/knowledge_preference_tool.py` — preference management tool | Tool that: accepts `action` (save/list/delete), parameters matching the preference schema. Returns JSON. Tool name: `manage_knowledge_preference`. | Tool registered. Saving, listing, and deleting preferences work correctly. | 0 | 100 | Tool registered + operations work |
| 3.5 | Write unit tests for preference tool | Test file: `tests/knowledge_center/test_knowledge_preference_tool.py`. Must test: save via tool, list via tool, delete via tool, invalid action returns error. | `scripts/run_tests.sh tests/knowledge_center/test_knowledge_preference_tool.py -q` passes. | 0 | 100 | Test run passes |
| 3.6 | Implement ask-before-promote flow in agent | When background review detects cross-project relevant knowledge: (1) check `KnowledgePreferenceManager.check_preference()`, (2) if allow → auto-promote (skip ask), (3) if deny → skip silently, (4) if no preference → add to review queue AND notify user with "New knowledge detected — promote to shared domain KB? [approve/reject/defer/never-for-this-domain]". "never-for-this-domain" creates a domain-level deny preference. | Start session, create cross-project knowledge. Verify: first time → asks user. After "approve" → next time auto-promotes. After "never-for-this-domain" → next time skips silently. | 0 | 100 | Preference flow works as described |
| 3.7 | Create `/knowledge-preferences` slash command | Shows all preferences, allows delete. Must work in CLI and gateway. | `hermes knowledge-preferences` shows list. Delete works. | 0 | 100 | Command works |
| 3.8 | Update background review prompt to include preference check | Modify the prompt in `agent/background_review.py` that reviews knowledge — add instruction to check preferences before queuing for review. Must not change the prompt for non-knowledge reviews (memory reviews stay the same). | Background review log shows "preference check: auto-promote" or "preference check: denied" entries. | 0 | 100 | Log entries confirm preference check |
| 3.9 | Write Phase 3 compliance report | File `docs/hermes-agent-standalone/reports/phase-3-preferences.md` with all issue rows. | File exists with all 9 issues. | 0 | 100 | File exists + complete |
| 3.10 | Localhost verification — preference flow works end-to-end | Start `hermes dashboard`, open `/chat`. (a) Create knowledge → asked to promote. (b) Approve → next time auto-promotes. (c) Create different knowledge → "never for this domain" → next time skips. Verify preference file updated correctly after each step. | All 3 scenarios verified. Preference JSON file has correct entries. | 0 | 100 | All scenarios verified + JSON correct |

**Phase 3 Completion Rule:** All 10 issues must be 100/0. Issues 3.3, 3.5 must pass tests. Issue 3.10 must show localhost end-to-end verification.

---

## Phase 4: Curator Extension for Domain Notes

**Goal:** Extend the curator review system to cover domain knowledge notes (Tier 2) — auto-archive stale, prompt review.

### Issue Breakdown

| Issue | Description | Detailed Requirements | Verification Gate | Done % | Remaining % | Evidence |
|-------|-------------|----------------------|-------------------|--------|-------------|----------|
| 4.1 | Extend curator to scan domain notes | Modify `agent/curator.py` to: during review loop, scan `~/ObsidianVault/HermesAgent/domains/*/` for `.md` files (excluding README.md, index.md, .review_queue.json). Track `last_modified` timestamp. Apply same stale/archive logic as skills: stale_after_days, archive_after_days. Must only touch agent-created domain notes (frontmatter `author: "Hermes Agent"` or `origin_project` field present). | Run curator manually (`hermes curator run`), verify it scans domain notes and reports stale ones. Does NOT touch README.md or index.md. | 0 | 100 | Curator scans domain notes + skips system files |
| 4.2 | Create domain note archive directory | Create `~/ObsidianVault/HermesAgent/domains/.archive/`. Curator moves stale notes here (not deletes). Must preserve frontmatter. Must create archive index `~/ObsidianVault/HermesAgent/domains/.archive/index.md`. | Directory exists. Archive index file exists. | 0 | 100 | Directory + index exist |
| 4.3 | Add curator restore command for domain notes | Extend `hermes curator restore` to accept `--domain` flag. Restores note from `.archive/` back to its domain directory. Must update domain index. | `hermes curator restore --domain <note-id>` restores file and updates index. | 0 | 100 | Restore works + index updated |
| 4.4 | Write unit tests for curator domain extension | Test file: `tests/knowledge_center/test_curator_domain.py`. Must test: curator scans domain notes, marks stale, archives, skips system files, skips non-agent-created notes, restore works. Must use only stdlib + pytest + unittest.mock. | `scripts/run_tests.sh tests/knowledge_center/test_curator_domain.py -q` passes. | 0 | 100 | Test run passes |
| 4.5 | Add domain knowledge usage tracking | Create `~/.hermes/knowledge_usage.json`. Schema: `{note_id: {view_count, use_count, last_used_at, domain, origin_project}}`. Updated when agent loads or references a domain note. Must use file locking. | After agent uses a domain note, usage JSON shows incremented view_count. | 0 | 100 | Usage tracking works |
| 4.6 | Curator uses usage data for review priority | Modify curator review loop to: prioritize notes with high view_count but old last_modified (popular but potentially stale). Low-view notes get lower priority. Must not skip review entirely — just reorder. | Curator review log shows "priority: high" for popular-stale notes. | 0 | 100 | Review log shows priority ordering |
| 4.7 | Write Phase 4 compliance report | File `docs/hermes-agent-standalone/reports/phase-4-curator.md` with all issue rows. | File exists with all 7 issues. | 0 | 100 | File exists + complete |
| 4.8 | Localhost verification — curator domain review works | Run `hermes curator run` on localhost. Verify: (a) domain notes scanned, (b) stale notes identified, (c) archive operation works, (d) usage tracking updates. | All 4 steps verified via log entries + file state. | 0 | 100 | All steps verified |

**Phase 4 Completion Rule:** All 8 issues must be 100/0. Issue 4.4 must pass tests. Issue 4.8 must show localhost verification.

---

## Phase 5: Integration E2E Test

**Goal:** End-to-end test of the full knowledge flow: work → create knowledge → detect relevance → ask user → promote → consume in another project.

### Issue Breakdown

| Issue | Description | Detailed Requirements | Verification Gate | Done % | Remaining % | Evidence |
|-------|-------------|----------------------|-------------------|--------|-------------|----------|
| 5.1 | Create E2E test script | Script: `tests/knowledge_center/test_e2e_knowledge_flow.py`. Simulates: (a) Agent works on Project A, creates knowledge note, (b) Relevance engine detects cross-project match with Project B, (c) No existing preference → adds to review queue, (d) User approves → promoted to domain KB, (e) Agent works on Project B, loads domain KB, finds the promoted knowledge, (f) Verifies knowledge was used. Must use only stdlib + pytest + unittest.mock. No network calls. | Script runs and passes. All 6 steps verified programmatically. | 0 | 100 | Test script passes |
| 5.2 | Create E2E test for deny preference flow | Script: `tests/knowledge_center/test_e2e_deny_preference.py`. Simulates: (a) Agent works on Project A, creates knowledge, (b) Relevance detects cross-project match, (c) User denies → stored as preference, (d) Agent works on Project A again, creates similar knowledge, (e) Preference match → skips silently (no ask, no promote). | Script runs and passes. All 5 steps verified. Preference file shows deny entry. | 0 | 100 | Test script passes + preference saved |
| 5.3 | Create E2E test for domain-level deny | Script: `tests/knowledge_center/test_e2e_domain_deny.py`. Simulates: (a) User sets "never for domain=frontend", (b) Any frontend knowledge from any project skips automatically. | Script runs and passes. | 0 | 100 | Test script passes |
| 5.4 | Create E2E test for curator lifecycle | Script: `tests/knowledge_center/test_e2e_curator_lifecycle.py`. Simulates: (a) Create domain note, (b) Fake timestamp to be old, (c) Run curator → marks stale, (d) Run curator again → archives, (e) Restore → note back in domain KB. | Script runs and passes. All 5 steps verified. | 0 | 100 | Test script passes |
| 5.5 | Token budget E2E measurement | Script: `tests/knowledge_center/test_e2e_token_budget.py`. Measures: (a) Tier 1 only (project context pack), (b) Tier 1 + Tier 2 (project + domain notes), (c) Tier 1 + Tier 2 + Tier 3 (all tiers). Each must be ≤ 3000 tokens total. If any exceeds, document which tier caused it and the mitigation. | Script runs. All three measurements ≤ 3000 tokens. If any exceeds, mitigation documented. | 0 | 100 | All measurements ≤ 3000 or mitigated |
| 5.6 | Write Phase 5 compliance report | File `docs/hermes-agent-standalone/reports/phase-5-e2e.md` with all issue rows. | File exists with all 6 issues. | 0 | 100 | File exists + complete |

**Phase 5 Completion Rule:** All 6 issues must be 100/0. All test scripts must pass. Token budget must be ≤ 3000.

---

## Phase 6: Final Acceptance

**Goal:** Full system verification — localhost, VPS, Obsidian graph, compliance report, documentation.

### Issue Breakdown

| Issue | Description | Detailed Requirements | Verification Gate | Done % | Remaining % | Evidence |
|-------|-------------|----------------------|-------------------|--------|-------------|----------|
| 6.1 | Verify `hermes dashboard` starts and serves all endpoints | `curl http://127.0.0.1:9119/` → 200. `curl http://127.0.0.1:9119/chat` → 200. `curl http://127.0.0.1:9119/api/sessions` → 200. No errors in startup log. | All 3 endpoints return 200. No startup errors. | 0 | 100 | HTTP status codes + clean log |
| 6.2 | Verify `/chat` embedded TUI works | Open `http://127.0.0.1:9119/chat` in browser. Send a message to a project session. Verify: (a) response streams, (b) domain knowledge loaded (check logs), (c) no errors in browser console. | Browser shows working chat. Logs show domain loading. No console errors. | 0 | 100 | Chat works + domain loading confirmed |
| 6.3 | Verify full knowledge flow in live chat | In `/chat`: (a) Work on Project A → create knowledge, (b) Relevance detects cross-project match, (c) Ask appears (or auto-promote if preference exists), (d) Approve → promoted to domain KB, (e) Switch to Project B → domain knowledge loaded. | All 5 steps complete in live chat. | 0 | 100 | Live chat flow verified |
| 6.4 | Verify Obsidian vault graph is valid | Open `~/ObsidianVault/HermesAgent` in Obsidian. Verify: (a) MOC.md links to all sections, (b) Domain index links to all 8 domains, (c) Project notes have domain tags, (d) No broken wikilinks, (e) No symlinks. | All checks pass in Obsidian graph view. | 0 | 100 | Obsidian graph valid |
| 6.5 | Verify no forbidden runtime URLs | `rg -n "7421\|7422" docs/hermes-agent-standalone/ ~/ObsidianVault/HermesAgent/ ~/.hermes/skills/` returns 0 matches. | 0 matches found. | 0 | 100 | rg output empty |
| 6.6 | Verify no secrets in generated files | `rg -n "(sk-[A-Za-z0-9]{20,}\|ghp_[A-Za-z0-9]{20,}\|AIza[A-Za-z0-9_-]{20,})" docs/ ~/ObsidianVault/HermesAgent/ ~/.hermes/knowledge_*.json` returns 0 matches. | 0 matches found. | 0 | 100 | rg output empty |
| 6.7 | Run full test suite | `scripts/run_tests.sh tests/knowledge_center/ -q` — all tests pass. | 0 failures, 0 errors. | 0 | 100 | All tests pass |
| 6.8 | Write final compliance report | File `docs/hermes-agent-standalone/reports/phase-6-final-acceptance.md`. Must include: (a) All 6 phases summary with Done/Remaining %, (b) Issue-by-issue table for all phases, (c) Localhost verification evidence, (d) VPS status (if applicable), (e) Residual risks (even if 0), (f) Token budget summary. | File exists with all required sections. | 0 | 100 | File exists + complete |
| 6.9 | Update project registry with knowledge center status | Update `docs/hermes-agent-standalone/03-project-registry.md` to note that knowledge center is active. Update each context pack to include `knowledge_tier: 1` field. | Registry updated. Context packs have knowledge_tier field. | 0 | 100 | Registry + packs updated |
| 6.10 | Update Obsidian MOC with knowledge center section | Add "Knowledge Center" section to MOC.md with links to domain index, review queue, preferences, curator. | MOC.md has knowledge center section with working links. | 0 | 100 | MOC.md updated + links work |

**Phase 6 Completion Rule:** All 10 issues must be 100/0. Issue 6.3 must show live chat verification. Issue 6.4 must show Obsidian graph valid. Issue 6.7 must show all tests pass.

---

## Overall Compliance Summary Template

```markdown
# Phase Compliance Summary

| Phase | Issues | Done % | Remaining % | Evidence | Localhost/VPS |
|-------|-------:|-------:|------------:|----------|---------------|
| 0: Safety Baseline | 12 | X | 100-X | [reports/phase-0](./reports/phase-0-safety-baseline.md) | localhost: ✅ |
| 1: Domain Layer | 14 | X | 100-X | [reports/phase-1](./reports/phase-1-domain-layer.md) | localhost: ✅ |
| 2: Promote Tool | 10 | X | 100-X | [reports/phase-2](./reports/phase-2-promote-tool.md) | localhost: ✅ |
| 3: Preferences | 10 | X | 100-X | [reports/phase-3](./reports/phase-3-preferences.md) | localhost: ✅ |
| 4: Curator | 8 | X | 100-X | [reports/phase-4](./reports/phase-4-curator.md) | localhost: ✅ |
| 5: E2E Tests | 6 | X | 100-X | [reports/phase-5](./reports/phase-5-e2e.md) | N/A (tests) |
| 6: Final Acceptance | 10 | X | 100-X | [reports/phase-6](./reports/phase-6-final-acceptance.md) | localhost: ✅ |
| **TOTAL** | **70** | **X** | **100-X** | — | — |

## Residual Risks
- [List any, or "0"]

## Scope Statement
100/0 means all 70 issues across 7 phases are complete, all tests pass, localhost verification succeeds, and Obsidian vault is valid.
```

---

## Human Tasks (After AI Completes All Phases)

These are tasks that require human action and cannot be automated:

| # | Task | When | Details |
|---|------|------|---------|
| H.1 | Open Obsidian vault and verify graph visually | After Phase 6 | Open `~/ObsidianVault/HermesAgent` in Obsidian app. Check that MOC, domains, projects all link correctly. Verify graph view looks clean. |
| H.2 | Review and approve first real knowledge promotion | During Phase 3 | When the system first asks "promote to shared KB?", review the knowledge content and decide whether to approve, reject, or set domain-level preference. This trains the system. |
| H.3 | Tune domain classifications | After Phase 1 | Review the domain assignments for your 40 projects. If any project is miscategorized (e.g., a project tagged `backend` that should also be `data`), update the project note's `domain:` frontmatter field. |
| H.4 | Set up curator schedule | After Phase 4 | Decide how often the curator should review domain notes. Add to `config.yaml`: `curator: { interval_hours: 24, stale_after_days: 30, archive_after_days: 90 }` (or your preferred values). |
| H.5 | Configure dashboard chat for knowledge center | After Phase 6 | If you want the dashboard to show knowledge center status, review the dashboard plugin configuration and enable any knowledge-related widgets. |
| H.6 | VPS verification (if applicable) | After Phase 6 | If you run Hermes Agent on a VPS, verify the knowledge center works there too: SSH to VPS, run `hermes dashboard --status`, verify all endpoints respond, run a test knowledge promotion. |
| H.7 | Backup strategy | After Phase 6 | Add `~/ObsidianVault/HermesAgent/` and `~/.hermes/knowledge_*.json` to your regular backup routine. These are now critical knowledge stores. |
| H.8 | Team onboarding (if applicable) | After Phase 6 | If other people use this Hermes Agent instance, share the knowledge center operating procedure: how to approve/reject promotions, how to review the queue, how to check domain KB. |

---

## File Map — What Gets Created/Modified

### New Files Created (AI)

| File | Phase | Purpose |
|------|-------|---------|
| `docs/ER_DIAGRAM.md` | 0 | ER diagram + data flow |
| `docs/hermes-agent-standalone/reports/phase-0-safety-baseline.md` | 0 | Phase 0 report |
| `tests/knowledge_center/__init__.py` | 0 | Test package |
| `tests/knowledge_center/test_knowledge_domains.py` | 1 | Unit tests |
| `tests/knowledge_center/test_domain_knowledge_loader.py` | 1 | Unit tests |
| `tests/knowledge_center/test_knowledge_relevance.py` | 2 | Unit tests |
| `tests/knowledge_center/test_knowledge_promote.py` | 2 | Unit tests |
| `tests/knowledge_center/test_knowledge_review.py` | 2 | Unit tests |
| `tests/knowledge_center/test_knowledge_preferences.py` | 3 | Unit tests |
| `tests/knowledge_center/test_knowledge_preference_tool.py` | 3 | Unit tests |
| `tests/knowledge_center/test_curator_domain.py` | 4 | Unit tests |
| `tests/knowledge_center/test_e2e_knowledge_flow.py` | 5 | E2E test |
| `tests/knowledge_center/test_e2e_deny_preference.py` | 5 | E2E test |
| `tests/knowledge_center/test_e2e_domain_deny.py` | 5 | E2E test |
| `tests/knowledge_center/test_e2e_curator_lifecycle.py` | 5 | E2E test |
| `tests/knowledge_center/test_e2e_token_budget.py` | 5 | E2E test |
| `agent/knowledge_domains.py` | 1 | Domain relevance matcher |
| `agent/knowledge_relevance.py` | 2 | Cross-project relevance engine |
| `agent/knowledge_preferences.py` | 3 | Preference manager |
| `tools/domain_knowledge_loader.py` | 1 | Domain loader tool |
| `tools/knowledge_promote.py` | 2 | Promote tool |
| `tools/knowledge_review.py` | 2 | Review queue tool |
| `tools/knowledge_preference_tool.py` | 3 | Preference management tool |
| `~/ObsidianVault/HermesAgent/domains/` (8 subdirs + READMEs + index) | 1 | Domain KB structure |
| `~/ObsidianVault/HermesAgent/domains/.review_queue.json` | 2 | Review queue |
| `~/ObsidianVault/HermesAgent/domains/.archive/` | 4 | Archive for stale notes |
| `~/.hermes/knowledge_preferences.json` | 3 | User preferences |
| `~/.hermes/knowledge_usage.json` | 4 | Usage tracking |
| Phase reports (6 files) | 0-6 | Compliance reports |

### Modified Files (AI)

| File | Phase | Change |
|------|-------|--------|
| `agent/system_prompt.py` | 1 | Add domain knowledge injection |
| `agent/background_review.py` | 2, 3, 8 | Add relevance check + preference check |
| `agent/curator.py` | 4 | Add domain note scanning |
| `hermes_cli/commands.py` | 2, 3 | Add slash commands |
| `cli.py` | 2, 3 | Add command handlers |
| `~/ObsidianVault/HermesAgent/MOC.md` | 1, 6 | Add domain entry + knowledge center section |
| `~/ObsidianVault/HermesAgent/projects/*.md` | 1 | Add `domain:` frontmatter field |
| `docs/hermes-agent-standalone/03-project-registry.md` | 6 | Add knowledge center status |

### Human-Modified Files

| File | Task | Change |
|------|------|--------|
| `~/.hermes/config.yaml` | H.4 | Add curator schedule config |
| `~/ObsidianVault/HermesAgent/projects/*.md` | H.3 | Tune domain assignments |
| Backup configuration | H.7 | Add vault + knowledge files to backup |

---

## Execution Order

```
Phase 0 (Safety) ──must pass──▶ Phase 1 (Domain Layer) ──must pass──▶ Phase 2 (Promote Tool)
                                                                                │
Phase 6 (Final) ◀── Phase 5 (E2E) ◀── Phase 4 (Curator) ◀── Phase 3 (Preferences)
     ▲
     │
     └── Human Tasks H.1 through H.8 run in parallel/after Phase 6
```

**Rules:**
- Each phase must be 100/0 on ALL issues before starting the next phase
- Tests must pass before declaring an issue 100/0
- Localhost verification must succeed before declaring an issue 100/0
- Human tasks cannot start until Phase 6 is 100/0
