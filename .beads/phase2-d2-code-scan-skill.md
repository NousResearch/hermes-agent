---
id: phase2-d2-code-scan-skill
title: Phase 2 D2 — code-scan SKILL.md: JIT orchestration skill
status: complete-and-committed
executor: delegate-coder
parallel_safe: false
base_branch: docs/ua-flywheel-phase1-phase2-plan
allowed_files:
  - skills/code-analysis/code-scan/SKILL.md
forbidden_files:
  - tools/skills_sync.py
  - tests/tools/test_skills_sync.py
  - scripts/code-scan/scan_project.py
  - scripts/code-scan/language_registry.py
  - scripts/code-scan/graph_schema.py
  - scripts/code-scan/extract_imports.py
  - .hermesignore
depends_on:
  - phase1-code-scan-completion-fix
  - phase2-d1-extract-imports
verification:
  - wc -l skills/code-analysis/code-scan/SKILL.md
  - grep -c "^|" skills/code-analysis/code-scan/SKILL.md
risk: low
---

# Phase 2 D2 — code-scan SKILL.md: JIT Orchestration Skill

## Context & Intent

**Why this bead exists.** Phase 1 provides the scan scripts (`scan_project.py`, `language_registry.py`, `graph_schema.py`) and Phase 2 D1 provides `extract_imports.py`. Agents cannot use these tools without orchestration instructions. This bead creates the JIT-loaded skill that teaches an agent _how_ to run the scan pipeline, read the outputs, and synthesize a structured summary — without loading any scan logic into context.

**Authority docs.** `.plans/phase-2-flywheel-ua-integration.md` (§D2: `skills/code-analysis/code-scan/SKILL.md`) defines scope: ≤80 lines, JIT-loaded via `agent/skill_commands.py`, orchestrates scan + import extraction + bounded LLM synthesis.

**Intent.** Create a single `SKILL.md` file at `skills/code-analysis/code-scan/SKILL.md` that defines the step-by-step orchestration flow: run `scan_project.py`, run `extract_imports.py`, read JSON artifacts, synthesize narrative fields (project name, one-line description, framework summary), render structured summary. The skill must be loadable on-demand only — never auto-injected.

**Non-goals.** No auto-injection. No dashboard. No React UI. No CLI command. No SQLite store. No tree-sitter/WASM. No always-on scanning. No changes to Phase 1 scripts. No modifications to Phase 1 test files.

## Implementation Details

### Target file

| File | Purpose | Max LOC |
|---|---|---|
| `skills/code-analysis/code-scan/SKILL.md` | JIT skill: orchestration instructions for scan pipeline | ≤80 |

### Required frontmatter

```yaml
---
name: code-scan
hermes.tags: [on-demand, code-analysis, project-mapping]
---
```

The `on-demand` tag is mandatory — ensures the skill is discovered but not loaded into every session context.

### Skill behavior (exact steps)

1. **Confirm target:** Ask user for target directory or use current working directory.
2. **Run scan:** `python scripts/code-scan/scan_project.py <target_dir> --output <temp_scan.json>`
3. **Run import extraction:** `python scripts/code-scan/extract_imports.py <temp_scan.json> > <temp_imports.json>`
4. **Read artifacts:** Read both JSON files using `read_file` or `terminal` + `cat`.
5. **Synthesize (LLM-only fields):** From the structured data, produce:
   - Project name (from directory name or `package.json`/`pyproject.toml` name field)
   - One-line description (inferred from frameworks + key files, not hallucinated)
   - Framework/stack narrative (from detected frameworks + language distribution)
6. **Render summary:** Output as structured markdown per the output format below.
7. **Clean up:** Note that temp files can be deleted; they are not tracked artifacts.

### Constraints the skill must enforce

- Never hallucinate file structures — only report what scan scripts return.
- If scan fails, report the error; do not guess.
- Respect `.hermesignore` rules (already enforced by `scan_project.py`).
- Only synthesize the three narrative fields (name, description, framework). Everything else is deterministic from JSON.

### Output format (prescribed)

```markdown
## Project: <name>
- **Description:** <one-line>
- **Languages:** <detected language distribution from scan JSON>
- **Frameworks:** <detected frameworks array>
- **Structure:** <top-level dirs + key files from scan JSON>
- **Import map:** <top 5 most-imported modules from extract_imports output>
- **Files:** <total_files> total, <files_with_imports> with imports
```

### Line budget check

The SKILL.md MUST NOT exceed 80 lines. Line count includes frontmatter, headers, body text, and code blocks. This is enforced during review.

## Complexity Tier

**T1** — Single-file authoring. No code logic — just structured markdown instructions. Estimated 2–3 subagent iterations. Requires coder subagent + Hermes line-count verification + reviewer signoff.

## Execution Engine

**Executor:** `delegate-coder` — Hermes dispatches a coder subagent with the exact SKILL.md content spec above.

**Process:**
1. Coder subagent writes `skills/code-analysis/code-scan/SKILL.md`.
2. Hermes verifies: `wc -l` ≤ 80, frontmatter present, `on-demand` tag present, all 7 orchestration steps included.
3. Hermes performs a dry-load test: confirms the file is discoverable by the skill manifest system (`python -c "from hermes_constants import get_hermes_home; ..."` or manual verify).
4. Reviewer subagent validates: line budget, spec compliance, scope guardrails, no forbidden-file touches.
5. Hermes presents evidence to JC.

**Subagent reliability preflight:**
- Task shape: single markdown file authoring
- Expected artifacts: 1 file, ≤80 lines
- `max_iterations`: 5 per subagent dispatch
- File-write: YES. Run-test: YES. Commit: NO.

## Required Inline Context

### Project context

- **Repo:** `/home/jarrad/.hermes/hermes-agent`
- **Current branch:** `docs/ua-flywheel-phase1-phase2-plan`
- **Skill loading:** Skills are loaded via `agent/skill_commands.py` as user messages, not system prompt. The frontmatter `hermes.tags` including `on-demand` controls discoverability.

### Existing dirty files — DO NOT TOUCH

```
tools/skills_sync.py                 # dirty
tests/tools/test_skills_sync.py      # dirty
```

### Contract test (line budget)

```bash
wc -l skills/code-analysis/code-scan/SKILL.md
# Must output ≤ 80

# Verify frontmatter
head -5 skills/code-analysis/code-scan/SKILL.md
# Must show --- name: code-scan hermes.tags: [...] ---

# Verify on-demand tag
grep "on-demand" skills/code-analysis/code-scan/SKILL.md
# Must match
```

## Dependencies

| Dependency | Type | Status |
|---|---|---|
| Phase 1 code-scan scripts | prerequisite | Completed |
| D1: extract_imports.py | prerequisite | Must be implemented and present before this file's workflow is valid |
| Skill system (`agent/skill_commands.py`) | runtime | Assumed present in hermes-agent |

## Test Obligations

### Contract tests (not unit tests, but deterministic checks)

| Check | Command | Pass criteria |
|---|---|---|
| File exists | `test -f skills/code-analysis/code-scan/SKILL.md` | Exit 0 |
| Line budget | `wc -l skills/code-analysis/code-scan/SKILL.md` | ≤80 |
| Frontmatter | `grep -q "^name: code-scan" skills/code-analysis/code-scan/SKILL.md` | Match found |
| On-demand tag | `grep -q "on-demand" skills/code-analysis/code-scan/SKILL.md` | Match found |
| References scan_project.py | `grep -q "scan_project.py" skills/code-analysis/code-scan/SKILL.md` | Match found |
| References extract_imports.py | `grep -q "extract_imports.py" skills/code-analysis/code-scan/SKILL.md` | Match found |
| No hallucination clause | `grep -qi "hallucin" skills/code-analysis/code-scan/SKILL.md` | Match found |
| Markdown whitespace | `python -c "..."` (no trailing whitespace) | Clean |

### RED/GREEN/FULL evidence required

- **RED:** File does not exist or has >80 lines
- **GREEN:** File exists, ≤80 lines, all required content present
- **FULL:** All contract tests pass, reviewer approves

## Verification Command

```bash
cd /home/jarrad/.hermes/hermes-agent

# Step 1: File exists
test -f skills/code-analysis/code-scan/SKILL.md && echo "FILE PASS" || echo "FILE FAIL"

# Step 2: Line budget ≤80
LINES=$(wc -l < skills/code-analysis/code-scan/SKILL.md)
[ "$LINES" -le 80 ] && echo "BUDGET PASS ($LINES lines)" || echo "BUDGET FAIL ($LINES lines)"

# Step 3: Required content
grep -q "^name: code-scan" skills/code-analysis/code-scan/SKILL.md && echo "FRONTMATTER PASS" || echo "FRONTMATTER FAIL"
grep -q "on-demand" skills/code-analysis/code-scan/SKILL.md && echo "TAG PASS" || echo "TAG FAIL"
grep -q "scan_project.py" skills/code-analysis/code-scan/SKILL.md && echo "SCAN PASS" || echo "SCAN FAIL"
grep -q "extract_imports.py" skills/code-analysis/code-scan/SKILL.md && echo "IMPORTS PASS" || echo "IMPORTS FAIL"
grep -qi "hallucin" skills/code-analysis/code-scan/SKILL.md && echo "NO-HALLUCINATION PASS" || echo "NO-HALLUCINATION FAIL"

# Step 4: Scope guardrail — only allowed file touched
git diff --name-only | grep -vE '^skills/code-analysis/code-scan/SKILL\.md$' | grep -vE '^(\.beads/|\.plans/)' && echo 'SCOPE FAIL' || echo 'SCOPE PASS'

# Step 5: Forbidden files check
git diff -- tools/skills_sync.py tests/tools/test_skills_sync.py 2>/dev/null | wc -l | grep -q "^[0-9][0-9]*$" && echo "FORBIDDEN PASS" || echo "FORBIDDEN FAIL"
```

### Expected pass criteria

1. File exists at exact path
2. Line count ≤80
3. All required content present (frontmatter, tags, pipeline steps, constraints)
4. Only allowed file modified
5. Forbidden files untouched

## Approval Evidence

### Before commit — present this evidence bundle to JC

**1. Line count:**
```bash
wc -l skills/code-analysis/code-scan/SKILL.md
```
Expected: ≤80.

**2. Content verification:**
```
FRONTMATTER PASS
TAG PASS
SCAN PASS
IMPORTS PASS
NO-HALLUCINATION PASS
```

**3. File content preview:**
```bash
cat skills/code-analysis/code-scan/SKILL.md
```

**4. Scope guardrail:**
```bash
git diff --name-only
# Only expected: .beads/phase2-d2-code-scan-skill.md, skills/code-analysis/code-scan/SKILL.md (and any planning files Hermes patches separately)
```

**5. Reviewer verdict:**
- [ ] Spec compliance (all orchestration steps, correct frontmatter, ≤80 lines)
- [ ] Scope preservation (no Phase 1 files, no forbidden patterns)
- [ ] Context budget (SKILL.md ≤80 lines, on-demand tag present)
- [ ] Existing dirty files not modified

**6. Commit gate:**
```
NO COMMIT, PUSH, OR MERGE until JC explicitly approves.
```

---

> **Bead execution readiness = this bead passes reviewer polish and JC approves execution.**
> **Bead completion = all verification commands exit 0 + reviewer PASS + JC commit approval.**
> Coder subagent has NO commit/push authority.
