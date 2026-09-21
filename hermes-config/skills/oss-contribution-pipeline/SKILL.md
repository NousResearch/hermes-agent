---
name: oss-contribution-pipeline
description: >-
  Build and operate an autonomous OSS contribution pipeline — a Hermes profile
  (leaf worker) that monitors upstream issues, scans repos for fix opportunities,
  files well-structured issues, and submits clean PRs. Covers profile setup,
  no_agent cron scripts, quality gates, sign-off conventions, and the full
  issue-detection-to-PR lifecycle.
license: MIT
compatibility: Hermes Agent
metadata:
  tags: [opensource, contributing, github, moss, automation, pipeline]
  related_skills: [kanban-ops, system-script-patterns, github-pr-workflow, hermes-command-analysis]
  version: "1.0.0"
---

# OSS Contribution Pipeline

## When to Use This Skill

- Setting up a new autonomous OSS contribution profile (like Moss)
- Creating no_agent cron scripts that monitor upstream repos for issues
- Building a repo scanner that finds fix opportunities across a portfolio
- Designing quality gates for OSS PRs
- Configuring sign-off conventions for contributions

## Architecture

```
Upstream Issues (GitHub) ──┐
                            ├──→ moss-issue-watch (no_agent cron, every 15m)
                            │       │
                            │       ├──→ Writes fixable P1/P2 bugs to moss-fix-queue.json
                            │       └──→ Delivers classified issues to Discord #build-log
                            │
                            ├──→ moss-fix-pipeline (agent cron, every 1h @ workdir=upstream clone)
                            │       │
                            │       ├──→ Reads queue, picks highest-priority unhandled issue
                            │       ├──→ Fetches issue details from GitHub
                            │       ├──→ Delegates fix to subagent (terminal+file toolsets)
                            │       ├──→ Verifies fix compiles/tests pass
                            │       ├──→ Pushes to Sahil-SS9/hermes-agent fork
                            │       └──→ Creates PR with `gh pr create`
                            │       └──→ Reports PR URL to Discord #build-log
                            │
Sahil's Public Repos ───────┘
                            ├──→ moss-repo-scan (no_agent cron, daily 06:00)
                                    ↓
                                Persistent clone cache (~/.hermes/data/moss-clones/)
                                Scans for: hardcoded constants, bare except:pass,
                                security-relevant TODOs, print() in prod (not tests)
                                Dedup via content-hash fingerprint — only new findings
                                    ↓
                                Delivered to Discord #build-log

P3 open PRs from prior work ─────→ Reviewed for merge progress (periodic ping)
```

## Fix Pipeline (moss-fix-pipeline — agent-driven cron)

Updated 2026-06-24. An agent-driven cron (`every 2h`, workdir=upstream clone) that turns upstream bug reports into merged PRs without duplicating work.

### How it works

**Before any fix work — duplication check:**
The pipeline checks if the issue already has a PR by ANYONE, not just Sahil-SS9:
```
gh issue view <number> --json state
gh search prs "Fixes #<number>" --repo <repo> --state open --json number,url,author
gh search prs "#<number>" --repo <repo> --state open --json number,url,author
```
If an open PR exists by another contributor, the issue is removed from queue with reason logged. If the issue is already closed on upstream, it's removed as already fixed.

**Fix cycle — one issue per run:**
1. Read queue file for pending candidates
2. Pick highest-priority item (P1 > P2 > P3, then oldest first)
3. Fetch issue details + comments via `gh issue view --comments`
4. Pull latest upstream main, create fix branch
5. Delegate fix to leaf subagent (terminal+file toolsets)

**Quality gate chain — 3 stages, ALL must pass:**
```
Stage 1 — simplify-swarm:
  3 parallel subagents (Hygiene, Clarity, Correctness).
  Apply SAFE and CAREFUL findings. Flag RISKY.

Stage 2 — hermaguard:
  Pre-scan + 3 adversarial agents (Edge Case Hunter, Adversarial Reviewer, Blast Radius).
  Fix ALL CRITICAL and HIGH findings. Re-run simplify-swarm on updated diff.

Stage 3 — Basic checks (always run even if skills unavailable):
  S1 (secrets), S2 (personal refs), C1 (conventional commits),
  C2 (British English), T1 (tests pass), B1 (branch current), F1 (focused diff)
```

**PR and delivery:**
6. Commit, push to fork, create PR with label `type/bug`
7. Remove item from queue
8. **Wait for CI** — run `gh pr checks <number> --repo <upstream> --watch`. If CI fails due to known flaky patterns, rebase + force-push. If CI fails on our code, fix.
9. Output a casual, friendly Discord message in this style:
   ```
   🔧 #<number> — <one-line problem + fix description>
   <PR URL>
   ```
   No "Fixed #N: title | PR: url" format. Keep it casual and human.

**CRITICAL — Merge Prevention:**
- NEVER enable auto-merge (`gh pr merge --auto`, `enablePullRequestAutoMerge`).
- NEVER merge the PR yourself. The pipeline submits and verifies only.

**Queue housekeeping:** Before picking next item, sweep queue: check `gh issue view <number> --json state` on each pending issue — if `CLOSED`, remove as already fixed.

**Retry:** 3 attempts per gate, then permanent skip and log reason.

## Setup Requirements

### Upstream clone + fork remote

The pipeline works from an upstream clone at `/home/kensei/repos/hermes-agent-upstream/`:

```bash
# Add fork remote for pushing fix branches
git remote add fork https://github.com/Sahil-SS9/hermes-agent.git
```

The`origin` remote points to `NousResearch/hermes-agent` (fetch only).
The `fork` remote points to `Sahil-SS9/hermes-agent` (fetch + push).

Remotes persist across runs — only add once.

## Profile Setup

### 1. Create the profile directory

```
~/.hermes/profiles/moss/
├── config.yaml       # Model, fallbacks, toolsets, default_skills, governance
├── profile.yaml      # Description, reports-to
├── SOUL.md           # Identity, mission, hard rules, voice, workflow
├── USER.md           # User profile
├── cron/             # Empty — no cron jobs run inside the profile
├── skills/           # Synced skills (opensource-contributions + 30+ others)
├── logs/             # agent.log, errors.log
├── sessions/         # Session DB (empty until first invocation)
├── memories/         # Memory store (empty until first invocation)
└── workspace/        # Working directory
```

### 2. config.yaml essentials

```yaml
agent:
  model: glm-5.2
  provider: opencode-go
  max_iterations: 30
  reasoning_effort: medium

fallback_providers:
  - provider: ollama-cloud
    model: deepseek-v4-pro
  - provider: ollama-cloud
    model: glm-5.2

enabled_toolsets:
  - terminal
  - file
  - web
  - search
  - skills
  - session_search
  - browser

default_skills:
  - opensource-contributions
  - github-issues
  - github-pr-workflow
  - github-code-review
  - github-auth
  - codebase-inspection
  - systematic-debugging
  - simplify-swarm
  - hermaguard

governance:
  eval_domains: []
```

### 3. SOUL.md structure

- **Identity**: Name, role (leaf worker), reports-to chain
- **Mission**: What the profile does
- **Hard Rules**: Never reference personal repos/infra, sign-off convention, always read CONTRIBUTING.md, quality gates, branch safety
- **Voice**: Tone for maintainers vs internal reporting
- **Workflow**: Task format, phase-by-phase execution
- **Tools**: What CLI tools are available
- **Boundaries**: What the profile does NOT do
- **Completion Handoff**: Structured report format for the parent lead

## No-Agent Cron Scripts

### Issue & PR Watcher (`moss-issue-watch.py`)

Polls GitHub for new open issues **and PR activity** on target repos. Tracks submissions, merges, comments, and reviews. Classifies issues. Tracks seen items to avoid duplicates.

**Output format (updated 2026-06-24):** To avoid flooding Discord with multiple issue lines per tick, the watcher outputs a single-line summary:
```
NousResearch/hermes-agent has 3 new issues · P1=0, P2=2, P3=1
```
Then attaches a full HTML report via `MEDIA:/path` tag (automatically rendered as a file in Discord). The HTML report (dark theme, clickable links, classification/priority pill badges) contains two tables: new issues and PR activity. Report path: `~/.hermes/cron/output/moss-watch-reports/moss-watch-YYYY-MM-DD.html`.

This pattern (short summary + MEDIA: HTML attachment) follows the same approach used by `kanban_daily_digest_noagent.py`, `token_health_render.py`, and `calendar_brief_format.py`.

**Key patterns:**
- State file at `~/.hermes/data/moss-issue-watch.json` — persists seen issues AND PRs across runs
- Migrates gracefully from old format (no `seen_prs` key) via `load_state()` migration check
- Classification: bug (always actionable), security (always actionable), needs-repro (only if reproduction steps exist), feature (skip if vague)
- **PR filtering:** Only tracks PRs authored by users in `TRACKED_AUTHORS` list (currently `Sahil-SS9`). Uses `--author` flag at the API level (`gh pr list --author Sahil-SS9`) instead of post-fetch Python filtering. All other upstream PRs are noise. This cuts output from 30+ PRs per tick to 1-2 and ensures Sahil's PRs aren't pushed out of the 50-result window by other contributors' activity.
- PR tracking: new PRs, state changes (OPEN→MERGED, OPEN→CLOSED), new comments/reviews on open PRs
- **No per-PR detail calls:** The `gh pr list` response includes `comments` and `reviews` arrays — use those counts directly for delta detection. Do NOT call `gh pr view` per PR — it's redundant and wastes API budget.
- Priority mapping: P1 > P2 > P3 from GitHub labels
- Silent on no new activity — only outputs when there's something to report
- Runs as `no_agent: true` cron — script IS the job, stdout is delivery
- **Frequency:** every 15m for better responsiveness

```python
# Core loop:
state = load_state_file()
for repo in TARGET_REPOS:
    issues = gh_issue_list(repo, state="open")
    for issue in issues:
        if issue already seen: skip
        classification = classify(issue)
        if classification is None: mark seen, skip
        mark seen, output for delivery
save_state()
```

### Repo Scanner (`moss-repo-scan.py`)

Scans a portfolio of repos for common fix opportunities. Uses a persistent clone cache (fetch + checkout, no re-clone per run). Deduplicates findings via content-hash fingerprint so the same TODO doesn't surface twice.

**Patterns to scan for:**
- Hardcoded constants (`max_models=50`, `timeout=120`, `max_iterations=25`)
- Bare `except:` followed by `pass` (multi-line aware via context check)
- TODO/FIXME with security-relevant keywords only (`security`, `crash`, `data loss`, `regression`, `auth`, `leak`, `race`) — generic TODOs suppressed
- `print()` statements in production code (skips test files and conftest)

**Dedup mechanism:**
- Each finding is fingerprinted via SHA-256 of `repo:filepath:pattern_type:first_3_context_lines`
- Fingerprints stored in state file under `reported_findings[repo]`
- Only findings with new fingerprints are surfaced to Discord
- Context lines (not line numbers) are used in the hash to survive line shifts

**Clone strategy:**
- Persistent cache at `~/.hermes/data/moss-clones/<repo-name>/`
- First run: `git clone`. Subsequent runs: `git fetch origin` + `git checkout main`
- No `rm -rf` per run — clones persist across runs
- Scan body wrapped in `try/finally` for crash safety

**P0/P1 PR patterns from NousResearch/hermes-agent (inspiration):**
- Auth credential write-through (multi-profile rotation)
- Nix dependency fixes
- Gateway command-line matcher hardening
- SQLite trigram tokenizer fallback
- Model picker caps removal
- Session message preservation across rotation
- Custom provider persistence fixes
- Skill rmtree scope guard
- CUA environment scrubbing
- Curator snapshot pruning
- Docker gateway takeover
- Stale lock eviction

## Quality Gates (MANDATORY — 3-STAGE CHAIN ENFORCED BY THE PIPELINE AGENT)

**The agent-driven pipeline MUST run all three stages programmatically before opening a PR.**
Do not treat these as documentation-only — the pipeline prompt must include the stage commands and resolution steps. If a stage fails, fix and re-run. No stage may be skipped unless the tool itself is unavailable.

| Stage | What | When | Tool |
|-------|------|------|------|
| 1 | **simplify-swarm** | After fix is written, before review | 3 parallel subagents (Hygiene, Clarity, Correctness). Apply SAFE+CAREFUL, flag RISKY. |
| 2 | **hermaguard** | After simplification | Pre-scan + 3 adversarial agents (Edge Case Hunter, Adversarial Reviewer, Blast Radius). Fix ALL CRITICAL+HIGH. Then re-run simplify-swarm on updated diff. |
| 3 | **Basic checks** | Always (even if stages 1-2 unavailable) | See table below — grep/pytest/git commands |

### Stage 3 — Basic Checks (always run)

| Gate | Check | Command |
|------|-------|---------|
| S1 | No secrets/tokens | `grep -rE "(sk-[a-zA-Z0-9]{20,}\\|ghp_[a-zA-Z0-9]{36}\\|xox[baprs]-[a-zA-Z0-9-]+)" .` |
| S2 | No personal refs | `grep -riE "(Sahil.?SS9\\|KenseiAgent\\|kensei\\.local\\|ops-lead@kensei)" .` |
| C1 | Conventional commits | `git log origin/main..HEAD --oneline \\| grep -vE "^(fix\\|feat\\|docs\\|test\\|refactor\\|chore)(\\(.+\\))?: "` |
| C2 | British English, no em-dashes | `git show --name-only \\| xargs grep -n "—"` |
| C3 | Signed off as human | Confirm issue/PR body is signed off as the human, no agent disclosure |
| T1 | Tests pass | `pytest tests/<affected-files> -v` (or compile-check if no test files touched) |
| B1 | Branch current | `git fetch origin && git merge-base --is-ancestor origin/main HEAD` |
| F1 | No unrelated changes | `git diff --stat origin/main..HEAD` — only fix files, no unrelated edits |

## Sign-Off Convention

All issues, PRs, and comments are signed off as the human operator. **No AI agent disclosure.** The work stands on its own merit.

**Correction history:** The original convention required a disclosure line: "Submitted by Moss (AI agent on behalf of Sahil Saghir)". This was removed by user request — they want contributions to stand on their own without agent attribution. The disclosure line was stripped from:
- Issue/PR templates (bug report, feature request)
- The Agent-Specific Rule section of the opensource-contributions skill
- Quality Gate C3 (changed from "agent disclosure present" to "signed off as human")

If a user asks to remove agent disclosure, apply the same three changes: templates, rules section, and gate C3.

## Reference Files

- `references/moss-pipeline-scripts.md` — detailed script locations, cron configs, and P0/P1 PR pattern reference for the Moss pipeline
- `references/gh-cli-patterns.md` — GitHub CLI commands refined through real usage: duplication detection, PR creation from forks, issue state checks, tracked-author PR filtering, commit author configuration, and branch naming

## Self-Validation Pattern — Evaluating Your Own Recommendations

When you produce a set of recommendations (e.g. "top 3 approaches"), critically assess each one against the actual code and system constraints before presenting them as final. This session's recommendations were:

| # | Recommendation | Verdict | Why |
|---|---|---|---|
| 1 | Fix scanner dedup | **Correct** | Without it the scanner is broken by design — re-reports same findings every run |
| 2 | Filter watcher + auto-kanban P1 bugs | **Partially wrong** | Filter is correct; auto-kanban violates system architecture (Moss is a leaf worker, triage pipeline exists) |
| 3 | Merge both scripts into one | **Wrong** | Different purposes, different cadences, shared boilerplate is ~30 lines not 70% |

**The self-validation checklist:**
1. Read the actual code — don't trust your memory of what you wrote
2. Check against system architecture — does the recommendation bypass existing pipelines (triage, routing, governance)?
3. Check for real bugs, not just stylistic preferences — the `rm -rf` without `try/finally` was a real bug, not a recommendation
4. Inflated claims are the most common error — "70% shared infrastructure" was wrong, it was ~30 lines out of 260
5. A recommendation that "sounds good" but violates the documented architecture is worse than no recommendation

## Feature PRs vs Bug-Fix PRs

The Moss pipeline handles bug-fix PRs (issue-watch → fix-queue → pipeline → PR). Feature PRs follow a different path:

### Feature PR workflow

1. **Analysis first** — use `hermes-command-analysis` skill to understand the feature area, find related PRs, and assess feasibility
2. **Branch from origin/main** — `git checkout origin/main -b feat/<name>` (not from local main, which may be stale)
3. **Implement with tests** — write code + test files, run existing tests to verify no regressions
4. **Quality gates** — Hermaguard (dispatch as async `delegate_task`), then manual self-review. Fix HIGH/CRITICAL findings, amend commit
5. **Push to fork** — `git push fork feat/<name> --force` (force-push after amending)
6. **PR creation** — `gh pr create --repo <upstream> --head <fork>:feat/<name> --base main --title "feat(scope): description" --body-file /tmp/pr_body.md`
7. **PR body** — include Design decisions, Hermaguard section, Test results with exact counts
8. **CI diagnosis** — after CI runs, classify failures as our-code vs pre-existing/flaky (see `hermes-command-analysis/references/ci-flaky-test-diagnosis.md`). Do NOT fix CI infrastructure issues by changing PR code.

### Key differences from bug-fix pipeline

| Aspect | Bug-fix (Moss) | Feature (direct) |
|---|---|---|
| Trigger | Issue watch cron | User request |
| Branch naming | `fix/issue-<n>-<desc>` | `feat/<desc>` |
| Quality gates | Full 3-stage chain (simplify-swarm + hermaguard + basic) | Hermaguard + manual self-review |
| Sign-off | As human, no agent disclosure | Same |
| PR body | Short (2-3 lines + PR URL) | Detailed (design decisions, Hermaguard section, test counts) |
| Force-push | No (pipeline pushes once) | Yes (amend after Hermaguard fixes) |

## Pitfalls

1. **Scanner dedup is mandatory** — without content-hash fingerprinting, the scanner re-reports the same findings every run. This is a design defect, not a missing feature. Always implement dedup before wiring the scanner cron.
2. **Persistent clone cache vs per-run clone** — use `~/.hermes/data/moss-clones/` with `git fetch` on subsequent runs. Per-run `git clone` to `/tmp/` wastes bandwidth (10 full clones daily) and risks temp-dir leaks if the script crashes mid-scan.
3. **Watcher PR noise** — tracking all upstream PRs produces 30+ items per tick. Filter to `TRACKED_AUTHORS` (e.g. `Sahil-SS9`) to cut to 1-2. The `gh pr list` response already has comment/review arrays — do NOT call `gh pr view` per PR for delta detection.
4. **TODO signal-to-noise** — scanning all TODOs produces a wall of noise. Only report TODOs with security-relevant keywords (`security`, `crash`, `data loss`, `regression`, `auth`, `leak`, `race`). Skip `print()` detection in test files.
5. **State file migration** — when adding new state keys (e.g. `seen_prs`), always include a migration check in `load_state()` that injects the missing key with a sane default. Old state files will exist in production.
6. **Silent failure** — no_agent cron scripts that fail silently (non-zero exit, no stdout) are invisible. Test the script standalone before wiring it up.
7. **Profile never invoked** — the Moss profile is fully configured but never triggered. It needs either a cron job, a kanban task, or manual invocation to start working.
8. **Do not auto-create kanban tasks from the watcher** — the watcher is a no_agent cron script. Auto-creating kanban tasks bypasses the triage pipeline (intake → triage → route → assign). P1 bugs should be surfaced to Discord for human or Octacon triage, not auto-assigned to Moss.
9. **Do not merge watcher and scanner into one script** — they serve different purposes at different cadences. Coupling them creates shared failure modes for no benefit. Shared boilerplate (state I/O, subprocess wrappers) can be extracted to a utility module if needed.
10. **Rate limiting** — `gh` CLI uses authenticated requests (5000/hour). The watcher does 2 calls per tick (issues + PRs) at 4 ticks/hour = 8/hour. The scanner does 10 fetches daily. Both are well within limits.

11. **Duplication detection must check everyone, not just Sahil** — the pipeline previously only checked for open PRs by `Sahil-SS9`. PR #52025 was a duplicate of #49659 by alt-glitch because it didn't check for existing PRs by other contributors. Always use `gh search prs` with both `"Fixes #<num>"` and `"#<num>"` patterns, against ALL authors, before starting fix work.

12. **Queue housekeeping is mandatory before picking next item** — issues that were open when added to the queue may have been closed or PR'd by others in the meantime. The pipeline must check `gh issue view <number> --json state` on pending items and remove closed ones. Left unchecked, the pipeline wastes cycles trying to fix already-resolved issues.

13. **Agent-driven pipeline crons can hit iteration limits** — the fix + simplify-swarm + hermaguard + basic checks chain consumes many agent steps. If the cron agent runs out of iterations before pushing the PR, the fix branch persists on disk but no PR is raised. The queue item stays for the next cycle. Mitigation: keep the fix scoped to minimal changes; if the fix needs cross-stack work (backend + frontend), split into two cycles or flag as too large for a single pipeline run.

14. **Discord output must be short** — agent-driven fix pipeline: 2-3 line max (emoji + issue + PR URL). No-agent issue watcher: single-line summary with priority counts + MEDIA:/path to HTML attachment. Long inline output gets ignored by users — they want signal, not noise.

17. **needs-repro issues are not actionable** — issues with the `needs-repro` label lack reproduction steps. The pipeline should skip these immediately (move to skipped with reason) rather than spending cycles investigating whether they're true bugs.

18. **CI flaky test patterns — do not fix CI infrastructure by changing PR code.** After the pipeline creates a PR, CI may show failing checks. Before attempting fixes, classify the failure against known pre-existing patterns:

    | Pattern | Signature | Action |
    |---------|-----------|--------|
    | Model list snapshot | `test_gmi_provider.py::test_provider_model_ids_falls_back_to_static_models` — `AssertionError: assert [...] == [...]` with model list mismatch | Upstream added models between branch creation and CI run. Passes locally. **Do not fix.** |
    | pytest tmpdir race | `test_protocol.py` — 28x `FileNotFoundError: /tmp/pytest-of-runner/...` | Pytest tmpdir fixture race on CI runner. Passes locally. **Do not fix.** |
    | Test timeout | `test_pet_generate.py` — `(140s exceeded; process tree SIGKILL'd)` | CI runner resource contention. Passes locally. **Do not fix.** |
    | Artifact upload | `Failed to CreateArtifact: Unable to make request: ETIMEDOUT` | Post-job cleanup failure. **Do not fix.** |

    Verify by running the failing test locally on the branch — if it passes locally, it's environmental. See `hermes-command-analysis/references/ci-flaky-test-diagnosis.md` for the full diagnosis procedure.

    **Rebase as the first recovery action:** If CI fails on a flaky pattern, rebase the PR branch onto latest `origin/main` and force-push. This re-triggers CI with a newer base that may have matching model snapshots and a fresh runner environment. Do NOT attempt code changes to fix environmental failures.

19. **Missing script file — cron works via prompt fallback but is fragile.** The `moss-fix-pipeline` cron has `script="moss-fix-pipeline.py"` set in its config, but the file does not exist on disk. The Hermes scheduler falls back to the cron's `prompt` field when the script file is missing, which is why the pipeline still works. However, this is fragile: if the scheduler's fallback behavior changes in a future Hermes update, the pipeline silently stops working. The fix is to create the script file at `~/.hermes/scripts/moss-fix-pipeline.py` with a minimal wrapper that invokes the agent-driven pipeline prompt, or to remove the `script` field from the cron config entirely. Either way, the cron should not have a dangling `script` reference.

20. **Subagent timeout recovery — check and verify partial work before shipping.** When a subagent times out (900s limit, 50 API calls) with the fix branch checked out, it may have left partial changes. Do not discard the branch — the fix work may be mostly correct. Recovery protocol:
    - Run `git diff --stat` to see what was touched. Read the diffs.
    - **Check for regressions first.** Subagents sometimes introduce unrelated changes (regex escape changes, comment removals, whitespace reformatting) in parts of the file they weren't supposed to touch. Verify by restoring the unmodified files and re-applying only the relevant hunks.
    - If the subagent ran tests and reported a failing test as "pre-existing" or "unrelated", verify with a baseline: `git stash && pytest <failing-test> -v -q && git stash pop`. A test that passes on HEAD but fails with the subagent's changes was introduced by those changes.
    - Complete the work: verify the fix logic, fix any regressions, commit, push, and raise the PR manually.
    - The fix may lack the quality gate chain (simplify-swarm, hermaguard) that the full pipeline runs. Run basic gates (S1/S2 secrets/refs, C1 conventional commits, T1 tests, B1 branch current, F1 focused diff) before pushing.
    - If the partial work is too incomplete or introduces too many regressions, discard the branch entirely (`git checkout main && git branch -D <branch>`) and wait for the next pipeline cycle.

16. **Subagent test-result misattribution is dangerous** — When a subagent runs tests and reports a failure as "pre-existing" or "unrelated to this change", **verify the claim before accepting it**. The 2026-06-24 incident (issue #49114 fix): a subagent changed a regex in an unrelated function (`collapseModelFamilies`), ran tests, got 1 failure, and blamed it on a pre-existing regex bug. The failure was actually introduced by the subagent's own accidental double-escape. The correct protocol:
    - Before the subagent modifies any code, capture the test baseline: `git stash && pytest <affected-tests> -v -q && git stash pop`
    - After the subagent reports test results, re-run the failing test(s) against the unmodified file to confirm they pass before the change
    - A single test failure "unrelated to the fix" that's in a function the subagent touched is suspect — treat it as introduced until proven otherwise
    - The pipeline SHOULD NOT accept a subagent's self-attestation of "pre-existing failure" at face value. Re-verify.

21. **contributor-check CI failure — commit email not in upstream AUTHOR_MAP.** The upstream `contributor-check / check-attribution` CI job verifies every commit author email against `scripts/release.py`'s AUTHOR_MAP. An unmapped email fails this check, which also fails the `All required checks pass` gate, blocking the PR. All other checks (tests, lint, e2e) pass green — only the attribution gate fails.

    **Diagnosis:**
    ```bash
    # Check what email was used in the PR commits
    gh pr view <number> --json commits --jq '.commits[] | "\(.authors[0].email)"'

    # Check what emails are in the upstream AUTHOR_MAP
    git show origin/main:scripts/release.py | grep -A50 "AUTHOR_MAP"
    ```
    If the commit email (e.g. `sahil@example.com`) does not appear in the AUTHOR_MAP, this is the failure.

    **Fix — amend + force-push with the registered email:**
    ```bash
    # Set correct email at repo level (overrides global config)
    git config user.email "218421507+Sahil-SS9@users.noreply.github.com"
    git config user.name "Sahil-SS9"

    # Amend and force-push
    git commit --amend --author="Sahil-SS9 <218421507+Sahil-SS9@users.noreply.github.com>" --no-edit
    git push fork HEAD --force-with-lease
    ```
    The amended commit triggers a fresh CI run — the attribution check will pass.

    **Prevention:** The upstream clone at `/home/kensei/repos/hermes-agent-upstream/` MUST have repo-level `user.email` set correctly. Local git config overrides global — do not rely on global config alone. Verify before every pipeline push: `git config user.email` inside the clone must return the noreply address. See `references/gh-cli-patterns.md` → Author Configuration for the full setup and recovery procedure.

    **Why this happens in the moss pipeline:** The pipeline cron runs with `workdir` set to the upstream clone. If that clone was ever configured with a different email (e.g. `sahil@example.com` from initial setup), every commit inherits it regardless of the VPS global git config.
