---
name: code-review
description: Review pull requests — diff analysis, impact assessment, backward compat, and test execution
version: 2.6.0
metadata:
  hermes:
    tags: [code-review, github, pull-requests, testing]
    related_skills: [requesting-code-review, grill-me]
---

# Code Review Methodology

Follow every step for every PR review. **Do not skip steps.** If a step is genuinely not applicable (e.g., no database changes → skip Migration Verification), state why in the review preamble.

---

## Mandatory: Review Output Structure

Every PR review MUST contain exactly these sections, in order:

### 1. Review Scope  (What I Checked)
State explicitly what was and wasn't examined. Helps the author trust the review.

```markdown
## Code Review – Scope

**PR:** #[number] — [title]
**Author:** @[username]
**Branch:** [source] → [target]
**Files changed:** [N] files (+[additions] -[deletions])

**Checks performed:**
- [x] Diff Analysis — [N] files read in full, [N] files spot-checked
- [x] Impact Assessment — downstream consumers identified
- [x] Backward Compatibility — checked for breaks
- [x] Dependency Scan — [new/updated] packages reviewed
- [x] Test Execution — [framework] ran: [✅ full pass / ⚠️ partial (N passed, M skipped — needs Docker/Redis/etc.) / ❌ failed / — skipped (no framework)]

**Files read in full:** file1.py, file2.js, ...
**Files spot-checked:** admin/*.py (pattern-match), ...
**Not reviewed:** docs/ folder (no code changes), ...

**Tests executed:** `[test command]` — [N] passed ([N] tests), [N] skipped ([reason]), [N] failed
```

### 2. PR Summary (Restate Your Understanding)
Before any criticism, demonstrate that you understand what the PR does.

```markdown
### PR Summary

This PR adds [feature/fix] by [mechanism]. The key changes are:
- [bullet on what changed and why]
- [bullet on what changed and why]
```

### 3. Findings (per severity, with inline comments)

Group findings by severity. **Every finding MUST also be posted as an inline comment on the diff**, quoting the exact code. The summary section below exists for scannability; the inline comments are the authoritative record.

Use this exact format per finding:

```
**File:Line** | Severity: 🔴 Critical / ⚠️ Warning / 💡 Suggestion / ✅ Nice
**Evidence:** `quoted new (proposed) code` — quote the `+` line from the diff, not the `-` line
**Issue:** One-sentence description of what's wrong.
**Impact:** What happens if this isn't fixed.
**Fix:** Concrete, actionable suggestion.
```

### 4. Final Verdict
End every review with exactly one of:

- **REQUEST_CHANGES** — Any 🔴 Critical or ⚠️ Warning exists.
- **COMMENT** — Only 💡 Suggestions and ✅ Niceties. Nothing blocking.
- **(Never APPROVE — flag for human review.)**

### 5. Self-Improvement Notes (when applicable)

When this review produced learnings (new toolchain discovered, methodology improved, project conventions cached), add a terse summary. Also post an inline comment at the first file where a structural/methodology change was triggered.

```markdown
> 🤖 **Self-improvement from this review:**
> - Cached toolchain for {owner}/{repo} ({runtime} + {package_manager} + {test_framework})
> - Added {new_check} to {methodology_section}
> - File environment PR for {missing_capability}: {PR_url}
```

**When to include:** any review where you discovered or improved something. Skip only if no learnings occurred.

**Structural improvements** (new methodology checks, updated rubrics, new patterns) → also post an inline comment at the first file that triggered the improvement, explaining what changed and why.

---

## Step 0 — Read Context

Before looking at any code, understand what the PR is trying to achieve:

1. **Read the PR title and description** — what does the author claim this does?
2. **Read linked issues/tickets** — what problem is being solved, and what was the agreed approach?
3. **Identify explicit design decisions** — note anything the author calls out as intentional
4. **Note what's declared out-of-scope** — don't flag things the author deliberately excluded

**If the PR description is missing, empty, or a one-liner** → flag as a finding. A PR without a description forces reviewers to reverse-engineer intent from the diff. The author should document what and why.

**Does the diff match the description?** If the author says "adds rate limiting" but the diff has no rate-limiting logic, that's a mismatch to flag.

**Draft / WIP PRs:** If the PR is marked as Draft or Work-in-Progress:
- Default to **COMMENT** verdict unless you find a security vulnerability
- Note in the Scope section that the review is **preliminary**
- Focus on **structural and design feedback** (architecture, approach, API contracts) rather than line-level nitpicks
- The code may change significantly before final review — don't exhaustively flag issues the author likely already knows about
- Do flag anything that would require a **fundamental redesign** if caught late

---

## Step 0.25 — GitHub API Access Verification

**Before doing anything else**, verify you can reach the GitHub API. Without this, Steps 0, 1, 5, 6, and inline comment posting will all fail silently or waste turns.

Run this detection sequence (cheapest first):

```bash
# 1. Check gh CLI auth
gh auth status 2>&1 | head -3

# 2. Check GITHUB_TOKEN env var
echo "${GITHUB_TOKEN:+SET}" || echo "UNSET"

# 3. Check .env file (Hermes config)
grep -q "^GITHUB_TOKEN=" /opt/data/.env 2>/dev/null && echo "IN_DOTENV" || echo "NOT_IN_DOTENV"

# 4. Check git credential store
grep -q "github.com" ~/.git-credentials 2>/dev/null && echo "IN_GIT_CREDS" || echo "NOT_IN_GIT_CREDS"
```

**Decision tree:**
- If `gh auth status` shows authenticated → use `gh` CLI for all API operations. Done.
- If `GITHUB_TOKEN` env var is set and non-empty → use `curl` + token for API calls. Done.
- If token found in `.env` or `~/.git-credentials` → extract it, export as `GITHUB_TOKEN`, use `curl`. Done.
- **If none of the above** → **STOP.** Do not proceed to toolchain discovery or diff analysis. Inform the user:
  > "I need GitHub API access to review this PR. Could you either:
  > 1. Set `GITHUB_TOKEN` in `/opt/data/.env` (uncomment and fill in), or
  > 2. Run `gh auth login` to authenticate the gh CLI?"
  Then wait for the user to provide credentials before continuing.

**Why this matters:** The review methodology depends on GitHub API access for *every* substantive step — fetching PR metadata, reading diffs, posting inline comments, checking merge conflicts, and fetching full file content for structured verification. Discovering the auth blocker after toolchain discovery wastes 3-5 turns. Discovering it after diff analysis wastes even more.

---

## Step 0.5 — Toolchain Discovery & Environment Readiness

Before reviewing code, determine how to build and test it. This prevents wasted turns on trial-and-error tool installs during the review.

### 0.5.1 — Check Project Index Cache

Load `references/project-index.md` and search for the repo being reviewed. If a cached entry exists:

- **Trust the cached toolchain** — use the recorded commands for `build`, `test`, `lint`, etc.
- **Run one verification command** (e.g., `<test_command> --version`) to confirm the toolchain is still valid.
- **On verification failure** — fall through to 0.5.2 (full discovery). Update the cache with the new toolchain.
- **On success** — skip 0.5.2, 0.5.3, and 0.5.4 entirely. Proceed to Step 1.

### 0.5.2 — Toolchain Discovery (First-Time or Cache-Miss)

Run these checks in order — cheapest first, escalate only if needed:

1. **Read project docs (free intelligence):** Check for `AGENTS.md`, `CONTRIBUTING.md`, `README.md`. These often document the exact setup commands. If found, use them directly.

2. **File-sniffing (deterministic, 1-2 tool calls):**
   - `mise.toml` / `.tool-versions` / `.nvmrc` → runtime + version + wrapper
   - `package.json` → Node; `pnpm-lock.yaml` vs `package-lock.json` vs `yarn.lock` → package manager
   - `go.mod` → Go; `Cargo.toml` → Rust; `*.csproj` / `*.sln` → .NET
   - `vitest.config.*` / `jest.config.*` / `pytest.ini` / `Makefile` test target → test framework
   - `Dockerfile` / `docker-compose.yml` → containerized workflow

3. **Interrogate toolchain (last resort):** Run `mise exec -- <tool> --version`, `go env`, etc. Costs turns if tools aren't installed — only use if files don't reveal the answer.

### 0.5.3 — Scan Known Tool Paths (Before Install)

Before attempting any install, check whether the tool binary already exists under persistent storage. Pod restarts preserve `/opt/data/` but don't preserve `$PATH` — binaries can be on disk but unreachable.

1. **Scan known persistent directories** for the tool binary:
   ```
   /opt/data/home/.local/bin/<tool>
   /opt/data/tooling/bin/<tool>
   /opt/data/tooling/<tool>/bin/<tool>
   ```
2. **If found at a known path** → add that directory to `$PATH` for the duration of the review session, verify with `which <tool>` or `<tool> --version`, and mark the tool as available. No install needed.
3. **If not found** → proceed to installation (0.5.4 below).

### 0.5.4 — Tool Installation & Caching

For each required tool discovered:

- **If the tool is NOT installed:** Install it to a persistent path under `/opt/data/tooling/`. For version managers like mise, install the manager first, then use it to install runtimes. Add the manager to PATH in every subsequent command **for this session** via `export`.
- **After installation, verify persistence:** Check whether the install path is in the container's **default** `$PATH` (not the session's augmented PATH). If `echo $PATH` does not include the install directory (`/opt/data/home/.local/bin/`, `/opt/data/tooling/bin/`, etc.), then:
  - **Immediately:** Record the absolute binary path and use it directly (e.g., `/opt/data/home/.local/bin/mise exec -- pnpm test`) so the current review isn't blocked.
  - **After the review (Step 8):** Flag as an environment gap — the tool binary exists on persistent storage but won't be reachable after a pod restart. File an environment PR to add the path to the container's default `$PATH` env var. Include this in the self-improvement notes.
- **If the tool installs in userspace** (mise, language runtimes via mise, package managers via corepack): Cache the paths and commands. No environment PR needed **if** the install path is in the default `$PATH`. Otherwise, follow the persistence verification step above.
- **If the tool requires system-level access** (Docker socket, kernel modules, root permissions, container image change): Do NOT attempt to install. File an environment PR (see Step 8d). In the meantime, note in the Review Scope that the relevant test execution / build step was skipped.

**Cached toolchain entry format** (saved to `references/project-index.md`):
```markdown
## {owner}/{repo}
- runtime: node 24 (via mise)
- package_manager: pnpm
- test: mise exec -- pnpm test
- build: mise exec -- pnpm build:packages
- lint: mise exec -- pnpm lint
- notes: Monorepo — build packages before running tests
- last_seen: {ISO date}
```

### 0.5.5 — Staleness Handling

- Entries with a `last_seen` date get trusted with a single verification command.
- If verification fails, fall back to 0.5.2 and update the entry.
- No TTL-based expiry — project toolchains rarely change.

---

## Step 1 — Diff Analysis

Read every changed line. For PRs with large diffs (>5000 lines or many files):

1. **Skim the file list first** — use `gh pr diff <N> --repo owner/repo --name-only` to see all changed files, grouped by package/module. This reveals the PR's structure and areas of concern.

2. **Fetch the full diff to a file** — `gh pr diff` output can be truncated in terminal (especially >1000 lines). Save it immediately:
   ```bash
   gh pr diff <N> --repo owner/repo > /tmp/pr<N>.diff
   ```
   Then read in chunks with `cat /tmp/pr<N>.diff | head -N | tail -M` to get sections. This avoids wasting turns on re-fetching truncated output.

3. **Read core logic files in full** — schema definitions, domain logic, business rules, and validation code. These are the highest-signal files. → Post inline comments for any finding.
4. **Spot-check admin/worker/boilerplate** — new CRUD routes and worker configs tend to follow patterns. Read one representative file of each pattern, then skim the rest for deviations.
5. **Read migration SQL carefully** — a few lines of SQL can have outsized impact. Compare every `ALTER TABLE` / `ADD COLUMN` against the schema diff for completeness.
6. **Read all test files** — tests reveal the author's understanding of edge cases. If the test covers it, the code likely handles it. If the test misses it, there's a gap.
7. **Cross-reference code changes with test coverage** — for every new or modified code path (function, conditional branch, error handler), verify a corresponding test exists. If not, flag it.
8. **Cross-reference with grep** — when you suspect a column, function, or constant is referenced in code but not defined in the diff, use `search_files` across the entire repo to confirm. Drizzle snapshot JSONs often contain columns that exist from prior migrations — verify by grepping the snapshot, don't assume it's missing.

9. **Verify structured file integrity** — for YAML, JSON, ConfigMap, and other structured files, fetch the full file content from the PR branch via the Git Blob API (see `references/yaml-configmap-verification.md`). The diff shows only changed lines, not the full structural context. Verify indentation consistency, block scalar boundaries (`|`, `>`, `|-`), and key naming conventions against the actual file.

10. **Context verification for files outside the diff** — When you need to read files *not* in the PR (enum definitions, type files, shared utilities) to verify a finding (e.g., "does `UserRole.USER` equal `'user'`?"), use sparse checkout for efficient access:
    ```bash
    git clone --depth 1 --filter=blob:none --sparse https://x-access-token:${GITHUB_TOKEN}@github.com/{owner}/{repo}.git /tmp/review-{repo}
    cd /tmp/review-{repo}
    git sparse-checkout set path/to/enums path/to/types  # only what you need
    git checkout {PR_BRANCH_OR_SHA}
    # read files, verify values, then clean up
    rm -rf /tmp/review-{repo}
    ```
    This is faster than a full clone and more capable than grep or Git Blob API when you need to read entire files for context. Use it when a finding depends on a value defined outside the diff (enum values, shared constants, type constraints).

#### Bot / Automated Dependency PRs (Renovate, Dependabot, etc.)

These PRs are high-frequency and low-complexity — the entire diff is mechanical version bumps plus lockfile regeneration. Follow a **fast, targeted** approach:

- **Signal-to-noise ratio is key.** The lockfile (`pnpm-lock.yaml`, `go.sum`, `Cargo.lock`, etc.) makes up 95%+ of the diff. Do NOT read it line by line. Spot-check the opening chunk to confirm catalogs and specifiers match the package.json changes, then trust the package manager.
- **Read the manifest files in full** (`package.json`, `go.mod`, `Cargo.toml`, `requirements.txt`, etc.) — these are the signal. Group the changes into categories:
  - **Toolchain version bumps** (Go, pnpm, Node, Rust) — these affect CI runners and build environments. Flag if the jump spans multiple minor versions.
  - **Peer dependency minimum tightening** (`>=8` → `>=8.21.0`) — can produce warnings for downstream consumers on older compatible versions. Note in the review but do not block on it (it's standard Renovate behaviour).
  - **Major version bumps** — rare in "non-major" PRs, but if present they need a Warning.
- **Verify lockfiles were updated alongside manifests.** A missing lockfile update breaks `--frozen-lockfile` CI. Quick check: confirm `pnpm-lock.yaml`, `go.sum`, or equivalent changed in the diff file list.
- **Verify catalog/workspace consistency.** In monorepos, if a `pnpm-workspace.yaml` catalog changed, check that all workspace packages that reference `catalog:` resolve correctly.
- **Default verdict: COMMENT** unless a breaking change (toolchain version unavailable in CI, peer dep tightening that excludes the project's own lockfile version, security-critical package with incompatible API change) is found.
- **No test execution required** — these PRs have no code logic changes.
- **Skip the detailed per-finding template** — just state the finding line + severity + short explanation in both the inline comment and the summary.
- **Shadow version references**: After bumping a dependency version, scan configuration files in the repo that may reference the old version. Common culprits: `nx.json` → `installation.version`, `.nvmrc`, `.tool-versions`, CI workflow files with pinned version strings, Dockerfiles with versioned ARG/ENV, Makefiles, and Helm chart `appVersion`. These shadow references create silent version drift between the installed package and what the tooling expects.
  - **Patch-level drift** (e.g. 20.4.6 → 20.4.9) and **pre-existing mismatches the PR didn't introduce** → flag as 💡 Suggestion. Note the drift exists, but don't block on it.
  - **Minor-level drift for a toolchain manager** (e.g. `pnpm`, `nx`, `go`) where the pinned version controls binary resolution or CI runner behavior (e.g. `nx.json` → `installation.version`) → flag as ⚠️ Warning. Even within the same major, the wrong minor version of a build orchestrator can silently change caching, dependency resolution, or task graph behavior. The `nx.json` `installation.version` field is the canonical example: it determines which Nx daemon binary runs, independent of what `pnpm-lock.yaml` installs, so drift between it and `package.json` can put CI in an inconsistent state.
  - **Major-level drift** (wrong major version selected) → flag as ⚠️ Warning.

#### ConfigMap Sync PRs

These PRs sync infrastructure configuration from local files or upstream sources into a ConfigMap via a bot/script. The diff is typically content embedded as YAML block scalars (`|`).

**Targeted review approach:**
- **YAML structural integrity first** — verify the manifest parses and that `apiVersion`, `kind`, `metadata`, `data` are all present at the top level. Validate block scalar markers and content indentation (see `references/yaml-configmap-verification.md`).
- **Content fidelity** — if you have access to the source (local files, upstream state), compare the embedded content against it. Check line counts, key section headers, and version strings to confirm the sync is faithful.
- **Version drift check** — compare version strings in the synced content against the source. Flag mismatches that cause a sync oscillation (source → ConfigMap → detected as divergent → new downgrade PR on the next cycle).
- **Data key naming** — for sync scripts that flatten paths into key names (e.g. `path/to/file.md` → `skill-path_to_file.md`), verify the convention is applied consistently across all keys. Cross-reference each key against the expected source path.
- **Default verdict: COMMENT** — unless the sync is structurally broken or will cause a sync oscillation loop.
- **No test execution required** — these PRs have no code logic.

Check for:
- **Hardcoded test paths** — New `.spec.ts` / `.test.ts` files that use `readFileSync` or `fs.readFileSync` with absolute paths (e.g. `/Users/username/...`). These break on CI and other machines. Flag as 🔴 Critical. Fix: use `new URL("./file.vue", import.meta.url)` or `import.meta.url`-relative resolution.
- **Bugs** — Logic errors, off-by-one, null/undefined handling, race conditions
- **Security** — Injection, auth bypass, secrets in code, SSRF, overly permissive IAM
- **Route restructuring regressions** — When a page moves to a new route (e.g. `/` → `/foo`), verify that access control, auth guards, and middleware from the original page were carried over. Common pattern: the old page had an inline redirect check (e.g. "if no access, navigate to `/doc`"), the new page at the new route omits it, and the login flow that previously gated on the same check is also simplified. This creates a double gap — both the page guard and the login redirect are gone. Always diff the old vs new page for access control logic, and check the login/auth flow for corresponding routing changes.
- **Performance** — N+1 queries, unbounded loops, memory leaks, wasteful resource allocation
- **Style** — Naming conventions, dead code, missing error handling, unnecessary complexity
- **Tests** — Are changes tested? Do tests cover edge cases or just the happy path?
- **Cleanliness** — Debugging artifacts (console.log, print, debugger, pdb), TODO/FIXME/HACK added but not addressed, commented-out code, accidental binary/blob commits

**Every finding → Post inline comment on the PR diff. Quote the code.**

---

## Step 2 — Impact Assessment

A change rarely lives in isolation. For each change, ask:
- What other modules, services, or systems depend on the changed code?
- Could this change break existing behaviour in a non-obvious way?
- Are there consumers of this code that won't be updated?
- Does this change affect shared infrastructure (VPC, IAM roles, DNS, certs)?

**Any concern → Post inline comment on the relevant line.**

---

## Step 3 — Backward Compatibility

If applicable, check that changes don't break existing consumers:
- **Terraform**: Would this destroy or recreate existing resources? Are state migrations needed?
- **Kubernetes**: Would a rolling update cause downtime? Are API version deprecations handled?
- **GitHub Actions**: Would this break existing workflow runs?
- **Docker**: Are environment variables or entrypoint commands changed?
- **Environment**: Are environment configs changed in a way that could break existing deployments?
- Are there any deprecation warnings or removed features?

**Any concern → Post inline comment.**

---

## Step 4 — Test Execution

If required and you have access to the codebase:

### 4.1 — Select tests by diff scope

Do NOT run the full test suite. Instead, determine the smallest set of tests that covers the changed code:

1. **Identify affected packages/modules** from the diff file list. In a monorepo, map each changed file to its workspace package (e.g., `packages/utils/src/string.ts` → `@gambit-group/utils`).
2. **Run tests for only those packages.** In most frameworks (vitest, jest, pytest, go test), you can target a specific package or directory:
   - `npx vitest run packages/utils/` — single package
   - `pnpm --filter @gambit-group/utils test` — pnpm workspace filter
   - `go test ./pkg/foo/...` — Go subpackage
   - `pytest tests/unit/test_foo.py` — single file
3. **If the diff spans packages, run each affected package's tests individually** — not the full workspace. This reduces the load from 38 suite files to 2-3.

### 4.2 — Report results with three tiers

Report test results in the Review Scope section:

- **Full pass:** `[test command]` — [N] passed, [N] failed
- **Partial (infra-limited):** `[test command]` — [N] passed ([N] tests), [N] skipped (needs [Docker/Redis/etc.]). Make it clear that tests DID run.
- **None possible:** `No test framework detected` or `Test execution environment unavailable`

**Never say "Skipped" when tests actually ran** — it erases the evidence. Use "N passed, M skipped" to keep the signal.

### 4.3 — Cross-reference

- Flag any tests that fail or are missing
- Cross-reference test results against new code — if a new feature has no test, flag it
- If the project needs infrastructure you don't have (Docker, cloud services), run whatever subset you CAN and report what was excluded

**Missing tests → Post inline comment on the untested code.**

**Toolchain patterns:** For mise-managed projects, see `references/mise-toolchain-pattern.md` for the full install → trust → exec workflow. For other toolchain managers (nix, asdf, devbox), apply the same detection logic from Step 0.5.2.

---

## Step 5 — Migration Verification

If the PR includes schema, model, or database changes:
- Run the project's migration/db generate command to verify the migration reflects the intended changes
- Check that no expected migration is missing
- **Cross-reference code references vs. migration SQL**: Search the diff for column names used in code (e.g. `onChainTxId`, `broadcastTxHash`) that appear in INSERT/UPDATE/SELECT statements. Cross-reference each against the migration SQL file. If a column is used in code but not added in the migration:
  - **Check the Drizzle snapshot JSON** — the column may already exist from a prior migration. The snapshot (`migrations/meta/*_snapshot.json`) reflects the full schema state and is the source of truth. If the column is in the snapshot, it's expected to already be in the database; the code is consuming a pre-existing column.
  - If the column is absent from BOTH migration SQL and snapshot JSON, flag it as a missing migration.
- Flag if migration files are absent when schema changes are present

**Any discrepancy → Post inline comment on the migration file or the code referencing it.**

---

## Step 6 — Base Branch Conflicts

- Check if the PR has merge conflicts with its base branch via `gh pr view <number> --json mergeable`
- If conflicts exist, flag as blocking and remind the author to resolve before review
- If `gh` CLI is unavailable or unauthenticated, fall back to `curl` + `GITHUB_TOKEN` against the GitHub REST API
- For diff fetching: `gh pr diff <number>` — if unavailable, use `curl` with `Accept: application/vnd.github.v3.diff` (see `references/github-api-inline-comments.md` for the exact pattern and a PR metadata one-liner)
- **Posting inline comments:** If `gh` CLI is unavailable for posting inline comments, use the individual-comment endpoint. See `references/github-api-inline-comments.md` for the full workflow.

**Conflicts → State in Review Scope. Do not attempt a full review of a conflicted PR — stop and request rebase.**

---

## Step 7 — Follow-up Review (Re-review)

When a PR has already been reviewed and receives new commits:

1. **Review only the new diff** — `git diff <previous-review-sha>...HEAD` shows what changed since last review. Do NOT re-read the entire PR unless the author explicitly requests a full re-review.
2. **Check every previous finding was addressed** — don't trust "all comments resolved"; verify each fix in the new diff. If a fix is incomplete, re-flag it.
3. **If findings were partially addressed** — note which were fixed and which remain open. Don't re-flag items that were correctly fixed.
4. **Update the Scope section** — note this is a follow-up review and what changed since last time.
5. **Verdict resets** — new findings can change the verdict even if the previous review was COMMENT.

**Full re-review exception:** If the author or any commenter explicitly says "please do a full re-review" or "I've made significant changes", ignore rule 1 and re-read the entire PR.

---

## Step 8 — Self-Improvement Loop

After posting the review, extract and apply learnings so every review makes you better than the last. This is not optional — it is how you get faster and sharper over time.

### 8a — Project Index Cache

For every project you review for the first time, record its toolchain in `references/project-index.md`. If the entry already exists, update it with any new discoveries.

**Template:**
```markdown
## {owner}/{repo}
- runtime: {node|python|go|dotnet|rust} {version} (via {mise|nvm|asdf|system})
- package_manager: {pnpm|npm|yarn|pip|cargo|go mod}
- test: {exact test command that worked}
- build: {build command if needed before tests}
- lint: {lint command if config detected}
- framework: {vitest|jest|pytest|go test|...}
- last_seen: YYYY-MM-DD
- notes: {quirks, special flags, known gotchas, docker needed}
```

**When to update:** during the review as infrastructure facts are discovered (live patching). This lets the *current* review benefit from cache entries filled earlier in the same review.

### 8b — Review Pattern Learning

If you made a mistake or missed something during this review, extract the lesson. Use these triggers with their confidence thresholds:

| Trigger | Threshold | Action |
|---------|-----------|--------|
| **D — Author instruction** | **1 instance** | Author told you something directly ("run build:packages first"). Apply immediately — highest confidence. Patch the skill or project-index.md right away. |
| **A — Author pushback** | **1 instance** | Author replied to a finding with a factual correction ("lint-staged handles that"). Reflect on why you missed it and add the counter-check. |
| **B — Self-detected inefficiency** | **2 instances** | You did the same sub-optimal thing twice in one review (e.g., 4 grep calls when a snapshot JSON check would suffice). Add a methodology check. |
| **C — Cross-review pattern** | **3+ instances across PRs** | Same finding type appeared in 3+ reviews of this repo. Add as a default check in project-index.md. |

For each trigger that fires:

1. **Identify the pattern** — what did you miss, and what check would have caught it?
2. **Write it generically** — the check should fire for ANY project that matches the pattern, not just the one that triggered it (see 8e).
3. **Patch the methodology** — add the check to the relevant step or Stack-Specific Checks section. Do this after the review is complete (post-mortem), not mid-review, to avoid over-fitting from a single data point.

**Cross-repo applicability:** When writing a methodology improvement, make it generic. A check triggered by a TypeScript monorepo ("monorepos using tsdown need a build step before tests") should fire for *any* monorepo with a build step, not just the one that triggered it. The skill is your global brain — improvements should transfer across projects.

### 8c — Project Convention Learning

Record project-specific norms that affect review quality. These go in `references/project-index.md` under the `notes:` field, or in a separate `references/conventions-{repo-name}.md` for complex projects.

What to capture:
- Code style rules not in linter config ("always use Result<T>", "prefer early return")
- Test patterns ("integration tests need Docker, skip locally", "use `getTestDB` fixture")
- Naming conventions ("services named *-controller", "migrations in `migrations/`")
- Architecture patterns ("database queries live in `queries/`, schemas in `schema/`")

### 8d — Environment Gap Handling

When a project requires tools not available in your environment:

**Userspace-installable tools** (mise, Node via mise, Go via mise, pnpm via corepack, Python via mise):
- Install to `/opt/data/tooling/` so they persist across sessions
- Cache the exact command in `references/project-index.md`
- No PR needed

**System-level dependencies** (Docker daemon, kernel modules, root-required packages, container image changes):
- Delegate a sub-agent to file a PR to `gambit-group/gitops` updating the Overseer configuration at `k8s/apps/hermes/agents/overseer/`
- PR body must include a structured justification:
  ```
  ## Reviewer Environment Request
  **Trigger:** Reviewing {PR URL} — {project}
  **Blocked capability:** {Step N — Test Execution, etc.} — project requires {tool}
  **Impact without this change:** {what's degraded}
  **Proposed fix:** {concrete config change}
  ```
- Ping `@gambit-group/devops` in the PR for human review
- Post a comment on the reviewed PR noting the limitation and linking to the environment PR
- Mention the PR in the review's Self-Improvement Notes so the author knows the gap was identified

**Technical workflow for filing the environment PR:**

1. **Clone the repo** (use GITHUB_TOKEN from env):
   ```bash
   git clone https://x-access-token:${GITHUB_TOKEN}@github.com/gambit-group/gitops.git
   cd gitops
   ```
2. **Create a branch** (name pattern: `overseer/<short-description>`):
   ```bash
   git checkout -b overseer/persist-tooling-path
   ```
3. **Set git identity** (ephemeral environments don't have one):
   ```bash
   git config user.email "overseer@gambit.com.my"
   git config user.name "Overseer"
   ```
4. **Apply the change** — use `patch` tool on the relevant YAML, Dockerfile, or kustomization file
5. **Commit and push**:
   ```bash
   git add -A
   git commit -m "chore(Overseer): <short description>"
   git push origin <branch-name>
   ```
6. **Create PR via API** (gh CLI not available):
   ```bash
   curl -s -X POST \
     -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "chore(overseer): <description>",
       "head": "<branch-name>",
       "base": "main",
       "body": "## Reviewer Environment Request\n\n**Trigger:** ...\n**Proposed fix:** ...\n\n@gambit-group/devops"
     }' \
     "https://api.github.com/repos/gambit-group/gitops/pulls"
   ```

### 8e — Self-Improvement Side Note

Post the improvement visibly for the PR author:

- **Terse summary:** In the review body (Section 5 of the output structure). Always include when learning occurred.
- **Inline comment:** When the improvement is structural (methodology change). Post at the first file where the pattern was noticed. Format: `💡 **Reviewer improvement:** ...`

---

## Stack-Specific Checks

### Application Code
- **API design**: Endpoint naming, request/response contracts, status codes
- **Error handling**: Meaningful error messages, graceful degradation, panic/exception safety
- **Data validation**: Input sanitization, type checks, boundary conditions
- **Auth/Authz**: Authentication checks in place, authorization scoped correctly
- **Dependencies**: New or updated packages — check for known vulnerabilities
- **Dependency lockfiles**: If package.json / Cargo.toml / requirements.txt changed, verify the matching lockfile was also updated (breaks `--frozen-lockfile` CI otherwise)
- **Shadow version references**: When a dependency version is bumped, scan for configuration files that reference the old version in the repo root and CI directories. Examples: `nx.json` → `installation.version`, `.nvmrc`, `.tool-versions`, `.github/workflows/*.yml`, `Dockerfile` versioned ARGs, `Makefile` version variables, `Chart.yaml` appVersion. Config files that determine tool behaviour can silently diverge from the installed package.
- **Logging/Observability**: Useful log messages, metrics, tracing where appropriate
- **Config**: Environment variables wired properly, no hardcoded secrets

### Terraform
- Resource definitions complete? Missing variables or outputs?
- Hardcoded values that should be variables?
- State locking and backend configuration correct?
- Destroy/recreate risk on existing resources?

### Kubernetes Manifests
- Resource limits and requests set?
- Security contexts (non-root, read-only root filesystem)?
- Image tags pinned (no `latest`)?
- Namespace scoping correct?
- **Referenced files exist**: For Kustomize resources, Helm values files, or any manifest that references another file (e.g. `install-v1.X.Y.yaml`), verify the referenced file actually exists in the repo.
- **ConfigMap data-key cross-references**: For ConfigMaps with inline data blocks, check that path-like references in one data key's content (e.g. `see references/foo.md`) resolve to another data key in the same ConfigMap. ConfigMap keys are flat — nested paths like `references/foo.md` won't exist as a file on disk unless the consuming application has special mapping logic. Flag any mismatches.
  - **Path-to-key mapping conventions**: When a sync script flattens file paths into ConfigMap data key names (e.g. `path/to/file.md` → `path_to_file.md` or `skill-path_to_file.md`), verify the mapping is consistent across all keys and that the consuming application can resolve them back to the expected paths. Cross-reference each data key against the source file path to catch mismatches.
- **Version consistency**: When bumping a component version, scan the repo for other files referencing the old version that may need updating.

### GitHub Actions
- Secrets properly referenced (not hardcoded)?
- Trigger scoping — workflow runs on correct events?
- Cache configuration optimal?

### Docker
- Base image pinned to specific digest or version tag?
- Unnecessary layers or tools that increase attack surface?
- Multi-stage builds where appropriate?

### IAM and RBAC
- Permissions scoped to least privilege?
- Resource-level restrictions applied?
- Condition keys used for extra security?

**Every finding → Post inline comment.**

---

## Rules

1. **Inline comments are mandatory for every finding.** The summary in the PR review body is for scannability; the inline comments are the authoritative source. A review with only a summary comment and no inline comments is incomplete.
   - **Posting inline comments programmatically:** When `gh` CLI is unavailable, use the GitHub REST API to post each inline comment individually. See `references/github-api-inline-comments.md` for the exact endpoints, fields, and Python/curl patterns. **Do NOT use the bundled `reviews` API with a `comments` array** — it can silently drop inline comments.

2. **Quote the problematic code** in every finding. Use the exact lines.

3. **State what was reviewed** in the Review Scope preamble. Include what was NOT reviewed (scope limitations).

4. **Restate the PR's purpose** before critiquing. This proves you understood it correctly.

5. **Don't flag style nitpicks** unless they affect readability, correctness, or maintainability.

6. **Don't invent problems.** If the PR looks good, say so. But still write the Review Scope and PR Summary sections to show what was checked.

7. **Never approve.** Flag for human review or request changes.

8. **End every review with:** `REQUEST_CHANGES` / `COMMENT`

9. **Don't opine on base branch targeting.** The reviewer's job is code correctness, not project process. Stating the source/target branches in the Scope section is fine (factual metadata). Flagging merge conflicts is fine (blocking condition). Saying "this should target develop not main" is out of scope — the author knows their branching strategy.

---

## Per-Finding Format (for Summary Section)

```
**File:Line** — Severity: 🔴 Critical / ⚠️ Warning / 💡 Suggestion / ✅ Nice
**Evidence:** `quoted new (proposed) code` — quote the `+` line from the diff, not the `-` line
**Issue:** One-sentence description.
**Impact:** What could go wrong.
**Fix:** Concrete suggestion.
```

When posting the inline comment, prefix with the severity icon so it's scannable in GitHub's diff view:

```
🔴 **Critical:** User input passed directly to SQL query — use parameterized queries.
```
```
⚠️ **Warning:** This error is silently swallowed. At minimum, log it.
```
```
💡 **Suggestion:** This could be simplified with a dict comprehension.
```
```
✅ **Nice:** Good use of context manager — ensures cleanup on exceptions.
```

---

## Verdict Decision

| Verdict | When |
|---------|------|
| **REQUEST_CHANGES** | Any 🔴 Critical or ⚠️ Warning finding exists. **Default for any real issue.** |
| **COMMENT** | Only 💡 Suggestions and ✅ Niceties. Nothing blocking. Use for draft PRs or informational reviews. |
| (Never APPROVE) | Always flag for human review. |

## Severity Rubric

Use to calibrate every finding consistently:

| Severity | Criteria | Examples |
|----------|----------|---------|
| 🔴 **Critical** | Causes data loss, security breach, crash in production, or breaks a core workflow for all users | SQL injection, hardcoded secrets, infinite loop on main path, auth bypass |
| ⚠️ **Warning** | Bug in a non-critical path, missing tests for new code, error handling that could fail silently, will cause issues in edge cases | Missing null check on rare path, unclosed resource, N+1 query on an admin-only page |
| 💡 **Suggestion** | Readability/ maintainability improvements, performance optimizations, refactoring ideas, missing docs, defensive programming | Variable naming, extractable helper function, add logging, future-proofing |
| ✅ **Nice** | Things done well — good patterns, clean naming, smart design decisions | Context manager usage, early return pattern, test coverage quality |

**Tiebreaker:** When unsure between Warning and Suggestion, **pick Suggestion**. Reviewers over-categorize more often than under — and a Suggestion can always be escalated during human review.

---

## ✅ Pre-Submit Checklist

Before posting the review, verify:

- [ ] Review Scope documented (what was checked, what wasn't)
- [ ] PR Summary written (restatement of understanding)
- [ ] Every finding posted as an inline comment on the diff
- [ ] Every inline comment has a severity prefix icon (🔴 ⚠️ 💡 ✅)
- [ ] Summary section lists all findings with Evidence / Issue / Impact / Fix
- [ ] Verdict stated: REQUEST_CHANGES or COMMENT
- [ ] Findings match verdict (any Critical/Warning → REQUEST_CHANGES, none → COMMENT)
- [ ] No findings exist only in the summary without a corresponding inline comment
- [ ] Every inline comment was confirmed posted (check API response for `id` field — the `reviews` API can silently drop bundled comments)
- [ ] (If re-review) Scope notes this is a follow-up; all previous findings checked
- [ ] Self-improvement: project-index.md updated with any new toolchain/convention discoveries
- [ ] Self-improvement: methodology gaps identified in this review have been patched into the skill (if applicable)
- [ ] Self-improvement: "I improved" side note added to review body (if learning occurred)

---

## Preamble Definition

Throughout this skill, **"the preamble"** means both:

- **Review Scope** — what was checked, files read in full vs. spot-checked, tests run, scope limitations
- **PR Summary** — one-paragraph restatement of what the PR does and how

These two sections are mandatory in **every** review. The Findings and Verdict always follow. For short PRs, Scope and Summary may be one line each instead of full headings, but both must be present.

## Short PR Exception

For PRs with ≤3 files and ≤100 lines changed total, the full template can be condensed but must still include the preamble:

```
## Code Review

**Scope:** [N] files reviewed in full. All checks performed: [list].
**Summary:** [one-paragraph restatement]

[Findings with inline comments]

**Verdict:** REQUEST_CHANGES / COMMENT
```

Rules still apply:
- Every finding must still be posted as an inline comment
- Scope and Summary (the preamble) must still be present
- Inline comments are the authoritative record
- Verdict must still end the review
