# Upstream-parity-merge: CI-gate traps that give false local greens

A parity merge (`fork/main` ← `NousResearch/main`, ~1500 upstream commits) is its own
class of PR. The merge resolution is the obvious work; the **CI gates fail for reasons a
naive local check passes**, and each one cost real time in the 2026-06-29 sync (PR #119).
Run these checks locally BEFORE pushing, with the exact CI semantics below.

## TRAP 1 — Live-tree submodule leak gives FALSE PASSES for deleted modules
**Symptom:** local full suite is green, but a CI test slice fails with
`ModuleNotFoundError: No module named 'agent.<x>'`.

**Cause:** the build worktree runs with `PYTHONPATH=$PWD` against the **parent**
`~/.hermes/hermes-agent/venv` (editable-installed against the live tree). The `agent`
package `__init__` correctly resolves to the worktree, but a **submodule the merge
DELETED** falls through to the live tree's still-present copy. A test importing that
deleted module **false-passes locally** and only fails on CI's clean checkout.

**Detect before pushing** — list modules present in live tree but not worktree HEAD, then
find ACTUAL imports (not comments) of them in tests:
```bash
comm -23 <(cd ~/.hermes/hermes-agent && git ls-files 'agent/*.py' 'hermes_cli/*.py' \
  'gateway/*.py' 'tools/*.py' 'plugins/**/*.py' | sort) \
  <(git ls-files 'agent/*.py' 'hermes_cli/*.py' 'gateway/*.py' 'tools/*.py' 'plugins/**/*.py' | sort)
# then grep tests/ for `^\s*(from|import)\s+<module>` — skip comment/docstring mentions
# and skip defensive `try: from new.path import X; except ModuleNotFoundError: from old import X`
```
**Confirm a single one leaks:** `PYTHONPATH=$PWD <venv>/bin/python -c "import agent.<x> as m; print(m.__file__)"`
— if `__file__` points at `~/.hermes/hermes-agent/...` it's leaking from the live tree.

**Fix:** if upstream intentionally deleted the module (check the deleting commit:
`git log --oneline --follow -- <path>` → e.g. `feat(providers): remove ...`), the fork's
test for it is STALE — retire just that test, keep tests for surviving siblings.

## TRAP 2 — gitleaks version + config auto-discovery (secret_scan)
**CI runs a PINNED gitleaks (8.18.4 as of 2026-06), NOT your local version.** Older
rulesets flag fixtures newer ones don't. The CI invocation has **no `--config`** — it
relies on gitleaks auto-discovering `.gitleaks.toml` from the `--source` scratch dir, which
only works because `.gitleaks.toml` is itself in the changed-files set and gets copied in.

**Reproduce CI exactly** (download the CI's version, scan from a scratch dir that includes
`.gitleaks.toml`):
```bash
VER=8.18.4; curl -fsSL "https://github.com/gitleaks/gitleaks/releases/download/v$VER/gitleaks_${VER}_darwin_arm64.tar.gz" -o /tmp/gl.tgz && tar xzf /tmp/gl.tgz -C /tmp gitleaks && mv /tmp/gitleaks /tmp/gl1844
MB=$(git merge-base fork/main HEAD); S=$(mktemp -d)
git diff --name-only --diff-filter=ACMR "$MB" HEAD | while read f; do [ -f "$f" ]||continue; mkdir -p "$S/$(dirname "$f")"; git show "HEAD:$f">"$S/$f" 2>/dev/null; done
cp .gitleaks.toml "$S/"; /tmp/gl1844 detect --source "$S" --no-git --no-banner --redact --exit-code 1
```
**Triage the findings (all were non-secret in #119):** test fixtures, doc placeholders
(`DISCORD_ALLOWED_USERS`, `curl -H "Authorization: Bearer <token>"`), public OAuth
**client IDs** (UUIDs / `*_OAUTH_CLIENT_ID` — sent in browser auth flows, NOT credentials),
and gitleaks matching `agent/redact.py`'s OWN private-key detection regex (the literal
`BEGIN ... PRIVATE KEY` PEM banner that the redactor itself scans for).

**Allowlist surgically — never blanket-allowlist a source file** (that masks a future real
leak). `.gitleaks.toml` `[allowlist]` supports: `stopwords` (substring of the secret),
`paths` (whole-file regex — fine for TEST/DOC fixtures), and `regexTarget="line"` +
`regexes` (line-anchored — use for source files so the rest of the file stays scanned).
For UUID client IDs use a UUID-anchored regex so real (non-UUID) secrets still trip.
**ALWAYS run a negative test** after editing: inject a real high-entropy `AKIA...`/40-char
key into one allowlisted source file and confirm gitleaks still exits 1.

## TRAP 3 — contributor-check email range + the `@users.noreply` skip is NARROW
The merge range `fork/main..HEAD --no-merges` enumerates EVERY upstream author (262 in
#119). Each must be in `scripts/release.py` AUTHOR_MAP or auto-skipped. **The auto-skip is
narrower than it looks** — replicate the CI's EXACT logic, don't approximate:
- `case` skip ONLY: `*teknium* *noreply@github.com* *dependabot* *github-actions* *anthropic.com* *cursor.com*`
- auto-resolve ONLY: `grep -qP '\+.*@users\.noreply\.github\.com'` — i.e. the **`+`-prefixed
  numeric form** (`123+user@users.noreply.github.com`). A **bare `User@users.noreply.github.com`
  (no `+`) is NOT auto-skipped** and DOES need an AUTHOR_MAP entry. (This bit #119: my sim
  blanket-skipped all `@users.noreply.github.com`; CI only skips the `+` form.)
- else must be present as `"email"` (exact case) in `scripts/release.py`.

**Resolve a GitHub username from an email** (most reliable = the commit's author.login):
```bash
SHA=$(git log --author="$EMAIL" --format=%H -1)
gh api "repos/NousResearch/hermes-agent/commits/$SHA" --jq '.author.login'
```
Add `"email": "login",  # PR #N (descriptor) — mapped during <date> upstream parity sync`.

## TRAP 4 — supply-chain scan fires on `setup.py` BY DESIGN (maintainer gate)
The supply-chain-audit scanner flags **any** modification to top-level `setup.py` /
`setup.cfg` / `sitecustomize.py` / `usercustomize.py` / `__init__.pth` as a critical
install-hook finding and **demands maintainer review — there is no allowlist** (it's the
litellm-attack "stop and look" gate). A parity merge inevitably touches `setup.py`. If the
file is byte-identical to upstream (`diff <(git show <target>:setup.py) setup.py`), this is
a **maintainer-judgment gate for Ace**, NOT something to route around in code. Surface it;
he approves or merges `--admin` over that one check once everything else is green.

## TRAP 5 — branch-protection required-context drift blocks the merge (not a CI failure)
Even with every CI job green, `gh pr merge` can fail with
`N of M required status checks are expected` — the merge is blocked by **branch protection
requiring check CONTEXTS that the current workflow no longer emits**. In #119 the fork's
`main` protection still required `test (1)`..`test (6)` while the workflow had been re-sharded
to `Python tests / Run tests slice 1/8`..`8/8`. Those stale contexts never report → forever
"expected" → every PR is unmergeable. Note **`enforce_admins: true` means `--admin` will NOT
bypass missing/"expected" contexts** (it only bypasses *failed* ones).

**Diagnose:** `gh api repos/<owner>/hermes-agent/branches/main/protection/required_status_checks`
→ compare its `contexts`/`checks` against the actual job names in `gh pr checks <N>`.

**Fix (own-fork maintenance):** realign the required contexts to the names CI actually emits,
via the targeted endpoint so you don't disturb `enforce_admins` / conversation-resolution /
`strict`:
```bash
gh api --method PATCH repos/<owner>/hermes-agent/branches/main/protection/required_status_checks \
  --input - <<'JSON'
{"strict": true, "contexts": ["Python tests / Run tests slice 1/8", "...8/8",
 "Python tests / e2e", "Python lints / ruff + ty diff", "sast", "secret_scan"]}
JSON
```
Deliberately leave the `setup.py` supply-chain check OUT of the required set (it's the
advisory maintainer-gate from TRAP 4, not a hard gate). **Merge a parity PR with `--merge`
(a real merge commit), NEVER `--squash`** — squashing collapses the 2-parent topology and
destroys the merge-base the NEXT upstream sync needs.

## General: how to find the REAL CI failure
- `gh pr checks <N> --repo Kyzcreig/hermes-agent` → which checks failed.
- Per-job log (works even mid-run for a completed job):
  `gh api repos/Kyzcreig/hermes-agent/actions/jobs/<JOB_ID>/logs` (the `--redact` on
  secret_scan hides finding detail — reproduce locally for those).
- Many CI gates are `workflow_call` orchestrated by `ci.yml`; some (e.g.
  `skills-index-freshness`) are `if: github.repository == 'NousResearch/...'` and **don't
  run on the fork at all** — don't waste time chasing those.
- semgrep `sast` gate scans `hermes/` which **doesn't exist on the fork** (package is
  root-level) → vacuously passes; harmless.
