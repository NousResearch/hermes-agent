---
name: public-repo-hygiene
description: "Audit full git history for secrets/PII before going public."
version: 1.0.0
author: WolfurX
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Security, Git, Secrets, Privacy, Audit, Pre-publish]
    category: security
    related_skills: [oss-forensics, 1password]
---

# Public Repo Hygiene

Audits a repository's **entire git history** — every blob reachable from any
ref, binaries included, plus commit and tag messages and author/committer
identities — before it becomes public. The working tree is not the repo:
deleted files, old branches, and force-push leftovers still hold blobs that
`ls-files` will never show, and they all ship the moment the repo goes public.

Pure Python stdlib + git. No scanner installs, no network.

## When to Use

- Before the first push of a repo that is or will become public.
- Before `gh repo create --public` or flipping a private repo's visibility.
- Before opening a PR from a fork (your commits ride along).
- User asks for a credentials or PII audit of their repositories.

## Prerequisites

`python3` and `git` on PATH. Optional: `~/.hermes/personal-patterns.txt`
with one regex per line for the user's own identity strings (see
`references/personal-patterns.example.txt`). Offer to create it on first
use — names, personal emails, usernames, phone prefixes — the scanner
always checks the current username and home path even without it.

## How to Run

```bash
# audit one repo (bare, mirror, or normal clone; cwd if no args)
python3 scripts/history_scan.py /path/to/repo

# audit a directory of mirror clones
git clone --mirror git@github.com:user/project.git /tmp/audit/project.git
python3 scripts/history_scan.py /tmp/audit

# with an explicit personal-patterns file
python3 scripts/history_scan.py --personal /path/to/patterns.txt /path/to/repo
```

Exit code is nonzero when anything hit, so it can gate a publish step — but
hits still need judgment (step 3 below); the exit code is a flag, not a verdict.

## Procedure

**1. Identity first — before the first commit.** Check what identity will be
baked into history:

```bash
git config user.email    # global AND any repo-local override
```

If the user wants their personal address out of public history, set the
GitHub noreply address (`ID+username@users.noreply.github.com`) before
committing, and point them at two GitHub settings that enforce it
server-side: **"Keep my email addresses private"** and **"Block command line
pushes that expose my email"**. Identity is baked into every commit at
commit time; fixing it later means rewriting and force-pushing.

**2. Scan the full history** with `scripts/history_scan.py` (see How to Run).
For a repo that already has a remote, prefer scanning a `--mirror` clone: it
holds exactly the refs a `git push` publishes.

**3. Judge the hits.** Placeholders (`${ENV_VAR}`, `sk-YOUR_KEY_HERE`,
documentation samples) are fine — say so and move on. For a real secret:

- **Not yet pushed:** remove it, recommit, rescan.
- **Already pushed anywhere public:** the secret is burned. **Rotate it at
  the provider first**, then rewrite history (`git filter-repo`), then
  force-push. Rewriting without rotating fixes nothing — assume it was
  scraped the moment it was pushed.

**4. Repo-level checks** before publishing:

```bash
git ls-files | grep -iE '\.env|creds|secret|key'   # committed config files
```

CI workflows must reference `${{ secrets.* }}`, never literals. Committed
databases and binaries are covered by the scan (bytes-level regex).

**5. Report** to the user: identities found in history, real hits vs
placeholders, and what was rotated/rewritten. Never print a full secret into
the conversation — the scanner already truncates fragments to 70 chars;
refer to hits by file path and pattern name.

## Pitfalls

- **Platform views lie.** A hosting site's file browser or API shows the
  default branch's tree, not history. Audit all refs (the scanner uses
  `--all`), or leaked blobs survive in old branches.
- **Forks inherit upstream noise.** In a fork, judge hits by whether *your*
  commits introduced them: `git log --all --author=<you> -p`.
- **Rewrite-then-rotate is backwards.** Rotate first; a rewritten secret is
  still burned if it was ever public.
- **The scanner is pattern-based.** High-entropy secrets with no known
  prefix can slip through; the content-email report and identity list are
  there to catch what patterns miss. Treat "0 hits" as necessary, not
  sufficient, for high-stakes repos.

## Verification

- `RESULT: 0 pattern hit(s)` for every repo, or every hit judged and
  reported as a placeholder.
- `git log --all --format='%ae %ce' | sort -u` lists only intended
  identities.
- After any history rewrite: rescan, and confirm the rotated credential's
  old value is dead at the provider.
