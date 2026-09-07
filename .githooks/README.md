# .githooks — secret guards

Two independent rules, enforced on commit and push:

1. **A real `.env` never enters the repository.** Only `*.example` / `*.sample` env
   files are allowed in git. — `secret-guard` (filename check)
2. **A fork commit never adds a high-confidence AWS / GitHub / Slack credential in
   file content.** — `content-scan` (content check)

| Hook | When | What it does |
| --- | --- | --- |
| `pre-commit` | `git commit` | `secret-guard --staged`: rejects the commit if a staged path is `.env`, `.env.<anything>`, or `.op.env` (not `*.example`/`*.sample`, not `.envrc`). |
| `pre-push` | `git push` | For every ref: `secret-guard` scans the tip tree + the new commits' filenames; `content-scan --commits` scans the **content** of files the new commits changed. Aborts the push on any hit. |

## `secret-guard` (filenames) — runnable by hand

```sh
sh .githooks/secret-guard --worktree      # what's on disk right now
sh .githooks/secret-guard --staged        # what's staged
sh .githooks/secret-guard --tree HEAD     # what a given commit's tree contains
```

## `content-scan` (content) — runnable by hand

```sh
sh .githooks/content-scan --commits origin/main..HEAD   # the gate: content the new commits changed
sh .githooks/content-scan --tree HEAD                   # full-tree audit (see note below)
sh .githooks/content-scan --worktree                    # working-tree audit
```

Three vendor families, each a high-confidence shape: AWS access key ids
(`AKIA…`/`ASIA…`, the AWS-doc `…EXAMPLE` placeholders excluded), GitHub tokens
(`ghp_`/`gho_`/`ghu_`/`ghs_`/`ghr_` + 36, `github_pat_` + 82; a ≥20-char single-char
run excluded), and structured Slack tokens (`xox[bpars]-<digits>-<digits>-<secret>`,
`xapp-1-…`, `xoxe.xox[bp]-…`).

**The gate scans only what the fork adds.** `--commits <range>` walks each commit in
the range and scans only the files that commit added or modified — so a key added in
one fork commit is caught even if a later commit renames or deletes the file, while
upstream's own credential-shaped test fixtures (which the fork never touches) are not
re-flagged on every push. `--tree` / `--worktree` are *full* scans for manual audits
and will surface those upstream fixtures — that is expected, not a finding.

### Allowlisting a confirmed fake / fixture

Put an inline marker on the **same line** as the value:

```py
FIXTURE_KEY = "AKIA................"   # nf-scan: allow  synthetic value used by tests
```

Whole-file and path exclusions are intentionally unsupported — the allowlist stays
small and reviewable, one line at a time.

## CI

`.github/workflows/nf-secret-scan.yml` runs `content-scan --commits` on every push
and PR (plus the scanner's own self-tests), so a push is checked even when local
hooks are not installed on the machine it came from.

## Activate local hooks (once per clone)

```sh
sh .githooks/install       # sets core.hooksPath = .githooks, chmods the hooks
```

Verify: `git config core.hooksPath` → `.githooks`.
Self-test: `sh .githooks/secret-guard --worktree && sh .githooks/tests/run.sh`.

## Emergency bypass

`git commit --no-verify` / `git push --no-verify` skip the hooks. That defeats the
rules — only for a deliberate, reviewed exception. `.gitignore` is still the backstop
for `.env` files.

## Layers (defense in depth)

1. `.gitignore` — `.env`, `.env.*`, `.op.env` ignored; `!*.example` / `!*.sample`.
2. `pre-commit` — blocks staging a secret file.
3. `pre-push` — blocks pushing a secret file (`secret-guard`) or a credential in new
   commit content (`content-scan`), whatever commit introduced it.
4. `nf-secret-scan` CI — the same `content-scan`, independent of local hook install.
5. `scripts/redact_handoff.py` — the handoff bundle (`D:\logs.zip`) is redacted with
   the agent's full production vocabulary before it is zipped; the zip is not created
   if a likely secret survives.

History was audited on 2026-09-06: `.env` has never been committed on any ref
(`AUDIT-2026-09-06-001`, `ERR-2026-09-06-001`).
