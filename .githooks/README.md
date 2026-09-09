# .githooks — secret guards

Two independent rules, enforced on commit and push:

1. **A credential-bearing file never enters the repository by its name.** — `secret-guard` (filename check)
2. **A fork commit never adds a high-confidence AWS / GitHub / Slack credential in
   file content.** — `content-scan` (content check)

| Hook | When | What it does |
| --- | --- | --- |
| `pre-commit` | `git commit` | `secret-guard --staged`: rejects the commit if a staged path matches the blocklist below. |
| `pre-push` | `git push` | For every ref: `secret-guard` scans the tip tree + the new commits' filenames; `content-scan --commits` scans the **content** of files the new commits changed. Aborts the push on any hit. |

## `secret-guard` (filenames) — runnable by hand

```sh
sh .githooks/secret-guard --worktree      # what's on disk right now
sh .githooks/secret-guard --staged        # what's staged
sh .githooks/secret-guard --tree HEAD     # what a given commit's tree contains
```

### Blocked filenames

| Class | Matches | Notes |
| --- | --- | --- |
| `.env` family | `.env`, `.env.<anything>`, `.op.env` | not `*.example` / `*.sample` / `*.template`, not `.envrc` |
| SSH private keys | `id_rsa`, `id_dsa`, `id_ecdsa`, `id_ed25519` | the `.pub` is allowed |
| PKCS#12 / keystores | `*.p12`, `*.pfx`, `*.pkcs12`, `*.jks`, `*.keystore` | |
| Cloud / service creds | `credentials.json`, `service-account.json`, `*-service-account.json`, `service-account-*.json`, `gcloud-service-key*.json` | |
| Auth stores | `.netrc`, `_netrc`, `.pgpass`, `.htpasswd` | |
| PEM / private keys | `*.pem`, `*.key`, `*.keypair`, `*.priv`, `*.pk8` | **except** public bundles (`cacert.pem`, `*-bundle.pem`, `fullchain.pem`, `chain.pem`, `cert.pem`, …) and **except** files under a `test/` / `tests/` / `fixtures/` / `testdata/` / `mocks/` / `spec/` / `e2e/` path (throwaway test certs by convention) |

A confirmed-safe file that still trips a rule: add its exact repo-relative path to
`ALLOWLIST` at the top of [`secret-guard`](secret-guard) (reviewed, line by line —
same discipline as content-scan's inline marker), or `--no-verify` for a one-off.
Self-tests: `sh .githooks/tests/secret-guard.sh`.

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

`.github/workflows/nf-secret-scan.yml` runs, on every push and PR, both hook
self-test suites, `secret-guard --tree`/`--range` (filenames), and
`content-scan --commits` (content) — so a push is checked even when local hooks
are not installed on the machine it came from.

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
   (`.gitignore` is convenience, not a boundary — `git add -f` bypasses it; the
   hooks are the boundary.)
2. `pre-commit` — blocks staging a credential-bearing filename.
3. `pre-push` — blocks pushing a credential-bearing filename (`secret-guard`) or a
   credential in new commit content (`content-scan`), whatever commit introduced it.
4. `nf-secret-scan` CI — the same `secret-guard` + `content-scan`, independent of
   local hook install.
5. `scripts/redact_handoff.py` — the handoff bundle (`D:\logs.zip`) is redacted with
   the agent's full production vocabulary before it is zipped; the zip is not created
   if a likely secret survives.

History was audited on 2026-09-06: `.env` has never been committed on any ref
(`AUDIT-2026-09-06-001`, `ERR-2026-09-06-001`).
