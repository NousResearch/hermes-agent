# .githooks — secret guard

**One rule: a real `.env` never enters the repository.** Only `*.example` /
`*.sample` env files are allowed in git.

Two gates enforce it:

| Hook | When | What it does |
| --- | --- | --- |
| `pre-commit` | `git commit` | Rejects the commit if a staged path is `.env`, `.env.<anything>`, or `.op.env` (not `*.example`/`*.sample`, not `.envrc`). |
| `pre-push` | `git push` | For every ref being pushed, scans the tip's **whole tree** and the newly added commits; aborts the push on any forbidden file. This is the "always check before pushing" gate. |

Both call `secret-guard`, which is also runnable by hand:

```sh
sh .githooks/secret-guard --worktree      # what's on disk right now
sh .githooks/secret-guard --staged        # what's staged
sh .githooks/secret-guard --tree HEAD     # what a given commit's tree contains
```

## Activate (once per clone)

Git does not run tracked hooks automatically. Enable them:

```sh
sh .githooks/install       # sets core.hooksPath = .githooks
```

Verify: `git config core.hooksPath` → `.githooks`.

## Emergency bypass

`git commit --no-verify` / `git push --no-verify` skip the hooks. That defeats the
rule — only for a deliberate, reviewed exception. `.gitignore` is still the backstop:
`.env` and `.env.*` are ignored (see the secrets block in `.gitignore`), so a plain
`git add .` will not pick them up even with hooks off.

## Layers (defense in depth)

1. `.gitignore` — `.env`, `.env.*`, `.op.env` ignored; `!*.example` / `!*.sample`.
2. `pre-commit` — blocks staging a secret file.
3. `pre-push` — blocks pushing one, whatever commit introduced it.

History was audited on 2026-09-06: `.env` has never been committed on any ref
(`AUDIT-2026-09-06-001`, `ERR-2026-09-06-001`).
