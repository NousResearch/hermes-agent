# Contributor email → GitHub login mappings

This directory replaces appending entries to `AUTHOR_MAP` in
`scripts/release.py`. The old dict caused constant merge conflicts when
several salvage PRs landed at once — every PR edited the same lines of the
same file. Here, **each mapping is its own file**, and file additions never
conflict.

## Adding a mapping

One file per commit-author email, under `emails/`:

```bash
python3 scripts/add_contributor.py <email> <github-login>
# or by hand:
echo "<github-login>" > contributors/emails/<email>
```

- File **name** = the exact commit-author email (as shown by `git log --format='%ae'`).
- File **content** = the GitHub login on the first non-comment line.
  Lines starting with `#` are comments (use them for the PR reference).

Example — `contributors/emails/jane.doe@example.com`:

```
janedoe
# PR #12345 salvage (gateway: fix session key routing)
```

## Rules

- Do NOT add new entries to `AUTHOR_MAP` in `scripts/release.py`. That dict
  is frozen legacy data; the release tooling merges it with this directory
  (directory entries win on duplicates).
- GitHub noreply emails (`<id>+<login>@users.noreply.github.com` and
  `<login>@users.noreply.github.com`) auto-resolve — no file needed.
- The `Contributor Attribution Check` CI job fails a PR whose commits carry
  an unmapped email; the failure message prints the exact command to run.
