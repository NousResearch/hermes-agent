# Genie — Self-Update Procedure

Standard GitHub tarball update via gh CLI. Runs silently.

1. Read local version from frontmatter `metadata.version`
2. Fetch remote version via gh CLI (or raw GitHub URL)
3. If versions match → **also compare script hashes** (see below)
4. Download tarball, extract, copy files
5. On failure → retry once, then report error
6. Output: `I updated Genie from version {old} to {new}`

## GitHub branch name

The repo's default branch is **`main`**, not `master`. Always use `main` in raw GitHub URLs:

```
https://raw.githubusercontent.com/indigokarasu/genie/main/scripts/genie.py
https://raw.githubusercontent.com/indigokarasu/genie/main/SKILL.md
```

Using `master` will return 404 or stale content.

## Script hash comparison (step 3 detail)

Version numbers can stay the same while the underlying script receives patches.
After confirming versions match, always compare the remote `genie.py` hash against
all local copies:

```bash
curl -sL "https://raw.githubusercontent.com/indigokarasu/genie/main/scripts/genie.py" | md5sum
md5sum ~/.hermes/scripts/genie.py
md5sum ~/.hermes/skills/ocas-genie/scripts/genie.py
md5sum ~/.hermes/profiles/indigo/skills/ocas-genie/scripts/genie.py
```

If the remote hash differs from any local copy, apply the update to all
local copies regardless of version match. Report what changed (e.g., "script
updated with same version — applied bugfix patches").

**Why:** The version in SKILL.md frontmatter is the *skill* version. The script
is a *component* that can be patched independently. Relying solely on version
comparison misses script-only fixes.

## Local-ahead scenario

If the local version is **newer** than remote (e.g., local 1.3.2 vs remote 1.3.0),
the local has patches not yet pushed to GitHub. Do NOT downgrade. Report:
"Local is ahead of remote — no update needed. Local may contain unpushed patches."
