# Manual Removal Trial Fallback

Use when `tools/repo-hygiene-authority/reversible-trial.mjs` is absent (as of 2026-07-24, it does not exist).

## Process

### 1. Verify no external consumers

Search ALL repos for references to candidate files:

```bash
# Search by filename
grep -rn 'candidate-filename' /home/deploy/lah-stack-repos/ --include="*.js" --include="*.mjs" --include="*.py" --include="*.json" 2>/dev/null | grep -v node_modules | grep -v ".codegraph"

# Search by relative path
grep -rn 'path/to/candidate' /home/deploy/lah-stack-repos/ 2>/dev/null | grep -v ".codegraph"

# Check imports from candidate directory
grep -rn "import.*from.*openclaw" /home/deploy/lah-stack-repos/ --include="*.js" --include="*.mjs" --include="*.py" 2>/dev/null | grep -v ".codegraph" | grep -v "__pycache__"
```

### 2. Capture baseline

```bash
# Record all candidate file hashes
find /path/to/candidate -type f -exec sha256sum {} \; > candidate-hashes.txt

# Record before-state
echo "SHA: $(git rev-parse HEAD)" > worktree-before.txt
git status --short >> worktree-before.txt
```

### 3. Preserve files

```bash
PRESERVE="/tmp/removal-trial-preservation"
mkdir -p "$PRESERVE"
cp -a /path/to/candidate "$PRESERVE/"
```

### 4. Run pre-removal validation

Run the repo's test suite to establish a passing baseline:

```bash
# Python
python3 -m pytest <test-suite> -x -q 2>&1 | tail -5

# Node
node --test --test-concurrency=1 2>&1 | tail -5
```

If tests fail pre-removal, document the pre-existing failure. Do NOT proceed if the failure is in candidate-referencing code.

### 5. Remove files

```bash
git rm -r /path/to/candidate/
```

### 6. Run post-removal validation

Run the EXACT SAME tests as step 4. Verify same pass/fail count.

```bash
# Same command as step 4
```

### 7. Check for unrelated changes

```bash
git diff --name-only  # Should only contain removed files
git status --short    # Should only show D <removed-files> + pre-existing untracked
```

### 8. Commit

```bash
git commit -m "MISSION_TAG — Remove <description> (<N> files, <L> lines)

Evidence: <evidence>
Rollback verified: <yes/no>
"
```

### 9. Verify rollback

EXECUTE rollback, don't just describe it:

```bash
cd /path/to/repo
git reset HEAD <candidate-path>/
git checkout -- <candidate-path>/

# Verify content integrity
sha256sum <restored-file>  # Must match hash from step 2

# Run tests again (should still pass)
python3 -m pytest <test-suite> -x -q

# Verify git status clean
git diff --name-only <candidate-path>  # Should be empty
```

### 10. Restore removal

After successful rollback verification, re-apply the removal:

```bash
git rm -r <candidate-path>/
git commit -m "MISSION_TAG — Remove <description> (with rollback verified)"
```

## Rollback Commands

```bash
# From committed removal
git checkout <previous-sha> -- <candidate-path>/

# From staged but uncommitted removal
git reset HEAD <candidate-path>/
git checkout -- <candidate-path>/

# From preserved files
cp -a /tmp/removal-trial-preservation/<candidate-dir>/ <repo>/<candidate-path>/
```

## Key differences from automated reversible-trial.mjs

| Aspect | Automated (22-phase) | Manual fallback |
|--------|---------------------|-----------------|
| Per-candidate isolation | Automatic sequential | Manual — remove as group or one-by-one |
| Structural validation | Syntactic checks on remaining files | Manual `grep` for remaining references |
| Operator scenario | Custom per-candidate CLI command | Repo-wide test suite |
| Cold-path review | Alternate entrypoints tested | Manual inspection |
| Decision engine | Programmatic `classifyCandidate()` | Human judgment |
| Rollback | Automated restore + hash verify | Manual `git checkout` + `sha256sum` |
