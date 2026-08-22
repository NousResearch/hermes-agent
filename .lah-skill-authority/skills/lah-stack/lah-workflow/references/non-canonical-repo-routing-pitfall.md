# Non-canonical Repo Routing Pitfall

## Symptom

The `lah-repo-router` returns `RESOLVED` with `HIGH` confidence, but the resolved
repository is WRONG. The agent proceeds to read source files that don't exist,
or commits changes to the wrong repo.

## Root Cause

The router only knows about repositories listed in `repo_mappings.json`. If
a mission operates in a checkout (or worktree) that is NOT in the canonical
mapping, the router will resolve by **prefix match** to whatever canonical
repo has the closest alias.

Example:

```
Mission: CLOE_LOT_E_...
Router:  CLOE_ prefix → "cloe" in canonicalMissionRepoMap → openclaw-runtime
Actual:  /home/deploy/lah-stack-repos/cloe-diagnostic-orchestrator (not in mapping)
```

The router doesn't know about `cloe-diagnostic-orchestrator` because it's
a non-canonical repo with its own worktrees and no remote configured.

## Detection

After receiving the router receipt, ALWAYS cross-check against the mission's
explicit metadata:

```bash
# Does the mission specify a working directory?
test -d "$MISSION_DIR" || echo "NOT FOUND"

# Does the mission specify a branch or HEAD SHA?
cd "$MISSION_DIR" && git branch --show-current
cd "$MISSION_DIR" && git rev-parse --short HEAD

# Does the resolved repo match?
cd "$ROUTER_RESOLVED_PATH" && git rev-parse --short HEAD
```

## Fix

Apply `MANUAL_OVERRIDE` when:

1. The mission's explicit metadata (branch, HEAD SHA, worktree path) uniquely
   identifies a checkout
2. That checkout differs from the router's resolved path
3. No other checkout shares the same metadata

```
decision: RESOLVED
decision_source: MANUAL_OVERRIDE (router returned <canonical_repo>;
    mission metadata matched <actual_repo> exclusively)
```

## Prevention

Add the repo to `repo_mappings.json` if it will receive recurring missions.
For one-off missions, the manual override is sufficient.
