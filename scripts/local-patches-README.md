# Local Patches Manifest

This directory stores `.patch` files and a `manifest.txt` for tracking
local source-tree modifications to `/usr/local/lib/hermes-agent/` that
survive across `hermes update` runs.

## Why this exists

Hermes Agent's git-based install path (`hermes update` → `git pull`)
will silently overwrite any uncommitted local modifications. If you've
applied a patch to fix a bug, optimize a workflow, or work around an
upstream regression, that work is lost on the next update — and you
won't notice until something breaks.

This directory + the `post-merge` git hook + the `hermes-preupdate.sh`
script solve that:

1. **`manifest.txt`** records the full commit SHA, source branch, and
   description of every local patch. The format is one line per patch:
   ```
   <full-sha>  <branch-name>  [optional description]
   ```

2. **`*.patch`** files are raw `git format-patch` outputs. They're the
   human-readable source of truth (and a fallback for when cherry-pick
   fails).

3. **`post-merge` git hook** (installed at
   `/usr/local/lib/hermes-agent/.git/hooks/post-merge`) runs after every
   successful `git pull` / `git merge`. It checks if a known
   `_build_minimax_oauth_aux_client` canary is present in
   `agent/auxiliary_client.py`; if missing, it cherry-picks the commit
   listed in `manifest.txt` (with `--3way` fallback) and warns the
   operator. If cherry-pick fails, it falls back to `git apply --3way`
   on the `.patch` file. If both fail, it logs and alerts.

4. **`hermes-preupdate.sh`** (in `~/.hermes/bin/`) is a read-only
   preflight check you can run before any `hermes update`. It snapshots
   the current git state + local patches to
   `~/.hermes/state-snapshots/<timestamp>-pre-update/`, verifies the
   hook is installed, verifies the canary is present, and reminds you
   to take a backup if one is older than 14 days.

## When to add a new entry

Every time you make a local change to a tracked file under
`/usr/local/lib/hermes-agent/` that you want to survive `hermes update`:

1. Commit the change on a feature branch (e.g.
   `fix/your-bug-name`) and push it to your fork
2. Add the full SHA + branch name to `manifest.txt`:
   ```
   25222e49068daa243a45850a43e92a6498e6abf5  fix/minimax-oauth-auxiliary-routing  PR #36779
   ```
3. Run `git format-patch -1 <sha>` from the fix branch and drop the
   resulting `.patch` file in this directory
4. Run `~/.hermes/bin/hermes-preupdate.sh --check` to verify the setup

## Recovery if upstream reverts your fix

The post-merge hook handles this automatically (see point 3 above). If
the hook itself fails:

```bash
cd /usr/local/lib/hermes-agent
git cherry-pick --3way <sha-from-manifest>
# or
git apply --3way ~/.hermes/patches/<patch-name>.patch
```

## Files in this directory

- `manifest.txt` — the source of truth for what to re-apply
- `minimax-oauth-auxiliary.patch` — the auxiliary OAuth routing fix
  (PR #36779, branch fix/minimax-oauth-auxiliary-routing)
