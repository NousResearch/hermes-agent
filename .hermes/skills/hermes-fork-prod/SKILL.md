---
name: hermes-fork-prod
description: Operate the personal Hermes fork that uses fork/main as an upstream mirror and fork/prod as the canonical deploy branch. Covers install, update, maintainer sync flow, and Telegram topic scoping.
version: 0.2.0
---

# Hermes fork prod

Use this skill when working with the personal Hermes fork hosted at `Git-on-my-level/hermes-agent`.

## Branch model

The fork uses this structure:

- `origin` remote → upstream Hermes: `NousResearch/hermes-agent`
- `fork` remote → personal fork: `Git-on-my-level/hermes-agent`
- `fork/main` → clean mirror of upstream main (+ rescued prod-only patches)
- `fork/prod` → integration branch = upstream + all custom patches (canonical deploy branch)
- local `prod` branch → should track `fork/prod`

**Key invariant:** `fork/prod` is always a **superset** of `fork/main`. Anything merged to prod survives syncs.

## What deploy agents should run

Deploy agents should run from `prod` and update with:

```bash
hermes update
```

On the `prod` branch, Hermes is patched so `hermes update` pulls from `fork/prod` instead of `origin/main`.

## Fresh install

Use the shared fresh-install script from the global memory repo:

```bash
~/hermes-global-memory/scripts/install-hermes-fork-prod.sh
```

What it does:
- clones `Git-on-my-level/hermes-agent`
- checks out branch `prod`
- renames remotes so:
  - `fork` points to the personal fork
  - `origin` points to upstream
- creates a venv
- installs Hermes editable
- symlinks `~/.local/bin/hermes`

## Converting an existing clone to prod

```bash
cd ~/.hermes/hermes-agent
git fetch origin --prune
git fetch fork --prune
git checkout prod || git checkout -b prod --track fork/prod
git reset --hard fork/prod
./venv/bin/python -m pip install -e .
```

## Maintainer update process

The prod branch is an **integration branch**: it always contains upstream main **plus** any custom patches you've merged directly to prod.

```
origin/main  ──rebase──→  fork/main  ──merge──→  fork/prod
                          (upstream mirror)     (upstream + our patches)
```

**Key invariant:** `fork/prod` is always a superset of `fork/main`. Commits merged directly to `prod` (via PR or GitHub merge button) are never lost — they're rescued onto `main` first, then flow into `prod` naturally.

### Prerequisites

- Clean working tree (script will refuse to run otherwise)
- GitHub branch rule for `prod` must allow push + force-push (for recovery scenarios)
- Local branches `main` and `prod` should exist and track their respective `fork/*` remotes

### Regular sync

Run the update script:

```bash
cd ~/.hermes/hermes-agent
./scripts/update-prod-branch.sh          # live run
DRY_RUN=1 ./scripts/update-prod-branch.sh  # preview what would happen
```

That script does:

1. **Fetch** `origin` and `fork` remotes (with prune)
2. **Check for upstream changes** — if none but prod has custom commits, skip to step 4
3. **Update `main`** — rebase local `main` onto `origin/main`, push `fork/main`
4. **Rescue prod-only commits** — find any commits on `fork/prod` not on `main`, cherry-pick them onto `main`, push updated `fork/main`
5. **Integrate into `prod`** — merge `main` into `prod` (fast-forward if possible, merge commit if histories diverged)
6. **Push** `fork/prod`

This ensures:
- Custom PRs merged to `prod` survive syncs (they get rescued to `main`, then merged forward)
- `main` stays a clean upstream mirror (all prod-only patches are also on `main`)
- No data loss — nothing is ever force-pushed away without being integrated first

### After resolving cherry-pick / merge conflicts

If step 4 (cherry-pick) or step 5 (merge) hits conflicts:

1. Fix the conflicting files
2. `git add <resolved-files>` → `git cherry-pick --continue` (or `git merge --continue`)
3. Repeat until all patches are applied
4. Re-run the script — it'll pick up where it left off

If conflicts are too numerous to resolve manually, use the clean reset approach below.

### Large syncs (100+ commits behind): clean reset + re-apply

When upstream has drifted so much that rebasing produces unmanageable conflicts:

1. **Analyze which prod patches survive** — identify non-merge commits on prod not in upstream:
   ```bash
   git log --oneline prod --not origin/main --no-merges
   ```

2. **Create an isolated worktree**:
   ```bash
   cd ~/.hermes/hermes-agent
   git worktree add -b sync/prod-vN /tmp/hermes-prod-sync origin/main
   ```

3. **Export old prod patches for reference**:
   ```bash
   mkdir /tmp/hermes-prod-sync/patches
   for hash in <still-needed-commit-hashes>; do
     git format-patch -1 "$hash" -o /tmp/hermes-prod-sync/patches/
   done
   ```

4. **Re-apply patches adapted to current code** — do NOT `git am` literally. Read each patch for INTENT and implement it in the current code structure. Use subagents (Codex preferred), batched by dependency group (infrastructure first, then gateway, then CLI, then trivial).

5. **Validate syntax after every subagent batch** (see `***` corruption pitfall below).

6. **Run tests**:
   ```bash
   python3 -m pytest tests/... -x -q -o "addopts="
   ```

7. **Push**:
   ```bash
   cd ~/.hermes/hermes-agent
   git checkout prod
   git reset --hard /tmp/hermes-prod-sync  # or git fetch fork && git reset --hard fork/sync/prod-vN
   git push fork prod --force-with-lease
   git worktree remove /tmp/hermes-prod-sync
   ```

### Why this over the old rebase-on-prod approach?

The previous workflow (`rebase prod onto main` then `force-push prod`) had a critical flaw: **any commit merged directly to prod was silently clobbered** on every sync. The force-push overwrote `fork/prod` with the rebased content, losing whatever was only on prod.

The new approach inverts the direction:
- **Old:** `main` → overwrite → `prod` (destructive)
- **New:** rescue `prod`-only commits → `main` → merge → `prod` (additive)

Benefits:
- PRs targeting `prod` are never lost
- `main` still stays clean (it's the upstream mirror + rescued patches)
- Normal case is fast-forward (no force-push needed day-to-day)
- Only recovery scenarios need `--force-with-lease`

## Telegram topic scoping

This fork supports inbound Telegram topic allowlists on the prod line.

It also supports session-scoped model settings for Telegram topics: each topic/session can retain its own `model`, `provider`, and `base_url` in the persisted session state. That means different topics can effectively run different models/providers, and `/new` clears conversation state without resetting the active model settings.

Add to config:

```yaml
platforms:
  telegram:
    extra:
      allowed_inbound_targets:
        - "CHAT_ID:THREAD_ID"
```

Behavior:
- the bot only responds to inbound group/topic messages from the listed topic(s)
- DMs remain allowed
- if omitted or empty, the bot responds anywhere it can see

## Multi-bot supergroup pattern

Example:
- m2 bot: `allowed_inbound_targets: ["-1003502111905:2"]`
- m4 bot: `allowed_inbound_targets: ["-1003502111905:7"]`

This allows multiple Hermes bots to share one Telegram supergroup while each bot only responds in its assigned topic.

## Worktree setup

Use worktrees for feature and PR branches, not for the live deploy checkout.

Example:

```bash
git worktree list
# /Users/dazheng/.hermes/hermes-agent                   [prod]
# /Users/dazheng/.hermes/hermes-agent-pr-foo            [feat/foo]
```

Recommended rule:
- `~/.hermes/hermes-agent` should be the canonical launchd-managed deploy checkout
- that checkout should stay on `prod`
- PRs and experiments should use separate worktrees
- avoid a dedicated `prod` worktree for the live service unless you also deliberately bind the venv, editable install, and launchd plist to that same path

Why:
- a split setup like `venv` in `~/.hermes/hermes-agent` but editable install / runtime code in `~/.hermes/hermes-agent-prod-worktree` is brittle
- `hermes update`, editable installs, and launchd path generation derive from the current project root
- if those roots drift apart, you can get path breakage like Python or module resolution failures tied to the worktree path

Important operational finding:
- a split install can happen where the venv lives under `~/.hermes/hermes-agent/venv` but the editable package points at `~/.hermes/hermes-agent-prod-worktree`
- launchd may then run Python from the main repo venv while importing code from the prod worktree
- this usually works until paths drift or the worktree disappears, then you can get brittle failures like Python/path not found during updates or restarts

If the desired state is "launchd-managed prod branch, not a worktree deployment", collapse back to a single checkout:

```bash
cd ~/.hermes/hermes-agent
git worktree remove ~/.hermes/hermes-agent-prod-worktree
git fetch fork --prune
git checkout prod || git checkout -b prod --track fork/prod
git reset --hard fork/prod
./venv/bin/python -m pip install -e '.[all]'
./venv/bin/python -m hermes_cli.main gateway install --force
```

Then verify all of these point to the same base repo path (`~/.hermes/hermes-agent`):
- `hermes --version` → `Project:`
- `pip show hermes-agent` → `Editable project location:`
- launchd plist `WorkingDirectory`
- imported module paths for `hermes_cli.main` / `gateway.platforms.telegram`

## Update behavior

`hermes update` on the prod branch should pull from `fork/prod`, not `origin/main` or `fork/main`.

Important nuance discovered in production:
- switching only the remote to `fork` is not enough
- the update command must also target branch `prod`
- if it switches to `main` before pulling, a deployed prod checkout can silently drift back onto the wrong line

Current detection order for prod installs:
1. explicit `~/.hermes/install-channel` containing `prod`
2. local `prod` branch upstream resolves to `fork/prod`
3. current branch is `prod` and `refs/remotes/fork/prod` exists
4. otherwise fall back to `main` / `origin/main`

So even if a user temporarily lands on a feature branch or detached HEAD, `hermes update` should converge back to the prod line when the install-channel metadata or prod tracking metadata is present.

If you are validating the live prod branch, inspect `hermes_cli/update_channel.py` and `hermes_cli/main.py` and confirm:
- prod resolves to `(channel="prod", remote="fork", branch="prod")`
- non-prod checkouts are switched to `prod` before update

macOS operational finding:
- for launchd-managed prod agents, the dangerous part is often not the git update itself but the post-update gateway restart
- a broken restart path can make `/update` succeed while Telegram stays down
- the reliable macOS behavior is: detached `launchctl kickstart -k gui/$UID/<label>` plus bounded restart healthchecks
- healthchecks should verify more than launchctl command exit status; use launchd state, fresh gateway PID, runtime status, and recent gateway log signals
- if restart verification fails, update output should explicitly say code update succeeded but gateway restart healthcheck failed, and print the failing checks plus recent log excerpt

If you ever see diverged histories:

```
git branch -vv
# * prod  7478c55d [fork/prod: ahead 8, behind 5] ...
```

The update command will auto-reset to match `fork/prod`. Manual fallback:

```bash
cd ~/.hermes/hermes-agent-prod-worktree
git fetch fork
git reset --hard fork/prod
```

## Remote branch deletion safety

Be careful deleting remote branches in this fork because one branch is literally named `prod` while the remote is named `fork`, so the remote-tracking ref appears as `fork/prod`.

That makes commands like:

```bash
git push fork --delete fork/prod
```

ambiguous-looking and easy to misuse. It deletes the remote branch named `prod` on remote `fork` (because `git push <remote> --delete <branch>` interprets the argument as a remote branch name, not a remote-tracking ref).

For destructive cleanup, prefer explicit ref syntax:

```bash
git push fork :refs/heads/stale-branch-name
```

Examples:

```bash
# delete the stale remote branch literally named "fork/prod"
git push fork :refs/heads/fork/prod

# delete the canonical prod branch only if you truly mean it
git push fork :refs/heads/prod
```

After any remote branch deletion, immediately verify with:

```bash
git fetch fork --prune
git branch -r | grep 'fork/.*prod'
```

## Verification checklist

After changing a bot to prod and/or topic scoping:

```bash
cd ~/.hermes/hermes-agent-prod-worktree  # or main repo if no worktree
git branch --show-current
git rev-parse HEAD
git rev-parse fork/prod
```

Expected:
- current branch is `prod`
- HEAD matches `fork/prod` unless there are new unpushed local commits

Then restart the gateway and verify:
- `/status` works in the assigned topic
- plain messages work in the assigned topic
- the bot ignores other topics in the same supergroup

Critical deployment check for topic allowlists:
- Do not assume `allowed_inbound_targets` is active just because it is present in config.
- Verify the live launchd service and editable install are actually pointing at the prod checkout/worktree that contains the allowlist code.
- Check launchd plist, pip show hermes-agent location, and imported gateway platform module paths all agree.

Failure mode to watch for:
- config contains `allowed_inbound_targets`
- a prod worktree exists
- but launchd still runs `~/.hermes/hermes-agent` on `main`
- result: topic allowlist config is silently ignored and `/new` can still respond in other topics

## Operational notes

- `/new` or `/reset` resets conversation state only; it does not claim topic ownership.
- Topic ownership is enforced by `allowed_inbound_targets`.
- If Telegram privacy mode was changed recently, remove and re-add the bot before testing.
- For deployed prod agents, do not manually switch them back to `main` for updates; use `hermes update` from `prod`.

## Pitfall: subagent patch tool `***` corruption

When delegating file edits to subagents (Codex, GLM subagents, etc.), their patch tools can silently replace actual code with `***` in files that contain redacted or placeholder content (API keys, auth types, token values). This produces syntactically invalid Python that passes visual review if you only look at surrounding context.

**Symptoms:**
- `SyntaxError: closing parenthesis ')' does not match opening parenthesis '{'`
- Lines like `api_key_env_vars=*** "ZAI_API_KEY"` or `models = _fetch_github_models(api_key=*** timeout=timeout)`
- Backslash-escaped quotes where none should exist: `(\\\"Z.AI / GLM\\\", ...`

**Prevention:**
- After every subagent batch, run syntax validation on ALL modified `.py` files:
  ```bash
  for f in <modified-files>; do
    python3 -c "import ast; ast.parse(open('$f').read()); print('$f ok')"
  done
  ```
- Search for literal `***` in modified files: `grep -n '=\\*\\*\\*' file.py`
- If a file is badly corrupted, reset it to upstream and re-apply only the intended change:
  ```bash
  git checkout origin/main -- path/to/corrupted_file.py
  # then manually apply just the prod-specific change
  ```

**Root cause:** The subagent's patch tool appears to match and replace content that looks like redacted placeholders, sometimes overwriting actual code that happens to be adjacent. Files with many redacted values (auth.py, models.py, doctor.py, setup.py) are highest risk.
