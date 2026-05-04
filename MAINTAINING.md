# Maintaining hermes-agent-cn

This fork tracks `NousResearch/hermes-agent` upstream `main`. We carry a small set of patches (see [`FORK_NOTES.md`](./FORK_NOTES.md)) that let the [Hermes v2 Web UI](https://github.com/Eynzof/hermes/tree/dev/v2/apps/web) and Chinese provider env vars work without runtime patching.

## Repo layout

```
origin    https://github.com/Eynzof/hermes-agent-cn.git    (fetch + push)
upstream  https://github.com/NousResearch/hermes-agent.git (fetch only)  ← push disabled
```

Push to `upstream` is intentionally blocked (`git remote set-url --push upstream no_push`). Don't try to push our patches there — they're meant for our fork. Upstream PRs are sent through normal `gh pr create --repo NousResearch/hermes-agent` flow.

## Routine: weekly upstream sync

`upstream-watch.yml` (GitHub Action) opens an issue every Monday 09:00 UTC if upstream is ahead. When that fires, run:

```bash
cd ~/Documents/GithubProjects/hermes-agent-cn
./scripts/sync-upstream.sh
```

The script:
1. `git fetch upstream`
2. Reports how far ahead `upstream/main` is
3. If ahead: attempts `git merge upstream/main` and bails on conflict

After a clean merge:
1. Run a smoke test (see "Smoke test" below)
2. `git push origin main`
3. Close the upstream-watch issue

## Smoke test (after every sync)

```bash
# 1. Install fork into a clean venv
python -m venv /tmp/hermes-cn-test
source /tmp/hermes-cn-test/bin/activate
pip install -e .

# 2. Start dashboard
hermes dashboard --no-open &
sleep 3

# 3. Verify our 7 patches still carry their endpoints
TOKEN=$(curl -sS http://127.0.0.1:9119/ | grep -oE '__HERMES_SESSION_TOKEN__="[^"]+"' | sed 's/.*="\(.*\)"/\1/')
HEADER="X-Hermes-Session-Token: $TOKEN"

curl -sS -H "$HEADER" http://127.0.0.1:9119/api/mcp-servers | jq           # P-005
curl -sS -H "$HEADER" http://127.0.0.1:9119/api/profiles/active | jq       # P-008
curl -sS -H "$HEADER" http://127.0.0.1:9119/api/fs/list | jq               # P-004
# /api/upload (P-002) — exercise via v2 web composer if available

# 4. Connect v2 web to this dashboard, verify gateway WS reaches main page
# (P-003 = no _DASHBOARD_EMBEDDED_CHAT_ENABLED gate)

# 5. Cleanup
kill %1
deactivate
rm -rf /tmp/hermes-cn-test
```

If any step fails, **don't push** — investigate which patch broke. Common cases below.

## Conflict scenarios

### Case A: Upstream refactors a function our patch touches

When `git merge upstream/main` reports conflicts in our patch's target file:

```bash
# See which of our [CN-fork] commits conflict
git log --oneline main..upstream/main -- <conflicted-file>
git log --oneline upstream/main..main -- <conflicted-file>

# Resolve each conflict by hand. The principle:
# - Preserve the upstream change
# - Re-apply our patch's intent on top
# Then:
git add <files>
git commit  # message should explain both the upstream change and how our patch rebased onto it
```

If our patch becomes much harder to maintain after an upstream refactor, consider:
- Re-architecting our patch (small redesign keeping the same external behavior)
- Filing an upstream PR that reduces the divergence (e.g. add a hook point our patch can plug into)

### Case B: Upstream silently merged something equivalent to one of our patches

This is **good news** — it means we can drop the patch.

```bash
# Identify which patch is now redundant
# Use FORK_NOTES.md as the menu of patches; spot-check each target file in upstream/main

# Drop the redundant commit
git rebase -i <commit-before-the-redundant-patch>
# In the editor, mark the redundant commit as `drop`

# Update FORK_NOTES.md: move the row into a "Resolved upstream" section + delete the per-patch detail
git add FORK_NOTES.md
git commit --amend --no-edit  # if there's a follow-up commit that needs to amend the rebase
```

This already happened to **P-001** (provider dict-vs-list bug) — upstream fixed it before our fork was created, so P-001 doesn't carry into this fork at all.

### Case C: Upstream renamed a file our patch targets

Worst case. Treat as a forced re-port:

```bash
# 1. Drop the old patch commit (rebase -i)
# 2. Find the new home of the same logic in upstream/main
# 3. Re-create the patch against the new file
# 4. Commit with the same `[CN-fork] P-NNN: ...` message and a body note explaining the file move
```

## Releasing

Currently we don't tag releases — users install via `pip install git+https://...`, which always pulls `main`.

When fork stabilizes (1-2 weeks of continuous upstream sync without surprises):
1. Pick a version tag matching upstream's calver scheme: `v2026.x.y+cn.N` (PEP 440 local version)
2. `git tag -a v<...>` + GitHub Release with release notes summarizing recent upstream changes + any fork-specific patch changes
3. (Future) Publish to PyPI as `hermes-agent-cn`

## Don't do

- ❌ Don't `git push upstream` (push URL is intentionally `no_push`)
- ❌ Don't `git rebase main` onto `upstream/main` — that's destructive history rewrite. Use `git merge upstream/main`.
- ❌ Don't squash the `[CN-fork] P-NNN` commits into a single bundle. Each patch is a separate commit on purpose so we can revert individually if upstream merges one.
- ❌ Don't add new patches without:
  1. A row in [`FORK_NOTES.md`](./FORK_NOTES.md)
  2. A "should we upstream?" note in the patch's section
  3. A smoke-test step in this file (if relevant)
- ❌ Don't modify the upstream README content. Only the fork banner at the top is ours; everything below is verbatim upstream.

## Reference

- Upstream: https://github.com/NousResearch/hermes-agent
- v2 Web UI consumer: https://github.com/Eynzof/hermes/tree/dev/v2/apps/web
- v2 docs that drive this fork's existence: [`v2/docs/web-public-deployment/`](https://github.com/Eynzof/hermes/tree/dev/v2/docs/web-public-deployment)
- Original runtime patch system (legacy, being phased out): [`v2/UPSTREAM_PATCHES.md`](https://github.com/Eynzof/hermes/blob/dev/UPSTREAM_PATCHES.md)
