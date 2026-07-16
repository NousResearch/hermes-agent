# Fork hygiene rules (zealchaiwut/hermes-agent)

This fork tracks `NousResearch/hermes-agent` and must stay conflict-free on
every `git merge upstream/main`. All fork functionality lives in
[`plugins/life_ops/`](plugins/life_ops/) (code), `optional-skills/` (agent
skills, additive directories only), `deploy/` (fork-owned ops dir), and new
files under `tests/`. Read this before touching ANY file that exists
upstream.

## Rules

1. **Never edit a file that exists upstream.** New behavior goes in
   `plugins/life_ops/` (code), `optional-skills/<category>/<name>/`
   (skills), or `deploy/` (ops). When unsure whether upstream owns a path:
   `git fetch upstream && git ls-tree upstream/main -- <path>`.
2. **Discord UI/commands**: extend `LifeOpsDiscordAdapter` in
   `plugins/life_ops/discord_adapter.py` — never touch
   `plugins/platforms/discord/adapter.py`. New slash commands must register
   on the tree BEFORE `super()._register_slash_commands()` so the base
   method's 100-command-cap accounting counts them (Discord error 30032
   kills ALL slash commands when the cap is blown).
3. **Cron jobs**: call the public `cron.jobs.create_job()` from plugin
   code/scripts, or let a skill self-register via the `cronjob` tool (see
   the perf-coach skill) — never add functions to `cron/jobs.py`.
4. **Need a core edit?** That is an upstream-PR-first situation. Do not
   land it locally except as a new entry in the documented-exceptions list
   below, and only when there is genuinely no plugin-side seam.
5. **Documented exceptions** — the entire permanent diff vs upstream; keep
   this list current and shrinking:
   - `cron/jobs.py` — per-job `tz` (IANA) support inside
     `compute_next_run()` (~26 lines, backward compatible). No upstream
     hook exists for per-job timezones.
   - `tests/hermes_cli/test_models.py` — determinism fix for
     `test_falls_back_to_static_snapshot_on_fetch_failure` (stubs the
     curated-manifest source).
6. **Before every upstream sync**, verify the diff is only the exceptions:

   ```bash
   git fetch upstream
   git diff upstream/main --stat -- $(git ls-tree -r --name-only upstream/main)
   ```

7. **Fork tests stay under `tests/`** (new files only) so upstream's
   pristine `testpaths = ["tests"]` and CI config pick them up.
   `tests/hermes/test_life_ops_canary.py` guards the private adapter
   internals `plugins/life_ops/discord_adapter.py` imports — if it fails
   after a sync, upstream renamed something; re-point the imports.

## Sync procedure

```bash
git fetch upstream
git checkout -b sync/upstream-$(date +%Y-%m) main
git merge upstream/main        # merge, not rebase — fork history is published
# expected conflicts: at most the two documented-exception files
python -m pytest tests/       # full suite incl. canary
MORNING_BRIEF_DRY_RUN=1 bash deploy/bin/morning-chain.sh --dry-run
```

After the merge, re-verify the Discord override still works: start the
gateway, confirm the `life_ops: registered Discord platform override` log
line, and spot-check `/done` and the bedtime/todo-closure buttons.
