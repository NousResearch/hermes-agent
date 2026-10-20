# Hermes Disk & Backup Hygiene

Use when `~/.hermes` is large, when `hermes doctor` reports a big `state.db` / WAL, or
when asked whether anything needs cleaning up.

## Diagnose before deleting

```bash
du -sh ~/.hermes/* 2>/dev/null | sort -h | tail -20
hermes doctor                      # reads the DB stat, not an estimate
hermes sessions stats              # sessions/messages/DB bytes
df -h /System/Volumes/Data         # macOS: free space lives on the Data volume
```

`du -sh ~/.hermes` on a heavily used profile is dominated by four things: `hermes-agent`
(source + venv + node_modules), `state.db`, redundant recovery copies, and browser
profiles. Sizes of 3-14 GB are normal, not broken.

Before promising any reclaim, compare `pragma freelist_count` with `pragma page_count` (both instant, straight from the header). A near-dense store — freelist under ~1% — has nothing for a VACUUM to return: those bytes are real data plus index, and ~373k messages of ~550 MB of text can sit inside a 1.6 GB file because the FTS trigram index is the bulk. The levers in that shape are `sessions.retention_days` (default 90) and the whole-store maintenance pass — not pruning, and not compaction reclaim.

## state.db: optimize-storage, then prove it

`hermes sessions optimize-storage --yes` migrates the search index to the compact v23
external-content layout and VACUUMs, typically reclaiming ~20-25% on a 300k-message store. Newer builds also expose the pass as `hermes sessions optimize`; `--help` is the arbiter of the current name and of the `--force` text, which warns that rewriting under a live writer can leave every agent refusing turns until all writers stop.

It **refuses while any process holds the DB** (current behaviour; the earlier
"safe against a running gateway" claim is wrong — the gateway holds it, so the command
exits 1 with `Refusing … another process is using …state.db`). It is therefore a
gateway-down operation: stop the gateway, run it, restart.

### Running it when your only shell is inside the gateway

The pass cannot be driven from the session that needs it. `hermes gateway stop` from inside a gateway
session is refused by a guard (SIGTERM would reach the command before it finishes), and `launchctl` is
refused too — `launchctl submit/bootstrap` is rejected inside a supervised gateway whatever the job
label, with an explicit warning not to switch verbs to get around it. Do not bypass; stage the work
instead:

1. Write a maintenance script under `~/.hermes/scripts/` that: takes a single-flight lock
   (`exec 9>~/.hermes/.offline-maintenance.lock; flock -n 9`), logs the before size, stops the
   gateway, waits for `lsof ~/.hermes/state.db` to report no holders (bounded loop, ~120s), runs the
   pass, restarts the gateway through a fallback chain — retry `hermes gateway start`, then
   `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.hermes.gateway.plist`, then
   `launchctl kickstart` — re-reads the size and `hermes doctor`, and writes everything to one stable
   log path.
2. Hand the user the arming step (or run the script from a non-gateway shell). A one-shot LaunchAgent
   at a quiet hour that `launchctl bootout`s itself when it finishes is the shape that cannot repeat
   nightly; arming it is theirs, not the gateway's.
3. Report the outcome from inside the script with `hermes send -t telegram "<text>"`. It reuses the
   platform credentials directly and needs **no running gateway**, so the result still reaches the
   user when the restart is the part that misbehaved. Verify that path once with a labelled test line
   before relying on it.
4. Confirm the pass ran from its own log and the restart from `hermes gateway status` — a restart that
   did not come back is the only failure the user actually notices.

On current builds `hermes sessions optimize` accepts only `--force` (no `--dry-run` preview);
`--help` is the arbiter of that surface.

Do **not** reach for `--force` to shortcut this. Rewriting under a live writer is what
produces the retired `state.db-wal` error that then fails every turn — a few hundred MB
is not worth risking the whole store.

Do **not** try to identify the layout yourself from row counts. `messages_fts` having one row per
message, alongside a populated `messages_fts_trigram`, is normal in **both** the old and the v23
external-content layout — external content still indexes one row per source row, so counts match
either way. A count-based "old layout" verdict is a false positive and has already sent one session
chasing a reclaim that did not exist.

The authoritative answer is the tool's own check: just run `hermes sessions optimize-storage --yes`.
It is cheap and idempotent — on a current store it prints `Search index is already on the compact
layout — nothing to do.` and exits 0. Only that output, or `_db_needs_fts_storage_upgrade()` in
`hermes_cli/update_cmd_maint.py`, tells you a rebuild is actually pending. Expect state.db to be
**flat or a few KB larger** in that case; a genuine rebuild is the only thing that reclaims ~60%.

A useful side effect even when there is nothing to reclaim: the run checkpoints the WAL to 0 bytes.

Never report the reclaim as done without verifying, read-only:

```python
import sqlite3; con = sqlite3.connect('file:'+db+'?mode=ro', uri=True)
con.execute('pragma quick_check').fetchone()            # ('ok',)
# messages == messages_fts (index complete, nothing dropped)
con.execute("select count(*) from messages_fts where messages_fts match 'hermes'")
con.execute("select count(*) from messages_fts_trigram where messages_fts_trigram match 'x'")
```

A WAL pinned at ~64 MB is its high-water mark, not a leak: the gateway holds the file,
so the checkpoint needs `hermes doctor --fix` run **from a non-gateway session**.

## Redundant recovery copies are the real disk hog

Each update can leave ~1.6 GB copies, and they stack up silently:

- `state.db.pre-update-emergency-<ISO>.bak` (one per update)
- `backups/pre-update-<stamp>.zip` (full archive)
- `state-snapshots/<stamp>-pre-update/state.db`

Keep the newest `.bak` as the safety net and purge older copies **only after the live DB
passes `quick_check`**.

Validate a copy before deleting it: a `.bak` carries no `-wal`/`-shm` sidecars, so a `mode=ro`
connection fails with `unable to open database file` and a perfectly good backup looks corrupt.
Connect with `?immutable=1` instead, run `pragma quick_check`, and count `sessions` to confirm it
holds the history you think it does (`file <bak>` is enough when you only need the header). A copy
that passes is a genuine restore point — deleting it is the user's decision, never housekeeping. When clearing a state snapshot, delete just its `state.db` —
`.env`, `auth.json` and `config.yaml` there are kilobytes and are the only rollback for
auth/config.

## Traps

- A 0-byte `*.lock` whose mtime is months old is **not** a blocker. They are flock files:
the lock is held by an open fd and released when the process exits, so the file lingers
harmlessly. Never "fix" maintenance by deleting a lock file.
- QA screenshots and snapshot text files get dropped straight into `~/.hermes/`; sweep them
by explicit path. `SOUL.md` is a real config file — a `rm ~/.hermes/*.md` glob destroys it.
- Untracked experiment dirs inside `~/.hermes/hermes-agent/` dirty `git status`; updates
run with `updates.non_interactive_local_changes: stash`, so removing them keeps updates
frictionless.
- `~/.hermes/hermes-setup` (macOS, ~11 MB) is the Tauri update-orchestrator binary, not
clutter.
- Doctor's "`state.db` is large — enable `sessions.auto_prune`" fires even when
`sessions.auto_prune` is already `true`; the actionable fix is optimize-storage.
- Check for pending updates with `hermes update --check` — currency is per release *channel*, so a
`[behind N]` count against `origin/main` is unreleased work, not a pending update (trap 10 in
`references/setup-review.md`). Updating restarts the gateway,
so never launch it from a gateway session — hand that step to the user.

## Sorting doctor's issue list into actionable vs noise

Doctor's closing issue list mixes real local problems with upstream facts. Classify before acting:

- **`state.db` is large** — the suggested fix (`sessions.auto_prune`) is usually already `true`;
the real fix is optimize-storage (gateway-down, above).
- **`<workspace> has N npm vulnerabilities`** — not locally fixable. Doctor says so itself
(`fix is an upstream lockfile bump — a local manual fix does not persist`), and `npm audit fix`
fails with `Cannot read properties of null (reading 'edgesOut')` (the arborist bug) because the
workspace has **no `package-lock.json`** — the repo uses a root `pm/lock.json` instead. Confirm
before spending time: `git diff --stat HEAD origin/main -- '*package-lock.json' '*pm/lock.json'`
returning empty proves updating will not clear them either. Report as upstream, do not churn
vendored deps in the installed checkout.
- **`Check <KEY> in .env`** — verify the credential yourself before believing it. A Gemini key
must start `AIza`; an `AQ.Ab…` value is an OAuth token and returns HTTP 401. Disabling a dead
key is a real fix and the issue list drops (it can clear a neighbouring Alibaba-CN warning in
the same pass). Comment the line out rather than deleting it so the value is recoverable.

When disabling a credential, edit `.env` from a script that prints only line numbers and
lengths — never echo the file, and never paste the value into a chat or a patch.
