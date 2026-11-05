Adds **hermes-watchdog** — cron jobs that fix themselves.

```
hermes plugins install hermes-watchdog
```

### What it does

Hermes already records cron failures and de-duplicates the alert. Nothing acts on them, so a job that fails on a transient timeout keeps failing on it until a human re-runs it by hand. Watchdog closes the loop with bounded verbs and a real success criterion:

**match → act → verify → record**

- **match** — each open incident (state `detected`/`alerted`) is matched to a playbook. Built-ins cover the failure types core classifies: rerun once on `timeout`/`delivery`/`agent`; pause + escalate on `script`; escalate on `auth`/`config`/`unknown`. A user playbook wins at equal specificity.
- **act** — `rerun_job` (fires the job through Hermes's own `cronjob` tool, the same path `hermes cron run` uses), `pause_job`, `check` (a user-authored argv list, no shell, per-step timeout), `escalate`.
- **verify** — a repair counts as `repaired` **only** when Hermes's own record agrees: the incident left the open states, or the job's latest execution finished successfully after the attempt started. Steps that ran without clearing the failure are recorded as `failed`, never as success.
- **record** — every attempt lands in the plugin's own SQLite ledger with its playbook, steps and outcome, and each incident carries a hard daily attempt budget so a repair loop cannot become a second failure mode.

### Surfaces used (all public)

| Surface | Registration |
|---|---|
| Tools | `ctx.register_tool` × 3 — `watchdog_status`, `watchdog_repair` (with `dry_run`), `watchdog_runbooks` |
| Hook | `ctx.register_hook("on_session_end")` — when a *cron* turn fails, repair that job in place instead of waiting for the next tick |
| Slash command | `ctx.register_command("watchdog")` — `/watchdog`, `/watchdog repair` |
| Skill | `ctx.register_skill` — bundled `hermes-watchdog:watchdog` |
| Desktop | `ctx.register` — a `panes` pane (right zone) and a `statusBar.right` chip, data via `ctx.rest` → `/api/plugins/hermes-watchdog/{status,repair,runbooks}` |
| Storage | `plugin_data_dir` / `plugin_db` — ledger + user playbooks only |

No core rebinds, no dashboard tab (`tab.hidden`), no middleware, no env vars, no Python dependencies.

### Rule 13 disclosure

- **Reads** this profile's cron store read-only — `cron.jobs`, `cron.incidents`, `cron.executions` — through late imports, so a moved module degrades to `upstream.ok: false` in the pane and tools rather than crashing.
- **Runs** the failing job through Hermes's own `cronjob` tool, or pauses it. `check` steps execute the argv list a **user-authored** playbook declares (no shell, bounded timeout). Nothing else is executed.
- **Writes** only `plugin-data/hermes-watchdog/` (repair ledger, saved playbooks).
- **Never** edits a job's prompt or schedule, never messages on the user's behalf, never touches another plugin's or core's data.

### Catalog rule 9 note

The cron reads are the only private imports. I opened #149262 asking for a read-only cron-health seam (`ctx.cron.jobs/incidents/latest_execution` or an `on_cron_run_finished` observer hook) and will rewire to it the day it lands — I wrote the request as the consumer of this plugin. Every read is wrapped, and the tests cover the "upstream unavailable" path.

### Validation

- `hermes plugins validate . --install-deps` → **passed** (manifest, loadable entrypoints, capability probe, declared tools/hooks match registrations, no collisions, security scan, no core override, desktop surface inside the SDK; the only warnings are the cron imports above).
- `python -m unittest discover -s tests` → **14/14 pass**, incl. "unverified repair is a failure, not a success", "budget exhausted skips without running", "dry run touches nothing", "upstream failure is reported not hidden".
- `scripts/validate_plugin_catalog.py` → entry valid.
- Smoke-tested against a real profile: 81 jobs, 1 open incident; the dry-run cycle matched it to `builtin-escalate-unknown` and planned exactly one step.

Pinned to `33hodl/hermes-watchdog@d64d01568de779c4320a16e0264788754c235017`. A ~90s demo video follows with the hackathon submission; the Desktop pane is opt-in (Settings → Plugins), as for every unified-package Desktop half.
