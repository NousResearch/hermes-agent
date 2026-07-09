# Safe CDP tab pruning after Kanban completion

**Module:** `tools/cdp_prune.py` · **Wiring:** `hermes_cli/kanban_db.py`
(`_maybe_prune_cdp_after_transition`) · **Tests:** `tests/tools/test_cdp_prune.py`,
`tests/hermes_cli/test_kanban_cdp_prune_hook.py`

Worker Brave/CDP lanes accumulate throwaway `about:blank` and
`chrome://newtab/` tabs as tasks run. This feature closes **only** those
clearly-disposable tabs after a Kanban task completes, and never touches a
real page.

## The cleanup rule (what future agents need to know)

> After a successful `kanban_complete`, disposable tabs (`about:blank`,
> `chrome://newtab/` and fork variants) on the lane's CDP endpoint may be
> auto-closed. **Real URLs are never closed.** At least one page target is
> always preserved per Brave process, so a lane's only context is never
> terminated. Do **not** hand-roll tab-closing in worker prompts — this is
> handled at the lifecycle layer.

### What is "disposable"

| Class       | Example                          | Closed by default? |
|-------------|----------------------------------|--------------------|
| `blank`     | `about:blank`, empty URL         | ✅ yes             |
| `newtab`    | `chrome://newtab/`, `brave://newtab/`, `chrome://new-tab-page/`, `edge://newtab/` | ✅ yes |
| `scratch`   | `data:text/html,...`             | ❌ no — only with `HERMES_CDP_PRUNE_SCRATCH=1` |
| `real`      | any `http(s)://`, `file://`, non-newtab `chrome://settings` etc. | ❌ never |
| `sensitive` | real URL matching a domain-critical hint | ❌ never (defense in depth) |

Non-page CDP targets (`service_worker`, `background_page`, `worker`,
`iframe`, `browser`) are never candidates.

### Safety invariants (enforced in `plan_prune`, covered by tests)

1. **Never close a real or sensitive URL.** The closable set is exactly
   blank + newtab (+ scratch only when explicitly opted in).
2. **Sensitive denylist as defense in depth.** Any URL whose host+path
   matches trading / marketplace / finance-IRS / health / auth-MFA /
   client-vendor hints is force-skipped even if a future rule widened the
   closable set. Over-matching here is safe — it only preserves more.
3. **Preserve ≥1 page per Brave process.** If every page target would be
   closed (e.g. a lane holding only its intentional initial `about:blank`),
   one disposable page is kept (preferring the `about:blank`). Coordinator
   lanes are one Brave process each, so "per process" == "per lane".
4. **Best-effort, non-blocking.** Any failure in the pruner is swallowed and
   never affects the board state transition.

## How it is wired

`kanban_db.complete_task` (and the truly-blocked path of `block_task`) call
`_maybe_prune_cdp_after_transition`, which delegates to
`tools.cdp_prune.prune_after_transition`. This is the universal completion
choke point, so it fires uniformly for CLI (`hermes kanban complete`),
in-process `kanban_complete` tool, and MCP completions.

> **Why not a plugin hook?** The `kanban_task_completed` plugin hook exists,
> but standalone plugins are opt-in via `plugins.enabled` (currently empty)
> and the `hermes kanban` CLI path skips plugin discovery, so a plugin would
> not fire for CLI completions without a config change. The direct
> dispatcher-adjacent wiring is robust across every completion path.

## Configuration (all off / dry-run by default)

| Env var | Default | Meaning |
|---------|---------|---------|
| `HERMES_CDP_PRUNE_ENABLED`  | `false` | Master switch for the runtime trigger. **Inert until set.** |
| `HERMES_CDP_PRUNE_DRY_RUN`  | `true`  | When enabled, still only log sanitized counts — close nothing. Set `0`/`false` to actually close. |
| `HERMES_CDP_PRUNE_SCRATCH`  | `false` | Also close synthetic `data:text/html` scratch tabs (completion only). |
| `HERMES_CDP_PRUNE_ON_BLOCK` | `false` | Also prune on a true block. Always blank/newtab only (never scratch). |
| `HERMES_CDP_PRUNE_ENDPOINT` | —       | Comma-separated explicit endpoints. Falls back to the worker's connected browser (`BROWSER_CDP_URL` / `browser.cdp_url`). |
| `HERMES_CDP_PRUNE_TIMEOUT`  | `5`     | Per-HTTP-call timeout (seconds, clamped 1–30). |

Because the default is **disabled + dry-run**, merging this code changes no
runtime behavior. Activating live pruning is a deliberate two-step opt-in
(`HERMES_CDP_PRUNE_ENABLED=1` and `HERMES_CDP_PRUNE_DRY_RUN=0`).

### Activation note (gateway restart)

CLI completions (`hermes kanban complete`) read env per-invocation, so no
restart is needed for that path. **In-process / gateway-embedded workers**
inherit env at process start — activating for those requires setting the env
in the gateway's service environment and **restarting the gateway**. Per the
Hermes ops rules, do not restart the gateway automatically; open a BLOCKED
R3 gate with the exact restart ask.

## Manual dry-run / audit (closes nothing)

```bash
# Audit the worker's own lane (uses browser.cdp_url):
python -m tools.cdp_prune

# Audit a specific lane; add --no-dry-run to actually close, --scratch for data: tabs:
python -m tools.cdp_prune --endpoint http://127.0.0.1:18838
```

Output is sanitized — lane `host:port`, per-class counts, and close/skip
totals only. Never URLs, titles, cookies, localStorage, tokens, page DOM,
or screenshots.

Example (real coordinator-lane audit — all dry-run, nothing closed):

```
lane=127.0.0.1:18832 would_close=4 counts=class:blank=4 class:newtab=1 ... preserved_last_page=True
lane=127.0.0.1:18838 would_close=2 counts=class:blank=2 class:real=4  ... preserved_last_page=False
lane=127.0.0.1:18833 would_close=0 counts=class:sensitive=1           ... preserved_last_page=False
```

`preserved_last_page=True` on 18832 shows the safety net firing: that lane
held only disposable tabs, so one was kept to preserve the lane context.
