# Hermes setup / architecture review

A read-only sweep of an installed Hermes, plus the traps that make a review wrong if you trust the
summary line instead of the artifact behind it.

## 1. The sweep (read-only, batchable)

```bash
hermes doctor              # config version, advisories, toolset deps, store sizes
hermes status              # model/provider/gateway/platforms/jobs (CONFIG view, see trap 7)
hermes cron list           # schedule, last run, dispatch punctuality
hermes curator status      # skill population, stale/archived, unmanaged counts
hermes sessions stats      # sessions, messages, DB size
du -sh ~/.hermes/* | sort -rh | head -12
ls -la ~/.hermes/          # markers, locks, stray backups
tail -4000 ~/.hermes/logs/gateway.error.log | grep -oE "ERROR[^:]*: ?.{0,90}" \
  | sed 's/[0-9]\{2,\}/N/g' | sort | uniq -c | sort -rn | head -12   # recurring signatures
cd ~/.hermes/hermes-agent && git fetch origin main && git rev-list --count HEAD..origin/main
```

Log-signature counting is the highest-yield single step: the doctor only reports static checks, so a
warning repeated hundreds of times per day is invisible to it.

Prove a config change that affects agent construction from the **log's before/after**, not from a
spawned agent: note the warning's timestamp, make the change, then re-read the log and confirm the same
session produced no further occurrence while it kept logging other activity. Pair that with the config
read-back (`hermes config get <key>`) and the exact source branch that emits the warning. A nested
`hermes -z` run from inside a gateway session is not a dependable probe for this — it is not the path
the gateway takes, so build the evidence from the log the gateway itself writes.

## 2. Traps (each one has produced a wrong review)

1. **`context.engine: default` is not a valid engine name.** The built-in is `compressor`; anything
   else is looked up as a plugin engine and, when missing, logs
   `Context engine 'X' not found — falling back to built-in compressor` on every agent build (214
   occurrences in one day here). Fix: `hermes config set context.engine compressor` (or remove the
   key) — the fallback is silent and correct, only the warning is the defect.
2. **The `state.db is large` doctor hint is a static string.** It always suggests
   `sessions.auto_prune`, which is already `true` on a default install. Verify the key before acting;
   the real levers are `sessions.retention_days` (default 90; 180 keeps ~6 months and grows the file
   steadily), `hermes sessions optimize` (FTS5 segment merge + VACUUM, non-destructive), and
   `hermes sessions prune` (deletes). VACUUM only fires when >25% of pages are reclaimable — check
   `pragma freelist_count / pragma page_count` before promising a reclaim. Attribute the file before
   recommending a retention drop: `select role, count(*), round(sum(length(content))/1048576.0) from
   messages group by role` shows whether it is tool output (almost always) versus conversation, and
   bucketing sessions by `started_at` against `strftime('%s','now','-90 days')` with a join to
   `messages` shows what the drop would actually free. `sessions` has **`started_at` (REAL epoch)**, not
   `created_at` — a query on `created_at` fails outright. Inspect the live store read-only with
   `sqlite3 'file:state.db?immutable=1'` while the gateway holds it; under `immutable=1`
   `pragma journal_mode` reports the base file (`delete`) even when `state.db-wal` proves WAL is
   active, so never infer journal mode that way.
   `optimize`/`prune` **refuse while any process holds `state.db`** (the gateway, Desktop, a nested
   CLI), and the refusal is the safety rail, not an obstacle: `--force` is documented as able to leave
   *every* agent refusing turns until all writers stop, so never use it to buy a window. Stop the
   writers from a shell **outside** the gateway process tree — arming a LaunchAgent
   (`launchctl bootstrap`/`submit`) from inside a supervised gateway is refused by design, and the
   guard rejects swapping verbs to get around it, so a scheduled maintenance window is prepared here
   and armed from a separate shell (or handed to the user). Decide which lever *can* reclaim before
   booking downtime: rows (retention/prune), index segments (optimize), and free pages (VACUUM) are
   three different budgets — see the `dbstat` breakdown in the `agent-cli-installation` skill,
   `references/state-disk-hygiene.md`.
3. **`.hermes-update-in-progress.lock`, `.hermes-update-paused-gateways.lock`, `.backup.lock` are
   kernel-lock sidecars, never deleted by design** (`marker_mutex_path`: `<marker>.lock`, deleting it
   would split the flock across two inodes). Their presence is NOT a stale lock. The live markers are
   the extension-less `.hermes-update-in-progress` and the `<stem>.<checkout>.json` pause records —
   check for those, and for marker liveness (PID + timestamp), before reporting a stuck update.
4. **A green doctor line is not a green runtime path.** `web.use_gateway: true` without a Nous
   entitlement still passes the static web checks, then errors at call time
   (`Firecrawl client initialization failed … tool gateway is unavailable`) and quietly serves a
   keyless rescue. Only the log shows this.
5. **`plugins.enabled` entries whose toolset reports "system dependency not met" are dead config**
   (e.g. `homeassistant` with no HASS deps). Enabling a plugin is not installing its dependencies.
6. **`skills.disabled` accumulates names of skills that are no longer installed.** Count the
   intersection before treating the list as a statement about the current skill tree.
7. **`hermes status` prints the config default model, not the live session's model.** A gateway
   `/model` switch writes a *per-session* override and leaves `config.yaml` untouched, so the two
   routinely disagree and only the session store says what is actually running: the non-secret fields
   (`model`, `provider`, `base_url`, `api_mode`) are persisted as `model_override` on the `SessionEntry`
   in `~/.hermes/sessions/sessions.json` (`agent:<profile>:<platform>:<chat_type>:<chat_id>` key) and
   lazily rehydrated on first use after a gateway restart, with credentials re-resolved through normal
   runtime provider resolution. `api_key` is never serialized there. Read that file to answer "what
   model is this chat on", `/new` to clear it, and `hermes config get model` for what new sessions
   inherit. Making a model the *default* is a config write, not a `/model` switch — see
   `references/model-provider-switching.md`.
8. **Credential warnings repeat per model-catalog refresh** and are easy to mistake for one-off
   noise: expired API keys (`auth failed (401)`), expired OAuth refresh tokens
   (`Claude Code OAuth refresh token is terminally invalid`), and providers that reject the token
   type (a classic `ghp_*` `GITHUB_TOKEN` is not usable by the Copilot API). `hermes auth list`
   shows the credential-per-provider view.
   First establish *which store* the warning comes from: `hermes auth status <provider>` can report
   `logged out` while a warning about that same provider keeps firing, because the dead token sits in an
   external tool's own credential store (a Claude Code OAuth login, a `gh` CLI token) rather than in
   Hermes' pool. Remove a pool entry with `hermes auth remove <provider> <id|label>` — it also
   suppresses env re-seeding, while the env var itself stays put — and fix an external login with
   `hermes auth add <provider>`. Re-run `hermes doctor` afterwards: the ⚠ list losing that entry is the
   read-back that the fix landed.
9. **`browser.cdp_url` is a live probe, not a config reading.** Chrome's DevTools HTTP endpoints
   (`/json/version`, `/json/list`) can answer 404 while the WebSocket upgrade on the same port
   returns `HTTP/1.1 101`. Probe the WS path before declaring browser automation impossible; the
   tool itself (`new_tab` + `page_info`) is the cheapest end-to-end check.
10. **"Behind `origin/main`" is not "an update is available".** Currency is measured against the
    release *channel*: `hermes update --check` answers it (`Up to date with the latest release
    (stable vX.Y.Z)`), and on the stable channel a checkout sitting dozens or hundreds of commits
    behind `origin/main` is expected — those are unreleased commits the channel deliberately does not
    take. Read `git rev-list --count HEAD..origin/main` as "how much unreleased main exists", never as
    a pending update, and do not reach for `--channel`/`--branch` to close the gap.
    `hermes update --plan` is read-only and prints the install kind, every running service and the
    exact restart command per profile; `git status --porcelain` shows what an update would have to
    stash.
11. **A repeating `lsp[<lang>] fresh diagnostics timed out` line is a budget problem, not a broken
    server.** Three `lsp` keys set that budget: `wait_timeout` (5s default — short for a large
    TypeScript repo on a busy machine), `warmup_timeout` (0 = reuse `wait_timeout`, but the *first*
    request also pays server spawn + initialize, which on a big project can need a minute), and
    `broken_retry_seconds` (0 = a failed server/root pair is skipped for the whole process lifetime,
    which is how one stall becomes a permanent streak). Raise them with `hermes config set` and read
    the live values back with `hermes lsp status`, which echoes `wait_timeout`. If a workspace really
    is too large to diagnose in budget, `lsp.exclude_roots` switches the layer off for that workspace
    alone instead of silencing every one.
12. **`~/.hermes/config.yaml` cannot be patched by the edit tools** (agent-write-protected). Use
    `hermes config set <key> <value>`. Two consequences: (a) `hermes config set` warns
    `not a recognized config key — it was saved anyway` for keys the schema list lacks but the
    *runtime still reads* (e.g. `web.use_gateway`, the legacy shim that maps to the managed "nous"
    web selection), so treat the warning as advisory and confirm the read path in source; and
    (b) setting a list-valued key writes a string, so edit YAML lists another way.
13. **The foreground `terminal` cap is 600s on current builds** — anything longer belongs in
    `background=true` + `notify_on_complete=true`. Cap figures quoted from older sessions drift;
    read the live tool schema instead of a remembered number.

14. **`display.platforms.<plat>.streaming` can only narrow `streaming.enabled`, never turn it on.**
    `StreamingConfig.enabled_for()` is `globally_enabled and (override is None or bool(override))`
    (`gateway/config.py`), so `streaming.enabled: false` plus
    `display.platforms.telegram.streaming: true` leaves live previews **off** and the per-platform key
    as dead config that reads as "on". Read both keys together and resolve the precedence in source;
    a review that quotes only the platform block reports the opposite of the truth.
15. **A drain timeout above the launchd clamp is capped, not honored.** `gateway.log` warns
    `restart_drain_timeout=180s exceeds the live launchd exit timeout (60s) … signal-driven stops will
    drain at most 50s` — launchd clamps per-user `ExitTimeOut`, so raising the plist does not buy a
    longer drain. Set `agent.restart_drain_timeout` to ~45 to match what actually happens; otherwise
    every restart SIGKILLs in-flight turns 50s after SIGTERM. One warning in the log is the read-back,
    so do not call a restart clean while it is still being emitted.

## 3. Safe-reclaim candidates (verify before deleting)

- `~/.hermes/cache/uv` — package-manager build cache; regenerates.
- `~/.hermes/sessions/request_dump_*.json` — debugging dumps written only when a dump was requested.
- superseded tool versions under `~/.hermes/tools/` — one tree per upgrade, newest wins; confirm the
  live driver path (e.g. `HERMES_CUA_DRIVER_CMD`, an app bundle) does not point into the tree you are
  removing before deleting it.
- `~/.hermes/logs/*` rotated generations once the recurring signatures are understood.

Never delete the `installs/<hash>` tree the running venv points at (`hermes doctor` prints the exact
path) and never delete a `.lock` sidecar. Pre-update `state.db*.bak` copies — validate them, then ask;
they are the only artifact that answers what the last update did — browser profiles, and the
`quick_check` + `immutable=1` validation for deciding whether a backup is still a real restore point
live in `references/disk-and-backup-hygiene.md`. Do not duplicate that list here.
