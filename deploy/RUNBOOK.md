# RUNBOOK — the test Hermes (:10119), its deploy loop, and the QA gateway

Scope: the `reconnect-optimization` integration branch and the **test** Hermes
instance. Production is not described here on purpose; see "Never touch" below.

## The pieces

| Thing | Where |
|---|---|
| Test checkout | `/home/claude/.hermes/hermes-agent-test` (branch `reconnect-optimization`) |
| Systemd user unit | `hermes-test-serve.service` (`systemctl --user status hermes-test-serve`) |
| Listener | `127.0.0.1:10119` — loopback only |
| Front door | `https://llmbox.tail291e4.ts.net:10119` (llmbox `tailscale serve` → ssh tunnel → this VM's loopback) |
| Profile | `hermes-test` (`~/.hermes/profiles/hermes-test/`: `config.yaml` + `.env`) |
| Deploy script | `deploy/deploy-test.sh` (in the checkout) |
| QA machine | `deploy/qa_gateway.py` (in the checkout) |
| Dashboard SPA | `hermes_cli/web_dist/` (build output, gitignored) |
| Webapp bundle | `apps/desktop/dist-webapp/` (build output, gitignored) |

### Never touch from anything in this program

- `https://…:9119` and `hermes-serve.service` — production backend + dashboard (default profile).
- `hermes-webui.service` (:9120).
- `/home/claude/.hermes/config.yaml` and `~/.hermes/.env` — the default profile's config/secrets.
- `/home/claude/.hermes/hermes-agent` — production's checkout. `deploy-test.sh` refuses it explicitly.

## Deploy loop

```bash
cd /home/claude/.hermes/hermes-agent-test
deploy/deploy-test.sh              # idempotent; safe to run twice back-to-back
```

What it does, every run:

1. `git fetch elight reconnect-optimization` + `git reset --hard elight/reconnect-optimization`.
2. Re-lock locally if `uv lock --check` fails (see "Known conditions").
3. `scripts/run-in-hermes-env bash -c 'pnpm install --frozen-lockfile'`.
4. `pnpm --dir web run build` → `hermes_cli/web_dist/` — **the dashboard SPA the
   :10119 UI is served from**.
5. `npm --prefix apps/desktop run build:webapp` → `apps/desktop/dist-webapp/` —
   the **desktop webapp bundle** (this is the card's `npm run build:webapp`).
   Note it is *not* what the :10119 dashboard serves: `dist-webapp` is served by
   the **gateway's** Webapp listener (`hermes_cli/gateway/webapp_listener`, a
   separate surface). Both are built here so a QA card can exercise either.
6. `systemctl --user restart hermes-test-serve`, then polls
   `http://127.0.0.1:10119/api/health` until it returns 200 (`HEALTH_TIMEOUT`,
   default 180 s).
7. Prints the deployed commit sha and subject.

Escape hatches: `SKIP_FRONTEND=1` (restart only — the builds are the slow part),
`REPO_DIR=… BRANCH=… HEALTH_TIMEOUT=…`. The unit and the port are *not*
overridable: the script fails closed on anything but `hermes-test-serve` and
`:10119`, so it can never restart a production unit.

### Why the unit is what it is

```
hermes-test-serve.service:
  run-in-hermes-env …/hermes-agent-test/hermes -p hermes-test dashboard \
      --isolated --skip-build --no-open --host 127.0.0.1 --port 10119
  Environment=HERMES_DASHBOARD_PUBLIC_URL=https://llmbox.tail291e4.ts.net:10119
```

- `-p hermes-test` — the test profile's own `config.yaml`/`.env`.
- `HERMES_HOME=%h/.hermes` — **required**. Run from a source checkout, the
  resolved default home is the checkout's own dev home (`<checkout>/.hermes`),
  so `-p hermes-test` would look for
  `<checkout>/.hermes/profiles/hermes-test`. Pin the machine root and `-p`
  selects the profile beneath it.
- `--isolated` — **required**. Without it, `hermes -p <name> dashboard` routes
  the launch to the machine-level dashboard (`main_dashboard.py`,
  `_route_named_profile_dashboard`): same process, default profile's config,
  production's auth block. `--isolated` runs a dedicated per-profile server.
- `--skip-build` — the unit serves what is on disk; the deploy loop does the
  building. It will refuse to start with no `hermes_cli/web_dist/index.html`.
- `HERMES_DASHBOARD_PUBLIC_URL` — declares the tailnet hostname so the
  DNS-rebinding Host check accepts it, and engages the auth gate (a declared
  public URL always does). The gate is answered by the bundled `basic` provider,
  whose credentials come from the test profile's `.env`
  (`HERMES_DASHBOARD_BASIC_AUTH_USERNAME` / `_PASSWORD` / `_SECRET`). Nothing
  production-shaped is used or needed.

## QA gateway (`deploy/qa_gateway.py`)

Stdlib + `websockets`, no privileged access, no writes. Run it inside the
checkout's environment so the WebSocket library comes from the same env the
server uses. The `cd` below is load-bearing: `run-in-hermes-env` composes the
environment for the *current* checkout, so invoking it by absolute path from
any other cwd fails with "the 'websockets' package is required":

```bash
cd /home/claude/.hermes/hermes-agent-test
scripts/run-in-hermes-env python deploy/qa_gateway.py ticket
scripts/run-in-hermes-env python deploy/qa_gateway.py open
scripts/run-in-hermes-env python deploy/qa_gateway.py phases --cycles 20
scripts/run-in-hermes-env python deploy/qa_gateway.py phases --cycles 20 --json
# from anywhere on the tailnet (Evan's laptop, a QA VM):
scripts/run-in-hermes-env python deploy/qa_gateway.py \
    --base-url https://llmbox.tail291e4.ts.net:10119 phases --cycles 20
```

- Signs in with `POST /auth/password-login` (provider `basic`), then uses the
  session cookie for `POST /api/webapp/socket-ticket` and `POST /api/auth/ws-ticket`.
- Credentials come from `~/.hermes/profiles/hermes-test/.env` (override with
  `--env-file`); environment variables of the same name win. They are never
  passed on the command line and never live in this repo.
- Tickets: `POST /api/webapp/socket-ticket` exists only on a surface whose
  `SurfacePolicy` sets `gateway_ticket` (the gateway's Webapp listener,
  `hermes webapp`). This dashboard answers 404 there, so `open`/`phases` mint
  with `POST /api/auth/ws-ticket` — the dashboard SPA's own path — and every
  report names the source it used. `ticket` times both endpoints and shows which
  one this surface serves.
- `phases` closes and re-dials `/api/ws` N times and prints, per cycle, the phase
  table named after the client-side schema in
  `apps/desktop/src/store/desktop-metrics.ts` (C1):
  `detect_ms pre_ipc_ms dial_ms ready_ms replay_barrier_ms usable_ms`, plus p50/p95.
  A gateway-side driver can only observe part of that schema, and says so in its
  output: `detect_ms` is 0.0 by construction (this driver closes the socket
  itself) and `replay_barrier_ms` is n/a (there is no client replay to wait
  behind). Both columns are filled by C1's client instrumentation.

## Verification checklist (test instance only)

```bash
systemctl --user status hermes-test-serve --no-pager     # active (running)
curl -sw '%{http_code}\n' -o /dev/null http://127.0.0.1:10119/api/health   # 200
curl -sk -o /dev/null -w '%{http_code}\n' https://llmbox.tail291e4.ts.net:10119/api/health  # 200
deploy/deploy-test.sh && deploy/deploy-test.sh           # 0 twice, sha printed each time
```

`/api/health` is on the shared public-paths allowlist, so it answers 200 behind
the auth gate — it is the only endpoint to trust for liveness. Everything else
under `/api/` redirects to the login without a session cookie.

## Known conditions

- **The test profile had been deleted, not merely stopped.** `hermes profile
  delete hermes-test` (2026-10-29) left a tombstone at
  `~/.hermes/profiles/.deleted/hermes-test` beside a directory that still held
  `.env`, `cron/`, `plugins/`, `runtime/`. A tombstoned profile resolves as
  "does not exist", and `hermes profile create hermes-test` refuses because the
  leftover dir carries profile identity — so the leftovers were neither usable
  nor replaceable. Clearing the tombstone (what `clear_named_profile_deleted`,
  i.e. a successful `profile create`, does) restored it. Never `rm` the profile
  directory: `cron/jobs.json` and `plugins/` live there; `config.yaml` is
  re-seeded by `hermes-inherit-default` from the default profile's models.
- **`uv lock --check` fails on the branch base.** The pinned uv (0.13.5) cannot
  vouch for the checked-in `uv.lock` here (the offline check cannot fetch a
  git-pinned dependency, and the online check reports the lock needs updating),
  in this checkout *and* in production's. `deploy-test.sh` re-locks locally when
  the check fails; the regenerated `uv.lock` stays local and is never committed
  or pushed. If the lock is fixed upstream, the step becomes a no-op.
- **First run after a fresh clone needs the environment once**:
  `bash setup-hermes.sh --runtime-only` in the checkout (stages uv/python/node
  tooling and the venv). `run-in-hermes-env` re-syncs automatically when the
  environment no longer matches its inputs.
- **npm/pnpm egress**: node 26.12.1 / npm 12.0.2 / pnpm 12.15.0 come from the
  pm tool store (`~/.hermes/tools/…`) via `run-in-hermes-env`, and the npm
  registry is reachable from this VM. Production's `--skip-build` history
  explains why the unit itself never builds: cross-building was once the only
  way to get a dist here, and keeping builds out of the unit keeps a broken
  front-end from taking the listener down.

## Recovery

```bash
journalctl --user -u hermes-test-serve -n 100 --no-pager     # why it exited
deploy/deploy-test.sh                                        # rebuild + restart
systemctl --user daemon-reload                               # after editing the unit
```

If `deploy-test.sh` fails in a front-end build, the checkout is left at the
fetched commit with the old dist on disk and the unit still serving it; re-run
after fixing the cause, or `SKIP_FRONTEND=1` to deploy the backend without
rebuilding the front-ends.
