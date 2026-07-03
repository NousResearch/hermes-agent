# Telegram Mini App — Operator Runbook

The Telegram Mini App is a small FastAPI sidecar for the Hermes gateway. It has
two independent layers:

1. **Status panel (default).** A loopback-only, read-only view of gateway health
   (running/busy/drainable, active agents) rendered in a Telegram Mini App.
2. **Action gate (opt-in, disabled by default).** A complete, tested mechanism
   for an allow-listed owner to *approve or reject* a pending dangerous-command
   approval — but its two gateway-side wiring points are not yet implemented, so
   today it records decisions without applying them. See
   [Enabling the action gate](#enabling-the-action-gate-current-state).

By default the sidecar is read-only: it serves redacted status, capabilities, an
approvals *preview*, and safe session/log snapshots, and resolves no approvals.

---

## Running the sidecar

The entrypoint is a foreground operator command — there is **no** launchd /
systemd / autostart on purpose:

```sh
python -m hermes_cli.telegram_miniapp.cli serve
```

It binds `127.0.0.1:9120` by default and serves the status panel. It requires
the dashboard extras (FastAPI + `uvicorn`). A missing `uvicorn` exits with a
friendly install message; a missing FastAPI surfaces earlier as a
`ModuleNotFoundError` at import (FastAPI is imported before that check), so
install both.

Flags (all defined in `hermes_cli/telegram_miniapp/cli.py`):

| Flag | Effect |
| --- | --- |
| `--https-smoke` (with `--public-base-url https://<host>`) | Opt-in, short-lived public HTTPS smoke test for **this run only**. The bind is still forced to `127.0.0.1` (front it with your own TLS terminator/tunnel). Mutually exclusive with `--enable-actions`. |
| `--enable-actions` | Enables the owner approve/reject action gate for **this run only**. Requires `telegram_miniapp.action_owners` to be set in config, or it exits. |

If the port is already in use, the CLI exits with an actionable message naming
the port to change (`telegram_miniapp.port`) rather than a raw traceback.

---

## Enabling the action gate (current state)

The action gate is a complete, tested mechanism that is **not yet wired to the
live gateway on either end**. This is the honest current state:

- **`--enable-actions`** (per-run CLI flag) registers the sidecar's decision
  POST route — but only when the sidecar is fully ready:
  `MiniAppSettings.actions_ready()` requires `--enable-actions` **and**
  `action_owners` **and** a bot token **and** a configured `HERMES_HOME`.
  Without it the route does not exist.
- **No gateway export.** The panel reads pending approvals from a signed
  `approvals_snapshot.json`, but no gateway code writes one today, so the panel
  falls back to a **preview** snapshot. A decision submitted against a preview
  (or a stale) snapshot is rejected by the bridge's target check.
- **No gateway resolve.** Even a validly-signed decision file is only *recorded*:
  no gateway code reads `telegram_miniapp.bridge_enabled` or drives
  `MiniAppBridge.run_cycle` / `resolve_gateway_approval_cas`, so nothing applies
  it to a live approval.

In short, the sidecar, the file bridge, the CAS resolver, and the redaction /
replay / conflict handling are all implemented and tested (`bridge.py`,
`tools/approval.py`), but the two gateway-side wiring points — exporting live
approvals into the snapshot, and running the resolve cycle — are future work.
`telegram_miniapp.bridge_enabled` is the durable switch reserved for the resolve
side; it is parsed but has **no effect yet**.

`enable_actions` and `public_smoke` are **never** read from `config.yaml` — they
exist only as per-run CLI flags, so a durable config can never silently expose
actions or a public origin.

To turn the gate **off**: drop `--enable-actions` (the route is gone on the next
run).

---

## Configuration

All keys live under `telegram_miniapp:` in `config.yaml` (see
`cli-config.yaml.example`). Defaults come from `MiniAppSettings`
(`hermes_cli/telegram_miniapp/server.py`) and `settings_from_config`
(`hermes_cli/telegram_miniapp/config.py`).

| Key | Default | Meaning |
| --- | --- | --- |
| `host` | `127.0.0.1` | Bind address. Forced to loopback in **both** modes (even `--https-smoke`); a non-loopback host without `--https-smoke` is rejected. |
| `port` | `9120` | Bind port. Change if in use. |
| `allowed_users` | `[]` | Telegram **user ids** allowed to *read* status. |
| `action_owners` | `[]` | Telegram **user ids** allowed to *approve* actions. |
| `cors_allowed_origins` | `http://127.0.0.1:5175`, `http://localhost:5175` | Allowed browser origins in loopback mode; `--https-smoke` overrides this to the smoke origin. |
| `auth_ttl_seconds` | `300` | initData freshness window for `POST /api/auth/telegram`. |
| `session_ttl_seconds` | `3600` | Sidecar session cookie lifetime. |
| `action_initdata_ttl_seconds` | `900` | initData freshness window per action POST. |
| `auth_rate_limit_per_minute` | `10` | Auth attempts / minute, keyed by client host + Host header (auth runs before a session exists). |
| `auth_global_limit` | `50` | Global lifetime cap on auth attempts (public smoke). |
| `action_rate_limit_per_minute` | `5` | Action POSTs / minute. |
| `status_rate_limit_per_minute` | `60` | Per-minute limit on public-smoke read endpoints (status, capabilities, approvals, sessions, logs). |
| `bridge_enabled` | `false` | Gateway-side bridge switch (see above). |

> The **owner** action id is `704305405` — this is the Telegram **user id**, not
> the `970532987` chat id. Put it in `action_owners`, not `allowed_users` alone.

`bridge_enabled` is **strictly** coerced: a quoted `"false"` / `"no"` / `"0"`
(easy to introduce via templating) stays **off**. Only `true` / `yes` / `on` /
`1` enable it.

---

## The file bridge

When the action gate runs, the sidecar and the gateway communicate through a
file-backed, HMAC-authenticated contract under `HERMES_HOME/miniapp/`:

```
HERMES_HOME/miniapp/
  approvals_snapshot.json   # signed, redacted snapshot of pending approvals
  decisions/                # owner-signed decision envelopes (inbox)
  receipts/                 # per-decision receipts (persistent replay guard)
  audit.jsonl               # append-only, redacted audit trail
```

- Directories are `0700`, files `0600`.
- Snapshots and decisions are HMAC-signed with a key derived from the bot token.
- Approval ids are **opaque and per-instance**: each pending approval gets a
  random nonce, so an id is bound to one specific approval instance and cannot
  be reused to resolve a re-issued identical command.
- `audit.jsonl` is redacted — it never contains the raw command, path, or
  session key.

---

## Security invariants

- **Fail-closed everywhere.** A malformed, expired, tampered, replayed, or
  stale-snapshot decision resolves nothing.
- **Redacted projections.** The public snapshot carries only opaque ids and
  fixed risk-tier copy — never the raw command, path, description, or session
  key. An unknown/spoofed risk tier is floored to `critical`.
- **Per-action proof.** Every action POST re-verifies fresh Telegram initData
  (window `action_initdata_ttl_seconds`).
- **Reject beats approve.** If the owner submits both an approve and a reject for
  one approval in the same tick, the reject wins (fail-closed to deny).
- **Replay guard.** A decision resolved in a prior run leaves a receipt on disk;
  a resubmission is reported `already_resolved` and never re-applied, and the
  authoritative receipt is never overwritten.
- **Bounded lifetime.** Decisions outside a two-sided TTL window (past *or*
  future-dated) are swept, matching the validation window.

---

## Verify & operate

Read the `miniapp` block of `GET /api/status`:

```json
"miniapp": { "mode": "read-only", "actions_enabled": false, "public_exposure": false }
```

- `mode`: `read-only` (default), `owner-action` (sidecar action route live), or
  `https-smoke`.
- `actions_enabled`: `true` when the **sidecar** is ready this run
  (`--enable-actions` + `action_owners` + a bot token + a configured
  `HERMES_HOME`). It does **not** assert that a gateway-side resolver is running
  (none is wired yet).
- `public_exposure`: `true` only under `--https-smoke`.

The frontend requires `actions_enabled === true` in addition to its own
capability check, so a stale/false status fails the action UI closed.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `requires uvicorn/FastAPI dashboard extras` (or a `ModuleNotFoundError` for FastAPI at startup — it is imported before the friendly check) | Install the dashboard extras (FastAPI **and** uvicorn). |
| `could not bind 127.0.0.1:9120 …` | Another process holds the port — stop it or set `telegram_miniapp.port` to a free port. |
| `sidecar is loopback-only unless --https-smoke is used` | You set a non-loopback `host` without `--https-smoke`. Use `--https-smoke` + `--public-base-url`, or keep `host: 127.0.0.1`. |
| `--enable-actions requires telegram_miniapp.action_owners` | Add the owner user id (`704305405`) to `action_owners`. |
| Actions submitted in the panel do nothing | Expected today: the gateway-side resolver is not yet wired, so decisions are validated and recorded but not applied. `--enable-actions` only enables submission. |
