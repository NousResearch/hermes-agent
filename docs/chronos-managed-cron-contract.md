# Chronos managed-cron — agent ↔ NAS wire contract

**Status:** authoritative wire spec for the Chronos cron provider.
**Audience:** the NAS-side implementer of the `agent-cron` endpoints
(`nous-account-service`) and anyone debugging the managed-cron path.

Chronos lets a hosted Hermes gateway **scale to zero** while idle and still
fire cron jobs. Instead of an in-process 60-second ticker, the agent asks NAS
to arm exactly **one external one-shot per job at that job's real next-fire
time**. NAS calls the agent back at fire time over an authenticated webhook;
the agent runs the job and re-arms the next one-shot. Between fires the agent
process can be fully stopped — it wakes only on a genuine fire.

The external scheduler NAS uses to implement the one-shots is an **internal NAS
implementation detail**. The agent never talks to it, never holds its
credentials, and never names it. The agent only knows the three NAS endpoints
below.

```
create/update/pause/resume/remove a cron job (agent side)
  │
  ▼
ChronosCronScheduler.reconcile()        ── agent computes next_run_at
  │  POST {portal}/api/agent-cron/provision   (auth: agent's Nous access token)
  ▼
NAS arms a one-shot for fire_at         ── NAS owns the scheduler + its creds
  │
  ⏰ at fire_at
  ▼
scheduler → POST {portal}/api/agent-cron/relay   (auth: scheduler signature, NAS-verified)
  │
  ▼
NAS mints a short-lived agent-audience JWT (purpose=cron_fire)
  │  POST {agent_callback_url}/api/cron/fire        (auth: that JWT)
  ▼
agent verifies the NAS JWT → store CAS claim → run_one_job → re-arm next one-shot
```

## Trust model (read this first)

| Hop | Who calls whom | Auth mechanism | Verified by |
|---|---|---|---|
| 1 | agent → NAS (`provision`/`cancel`/`list`) | the agent's existing **Nous Portal access token** (Bearer) — for a hosted agent this is the **bootstrap-session token** NAS planted in `auth.json` (client `hermes-cli-vps`), NOT an `agent:*` client token | NAS (its normal agent-token path) |
| 2 | scheduler → NAS (`relay`) | the scheduler's request **signature** | NAS (the signature path it already has) |
| 3 | NAS → agent (`/api/cron/fire`) | a **short-lived NAS-minted JWT** (`aud=agent:{instance_id}`, `purpose=cron_fire`) | agent (PyJWT against NAS JWKS) |

> **Which token, exactly (hop 1).** A hosted agent never holds an `agent:{instance_id}`
> OAuth client credential — that shape is minted only by the interactive dashboard
> auth-code grant (a browser user). For all of its own outbound portal calls the
> agent uses the **bootstrap-session access token** (`resolve_nous_access_token`),
> minted under the bootstrap-only client `hermes-cli-vps` and seeded into the
> container on first boot. NAS therefore must resolve the calling agent's instance
> id from EITHER an `agent:{id}` client (self-hosted/dashboard callers) OR — for the
> bootstrap token — from `AgentInstance.bootstrapSessionId` matching the token's
> session id (`sid`), org-scoped. The fire JWT minted at hop 3 still carries
> `aud=agent:{instance_id}` regardless. (Gating hop 1 on an `agent:*` client alone
> 403s every real hosted-agent provision — see `src/server/agent-cron/instance-auth.ts`.)

Why NAS-mediated rather than scheduler→agent direct: the scheduler signs with
**NAS's** keys, which the agent does not (and should not) hold. The agent can
only verify a **NAS-minted** token — a trust path it already has. This keeps
all scheduler credentials inside NAS. (Full rationale: the plan's DQ-4.)

No new secret is introduced on the agent: hop 1 reuses the token the agent
already uses for the portal, and hop 3 reuses the NAS-JWT verification the agent
already performs.

---

## Endpoint 1 — `POST /api/agent-cron/provision`  (agent → NAS)

Arm (or re-arm, idempotently) exactly one one-shot for a job.

- **Auth:** `Authorization: Bearer <agent Nous access token>`. NAS validates via
  its normal agent-token path and scopes the row to the calling agent/org.
- **Request body:**
  ```json
  {
    "job_id": "ab12cd34",
    "fire_at": "2026-06-18T12:34:56+00:00",
    "agent_callback_url": "https://agent-xyz.fly.dev",
    "dedup_key": "ab12cd34:2026-06-18T12:34:56+00:00"
  }
  ```
  - `fire_at` — ISO 8601, **agent-computed**. May be sub-minute in the future;
    NAS must honor second-granularity (the agent owns the time, so there is no
    1-minute scheduler floor).
  - `agent_callback_url` — the agent's own publicly-reachable base URL. NAS
    POSTs `{agent_callback_url}/api/cron/fire` at fire time.
  - `dedup_key` — `"{job_id}:{fire_at}"`. NAS **upserts by `(agent_id, job_id)`**
    so re-arming the same fire is idempotent (no duplicate one-shots). A new
    `fire_at` for the same `job_id` replaces the prior arm.
- **Action:** arm one one-shot to fire at `fire_at`, destined for the NAS
  **relay** route (Endpoint 3) — NOT the agent directly, so NAS stays in the
  loop to mint the agent JWT. Persist `(agent_id, job_id, schedule_id,
  agent_callback_url)`.
- **Response:** `200 {"schedule_id": "<opaque>"}`.

## Endpoint 2 — `POST /api/agent-cron/cancel`  (agent → NAS)

- **Auth:** same as Endpoint 1.
- **Body:** `{"job_id": "ab12cd34"}`.
- **Action:** cancel the armed one-shot for `(agent_id, job_id)` and delete the
  row. Idempotent — cancelling an unknown job is a 200 no-op.
- **Response:** `200 {"ok": true}`.

## Endpoint 3 — `POST /api/agent-cron/relay`  (scheduler → NAS, the fire relay)

- **Auth:** the scheduler's request **signature**, verified by NAS with the
  signature path it already has. This is the trust boundary for the fire — a
  forged relay call must be rejected here.
- **Action:**
  1. Look up `(agent_id, job_id) → agent_callback_url` from the persisted row.
  2. Mint a **short-lived** JWT: `aud = "agent:{instance_id}"`,
     `iss = {portal_url}`, `purpose = "cron_fire"`, small `exp` (≈60–120s),
     signed with NAS's normal asymmetric signing key (published via JWKS).
  3. `POST {agent_callback_url}/api/cron/fire` with
     `Authorization: Bearer <that JWT>` and body `{"job_id": "...", "fire_at": "..."}`.
  4. Treat a non-2xx agent response as a **retryable** failure (let the
     scheduler retry the relay). The agent's store CAS de-dupes a double fire,
     so retries are safe.
- **Response to the scheduler:** 2xx once the agent POST is accepted (202), so
  the scheduler does not retry a delivered fire.

---

## Inbound `POST /api/cron/fire`  (NAS → agent) — agent side, already implemented

This is the agent endpoint NAS calls in Endpoint 3 step 3. Served by the
**dashboard app** (`hermes_cli/web_server.py`) — the agent's always-reachable
public HTTP surface on hosted deployments (the gateway may be idle/scaled down);
it is in `PUBLIC_API_PATHS` so the dashboard cookie gate lets the bearer-JWT
callback through to the verifier. (Also registered on the optional
`APIServerAdapter` for self-host API-server deployments.) The verifier is
`plugins/cron_providers/chronos/verify.py`.

- **Auth:** `Authorization: Bearer <NAS-minted JWT>`. The agent verifies:
  - signature against the NAS JWKS (`cron.chronos.nas_jwks_url`),
  - `aud` == `cron.chronos.expected_audience` (this agent's
    `agent:{instance_id}`),
  - `iss` == `cron.chronos.portal_url`,
  - `exp` / `nbf` (30s leeway),
  - `purpose == "cron_fire"` — a general agent JWT (no/other purpose) is
    rejected so it can't be replayed against this endpoint.
- **Body:** `{"job_id": "ab12cd34", "fire_at": "..."}` (only `job_id` is used).
- **Behavior:**
  - invalid/missing/forged/expired/wrong-aud/wrong-purpose token → **401**, no
    execution.
  - missing `job_id` → **400**.
  - valid → **202 `{"status": "accepted", "job_id": "..."}`** immediately, and
    the job runs in the background. 202-before-run means a long agent turn never
    trips the relay's HTTP timeout.
- **At-most-once:** the agent claims the job with a store-level compare-and-set
  (`claim_job_for_fire`) before running. A relay/scheduler retry that arrives
  while the first fire is in flight (or after it completed) loses the claim and
  does not double-run.

---

## At-most-once & re-arm semantics

- **Recurring (cron/interval):** on fire, the agent advances `next_run_at`
  (under its store lock) as part of the claim, runs the job, then re-provisions
  a one-shot for the new `next_run_at`. A duplicate relay for the old `fire_at`
  finds the claim taken / time advanced and is dropped.
- **One-shot (`30m`, `+90s`, etc.):** fires once; `mark_job_run` marks it
  completed. No re-arm.
- **`repeat.times = N`:** `mark_job_run` deletes the job at the limit, so
  `get_job` returns `None` after the final fire → the agent does **not** re-arm
  → the schedule stops cleanly with no orphaned one-shot.
- **Multi-replica agents:** the store CAS makes the fire at-most-once across N
  gateway replicas sharing one `HERMES_HOME` — exactly one replica runs each
  fire.

## Reconcile (self-healing)

The agent reconciles desired (`jobs.json`) vs armed on:
- `start()` (gateway boot / wake),
- every successful job mutation (`on_jobs_changed`),
- piggybacked after each fire (re-arm).

Reconcile arms missing/changed-time jobs and cancels orphans. A missed
provision (transient NAS error) self-heals on the next reconcile. There is **no
periodic wake** of a sleeping agent — that would negate scale-to-zero.

## Config (agent side)

All non-secret (`cron.chronos.*` in `config.yaml`); the agent holds no scheduler
credentials. For hosted agents NAS sets these at provision time:

| key | meaning |
|---|---|
| `cron.provider` | `"chronos"` to activate (empty = built-in ticker; `"none"` = scheduler disabled) |
| `cron.chronos.portal_url` | NAS base URL (also the expected JWT `iss`) |
| `cron.chronos.callback_url` | the agent's own public base URL for NAS→agent fires |
| `cron.chronos.expected_audience` | this agent's JWT `aud` (`agent:{instance_id}`) |
| `cron.chronos.nas_jwks_url` | NAS JWKS for verifying the fire JWT |

If `callback_url` / `portal_url` is blank or the agent has no Nous login,
`is_available()` returns False and the resolver falls back to the built-in
in-process ticker — cron never silently loses its trigger (only the explicit
`cron.provider: none` disables it).

## Escape hatch (not default)

The inbound `/api/cron/fire` verifier is pluggable (`get_fire_verifier()`). If
relay volume through NAS ever saturates, a direct scheduler→agent mode with a
per-job NAS-minted cron-key can replace the NAS-JWT verifier with **no change to
the webhook handler**. NAS-mediated (this contract) is the default.
