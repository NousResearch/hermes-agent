# Hermes Orchard 🌳

A multi-tenant **control-plane for [Hermes Agent](https://github.com/NousResearch/hermes-agent)**.

Hermes is, by design, a *single-tenant personal agent* (its own `SECURITY.md`
says so). Orchard puts a fleet of them behind **one Mattermost bot**: every
employee gets their **own isolated, permission-locked Hermes agent**, woken on
demand and put back to sleep when idle — so you can serve thousands of people
from a handful of servers without anyone touching anyone else's data.

> Status: **working MVP / scaffold.** The local loop runs end-to-end against a
> real model. The production isolation backend (containers/microVMs) and a few
> ops pieces are stubbed with clear TODOs. See [Status](#status).

---

## Why not just one shared bot?

Because a shared Hermes gateway isolates *conversations* per user but shares one
`HERMES_HOME` — so skills, files, cron, memory and credentials are **common**,
and the agent runs arbitrary shell. That fails "nobody can access anyone else's
stuff." Real isolation needs **one `HERMES_HOME` + one sandbox per employee**.
Orchard automates exactly that and adds the routing/lifecycle layer on top.

---

## Architecture

```
                 ┌──────────────────────────────────────────┐
   Mattermost ──▶│  Ingress (ONE bot: WS in, REST out)       │
   (one bot)     └───────────────────┬──────────────────────┘
                                      │ InboundMessage
                 ┌────────────────────▼──────────────────────┐
                 │  Router   sender_id → Employee (Registry)   │
                 │           rejects unprovisioned senders      │
                 └────────────────────┬──────────────────────┘
                                      │ handle(employee, session, text)
                 ┌────────────────────▼──────────────────────┐
                 │  Supervisor   wake-on-demand · keep-alive ·  │
                 │               idle-sleep · max-active cap ·   │
                 │               LRU eviction                    │
                 └────────────────────┬──────────────────────┘
                                      │ ensure_ready / send / sleep
                 ┌────────────────────▼──────────────────────┐
                 │  Backend (isolation)  local | docker | …     │
                 └───┬───────────────┬───────────────┬────────┘
                     ▼               ▼               ▼
              ┌───────────┐   ┌───────────┐   ┌───────────┐
              │ worker    │   │ worker    │   │ worker    │   ← one sandbox/employee
              │ daemon    │   │ daemon    │   │ (asleep)  │
              │ HERMES_   │   │ HERMES_   │   │           │
              │ HOME=/…/a │   │ HOME=/…/b │   │           │
              └─────┬─────┘   └─────┬─────┘   └───────────┘
                    │ hermes -z -c <session>  (headless, per tenant)
                    ▼
             Internal LLM endpoint (vLLM / SGLang / DeepSeek weights)
```

Components (all in [`orchard/`](orchard/)):

| Module | Role |
|---|---|
| `provisioner.py` + `security.py` | create & **lock down** a per-employee `HERMES_HOME` (0700 dirs, 0600 secrets, optional chown to a dedicated OS user) |
| `registry.py` | SQLite: `mm_user_id` → employee (the allowlist / source of truth) |
| `supervisor.py` | worker lifecycle: **wake on demand**, keep-alive, idle-sleep, hard concurrency cap + LRU eviction |
| `backends/` | isolation: `local` (subprocess), `docker` (container/employee), microVM (future) |
| `worker_daemon.py` | agent server that runs **inside** the sandbox; speaks a tiny unix-socket JSON protocol; invokes Hermes headless |
| `ingress/` | `mattermost` (single bot) + `cli` (local test harness) |
| `router.py` | ties it together; refuses unprovisioned senders |

---

## Security model

Two levels — don't confuse them:

1. **Filesystem permissions** (`security.py`): each `HERMES_HOME` is `0700`,
   secrets `0600`. On a shared OS user this prevents *accidental* cross-tenant
   reads but is **not** a boundary against a prompt-injected agent (same UID can
   `cat` another home). Fine for dev / small trusted teams.

2. **OS / kernel boundary** (the real one, via `backends/`):
   - `run_as_user`: drop each worker to a dedicated unprivileged OS user.
   - **`docker` backend**: one hardened container per employee (`--cap-drop ALL`,
     `no-new-privileges`, read-only rootfs, per-tenant bind mount, no open net).
   - **microVM** (Firecracker/gVisor, roadmap): strongest isolation + fast
     snapshot-restore — the right primitive for many untrusted tenants at scale.

The agent executes arbitrary shell, and messages are untrusted input, so **the
sandbox is the security boundary — not any in-process check.** (Same stance as
Hermes' own `SECURITY.md`.)

### Control-channel isolation (incident 2026-07-02)

A pen-test found a **confused-deputy / lateral-movement** path: the filesystem
sandbox held (Alice couldn't read Bob's files), but Alice's shell reached **Bob's
worker socket** in a shared `run/` dir and asked *Bob's* worker to read the file
and hand it back. Fixes (defense-in-depth):

1. **Socket moved into the tenant home** (`<home>/run/worker.sock`, 0600) — off
   the shared path.
2. **Sandbox denies the whole `data/` root** (all homes + registry DB), not just
   `employees/`.
3. **Per-worker token** on the socket protocol — a caller that reaches the socket
   still can't drive the worker without the secret.
4. **Token delivered over stdin, never env** — a same-UID sibling can read another
   process's environment with `ps eww`, so an env-var token leaks; stdin doesn't.

Findings worth remembering: macOS seatbelt `file-*` denies do **not** block a
unix-socket `connect()` (it's a network op) — the token, not the sandbox, is what
stops the socket call locally. And **same-UID is not a boundary**: the local
backend is dev-only. Production MUST use `run_as_user` (per-worker OS user → the
socket's 0600 owner-UID blocks siblings, and env/procs aren't cross-readable) or
the container/microVM backend (separate namespaces). A `SO_PEERCRED`/`LOCAL_PEERCRED`
peer-UID check on the daemon is a further hardening TODO.

### Blast-radius confinement (incident 2026-07-02 #2)

The first sandbox was **allow-default, deny-a-few** — so a worker could read the
operator's *entire laptop* (`~/.ssh`, `~/.aws`, the global `~/.hermes` secrets +
`state.db` of all conversations, other repos, other users). Fixed by inverting to
**deny the whole home root (`/Users`), allow only what's needed**:

- deny all file access under `/Users`;
- re-allow **bare metadata/stat** under `/Users` (so Hermes' walk-up `.git` probe
  doesn't crash — no content, no writes);
- re-allow **read-only** the specific runtime paths (`hermes` checkout, orchard
  venv/pkg, `~/.local`, `~/.hermes/node` + `bin`) — tool binaries, not user data;
- re-allow **read+write only this tenant's own home**.

Verified: a confined worker reads its own workspace but gets `BLOCKED` on
`~/.hermes/config.yaml`, `state.db`, `~/.ssh`, and sibling homes. Residual (dev
backend): directory **names** under `/Users` are still stat-able (metadata), and
same-UID caveats above still apply — the container backend removes both by
construction (the worker only ever sees its mounted home + the base image).

Model data stays on-prem: workers point at your **internal** OpenAI-compatible
endpoint (`llm.base_url`); nothing goes to a third-party API.

---

## Quickstart (local, end-to-end)

Requires: a Hermes checkout with its venv, and a reachable model endpoint.

```bash
uv venv --python 3.11 && uv pip install -e '.[dev]'
pytest -q                       # unit tests (no network)

# edit scripts/demo.config.yaml: set hermes_bin + llm.base_url
export ORCHARD_LLM_API_KEY=...  # your endpoint key
./scripts/demo_local.sh         # provisions alice+bob, chats as both
```

Manual:

```bash
orchard --config config.yaml provision alice --mm-user <MM_USER_ID> --name "Alice"
orchard --config config.yaml list
orchard --config config.yaml serve --ingress mattermost   # or: --ingress cli
```

---

## Production path

1. `backend: docker`, build a worker image (Hermes + orchard baked, `.pyc`
   precompiled), pre-pull to every node.
2. Set `security.run_as_user` and/or rely on container UID; keep `require_provisioned: true`.
3. Point `llm.base_url` at your internal inference cluster (size KV-cache for
   your peak concurrency × context — that's the real capacity wall, see notes).
4. Tune `supervisor`: `max_active_workers` (RAM guard), `idle_ttl_seconds`
   (keep-alive window), `warm_pool_size` (pre-warmed sandboxes for burst).
5. Orchestrate with k8s/Nomad; the Supervisor's wake/sleep maps onto
   scale-to-zero pods. Add per-tenant quotas + centralized audit logging.

**Warm-worker optimization:** today each message re-inits Hermes via `hermes -z`
(seconds of startup). Swap `worker_daemon` to drive Hermes over **ACP**
(persistent JSON-RPC) so the agent stays initialized between messages — turns
the "warm" state into true zero-init latency.

---

## Skill secrets (per-employee tokens)

Custom skills declare the data source + token they need in `SKILL.md`:

```yaml
metadata:
  orchard:
    data_sources: [{name: jira, url: https://jira.corp/api}]
    secrets:
      - {env: JIRA_TOKEN, label: "Jira API token", required: true}
```

Employees provide their **own** token **without ever typing it into chat**:

1. In Mattermost: `/secret set JIRA_TOKEN` (name only — a slash command, args
   aren't posted to the channel; a pasted value is refused + flagged for rotation).
2. The bot replies (ephemeral) with a **one-time, short-TTL HTTPS link**.
3. The employee opens it in a browser and enters the token in a password field.
4. It's stored in their **own confined home** (`secrets.json`, 0600) and injected
   into their worker as an env var at run time. `/secret list`, `/secret rm NAME`.

The value never enters the channel, the LLM context, the transcript/`state.db`,
or the logs (verified end-to-end). Storage is behind `SecretStore` so `local`
swaps for **Vault** later; the secret file rides with the tenant home to S3.
Same-UID caveat from the sandbox section applies to the injected env var — the
container backend removes it. Admin API: `GET/DELETE /api/employees/{id}/secrets`
(names + set/missing status only, never values).

### Integrations (tokens per-employee, config shared)

The dashboard shows an **Integrations** catalog (GitLab / Jira / Wiki — see
`integrations.yaml`). Each integration splits its fields:

- **Secret fields (tokens)** → entered by each employee via the one-time link,
  stored per-tenant, injected into their worker per turn. This is the ONLY thing
  an employee enters.
- **Non-secret fields (URLs, emails)** → **common org config**, one `value` in the
  catalog for everyone, injected into all workers. Not stored per-employee.

So a worker gets `GITLAB_URL` (shared) + its own `GITLAB_TOKEN` (personal). Cards
show configured/not (by the employee's tokens), the shared values inline, and
tokens as set/missing. API: `GET /api/employees/{id}/integrations`,
`POST .../integrations/{iid}/link`, `DELETE .../integrations/{iid}` (tokens only).

## Status

Done & tested:
- ✅ Provisioning + filesystem hardening (0700/0600, idempotent, path-contained)
- ✅ Registry (allowlist), unprovisioned-sender rejection
- ✅ Supervisor: wake-on-demand, keep-alive, idle reaper, capacity cap + LRU eviction (unit-tested)
- ✅ Local backend + worker daemon; **end-to-end chat works against a real model**
- ✅ CLI ingress (test harness)

Scaffolded / TODO:
- 🔧 `docker` backend: run wired, socket-RPC into container + readiness poll TODO
- 🔧 Mattermost ingress: written to the v4 REST/WS shape, **untested vs a live server**
- 🔧 ACP-based warm worker (kill per-message init cost)
- 🔧 Per-tenant quotas / cost metering, centralized audit log
- 🔧 microVM backend + snapshot-restore; k8s manifests; warm-pool implementation
