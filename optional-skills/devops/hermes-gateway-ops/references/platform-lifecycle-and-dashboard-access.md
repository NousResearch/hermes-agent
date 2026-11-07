# Platform lifecycle & dashboard access (hosted Hermes instance)

Use when the symptom arrives from the hosting platform rather than from the
box: opening the dashboard from the portal is refused ("another operation is
in progress for this agent"), a platform action (start/stop/restart/backup/
update) refuses or hangs, or a restart happened that nobody asked for locally.

## 1. Attribute the message before diagnosing

First, locate where the exact error text lives:

- Hit under the Hermes install (Python sources or the dashboard's bundled
  web assets) → upstream Hermes behaviour; follow the code path.
- Hit under your own scripts, skills, or profile configs → yours; fix there.
- **No hit anywhere → the hosting platform's control plane emits it.** Stop
  searching for a local cause: the control plane keeps its own per-instance
  operation record, and this box cannot clear it.

Search both code and the web assets — a UI-facing string can live in the
dashboard's JS bundle rather than Python, and searching only `*.py`
manufactures a false "it must be the platform".

## 2. Prove the instance healthy from outside

```bash
curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' https://<instance-dashboard-url>/
curl -s https://<instance-dashboard-url>/api/health
```

Expected: `302` to the login page with a provider parameter — a normal
SSO redirect, i.e. the dashboard itself is serving. `/api/health` should
answer with an ok payload. A login attempt that fails on a missing session or
verification cookie is transient: retrying the URL is the fix, not a repair.

Then read the box's own state (all read-only):

- `gateway_state.json` — `gateway_state`, `restart_requested`,
  `active_agents`, `served_profiles`, and per-platform `state` entries
  (`<profile>:telegram: connected` on a multiplexer)
- `state/gateway.lifecycle.json` — `phase` and `prior_unclean_exit`
- `state/gateway.heartbeat` — `updated_at` must be seconds-fresh; `mem.rss_kib`
  for memory pressure

## 3. Read the platform's own view (API, not the web UI)

When Hermes was set up through the hosting platform's OAuth
(`hermes setup --portal`), `auth.json` already holds an agent OAuth token —
it authenticates the platform's REST API:

```python
import json, urllib.request
d = json.load(open('auth.json'))['providers']['nous']
req = urllib.request.Request(d['portal_base_url'] + '/api/agents',
                             headers={'Authorization': 'Bearer ' + d['access_token']})
print(urllib.request.urlopen(req, timeout=20).read().decode())
```

Returns each instance's `id`, `name`, `status`, `dashboardUrl`, and gateway
state as the platform sees it. Treat the platform's state as *their* view:
`unknown` there with a healthy local instance means the platform could not
read it — evidence, not proof of a fault. The platform's HTML pages may sit
behind a bot checkpoint and its MCP endpoint needs a browser OAuth round-trip;
the REST route above is the one that answers headlessly — don't spend the
session fighting the other two.

## 4. The drain marker (`.drain_request.json`)

The platform's only in-band channel into a running gateway: it POSTs the
dashboard's drain endpoint, which writes a `.drain_request.json` marker in the
Hermes home; a gateway watcher reads it and flips `gateway_state` to
`draining`. Payload: `action`, `requested_at`, `principal`, `epoch`.

- **Foreign epoch = stale by design.** `epoch` identifies the machine
  instantiation that wrote the marker. After a container/VM restart the fresh
  gateway computes a different epoch and ignores the marker rather than
  parking itself in `draining` forever — so a leftover marker coexisting with
  `gateway_state: running` is normal, not a fault. Same-epoch markers expire
  after an hour.
- **Cancel it through the contract** (idempotent, keeps the staleness rules
  authoritative):

```bash
cp -a <HERMES_HOME>/.drain_request.json <HERMES_HOME>/drain_request.orphan-$(date +%s).json
HERMES_HOME=<home> python -c "import sys;sys.path.insert(0,'<hermes-install>');from gateway.drain_control import clear_drain_request, drain_requested;print(clear_drain_request(), drain_requested())"
```

  Verify after: `gateway_state.json` still `running`, `/api/health` still ok.
  A marker file missing the expected fields reads as drain-active (fail-safe),
  so never hand-write a "clean" marker — clear it or leave it to expire.

## 5. Restart-leftover forensics

What a platform restart leaves behind, and what each artifact tells you:

| Path (under the Hermes home) | Read for |
|---|---|
| `gateway-starts.log` | one monotonic stamp per gateway start — pairs of starts seconds apart = a restart loop |
| `logs/gateway-exit-diag.log` | unclean-exit diagnostics: last heartbeat memory, suspected OOM |
| `logs/container-boot.log` | per-profile prior state / action / prior exit at each boot |
| `logs/action-*.log` | dashboard-spawned long actions (doctor, backup, security-audit, update); a log that stops mid-progress = the action was killed by the restart — how a platform-side operation ends up unfinished |
| `.backup.lock` | `hermes backup`'s cross-process slot — an flock, released when the process dies, so a leftover zero-byte FILE is harmless |
| `state/dashboard_clients.heartbeat` | when a dashboard client was last connected (brackets who triggered a dashboard action) |

A dashboard action killed mid-run plus a drain marker written minutes before
the restart is the signature of a platform-coordinated sequence
(backup → drain → restart) whose completion signal was lost. Report it as the
platform's unfinished operation; the local artifacts are only evidence.

## 6. Escalation ladder and how to report it

1. Retry (an operation record can age out on its own).
2. Open the dashboard at its `dashboardUrl` directly — this bypasses the
   platform's gate and is the immediate unblock for the user.
3. The user stops then starts the instance on the platform's agents page:
   that resets the instance's operation state. Say the cost plainly (container
   restart, ~a minute of downtime, persistent data untouched).
4. Only then is it a wedged record on the platform side — their support.

Report it the way the user reads it: what the block is, that the instance and
all its message lanes are healthy, the one-link workaround, and what to do if
it persists. No file paths, job ids, or internal names in the user-facing
report — that detail belongs in your working notes.
