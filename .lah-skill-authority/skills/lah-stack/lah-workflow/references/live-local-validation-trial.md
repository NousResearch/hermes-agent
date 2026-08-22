# Live Local Validation Trial — Pattern

## Overview

A Live Local Validation Trial validates a real (non-mocked) service stack
locally before production wiring. It requires explicit operator approval
before any container, socket, or live HTTP call.

### When to use

- You have offline tests passing but need to verify real service integration
- A Docker container, REST API, or database needs to run briefly for validation
- You need to confirm wiring, error handling, and evidence recording work
- The mission plan includes a "live validation" phase

### When NOT to use

- Pure offline missions with no service dependencies — skip directly to tests
- Production deployments — this is for local validation only
- Scheduler or automatic discovery validation — those need separate approval

---

## Phase 1: Pre-flight Inspection

Before drafting an approval request, gather:

```bash
# 1. Docker status
docker ps  # what's already running?

# 2. Port availability
ss -tlnp | grep -E '<PORT1>|<PORT2>'

# 3. Relevant config files (read-only, no edits)
cat infra/docker-compose.yml
cat src/lah_discovery/api/app.py          # or equivalent entrypoint

# 4. Hermes/OpenClaw provider config (read-only)
cat ~/.hermes/config.yaml
cat src/services/lah-discovery-readonly-client.js

# 5. Related service .env.example
```

Compile into a table:

| Item | Status | Detail |
|------|--------|--------|
| No <service> container running | ✅/❌ | Port X free/in use |
| No REST API process | ✅/❌ | Port Y free/in use |
| Hermes providers config | ✅ | Read-only: providers: {} |
| OpenClaw client mode | ✅ | DISABLED/MOCK (default) |

---

## Phase 2: Approval Request

Draft an approval message with:

1. **Pre-flight results** — the table above
2. **Validation plan** — numbered steps with exact curl commands
3. **Approved test URL** — safe, static URL (e.g. `https://example.com`)
4. **Ports** — exact ports to bind, all to 127.0.0.1
5. **No-secret logging policy** — what will and won't appear in logs
6. **Rollback plan** — exact commands to stop everything

### Approval request template

```
## LIVE_LOCAL_VALIDATION — Approval Required

### Pre-flight inspection complete

**Current service state:**
- No <service> container running (port X free)
- No <other-service> process running (port Y free)

**Validation Plan (if approved)**

1. Start <service> container (port X, 127.0.0.1 only)
2. Start <other-service> (port Y, 127.0.0.1 only)
3. Run validation:
   - `curl http://127.0.0.1:Y/health` — expect 200
   - `curl -X POST .../discover/...` — expect items
   - `curl -X POST .../extract` — expect result
4. Do NOT run scheduler, automatic discovery, or any non-approved path
5. Stop all services
6. Verify rollback

**Approved test URL:** https://example.com
**Ports:** X, Y — both bound to 127.0.0.1
**No secrets in logs:** <list exactly what is logged>

### Rollback Plan

```
docker stop <container> && docker rm <container>
kill <PID>  # if started manually
ss -tlnp | grep -E '<PORT1>|<PORT2>'  # verify nothing listening
```
```

---

## Phase 3: Start Services (localhost-only)

### Docker container

```bash
docker pull <image>:<tag>
docker run -d --name <name> -p 127.0.0.1:<PORT>:<PORT> <image>:<tag>
```

Verify startup:

```bash
docker logs <name> | tail -5
# Look for: "Application startup complete" or equivalent signal
```

### Python REST API (manual launch)

Use `background=true` in terminal, then verify with health check.

```bash
# Start
cd /path/to/repo
source .venv/bin/activate
python scripts/run_api.py --port Y --bind 127.0.0.1

# Verify
curl -s http://127.0.0.1:Y/health
```

---

## Phase 4: Run Validation

Execute each step in order. Log results in a table:

| Step | Command | Expected | Actual | PASS/FAIL |
|------|---------|----------|--------|-----------|
| 1 | `GET /health` | 200 + {"status":"ok"} | ... | ... |
| 2 | `POST /discover/unknown` | 404 | ... | ... |
| 3 | `POST /discover/<source>` | 200 + items | ... | ... |
| 4 | `POST /extract <url>` | 200 + result | ... | ... |
| 5 | Evidence check | records present | ... | ... |

Key assertions:
- Unknown routes → 404, not 500
- Network errors → structured error response, not crash
- Evidence recording only fires on successful extraction
- JSONL file exists and is parseable

### Validate error handling

```bash
# Bad request
curl -s -X POST http://127.0.0.1:Y/extract -H "Content-Type: application/json" -d '{"bad": "payload"}'

# Unknown source
curl -s -X POST http://127.0.0.1:Y/discover/unknown-source
```

---

## Phase 5: Stop Services

```bash
# Stop Docker container
docker stop <container>
docker rm <container>

# Stop manual process via process(action='kill')
process(action='kill', session_id='<session_id>')
```

---

## Phase 6: Rollback Verification

```bash
# No listener on approved ports
ss -tlnp | grep -E '<PORT1>|<PORT2>' || echo "(clean)"

# No remaining containers
docker ps -a --filter name=<name>

# Tests still pass (regression check)
cd /path/to/repo
pytest -q
```

### Rollback report template

| Step | Result |
|------|--------|
| Stop service 1 | ✅ |
| Stop service 2 | ✅ |
| No listener on port X | ✅ |
| No listener on port Y | ✅ |
| No remaining containers | ✅ |
| Tests still passing (N) | ✅ |

---

## Output

After completion, produce:

1. **Validation results** — table with PASS/FAIL per step
2. **Findings** — any issues discovered (e.g. "Crawl4AI 0.9.0 binds to 127.0.0.1 inside container")
3. **GO/NO-GO assessment** — table of criteria with verdict per criterion
4. **Operator packet** — in `docs/operator/<MISSION>.md`
5. **Commit** — the launcher script and operator packet

### GO/NO-GO criteria

| Criterion | Verdict |
|-----------|---------|
| REST API serves | GO |
| Pipeline composes | GO |
| Evidence wired | GO |
| Error handling works | GO |
| All endpoints respond | GO |
| Rollback verified | GO |
| Service connectivity | ⚠️ NO-GO — needs fix |

---

## Pitfalls

- **Crawl4AI may bind to 127.0.0.1 inside the container.** Docker port mapping `-p 127.0.0.1:X:X` maps to the container's `0.0.0.0:X`, but if the service listens on the container's `127.0.0.1` it's unreachable. Check with `docker logs` for the actual bind address.
- **Crawl4AI versions may switch from REST to MCP protocol.** Verify endpoint availability (`/crawl`, `/md`) before committing to a version.
- **`process(action='wait')` blocks your turn.** Use `notify_on_complete=true` or `process(action='poll')` instead.
- **Shell backgrounding (`&`) is not supported** in foreground terminal commands. Always use `background=true`.
- **Clean up temp files** (`.env`, JSONL dbs, certificates) created during the trial.
- **Do not commit launcher scripts** that contain API keys, tokens, or passwords.