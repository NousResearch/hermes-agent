# Docker Compose Safe Deployment Alignment Pattern

**Use when:** Rebuilding a Docker image from certified source, recreating a single service, proving a new route/feature is deployed and active, and verifying safety/data preservation across the transition.

Designed for read-only deployment-alignment missions where no production source code changes are made — only image rebuild + container recreate + validation.

## Prerequisites

- Compose project and service name known
- Repository contains the certified source at a known HEAD
- Running container already exists (this is an update, not a greenfield deploy)
- Data directory is bind-mounted (not a named volume) for easy pre/post comparison

## Phase 1 — Audit (pre-build)

### 1a. Repository routing

Confirm git root, branch, HEAD, and dirty state. Record all for immutability proof.

```bash
cd <repo> && git rev-parse --show-toplevel
git branch --show-current
git rev-parse --short HEAD
git status --short | wc -l   # dirty count
```

### 1b. Compose and Dockerfile audit

Inspect each in full:

| Artifact | What to verify |
|----------|----------------|
| `docker-compose.yml` | Build context, container_name, ports, volumes, env_file, environment, restart, command |
| `Dockerfile` | FROM, COPY strategy (`.` vs selective), WORKDIR, EXPOSE, CMD/ENTRYPOINT |
| `.dockerignore` | Does not exclude required source files (route files, server entry point) |

Key checks:
- Build context resolves to the source tree with the certified files
- Dockerfile copies `src/` into the image (no `.dockerignore` exclusion of critical paths)
- No source bind mount that would overwrite the newly built `/app/src/...`
- `/app/data` (or equivalent) is a persistent bind mount, NOT ephemeral

### 1c. Pre-deployment snapshot

Capture these into an evidence directory:

```
CONTAINER_ID=$(docker inspect <container> --format '{{.Id}}')
IMAGE_ID=$(docker inspect <container> --format '{{.Image}}')
IMAGE_TAG=$(docker inspect <container> --format '{{.Config.Image}}')
CREATED=$(docker inspect <container> --format '{{.Created}}')
STATUS=$(docker inspect <container> --format '{{.State.Status}} {{.State.Running}} RestartCount={{.RestartCount}}')
PORTS=$(docker inspect <container> --format '{{json .NetworkSettings.Ports}}')
MOUNTS=$(docker inspect <container> --format '{{json .Mounts}}')
RESTART_POLICY=$(docker inspect <container> --format '{{json .HostConfig.RestartPolicy}}')

# Safety env (filter to safety flags, never print secrets)
docker exec <container> sh -c 'echo "EXOCLICK_LIVE_ENABLED=${EXOCLICK_LIVE_ENABLED:-unset}"'

# Pre-deployment route probe (prove the new route is NOT yet present)
curl -s -w '\nHTTP_STATUS: %{http_code}\n' -X POST http://<host>:<port>/<route-path> ...

# Data directory state
ls -la <host-data-path>/
stat <host-data-path>/
```

### 1d. Safety preflight

| # | Check | Enforce |
|---|-------|---------|
| 1 | Live/mutation gate is false | `EXOCLICK_LIVE_ENABLED=false` in container env |
| 2 | Provider credentials present but never printed | List them as "SET" without values |
| 3 | Admin/auth key set | Confirm, never print |
| 4 | Data bind mount source exists | `test -d <path>` |
| 5 | Data not empty (if previously non-empty) | `ls <path> \| wc -l` |
| 6 | No `docker compose down -v` | Document as forbidden |
| 7 | No `docker volume rm` | Document as forbidden |
| 8 | No `docker system prune` | Document as forbidden |
| 9 | No unrelated container recreate | `docker ps` before/after comparison |
| 10 | No real action_id / approval_id in probes | Use `nonexistent-*` prefix diagnostics |

## Phase 2 — Build

### 2a. Fresh build

```bash
docker compose build --no-cache <service-name>
```

Use `--no-cache` to guarantee fresh source copy — the `COPY . .` layer is not reused from a previous build that may have omitted the target files.

### 2b. Image verification

```bash
# New image ID
docker images <image-tag> --format '{{.ID}}'

# Verify route file exists in image
docker run --rm <image-tag> sh -c 'test -f /app/src/routes/<target-route>.js && echo EXISTS'

# Verify router import in server entry point
docker run --rm <image-tag> sh -c 'grep "<import-name>" /app/src/server.js'

# Verify route mount in server entry point
docker run --rm <image-tag> sh -c 'grep "app.use.*<mount-path>" /app/src/server.js'
```

### 2c. Fingerprint comparison

| Check | Command |
|-------|---------|
| Repo route file | `sha256sum <repo>/src/routes/<route>.js` |
| Image route file | `docker run --rm <tag> sha256sum /app/src/routes/<route>.js` |
| Repo server entry | `sha256sum <repo>/src/server.js` |
| Image server entry | `docker run --rm <tag> sha256sum /app/src/server.js` |

**Required:** Repository and image fingerprints must match for all critical files. Mismatch = BLOCKED.

## Phase 3 — Controlled Single-Service Recreate

### 3a. Record pre-recreate state

```bash
echo "Old container ID: $(docker inspect <container> --format '{{.Id}}')"
echo "Old image ID: $(docker inspect <container> --format '{{.Image}}')"
```

### 3b. Recreate

```bash
docker compose up -d --no-deps --force-recreate <service-name>
```

**Parameters explained:**
- `-d` — detached mode
- `--no-deps` — do not recreate dependent services
- `--force-recreate` — force container recreation even if compose considers it unchanged

### 3c. Post-recreate verification

```bash
echo "New container ID: $(docker inspect <container> --format '{{.Id}}')"
echo "New image ID: $(docker inspect <container> --format '{{.Image}}')"
echo "Status: $(docker inspect <container> --format '{{.State.Status}} {{.State.Running}} RestartCount={{.RestartCount}}')"
echo "Mounts: $(docker inspect <container> --format '{{json .Mounts}}')"
echo "Ports: $(docker inspect <container> --format '{{json .NetworkSettings.Ports}}')"
```

**Required:**
- New container ID differs from old container ID
- New image ID is the freshly built image
- Mount source and destination are unchanged
- Port mapping is unchanged
- EXOCLICK_LIVE_ENABLED (or equivalent safety flag) remains false
- Unrelated containers remain untouched (compare `docker ps` before/after)

## Phase 4 — Post-Deployment Validation

### 4a. Startup validation

| Check | Method |
|-------|--------|
| Container remains running | `docker inspect` → Status=running, RestartCount=0 |
| Process is correct | `docker exec <container> ps aux \| grep node` |
| No crash loop | Check RestartCount after 5-10 seconds |
| No fatal import error | `docker logs <container> --tail 50` → grep for `Error`, `SyntaxError`, `Cannot find`, `ENOENT` |
| Server startup event | Look for `server_started` or equivalent log event |
| Runtime reachable | `curl -s -o /dev/null -w '%{http_code}' http://<host>:<port>/<known-route>` |

### 4b. Deployed source proof

Execute the same fingerprint checks from Phase 2c INSIDE the running container:

```bash
docker exec <container> sha256sum /app/src/routes/<route>.js
docker exec <container> sha256sum /app/src/server.js
```

Compare against the repository fingerprints captured in Phase 2c. They must match.

Also verify:
```bash
docker exec <container> sh -c 'grep "<import-name>" /app/src/server.js'
docker exec <container> sh -c 'grep "app.use.*<mount-path>" /app/src/server.js'
```

### 4c. Route activation proof (two-step)

**Step 1 — Unauthenticated probe:**
```bash
curl -s -w '\nHTTP_STATUS: %{http_code}\n' -X POST http://<host>:<port>/<route-path> \
  -H 'content-type: application/json' \
  -d '{"<required-body-field>":"<nonexistent-diagnostic-id>"}'
```

Expected: HTTP **401** or **403** (`AUTH_REQUIRED` or similar auth middleware response).
IMPORTANT: It must NOT be a raw Express 404. If it's 401/403, the route is mounted and auth middleware intercepts correctly. If it's 404, the route may not be mounted.

**Step 2 — Authenticated probe with nonexistent ID:**
```bash
ADMIN_KEY=$(docker exec <container> sh -c 'echo "$ADMIN_API_KEY"')
curl -s -w '\nHTTP_STATUS: %{http_code}\n' -X POST http://<host>:<port>/<route-path> \
  -H 'content-type: application/json' \
  -H 'x-admin-api-key: '"$ADMIN_KEY" \
  -d '{"<required-body-field>":"<nonexistent-diagnostic-id>"}'
```

Expected: A deterministic fail-closed response from the handler (e.g. `PACKET_NOT_FOUND`, `MISSING_ACTION_ID`, `APPROVAL_INVALID`). The key proof is:
- Structured JSON response (not raw Express 404)
- `live_sent: false` (no provider mutation)
- No execution receipt created

### 4d. Existing route regression

Probe a bounded set of existing read-only/diagnostic routes that were known to work before the recreate:

```bash
curl -s -o /dev/null -w '%{http_code}' http://<host>:<port>/<known-route-1>
curl -s -o /dev/null -w '%{http_code}' -H 'x-admin-api-key: <key>' http://<host>:<port>/<known-route-2>
```

Accept 200, 401, 403, or 404 where appropriate — the goal is to prove the Express app started and old routes still respond (not 500/timeout).

### 4e. Provider safety proof

Search all container logs since the recreate timestamp for any mutation events:

```bash
docker logs <container> --since <ISO-timestamp> 2>&1 | grep -i -E \
  'CAMPAIGN_PAUSE_SENT|CAMPAIGN_PLAY_SENT|live_sent.*true|provider.mutation|execution_receipt|receipt_created'
```

Required: **Zero** provider mutation events, **zero** execution receipt events.

### 4f. Data preservation proof

Compare pre- and post-recreate data state:

| Check | Method |
|-------|--------|
| Same host source directory mounted | Compare `docker inspect` Mounts |
| Same ownership/permissions | `stat <host-data-path>/` |
| File count unchanged | `ls -1 <host-data-path>/ \| wc -l` |
| Pre-existing files still present | Check specific known files (`.gitkeep`, config JSONs, etc.) |
| No data directory recreated empty | File count > 0 if previously > 0 |
| No persistent store deleted | Check for known data files (packets, receipts, ledgers) |

## Phase 5 — Rollback Readiness

Rollback is mandatory if any critical validation fails:
- Container does not stay running
- Route source absent or fingerprint mismatch
- Route mount absent in server entry point
- Route still returns raw 404 (not auth/handler response)
- Live gate becomes true
- Data mount changes or persistent data disappears
- Startup logs show fatal regression
- Existing routes broadly fail (500s, timeouts)
- Provider mutation event appears

**Rollback procedure:**
1. Preserve all current evidence
2. Restore the previous image: either `docker tag` the old image back to `:latest`, or rebuild from a known-good git state
3. Recreate only the target service: `docker compose up -d --no-deps --force-recreate <service>`
4. Preserve the data bind mount
5. Confirm the old runtime is reachable again
6. Report failure + rollback result

**Pitfall:** When rebuilding with `--no-cache` and the same `:latest` tag, the old image is immediately untagged by Docker. To preserve the old image for rollback, tag it BEFORE the build:
```bash
docker tag <image-tag>:latest <image-tag>:rollback-$(date +%Y%m%d)
```

## Evidence Directory Structure

```
/tmp/<mission-name>/<timestamp>/
├── pre-deployment-snapshot.json     # Phase 1c capture
├── build-result.txt                 # Phase 2a+2b+2c
├── recreate-result.txt              # Phase 3a+3c
├── startup-validation.txt           # Phase 4a
├── deployed-source-proof.txt        # Phase 4b
├── route-activation-proof.txt       # Phase 4c
├── route-regression.txt             # Phase 4d
├── provider-safety.txt              # Phase 4e
├── data-preservation.txt            # Phase 4f
├── rollback-readiness.txt           # Phase 5
└── post-deployment-snapshot.txt     # Phase 4g (summary)
```

## Safety Constants

When coding deploy scripts or validation helpers, embed these:

```javascript
const SAFETY = Object.freeze({
  image_never_delete_before_validation: true,
  never_use_down_v: true,
  never_prune: true,
  never_touch_unrelated_services: true,
  never_use_real_production_ids_in_probes: true,
  never_print_secrets: true,
  data_mount_never_overwrite: true,
  live_gate_must_stay_false: true,
  rollback_before_delete_old_image: true,
});
```

## Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Old image untagged by tag-overwrite | Old image ID resolves to "no such image" after build | Tag old image as `rollback-<date>` BEFORE rebuild |
| `--no-deps` omitted in compose | Unrelated services recreated alongside target | Always use `--no-deps --force-recreate <service>` |
| Route returns 404 in probes | Raw Express 404 (route not mounted) vs structured handler 404 (different JSON) | Check response body: `{"ok":false,"status":"..."}` = handler active. Raw `Cannot POST /...` = not mounted |
| `.dockerignore` blocks route files | Route missing from new image | Check `.dockerignore` content. If absent, verify with `docker run --rm <image> find /app` |
| Compose override files add/remove services | `docker compose config --services` differs from expected | Check `docker compose ls` for all config files — overrides may add qdrant or other services |
| Data dir file count differs pre/post | New container auto-creates log/store files on startup | Compare known persistent files (not auto-generated logs). Distinguish expected growth from data loss |
| `docker exec` fails after recreate | New container may use a different user or have different PATH | Wait 3-5 seconds for full startup, then retry. Use `docker exec <container> sh -c` not bare `docker exec <container> <command>` |
| Provider safety search shows false positives | Logs contain source references like `execute_campaign_pause` in static text or JSDoc | Distinguish harmless static source text from actual runtime events with `live_sent` field check |
| `env_file` overrides Dockerfile `ENV` at runtime | Image env shows the correct `GIT_COMMIT` (`docker run --rm <image> echo $GIT_COMMIT`) but the CONTAINER shows a stale value (`docker exec <c> echo $GIT_COMMIT`). Cause: compose `env_file: .env` re-injects env at runtime, and a stale `GIT_COMMIT=` line in `.env` (e.g. an old commit from an unrelated context) beats the Dockerfile ENV. Observed 2026-08-01 live activation: image had 2855674, container reported 5d21a47b. | Always check BOTH image env AND container env — they can differ. Fix is config-only: `cp .env .env.bak-<date>-<sha>`, align the `GIT_COMMIT=` line to the deployed SHA, recreate the container (`docker stop && docker rm && up -d`). Verify both again after recreate. The image is NOT wrong when only the container env mismatches — do not rebuild. |
| Dirty canonical checkout contaminates the build | `COPY . .` ships the dirty working tree AND local `node_modules/` into the image when building from a checkout that has uncommitted changes (even an EARLIER state than the target merge) | Build from a clean detached worktree at the exact target SHA (`git worktree add --detach /tmp/<mission>-<sha> <sha>` → status 0, node_modules ABSENT → `GIT_COMMIT=<sha> docker compose build`), then deploy from the canonical checkout (where `.env` and `data/` live) with `docker compose up -d --no-deps --no-build` to reuse the already-tagged image. Remove the worktree after. |
| `docker-compose.override.yml` redirects the data mount to the CANONICAL checkout | The compose project runs from a deploy worktree (label `com.docker.compose.project.working_dir` = `<worktree>/lah-openclaw-mvp`), but a generated `docker-compose.override.yml` (e.g. by `bin/deploy-lah-openclaw-mvp-exact-sha.mjs`) forces `volumes: - /home/deploy/lah-stack-repos/<repo>/lah-openclaw-mvp/data:/app/data`. Persisted app data (decision-records/, campaign-ledger, packets, receipts) therefore lives in the CANONICAL checkout data dir, NOT the deploy worktree's `data/` (which holds only .gitkeep/campaign-ledger/refs). Hunting for post-E2E artifacts in the worktree data dir returns "Path not found" and can be mistaken for a persistence failure. | Before hunting for persisted artifacts: `docker inspect <c> --format '{{json .Mounts}}'` AND `docker inspect <c> --format '{{index .Config.Labels "com.docker.compose.project.config_files"}}'` — read the override to learn the real data source path. Verify data preservation and locate new records against the CANONICAL data dir. Observed 2026-08-08 (CLOE_LAHB_AUTONOMOUS_AFFILIATE_RUNTIME_E2E_V1 LOT 8: decision record was written to `/home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp/data/decision-records/` while the deploy worktree's data/ never changed). |
