# Enterprise Production Hardening Implementation Plan

> **For Hermes:** Use subagent-driven-development/adaptive orchestration to implement this plan task-by-task. Do not use Gemini for any subagent. Use local Qwen/OpenRouter non-Gemini models only.

**Goal:** Complete the remaining production-readiness gaps: security/session hardening for the API server and deployment artifacts for production operation.

**Architecture:** Hermes Agent is a Python CLI/gateway application with an aiohttp OpenAI-compatible API adapter. This plan keeps compatibility with existing bearer API-key clients, avoids introducing a heavyweight auth system prematurely, and adds production controls around the existing security boundary: shorter token guidance, optional revocation for API tokens, CSRF protection for cookie-authenticated browser requests if enabled later, and hardened deployment examples. Deployment artifacts must be safe-by-default, profile-aware, and avoid committing secrets.

**Tech Stack:** Python 3.11+, aiohttp, pytest, ruff, Docker/Compose samples, nginx/Caddy examples, shell scripts.

---

## Phase 0: Preserve WIP and baseline

### Task 0.1: Confirm branch and untracked state

**Objective:** Ensure all work remains on `prod/adaptive-production-ready-20260708` and unrelated `prototypes/` is not staged.

**Files:**
- Inspect only: git state

**Steps:**
1. Run `git status --short --branch`.
2. Confirm branch is `prod/adaptive-production-ready-20260708`.
3. Confirm `prototypes/` remains untracked and excluded from later `git add`.

**Verification:** No accidental staging of `prototypes/`.

---

## Phase 1: API server auth/session hardening

### Task 1.1: Add explicit API server production-security config knobs

**Objective:** Make auth behavior self-documenting and configurable without breaking current API-key clients.

**Files:**
- Modify: `gateway/platforms/api_server.py`
- Test: `tests/gateway/test_api_server_security.py`

**Behavior:**
- Continue supporting `Authorization: Bearer <API_SERVER_KEY>`.
- Add optional revoked token digests via config/env:
  - `revoked_token_sha256` / `API_SERVER_REVOKED_TOKEN_SHA256`
  - comma-separated SHA-256 digests only, never raw tokens.
- Add token age guidance/config for deployments using short-lived externally minted API keys:
  - `max_bearer_token_age_seconds` / `API_SERVER_MAX_BEARER_TOKEN_AGE_SECONDS`
  - Do not reject opaque API keys by default; expose in capabilities/security docs.
- Add optional cookie auth mode for reverse-proxy/browser deployments only if explicitly configured:
  - `cookie_name` / `API_SERVER_COOKIE_NAME`
  - `csrf_header` / `API_SERVER_CSRF_HEADER`
  - `csrf_cookie` / `API_SERVER_CSRF_COOKIE`

**TDD steps:**
1. Write tests proving revoked token digests reject an otherwise valid API key.
2. Write tests proving non-mutating requests do not require CSRF.
3. Write tests proving mutating cookie-authenticated requests require matching CSRF header/cookie.
4. Run the new tests and verify RED before implementation.
5. Implement minimal helpers and config parsing.
6. Run the new tests and existing API server tests.

**Verification:**
- `python -m pytest tests/gateway/test_api_server_security.py -q`
- Existing API server tests remain green.

### Task 1.2: Improve capabilities/health documentation surface

**Objective:** Make clients and operators aware of auth mode, CSRF cookie mode, token revocation support, and health endpoints.

**Files:**
- Modify: `gateway/platforms/api_server.py`
- Test: `tests/gateway/test_api_server_security.py`

**Behavior:**
- `/v1/capabilities` includes a `security` block:
  - bearer auth required yes/no
  - token revocation supported yes/no
  - cookie auth configured yes/no
  - CSRF required for cookie mutating requests yes/no
  - recommended token max age value when configured
- `/health` remains unauthenticated and lightweight.
- `/health/detailed` remains unauthenticated but does not leak secrets.

**Verification:** Dedicated tests assert new fields and no secret values.

---

## Phase 2: Deployment artifacts

### Task 2.1: Add production Dockerfile

**Objective:** Provide a secure, reproducible container image for Hermes Agent.

**Files:**
- Create: `Dockerfile`
- Create/modify docs: `docs/deployment/production.md`

**Requirements:**
- Python 3.11 slim base.
- Non-root runtime user.
- No secrets baked into image.
- Healthcheck uses `/health`.
- Exposes API server port `8642`.
- Uses `hermes gateway run` by default.
- Installs package with messaging/API dependencies needed for gateway use.

**Verification:**
- `docker build` if Docker is available; otherwise static validation and documented skip.

### Task 2.2: Add docker-compose production sample

**Objective:** Give operators a safe default deployment with volume mounts and env-file separation.

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.production.example`

**Requirements:**
- Service `hermes-agent`.
- Bind `127.0.0.1:8642:8642` by default; no public wildcard exposure.
- Mount persistent Hermes home volume.
- Use env-file for secrets; example values must be placeholders.
- Set `API_SERVER_KEY` placeholder and documented requirement for public exposure.

**Verification:**
- `docker compose config` if available; otherwise YAML parse/static review.

### Task 2.3: Add reverse proxy examples

**Objective:** Provide nginx and Caddy examples for TLS termination, headers, body limits, and WebSocket/SSE-friendly proxying.

**Files:**
- Create: `deploy/nginx/hermes-agent.conf`
- Create: `deploy/caddy/Caddyfile`

**Requirements:**
- Proxy to `127.0.0.1:8642`.
- Preserve `Authorization`, `X-Hermes-Session-Id`, and `X-Hermes-Session-Key`.
- Disable buffering for SSE endpoints.
- Include security headers.
- Include comments reminding users to set strong API key and restrict CORS.

**Verification:** Static review; nginx/caddy syntax check if binaries available.

### Task 2.4: Add backup and log rotation artifacts

**Objective:** Provide operational maintenance examples for persistent state and logs.

**Files:**
- Create: `deploy/scripts/backup-hermes.sh`
- Create: `deploy/logrotate/hermes-agent`

**Requirements:**
- Backup script archives config/state/sessions/skills/logs while excluding secrets by default unless `INCLUDE_SECRETS=1`.
- Writes timestamped tar.gz to configurable backup directory.
- Uses safe shell flags and profile-aware `HERMES_HOME` override.
- Logrotate rotates Hermes logs with compression and retention.

**Verification:**
- `bash -n deploy/scripts/backup-hermes.sh`
- Run backup script against a temp fake HERMES_HOME to confirm it creates archive and excludes `.env` by default.

### Task 2.5: Document production deployment and healthchecks

**Objective:** Make production operation actionable.

**Files:**
- Create: `docs/deployment/production.md`

**Requirements:**
- Explain API key auth, revocation, token lifetime recommendation, cookie/CSRF policy.
- Explain `/health`, `/health/detailed`, `/v1/capabilities`.
- Include Docker, Compose, reverse proxy, backup, log rotation, and systemd notes.
- Include “do not expose without API_SERVER_KEY” warning.

**Verification:** markdown link/path sanity via static review.

---

## Phase 3: Verification and review

### Task 3.1: Run focused automated verification

**Commands:**
- `python -m pytest tests/gateway/test_api_server_security.py tests/gateway/test_api_server.py tests/gateway/test_api_server_bind_guard.py -q`
- Existing focused suite from previous phase:
  - `python -m pytest tests/gateway/test_gateway_workstream_progress.py tests/tools/test_delegate.py tests/tools/test_tts_speed.py tests/gateway/test_gateway_inactivity_timeout.py tests/test_agent_activity_summary.py tests/hermes_cli/test_command_aliases.py -q`
- `python -m py_compile` for changed Python files.
- `python -m ruff check` for changed Python files.
- Static secret/security grep on added diff lines.

### Task 3.2: Independent non-Gemini review

**Objective:** Use fresh-context reviewers without Gemini.

**Subagents:**
- Security reviewer: model `nvidia/nemotron-3-ultra-550b-a55b:free` or local Qwen if routed.
- Deployment reviewer: model `openrouter/owl-alpha` or Qwen Coder.
- Code quality reviewer: model `qwen/qwen3-coder:free`.

**Verification:** Parent validates claims with direct file reads/tests before finalizing.

---

## Phase 4: GitHub branch delivery

### Task 4.1: Stage intended files only

**Objective:** Commit all intended production-readiness changes and exclude unrelated artifacts.

**Files to stage:**
- Existing adaptive hardening files from prior phase.
- New security/deployment files from this plan.
- Exclude `prototypes/`.

### Task 4.2: Commit and push

**Commands:**
- `git add <explicit files>`
- `git commit -m "feat: harden production deployment and API security"`
- `git push -u origin prod/adaptive-production-ready-20260708`

**Verification:**
- Confirm commit exists on branch.
- Confirm push succeeds or report auth failure without exposing credentials.
