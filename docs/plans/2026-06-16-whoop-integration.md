# WHOOP Integration Implementation Plan

> **For Hermes:** Use subagent-driven-development skill or native Claude Code/Codex to implement this plan task-by-task.

**Goal:** Add a bundled WHOOP integration to Hermes so Graham can authenticate once and let agents read WHOOP recovery/sleep/cycle/workout data through safe tool calls.

**Architecture:** Mirror the existing Spotify integration: provider-specific OAuth helpers live in `hermes_cli/auth.py`; API client and tool schemas live in a bundled plugin under `plugins/whoop`; tool availability gates on `get_auth_status("whoop")`. Keep the first slice read-only. No medical advice; expose raw health metrics plus lightweight summaries only.

**Tech Stack:** Python 3.11, `httpx`, Hermes plugin loader, `tools.registry.tool_result/tool_error`, pytest.

---

## Official WHOOP API Facts Captured

- Developer app required in WHOOP Developer Dashboard.
- Client secret must not be logged or pasted into chat. Credentials belong in local CLI prompts, env vars, or approved secret stores.
- OAuth authorization URL: `https://api.prod.whoop.com/oauth/oauth2/auth`
- OAuth token URL: `https://api.prod.whoop.com/oauth/oauth2/token`
- API base: `https://api.prod.whoop.com/developer/v2`
- Redirect URI must match the dashboard.
- OAuth `state` must be eight characters when generated manually.
- Collections paginate with `records` plus `next_token`; follow-up requests use `nextToken`.
- Documented collection example: `GET /activity/sleep?start=...&end=...`
- Default rate limits: 100 requests/minute and 10,000/day.

---

## Task 1: Add RED auth tests

**Objective:** Define the auth contract before touching production code.

**Files:**
- Create: `tests/hermes_cli/test_whoop_auth.py`
- Reference: `tests/hermes_cli/test_spotify_auth.py`
- Modify later: `hermes_cli/auth.py`

**Step 1: Write failing tests**

Cover:
- `get_auth_status("whoop")` returns logged-out status when no state exists.
- `resolve_whoop_runtime_credentials()` raises `AuthError` when unauthenticated.
- env resolution accepts `HERMES_WHOOP_CLIENT_ID`, `WHOOP_CLIENT_ID`, `HERMES_WHOOP_CLIENT_SECRET`, `WHOOP_CLIENT_SECRET`, redirect/base URL overrides.
- expired tokens refresh through the WHOOP token URL and persist updated state.
- state generator returns exactly 8 characters.

**Step 2: Run RED**

```bash
python -m pytest tests/hermes_cli/test_whoop_auth.py -q -o 'addopts='
```

Expected: FAIL due missing WHOOP symbols.

---

## Task 2: Implement WHOOP auth helpers

**Objective:** Add service-provider auth state support without changing inference provider behavior.

**Files:**
- Modify: `hermes_cli/auth.py`

**Implementation notes:**
- Add constants:
  - `DEFAULT_WHOOP_AUTH_URL`
  - `DEFAULT_WHOOP_TOKEN_URL`
  - `DEFAULT_WHOOP_API_BASE_URL`
  - `DEFAULT_WHOOP_REDIRECT_URI = "http://127.0.0.1:43828/whoop/callback"`
  - `WHOOP_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120`
  - `DEFAULT_WHOOP_SCOPE = "read:profile read:cycles read:recovery read:sleep read:workout"` unless docs/tests prove different scope names.
- Add `"whoop": "WHOOP"` to `SERVICE_PROVIDER_NAMES`.
- Mirror Spotify helper shape where practical:
  - `_whoop_client_id(...)`
  - `_whoop_client_secret(...)`
  - `_whoop_redirect_uri(...)`
  - `_whoop_api_base_url(...)`
  - `_whoop_auth_url(...)`
  - `_whoop_token_url(...)`
  - `_whoop_state_token()` returning exactly 8 chars.
  - `_whoop_token_payload_to_state(...)`
  - `_refresh_whoop_oauth_state(...)`
  - `resolve_whoop_runtime_credentials(...)`
  - `get_whoop_auth_status()`
- Fail closed when client secret is absent during code exchange/refresh.
- Do not print secrets; redact secret presence as boolean only.

**Step 3: Run GREEN**

```bash
python -m pytest tests/hermes_cli/test_whoop_auth.py -q -o 'addopts='
```

Expected: PASS.

---

## Task 3: Add WHOOP API client tests

**Objective:** Lock request paths, pagination, refresh-on-401, and error formatting.

**Files:**
- Create: `tests/tools/test_whoop_client.py`
- Create later: `plugins/whoop/client.py`

**Step 1: Write failing tests**

Cover:
- `WhoopClient.request("GET", "/activity/sleep", params={...})` uses bearer auth and base URL.
- 401 triggers one forced refresh and retries once.
- 429 errors include retry/rate-limit context.
- `paginate_collection()` accumulates records and follows `next_token` via `nextToken`.
- `max_pages` stops runaway pagination.

**Step 2: Run RED**

```bash
python -m pytest tests/tools/test_whoop_client.py -q -o 'addopts='
```

Expected: FAIL due missing `plugins.whoop.client`.

---

## Task 4: Implement `plugins/whoop/client.py`

**Objective:** Provide a thin, testable WHOOP Web API client.

**Files:**
- Create: `plugins/whoop/client.py`

**API shape:**
- Exceptions: `WhoopError`, `WhoopAuthRequiredError`, `WhoopAPIError`.
- Runtime resolver: `resolve_whoop_runtime_credentials()` from `hermes_cli.auth`.
- `request(method, path, params=None, json_body=None, allow_retry_on_401=True)`.
- Convenience methods:
  - `get_profile()` -> `GET /user/profile/basic`
  - `get_cycle(cycle_id)` -> `GET /cycle/{cycle_id}` if official docs confirm; otherwise skip get-by-id.
  - `get_cycles(...)` -> `GET /cycle`
  - `get_recovery(...)` -> `GET /recovery`
  - `get_sleep(...)` -> `GET /activity/sleep`
  - `get_workouts(...)` -> `GET /activity/workout`
- `paginate_collection(path, params, max_pages=1)`.

**Step 3: Run GREEN**

```bash
python -m pytest tests/tools/test_whoop_client.py -q -o 'addopts='
```

---

## Task 5: Add tool schema and handler tests

**Objective:** Define a minimal read-only tool surface for agents.

**Files:**
- Create: `tests/tools/test_whoop_tools.py`
- Create later: `plugins/whoop/tools.py`

**Tools:**
- `whoop_profile`
- `whoop_cycles`
- `whoop_recovery`
- `whoop_sleep`
- `whoop_workouts`

**Required behavior:**
- Tool handlers return `tool_result` JSON on success and `tool_error` JSON on validation/API/auth failures.
- `list` action accepts `start`, `end`, `limit`, `nextToken`, `max_pages`.
- `latest` action uses the documented latest endpoint if confirmed; otherwise it lists with `limit=1` and labels this as derived latest.
- Clamp `limit` to documented max if known, otherwise conservative 25.
- Clamp `max_pages` to 1-10.

**Step 1: Run RED**

```bash
python -m pytest tests/tools/test_whoop_tools.py -q -o 'addopts='
```

---

## Task 6: Implement `plugins/whoop/tools.py` and plugin registration

**Objective:** Register WHOOP tools in Hermes' plugin system.

**Files:**
- Create: `plugins/whoop/tools.py`
- Create: `plugins/whoop/__init__.py`
- Create: `plugins/whoop/plugin.yaml`
- Modify: `toolsets.py` if explicit toolset listing is required by current loader/tests.

**Registration pattern:**
- Copy the Spotify plugin structure:
  - `_check_whoop_available()` calls `get_auth_status("whoop")` and returns `logged_in`.
  - `register(ctx)` registers each tool with `toolset="whoop"`.
  - Emoji optional, keep boring if unsure. This is fitness telemetry, not a disco ball.

**Step 2: Run GREEN**

```bash
python -m pytest tests/tools/test_whoop_tools.py tests/providers/test_plugin_discovery.py -q -o 'addopts='
```

---

## Task 7: Add docs

**Objective:** Tell Graham/future users how to set it up without leaking secrets.

**Files:**
- Create: `website/docs/user-guide/features/whoop.md` or matching feature-doc location.
- Modify: sidebar/index only if required by docs build.

**Docs content:**
- Create WHOOP Developer app.
- Redirect URI: `http://127.0.0.1:43828/whoop/callback`.
- Set credentials locally:

```bash
hermes config env-path
# then edit the local .env or export locally:
export HERMES_WHOOP_CLIENT_ID="..."
export HERMES_WHOOP_CLIENT_SECRET="..."
```

- Run:

```bash
hermes auth whoop
```

- Tool examples:
  - `whoop_profile`
  - `whoop_recovery action=latest`
  - `whoop_sleep action=list start=... end=...`

**Secret-safe warning:** Never paste WHOOP client secret/token into chat.

---

## Task 8: Final targeted verification

**Objective:** Prove the first slice is safe and works locally.

Run:

```bash
python -m pytest \
  tests/hermes_cli/test_whoop_auth.py \
  tests/tools/test_whoop_client.py \
  tests/tools/test_whoop_tools.py \
  -q -o 'addopts='

git diff --stat
git diff --check
```

Expected:
- WHOOP tests pass.
- `git diff --check` clean.
- No unrelated files modified.
- No secrets in diff.

---

## Blast Radius

- **Local-only code branch/worktree:** `/Users/admin/.openclaw/workspace/projects/howard-hermes-migration/repo/hermes-agent-whoop` on `feat/whoop-integration`.
- **No production routing/gateway/cron changes** in first slice.
- **No external sends.** OAuth only opens browser/local callback when Graham runs `hermes auth whoop` himself.
- **Credential risk:** client secret is required by WHOOP; all UX must direct Graham to local `.env`/CLI only. Never chat.
- **Health-data risk:** WHOOP data is personal health telemetry. Treat outputs as private to Graham/Personal Tracking unless explicitly authorized.

---

## Completion Criteria

- `hermes auth whoop` exists or a clearly documented `hermes auth` service-provider flow accepts `whoop`.
- WHOOP tools only appear/execute when authenticated.
- Profile/cycle/recovery/sleep/workout reads are available to agents.
- Tests pass for auth, client, tools, and plugin registration.
- Docs include secret-safe setup and redirect URI.
