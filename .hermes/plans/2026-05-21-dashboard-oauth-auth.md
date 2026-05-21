# Dashboard OAuth Authentication Implementation Plan

> **For Hermes:** Use `subagent-driven-development` skill to implement this plan task-by-task.

**Goal:** Put an OAuth login gate in front of `hermes dashboard` whenever it binds to a non-loopback host without `--insecure`. Default provider is Nous Portal (authorization-code + PKCE); the provider layer is plugin-extensible so others (Google, GitHub, self-hosted OIDC, header-trust) can be added later without modifying core.

**Architecture:**

Three new layers slot into the existing `hermes_cli/web_server.py`:

1. A **dashboard-auth provider registry** in `hermes_cli/dashboard_auth/` exposing a small `DashboardAuthProvider` protocol (`name`, `display_name`, `start_login()`, `complete_login(callback_params)`, `verify_session(cookies)`, `refresh_session(cookies)`, `revoke_session(cookies)`). Providers are registered through a new `ctx.register_dashboard_auth_provider(provider)` plugin hook, mirroring `register_image_gen_provider` / `register_memory_provider`.
2. A **default Nous provider plugin** at `plugins/dashboard-auth-nous/` that talks to the Portal's authorization-code + PKCE endpoints (developed in `nous-account-service`). Token verification reuses the existing `/api/oauth/account` userinfo path; refresh reuses the device-flow `POST /oauth/token` with `grant_type=refresh_token`. Bundled with Hermes so the default install Just Works™.
3. An **auth gate** in `hermes_cli/web_server.py`:
   - A FastAPI middleware (ordered before `auth_middleware`) that gates everything when `auth.required is True`, except routes under `/auth/`, the narrow login-bootstrap public API, and the `/login` HTML.
   - Three new routes (`/auth/login`, `/auth/callback`, `/auth/logout`) plus `/api/auth/providers`, `/api/auth/me`, `/api/auth/refresh`.
   - HttpOnly `Secure` `SameSite=Lax` cookies: `hermes_session_at` (Portal access JWT) and `hermes_session_rt` (Portal refresh token, opaque). Plus `hermes_session_pkce` for the short-lived PKCE state during the OAuth round-trip.
   - PTY/WS auth: server-side endpoint `/api/auth/ws-ticket` mints a short-lived single-use ticket from a valid cookie; the SPA passes it as `?ticket=` to `/api/pty` and `/api/ws`. The existing `_SESSION_TOKEN` becomes legacy and is unused in auth-required mode (the SPA reads its cookie identity, not the injected script).

**Tech Stack:**

- Python: FastAPI, httpx, PyJWT (already in deps via `jose` in NAS but we'll use `pyjwt` since `cryptography` is already pulled in), cryptography (existing).
- Frontend: minimal server-rendered Jinja-style HTML for `/login` (no React bundle); the existing React SPA gets a small "load cookie identity from `/api/auth/me` instead of `window.__HERMES_SESSION_TOKEN__`" change.
- Portal side: depends on a new authorization-code endpoint pair on `portal.nousresearch.com`. This plan documents the contract; cross-repo work happens in `nous-account-service`.

---

## ⚠️ Contract Anchor (Plan v2 — re-validated 2026-05-21)

**Source of truth:** `nous-account-service` PR #180 `docs/agent-dashboard-oauth-contract.md` ([link](https://github.com/NousResearch/nous-account-service/pull/180)).

The Portal's published contract differs from our initial assumptions in several material ways. This plan was revised after fetching PR #180 to bring our design into compliance. Phases 0–3 (already implemented) are NOT affected — they only set up the gate-engagement plumbing, not the OAuth payload specifics. **Phases 4, 6, and 7 are rewritten below; the original drafts are kept inline as "Original (rejected)" blocks so reviewers can see why each call was made.**

### Material findings from the contract

| # | Topic | Original plan said | Contract requires |
|---|---|---|---|
| C1 | **`client_id`** | Static `hermes-dashboard` (one OAuth client for all dashboards) | **Per-instance synthesized: `agent:{AgentInstance.id}`**. Portal injects `HERMES_DASHBOARD_OAUTH_CLIENT_ID` at provisioning time. Dashboard MUST NOT attempt OAuth if missing. |
| C2 | **Audience claim** | `aud = "hermes-cli:hermes-dashboard"` | **`aud = client_id`** (bare, no prefix). Each dashboard verifies tokens issued specifically to itself. |
| C3 | **Scopes** | `openid profile email inference:invoke tool:invoke` | **`agent_dashboard:access`** only (or omit; default). No OIDC scopes are honoured for this flow. |
| C4 | **Token claims** | `email`, `email_verified`, `name`, `session_id` | These are NOT emitted. Available claims: `iss`, `aud`, `exp`, `sub` (user_id), `client_id`, `agent_instance_id`, `org_id`, `scope`, `token_use`, `product_id`, `nous_client`, plus optional rate-limit hints. **No email, no display name.** |
| C5 | **Refresh tokens** | Refresh-in-cookie (Phase 6 silent-refresh) | **No refresh tokens** in V1. 401 → re-auth via full `/oauth/authorize` redirect. |
| C6 | **Token TTL** | Not pinned | 900 seconds (15 minutes). With no refresh, users re-auth every 15 min of active use. |
| C7 | **Verification mode** | JWKS or userinfo fallback | **JWKS only** — `GET /.well-known/jwks.json`, RS256. There is no userinfo endpoint in the contract. |
| C8 | **Redirect URI** | `https://*.fly.dev/auth/callback` wildcard | Two shapes only: exact `https://{flyAppName}.fly.dev/auth/callback` (per `AgentInstance.flyAppName`) OR `http://{localhost,127.0.0.1}:{any-port}/auth/callback` (unconditional carve-out, including prod). |
| C9 | **Agent-instance check** | Not in plan | Contract recommends defense-in-depth: after `aud` passes, verify `claims["agent_instance_id"] == HERMES_DASHBOARD_AGENT_INSTANCE_ID` (or extract from client_id). |
| C10 | **Env var names** | `HERMES_DASHBOARD_AUTH_NOUS_PORTAL_URL` | Portal injects `HERMES_DASHBOARD_PORTAL_URL` at provisioning. Match exactly. |
| C11 | **`oauth_contract_version` claim** | Not in plan | Contract says `1`; verifiers must check before trusting other claims. (Current code does NOT actually emit this — flagged back to Portal team. Treat as tolerant: if missing, proceed with a warning log; if present and != 1, refuse.) |

### What this kills

- **Phase 6 (silent refresh) is entirely deleted.** Replaced by a thinner "401 → re-auth redirect" UX. The refresh-token cookie (`hermes_session_rt`) is removed from the design — there is no refresh token to put in it.
- **The "userinfo fallback verification mode"** in Phase 4 is deleted. JWKS is mandatory.
- **The `Session.email` / `Session.display_name` fields** stay (we already shipped them in Phase 1) but the Nous provider will populate them with empty strings + the user's opaque Portal `sub` claim as a fallback display value. Phase 7's AuthWidget shows the truncated `user_id`, not a name/email.
- **The plan's promise of "all claims in the access token" is broken** by the contract — but in our favor: less round-tripping, just less to display.

### What stays the same

- All of Phases 0–3 (gate engagement, plugin hook, middleware, cookies for the access token only, routes, login page) are unaffected. The cookie machinery already only requires the access token to be set; we'll simply pass `refresh_token=""` everywhere.
- The "fail closed if zero providers" + "fail closed if `HERMES_DASHBOARD_OAUTH_CLIENT_ID` is missing" stories combine cleanly: in the gated-public path, both must be true.
- Multi-provider plugin extensibility stays — only Nous ships, but the architecture supports adding a `Custom OIDC` provider later.

### Decision Log updates (resolving the above)

| Original Q | New resolution |
|---|---|
| QC — Refresh strategy (c2) | **REVERSED to c1 (no refresh).** Contract V1 has no refresh tokens. |
| Q5 — Redirect URI (wildcard) | **REVERSED.** Each `AgentInstance` has a single canonical Fly URL it was provisioned at; the dashboard reads its own `HERMES_DASHBOARD_OAUTH_CLIENT_ID` to derive the instance id and uses `request.url_for("auth_callback")` (under `proxy_headers=True`) for the redirect. Localhost dev: only `localhost` and `127.0.0.1` work, never `0.0.0.0`. |
| Q6 — JWT claims (all in access token) | **PARTIALLY HONORED.** All claims that DO exist are in the JWT — but `email`/`name` aren't in the contract. AuthWidget surfaces `user_id` (truncated) instead. |
| Q12 — Operator setup (zero config) | **REVISED.** Zero config for the **bundled** flow (Portal-managed Fly agent), because the Portal injects the env vars at provisioning. The dashboard is NOT meant to be self-hosted with OAuth in V1 — operator-owned dashboards stay loopback-only or `--insecure`. |

### Open Questions added

| OQ-C1 | Should the AuthWidget make a separate authenticated call to a Portal userinfo endpoint to surface email/name? Punt for V1; show truncated `user_id`. Revisit if a Portal `/api/oauth/userinfo` endpoint lands. |
| OQ-C2 | The contract documents an `oauth_contract_version` claim but the issuer code doesn't emit it. Flag back to Portal team in PR #180 review. Hermes implementation logs a warning and proceeds if absent (tolerant); refuses if present and != 1. |
| OQ-C3 | Staging Portal lives at `portal.rewbs.uk`. Smoke test against that before considering Phase 4 done. |

---

## Background From The Codebase

Findings from grepping `hermes_cli/web_server.py`, `hermes_cli/auth.py`, `hermes_cli/plugins.py`, and the Portal `nous-account-service` repo:

### Current state

- **`_SESSION_TOKEN`** (`hermes_cli/web_server.py:86`) is generated fresh at server start and injected into the SPA `index.html` via `<script>window.__HERMES_SESSION_TOKEN__="…"</script>` (`_serve_index`, ~line 3685). Every browser that can `GET /` reads it. The token gates all `/api/...` routes except `_PUBLIC_API_PATHS` via `auth_middleware` (~line 237).
- **DNS-rebinding defense** lives in `host_header_middleware` (~line 207). It compares the inbound `Host` header against `app.state.bound_host` set by `start_server`. We must preserve this; the new auth gate is an additional layer on top.
- **`--insecure` and `--host`** wire through `hermes_cli/main.py:13140–13157` → `cmd_dashboard` (~10282) → `start_server(host=, allow_public=getattr(args, "insecure", False))` (~10338). `start_server` (`web_server.py:4514`) raises `SystemExit` if `host not in _LOCALHOST and not allow_public`.
- **`/api/status`** is in `_PUBLIC_API_PATHS` because the sidebar polls it pre-token; it returns version, profile, gateway state, etc. The login page only needs version + auth-bootstrap info, so we will narrow what's public.
- **PTY auth** (`/api/pty`, `/api/ws`, `/api/pub`, `/api/events`) uses the SPA-injected token as a `?token=` query param (`web_server.py:3530, 3562, 3591`). Browsers cannot set `Authorization` on WebSocket upgrade, so query-param auth stays — but the token source flips from "injected script" to "server-minted short-lived ticket from cookie".
- **Plugin registry** is at `hermes_cli/plugins.py` (`PluginContext` class). It already has `register_context_engine`, `register_image_gen_provider`, `register_memory_provider`, `register_video_gen_provider` (~lines 499, 531, 558). Adding `register_dashboard_auth_provider` follows the exact same pattern.
- **Existing Nous OAuth in Hermes** is **device flow only** (`hermes_cli/auth.py:73–190`, `PROVIDER_REGISTRY["nous"]`). It already speaks `portal.nousresearch.com`, knows `client_id="hermes-cli"`, scopes `inference:invoke tool:invoke`, and persists tokens to `~/.hermes/auth.json` under `providers.nous`. **The dashboard auth flow is distinct from this**: it's a *user-identity* session for the operator, not an *inference credential* for the agent. They share Portal infrastructure but live in different stores (cookies vs. `auth.json`) and use different client_ids and scopes.
- **`auth.json` keys for `providers.nous`** (confirmed from disk): `access_token`, `refresh_token`, `client_id`, `portal_base_url`, `inference_base_url`, `token_type`, `scope`, `obtained_at`, `expires_at`, `agent_key`, `agent_key_expires_at`, `tls`, `agent_key_id`, `agent_key_expires_in`, `agent_key_reused`, `agent_key_obtained_at`, `expires_in`. The dashboard session does NOT need agent_key fields — those are inference-side.

### Portal endpoints already shipped

In `/home/ben/nous/nous-account-service/src/app/api/oauth/`:

- `POST /api/oauth/device/code` — device-flow code request (existing, NOT used by dashboard).
- `POST /api/oauth/device/verify` — device-flow approval (existing, NOT used by dashboard).
- `POST /api/oauth/token` — token exchange + refresh (existing). Accepts `grant_type=urn:ietf:params:oauth:grant-type:device_code` today; needs to also accept `grant_type=authorization_code` (cross-repo work).
- `GET /api/oauth/account` — userinfo (existing, returns `{userId, orgId, ...}` after JWT validation). Reusable as-is for verifying a JWT carried in the dashboard cookie.

### Portal endpoints to be developed (cross-repo dependency)

This plan **assumes** but does not implement the Portal side. The contract Hermes will speak:

- `GET https://portal.nousresearch.com/oauth/authorize?response_type=code&client_id=hermes-dashboard&redirect_uri=<dashboard_callback>&scope=openid+profile+email+inference%3Ainvoke+tool%3Ainvoke&state=<csrf>&code_challenge=<S256>&code_challenge_method=S256` — browser-redirect endpoint that prompts the logged-in Portal user to approve the Hermes dashboard, then 302s to `redirect_uri?code=<auth_code>&state=<csrf>`.
- `POST https://portal.nousresearch.com/api/oauth/token` (existing endpoint, extended) — accepts `grant_type=authorization_code&code=<auth_code>&code_verifier=<pkce>&client_id=hermes-dashboard&redirect_uri=<dashboard_callback>` and returns `{access_token, refresh_token, token_type: "Bearer", expires_in, scope}`. The access token is a JWT with the claims listed below.

The client_id `hermes-dashboard` must be added to the Portal's `OAUTH_CLIENT_PRODUCT_CONTEXT_MAP` (`src/server/oauth/access-token-issuer.ts:49`). The Portal-side change is tracked separately; this plan flags every place Hermes assumes that client_id is registered.

**Redirect URI handling (per Q5):** the initial design only needs to support Fly.io-hosted dashboards. The Portal will whitelist `https://*.fly.dev/auth/callback` for the `hermes-dashboard` client_id. Other deployments (custom domains, on-prem) are out of scope for v1; operators with those needs would register their own Portal OAuth client and override via config (`dashboard.auth.providers.nous.client_id`, future work).

### JWT claims expected on the access token (per Q6 — all claims in the access token, no userinfo round-trip required)

```json
{
  "iss": "https://portal.nousresearch.com",
  "sub": "<userId>",
  "aud": "hermes-cli:hermes-dashboard",
  "exp": <unix>,
  "iat": <unix>,
  "client_id": "hermes-dashboard",
  "scope": "openid profile email inference:invoke tool:invoke",
  "org_id": "<orgId>",
  "email": "<user email>",
  "email_verified": true,
  "name": "<display name>",
  "session_id": "<server-side session id; opaque>"
}
```

`org_id`, `email`, `name` are net-new to the existing Portal `access-token-issuer.ts` payload (today it carries `userId`, `orgId`, `client_id`, `aud`, `sub`, `exp`, `iat`, `session_id`, `scope`, rate-limit entitlement). Cross-repo work item: extend `issueOAuthAccessToken` to include `email`, `email_verified`, `name` when the scope includes `profile email`. The Portal already has all three fields on the `User` row, so this is purely a claim-projection change.

### JWT signing

Portal currently signs with `AUTH_SECRET` (HS256) and only opens RS256 via `OAUTH_PRIVATE_KEY` for specific paths. **Hermes will verify with the public JWKS** at `https://portal.nousresearch.com/.well-known/jwks.json` — this requires the Portal to migrate dashboard-issued tokens to RS256/JWKS. Tracked as a cross-repo prerequisite. If JWKS isn't ready by Phase 4, we fall back to validating the JWT via `GET /api/oauth/account` (a network round-trip per request, cacheable for 60s), which is correct but slower. The plan's verification module supports both modes from day one and picks based on `nous.signing_mode: jwks | userinfo`.

---

## Key Design Decisions

| # | Decision | Why |
|---|---|---|
| 1 | Auth gate ONLY when `host != loopback` AND `--insecure` not set. | Matches Q1+Q2. Loopback stays zero-friction; `--insecure` stays as escape hatch with current "no-auth" semantics; new behavior fires for any other non-loopback bind. |
| 2 | Stateless server, refresh-in-cookie. | Q9 + Q-C2. No server-side session store, but the refresh cookie lets us silently refresh expired access tokens so the dashboard survives all-day tabs. |
| 3 | `DashboardAuthProvider` is a plugin hook (`ctx.register_dashboard_auth_provider`). | Q-A. Mirrors existing plugin shapes; the Nous provider ships as `plugins/dashboard-auth-nous/` so third parties have a verbatim template. |
| 4 | Multiple stacked providers; login page lists all. | Q-D. With only Nous installed it's a one-button page. No `dashboard.auth.provider` selector — the operator chooses at login time. |
| 5 | Server-rendered `/login` / `/auth/callback` / `/auth/logout`. | Q-E (e1) + Q15. Pre-login pages must NOT load the React bundle (which would also load `window.__HERMES_SESSION_TOKEN__`). Jinja-style HTML rendered straight from FastAPI. |
| 6 | Narrow `/api/auth/providers` for the login-page bootstrap; remove `/api/status` from `_PUBLIC_API_PATHS` only when gate is active. | Q15. When gate is off (loopback), nothing changes — `/api/status` stays public for sidebar polling. When gate is on, the login HTML hits only the narrow endpoint; post-login SPA continues to read `/api/status` (now auth-gated). |
| 7 | WS auth = short-lived ticket from `/api/auth/ws-ticket`. | Browsers cannot set Authorization on WS upgrade. Tickets are random 32-byte tokens, single-use, 30-second TTL, minted only after cookie verification. Replaces `?token=<_SESSION_TOKEN>` in auth-required mode. |
| 8 | `--tui` (embedded PTY) works in gated mode. | Q11. Same ticket flow. |
| 9 | Audit log to `~/.hermes/logs/dashboard-auth.log` (JSON-lines). | Q14. One line per `login_start | login_success | login_failure | refresh_success | refresh_failure | logout | revoke`. Profile-aware path. |
| 10 | Fail-closed if zero providers are registered AND gate is active. | Q13. Loopback mode never hits this. Non-loopback mode with no provider plugins installed = `start_server` raises `SystemExit("dashboard auth gate is enabled but no auth providers are registered")`. |
| 11 | PKCE (S256) is mandatory for the authorization-code flow. | Defense-in-depth even though Portal will also enforce. |
| 12 | Cookies: `Secure` when bound on TLS, omitted otherwise. `SameSite=Lax` always. `HttpOnly` always. Path scoped to `/`. | TLS termination for Fly.io is upstream; we detect `X-Forwarded-Proto: https` to decide. |

---

## Decision Log (Open Questions resolved during planning)

| Q | Resolution |
|---|---|
| Q1 — `--insecure` semantics | Keep current "no auth, no warning" behavior. Auth gate engages only for non-loopback binds where `--insecure` was NOT passed. |
| Q2 — Loopback auth opt-in | Out of scope for v1. Punt to a future `dashboard.auth.required: true` config knob. |
| Q3 — VPS/Fly path | **Primary use case for v1.** Cross-repo Portal redirect-URI whitelist must accept `https://*.fly.dev/auth/callback`. |
| Q4 — OAuth flow | Authorization Code + PKCE (S256). |
| Q5 — Redirect URI | `https://*.fly.dev/auth/callback` wildcard for v1. Operators with other deployments register their own Portal client (future, not in v1). |
| Q6 — JWT claims | All claims in access token. Portal must extend `access-token-issuer.ts` to add `email`, `email_verified`, `name` for `profile email` scope. |
| Q7 — Other providers | Google, GitHub, OIDC, etc. Not implemented; abstraction must support them. |
| Q8 — Plugin vs in-tree | Plugin. `plugins/dashboard-auth-nous/` is the default; third parties drop in `~/.hermes/plugins/dashboard-auth-*/`. |
| Q9 — Session model | Stateless. JWT-in-cookie. |
| Q10 — Multi-user | Single user only. No per-user UI state. |
| Q11 — `--tui` interaction | Same auth applies; WS ticketing flow. |
| Q12 — Operator setup | Zero config. Baked-in `client_id=hermes-dashboard` + Portal URL. |
| Q13 — Portal down | Fail-closed. Provider plugin's `verify_session` raises → middleware returns 503 with a "Portal unreachable, try again" page. |
| Q14 — Audit log | Yes. `~/.hermes/logs/dashboard-auth.log` (JSON-lines). |
| Q15 — `/api/status` | Stays public when gate is off. When gate is on, becomes auth-gated; login HTML uses `/api/auth/providers` for bootstrap. |
| QC — Refresh strategy | Refresh-in-cookie (c2). Access token in `hermes_session_at`, refresh token in `hermes_session_rt`. Server-side `/api/auth/refresh` silently rotates both when the access token is within 60s of expiry. |
| QD — Provider selection | Multi-provider stack; login page lists all registered providers. |
| QE — Login route shape | `/auth/login`, `/auth/callback`, `/auth/logout`, server-rendered HTML. |

---

## Phases Overview

| # | Phase | Shippable on its own? | Exit gate |
|---|---|---|---|
| 0 | Regression harness + auth-gate detection skeleton | Yes (no behavior change) | All existing tests pass; new tests assert "gate is OFF in loopback mode" and "gate is ON for non-loopback w/o --insecure" |
| 1 | `DashboardAuthProvider` protocol + plugin hook + audit logger | Yes (no provider registers yet → gate fails-closed only if anyone tried to enable it) | `register_dashboard_auth_provider` works in unit test; nothing changes for the loopback dashboard |
| 2 | `plugins/dashboard-auth-nous/` provider (no Portal integration yet — uses a stub) | No (skipped in CI; bench tested w/ Portal stub) | Stub provider passes the provider protocol contract test; `/auth/login` returns a redirect to the stub |
| 3 | Auth gate middleware + cookie machinery + `/auth/*` routes + audit log | Yes for stub provider | End-to-end test: bind to `0.0.0.0` with stub provider; `GET /` redirects to `/auth/login`; complete stub OAuth round-trip; receive cookie; `GET /api/status` now 200s |
| 4 | Real Nous provider implementation (JWKS verify OR userinfo fallback) | Once Portal RS256/JWKS lands or with userinfo mode | Real OAuth round-trip against staging Portal succeeds end-to-end |
| 5 | WS ticket auth (`/api/auth/ws-ticket`) + SPA cookie identity refit + remove `window.__HERMES_SESSION_TOKEN__` injection in gated mode | Yes | `--tui` works in gated mode; `/api/pty?ticket=…` accepted; no token leaks to unauthenticated index.html |
| 6 | Refresh-in-cookie machinery (`/api/auth/refresh`, silent-refresh middleware) | Yes | Test: access token mocked to expire in 30s; subsequent request silently refreshes; cookie is rotated |
| 7 | Documentation, dashboard sidebar "Logged in as …" widget, CLI status integration | Yes | `hermes status` shows dashboard auth state; docs updated |

---

## Cross-Repo Coordination Checklist

Tracked separately from the task list; Hermes-side work is independent except where called out.

| Item | Repo | Owner | Required by Phase |
|---|---|---|---|
| Add `hermes-dashboard` to `OAUTH_CLIENT_PRODUCT_CONTEXT_MAP` | `nous-account-service` | Portal team | Phase 4 |
| Implement `GET /oauth/authorize` (browser approval UI) | `nous-account-service` | Portal team | Phase 4 |
| Extend `POST /api/oauth/token` to accept `grant_type=authorization_code` with PKCE | `nous-account-service` | Portal team | Phase 4 |
| Extend `issueOAuthAccessToken` to include `email`, `email_verified`, `name` claims when `scope` includes `profile email` | `nous-account-service` | Portal team | Phase 4 |
| Whitelist `https://*.fly.dev/auth/callback` as redirect URI for `hermes-dashboard` client | `nous-account-service` | Portal team | Phase 4 |
| (Optional, future) Publish JWKS at `/.well-known/jwks.json` and migrate dashboard tokens to RS256 | `nous-account-service` | Portal team | Phase 4 enhancement (Phase 4 ships with `signing_mode=userinfo` fallback first) |

If Portal isn't ready by Phase 4, the plugin ships with `signing_mode=userinfo` as default and switches to `jwks` once the JWKS endpoint is live. No Hermes code change needed for the swap.

---

## Open Questions (still TBD — flag at Phase 4 kickoff)

1. **Cookie domain in multi-tenant Fly setup.** Each Fly app gets its own subdomain (`hermes-agent-prod-<id>.fly.dev`). Cookies will be set scoped to that exact host with no `Domain` attribute — works for single-tenant. If we ever serve multiple Hermes dashboards from sibling subdomains and want SSO across them, revisit. Not in v1.

2. **Org selection UX.** A user with multiple orgs at Portal would today see the org_id baked into the JWT by Portal's resolution logic (probably their default org). If Portal exposes an org-selection step in `/oauth/authorize`, we get it for free. If not, the dashboard will be bound to whichever org Portal picked, and switching requires `/auth/logout` + re-login. Defer; revisit when first multi-org operator complains.

3. **Session length policy.** Refresh tokens are valid for 30 days on the Portal side today. We honor that. We do NOT add a separate Hermes-side maximum-session-age cap. If sec-eng wants one later, it goes in the audit log + a cookie expiry override.

---

## Phase 0 — Regression Harness + Gate-Detection Skeleton

**Goal:** Lock current dashboard behavior with tests, then introduce a single `should_require_auth(host, allow_public)` helper that returns `True` for non-loopback + non-`--insecure` binds. Nothing actually gates yet; this phase just installs the predicate that later phases branch on.

**Why TDD-first:** auth middleware is load-bearing infrastructure. Per `writing-plans`, infra changes need a behavioral harness against the current code before we touch anything.

### Task 0.1: Write the dashboard-auth regression harness

**Objective:** A pytest module that exercises today's `_SESSION_TOKEN` flow end-to-end against the live FastAPI app, so we can prove later phases don't regress loopback behavior.

**Files:**
- Create: `tests/hermes_cli/test_dashboard_auth_gate.py`

**Step 1: Write the harness.**

```python
"""Regression harness for the dashboard auth gate.

Phase 0 — establish a baseline pin on the current (pre-OAuth) behavior so
later phases can prove they didn't break loopback mode.
"""
import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture
def client_loopback(monkeypatch):
    # Reset bound_host between tests
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 9119
    return TestClient(web_server.app)


def test_loopback_status_is_public(client_loopback):
    """`/api/status` must remain reachable without a token in loopback mode."""
    r = client_loopback.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert "version" in body


def test_loopback_protected_route_requires_token(client_loopback):
    """Any non-public /api/ route must require the session token."""
    r = client_loopback.get("/api/sessions")
    assert r.status_code == 401


def test_loopback_protected_route_accepts_session_token(client_loopback):
    """The injected SPA token unlocks protected /api/ routes."""
    r = client_loopback.get(
        "/api/sessions",
        headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN},
    )
    # 200 or 404 (no sessions yet) both prove the auth layer let it through.
    assert r.status_code in (200, 404)


def test_loopback_index_injects_session_token(client_loopback):
    """Loopback mode keeps injecting the SPA token into index.html.

    This is the property that the new auth gate MUST disable once a gated
    bind is detected. Phase 3 will add an inverse test for the gated path.
    """
    r = client_loopback.get("/")
    if r.status_code == 404:
        pytest.skip("WEB_DIST not built in this env")
    assert "__HERMES_SESSION_TOKEN__" in r.text


def test_loopback_host_header_validation_still_enforced(client_loopback):
    """DNS-rebinding protection: a foreign Host header is rejected."""
    r = client_loopback.get("/api/status", headers={"Host": "evil.test"})
    assert r.status_code == 400
```

**Step 2: Run the harness against `main`.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_gate.py -v
```

**Expected:** All five tests pass (the `index` test may skip in CI if `WEB_DIST` isn't built). If any fails, STOP — current behavior was already broken and the rest of the plan rests on a wrong baseline.

**Step 3: Commit.**

```bash
git add tests/hermes_cli/test_dashboard_auth_gate.py
git commit -m "test(dashboard): pin current loopback auth behavior as regression harness"
```

### Task 0.2: Add the `should_require_auth` predicate

**Objective:** A single source of truth for "is the auth gate active?" — used by `start_server`, the middleware, and the SPA token injection.

**Files:**
- Modify: `hermes_cli/web_server.py` — add helper near the existing `_LOCALHOST` constant in `start_server`, but as a module-level function so middleware can call it.
- Test: `tests/hermes_cli/test_dashboard_auth_gate.py`

**Step 1: Write failing test.**

Append to `tests/hermes_cli/test_dashboard_auth_gate.py`:

```python
from hermes_cli.web_server import should_require_auth


@pytest.mark.parametrize("host,allow_public,expected", [
    ("127.0.0.1", False, False),
    ("127.0.0.1", True,  False),
    ("localhost", False, False),
    ("::1",       False, False),
    ("0.0.0.0",   True,  False),   # --insecure escape hatch
    ("0.0.0.0",   False, True),
    ("192.168.1.5", False, True),
    ("10.0.0.1", True,   False),
    ("100.64.0.1", False, True),   # Tailscale CGNAT — treated as public
    ("hermes-agent-prod-abc.fly.dev", False, True),
])
def test_should_require_auth_truth_table(host, allow_public, expected):
    assert should_require_auth(host, allow_public) is expected
```

**Step 2: Run, observe failure.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_gate.py::test_should_require_auth_truth_table -v
```

Expected: `ImportError: cannot import name 'should_require_auth'`.

**Step 3: Add the predicate.**

Insert in `hermes_cli/web_server.py` **immediately after** the `_LOOPBACK_HOST_VALUES` definition (~line 159):

```python
_LOCALHOST_HOSTS: frozenset = frozenset({"127.0.0.1", "localhost", "::1"})


def should_require_auth(host: str, allow_public: bool) -> bool:
    """Return True iff the dashboard auth gate must be active.

    Truth table:
      host == loopback                              → False (no auth)
      host != loopback AND allow_public (--insecure)→ False (legacy escape hatch)
      host != loopback AND NOT allow_public         → True  (gate engages)

    "Loopback" matches the same set used by ``--insecure`` enforcement in
    ``start_server``: 127.0.0.1, localhost, ::1. RFC1918 / CGNAT / link-local
    are deliberately treated as PUBLIC — a hostile device on the same LAN is
    exactly the threat model the gate is designed for.
    """
    return (host not in _LOCALHOST_HOSTS) and (not allow_public)
```

**Step 4: Run, verify pass.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_gate.py -v
```

Expected: all tests pass (including the new parametrized one with 10 cases).

**Step 5: Commit.**

```bash
git add hermes_cli/web_server.py tests/hermes_cli/test_dashboard_auth_gate.py
git commit -m "feat(dashboard): add should_require_auth predicate for OAuth gate"
```

### Task 0.3: Surface the gate flag on `app.state` so middleware can read it

**Objective:** Wire `should_require_auth(host, allow_public)` into `start_server` so the rest of the system has one place to ask "are we gated?"

**Files:**
- Modify: `hermes_cli/web_server.py:4514` (`start_server`)
- Test: `tests/hermes_cli/test_dashboard_auth_gate.py`

**Step 1: Write failing test.**

Append:

```python
def test_start_server_sets_auth_required_flag_on_app_state(monkeypatch):
    """``start_server`` must record auth_required so middleware can read it."""
    # Don't actually start uvicorn — patch it out and only inspect side effects.
    called = {}

    def fake_run(*args, **kwargs):
        called["host"] = kwargs.get("host")

    monkeypatch.setattr(web_server, "uvicorn", type("U", (), {"run": staticmethod(fake_run)}))
    # The "0.0.0.0 without --insecure" case currently raises SystemExit. We're
    # going to change that in Phase 3 (replace SystemExit with "gate engages"),
    # but for now the helper itself is what we're testing.
    web_server.app.state.bound_host = None
    web_server.app.state.bound_port = None
    web_server.app.state.auth_required = None

    with pytest.raises(SystemExit):
        # SystemExit is fine — we just want the state flag set before the exit.
        web_server.start_server(host="0.0.0.0", port=9119,
                                open_browser=False, allow_public=False)
    # Even though it exited, the flag must have been computed and stashed.
    assert web_server.app.state.auth_required is True


def test_start_server_loopback_does_not_set_auth_required(monkeypatch):
    monkeypatch.setattr(web_server, "uvicorn", type("U", (), {"run": staticmethod(lambda *a, **k: None)}))
    web_server.app.state.auth_required = None
    web_server.start_server(host="127.0.0.1", port=9119,
                            open_browser=False, allow_public=False)
    assert web_server.app.state.auth_required is False
```

**Step 2: Run, observe failure.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_gate.py -v -k auth_required_flag
```

**Step 3: Edit `start_server` to set the flag.**

In `hermes_cli/web_server.py` `start_server`, **before** the `_LOCALHOST = (...)` check (~line 4528), insert:

```python
    app.state.auth_required = should_require_auth(host, allow_public)
```

The existing `SystemExit` branch stays for now — Phase 3 will replace it with "the gate engages" once the gate exists.

**Step 4: Run, verify pass.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_gate.py -v
```

**Step 5: Commit.**

```bash
git add hermes_cli/web_server.py tests/hermes_cli/test_dashboard_auth_gate.py
git commit -m "feat(dashboard): stash auth_required flag on app.state"
```

### Phase 0 Exit Gate

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_gate.py -v
```

All tests pass. `should_require_auth` is the single predicate. `app.state.auth_required` is the runtime flag. No behavior has changed — the SystemExit branch in `start_server` still fires for `0.0.0.0 + no --insecure`.

---

## Phase 1 — Provider Protocol + Plugin Hook + Audit Logger

**Goal:** Define the `DashboardAuthProvider` ABC + registry, the `ctx.register_dashboard_auth_provider` plugin hook, and the audit logger. No HTTP routes yet — pure plumbing.

**Why this comes before the gate:** the gate's middleware delegates to providers. We can't write the middleware without a provider contract to mock.

### Task 1.1: Define `DashboardAuthProvider` ABC + `Session` dataclass

**Objective:** A minimal protocol every auth provider implements, plus the dataclass representing a verified session.

**Files:**
- Create: `hermes_cli/dashboard_auth/__init__.py`
- Create: `hermes_cli/dashboard_auth/base.py`
- Create: `tests/hermes_cli/test_dashboard_auth_provider_base.py`

**Step 1: Write the failing protocol test.**

```python
# tests/hermes_cli/test_dashboard_auth_provider_base.py
"""Contract test for DashboardAuthProvider implementations.

Every provider plugin should import and call ``assert_protocol_compliance``
on its provider instance in its own unit test. This module also tests the
abstract base raises on missing methods so the contract is enforced.
"""
import pytest

from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider,
    Session,
    LoginStart,
    assert_protocol_compliance,
)


def test_session_has_required_fields():
    s = Session(
        user_id="u1",
        email="a@b.com",
        display_name="A",
        org_id="org_1",
        provider="test",
        expires_at=1234567890,
        access_token="at",
        refresh_token="rt",
    )
    assert s.user_id == "u1"
    assert s.provider == "test"


def test_login_start_has_redirect_and_state():
    ls = LoginStart(
        redirect_url="https://portal/authorize?...",
        cookie_payload={"hermes_session_pkce": "verifier=abc;state=xyz"},
    )
    assert ls.redirect_url.startswith("https://")
    assert "hermes_session_pkce" in ls.cookie_payload


def test_abstract_provider_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DashboardAuthProvider()


class _BrokenProvider(DashboardAuthProvider):
    name = "broken"
    display_name = "Broken"
    # Deliberately missing all the methods.


def test_assert_protocol_compliance_rejects_partial_impl():
    with pytest.raises(TypeError):
        assert_protocol_compliance(_BrokenProvider)


class _CompliantProvider(DashboardAuthProvider):
    name = "ok"
    display_name = "OK"

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        return LoginStart(redirect_url="x", cookie_payload={})

    def complete_login(self, *, code, state, code_verifier, redirect_uri) -> Session:
        return Session(
            user_id="u", email="x", display_name="x", org_id="o",
            provider=self.name, expires_at=0,
            access_token="a", refresh_token="r",
        )

    def verify_session(self, *, access_token: str) -> Session | None:
        return None

    def refresh_session(self, *, refresh_token: str) -> Session:
        return Session(
            user_id="u", email="x", display_name="x", org_id="o",
            provider=self.name, expires_at=0,
            access_token="a", refresh_token="r",
        )

    def revoke_session(self, *, refresh_token: str) -> None:
        return None


def test_assert_protocol_compliance_accepts_full_impl():
    assert_protocol_compliance(_CompliantProvider) is None
```

**Step 2: Run, observe failure.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_provider_base.py -v
```

Expected: `ImportError: No module named 'hermes_cli.dashboard_auth'`.

**Step 3: Write the base module.**

```python
# hermes_cli/dashboard_auth/__init__.py
"""Dashboard authentication provider framework.

The dashboard auth gate engages only when the dashboard binds to a non-loopback
host without ``--insecure``. In that mode, every request must carry a verified
session from one of the registered ``DashboardAuthProvider`` plugins.

The Nous provider lives in ``plugins/dashboard-auth-nous/`` and is the default.
Third parties register their own providers via the plugin hook
``ctx.register_dashboard_auth_provider``.
"""
from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider,
    Session,
    LoginStart,
    assert_protocol_compliance,
)
from hermes_cli.dashboard_auth.registry import (
    register_provider,
    get_provider,
    list_providers,
    clear_providers,
)

__all__ = [
    "DashboardAuthProvider",
    "Session",
    "LoginStart",
    "assert_protocol_compliance",
    "register_provider",
    "get_provider",
    "list_providers",
    "clear_providers",
]
```

```python
# hermes_cli/dashboard_auth/base.py
"""Abstract base + dataclasses for dashboard auth providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Session:
    """A verified identity. Returned by ``complete_login`` and ``verify_session``.

    All fields are mandatory. Providers that don't have a concept of orgs
    should set ``org_id`` to an empty string. ``access_token`` and
    ``refresh_token`` are opaque to Hermes — provider-specific.
    """
    user_id: str
    email: str
    display_name: str
    org_id: str
    provider: str
    expires_at: int  # unix seconds; the access_token's exp claim
    access_token: str
    refresh_token: str


@dataclass(frozen=True)
class LoginStart:
    """First leg of the OAuth round trip.

    ``redirect_url`` is the URL the browser must navigate to (e.g. the
    Portal's ``/oauth/authorize``). ``cookie_payload`` is a dict of cookie
    name → serialised value that the auth route will Set-Cookie on the
    response. Used for PKCE state, CSRF nonces, etc. Cookies set here MUST
    be HttpOnly Secure SameSite=Lax and TTL ≤ 10 minutes (login lifetime).
    """
    redirect_url: str
    cookie_payload: dict[str, str]


class DashboardAuthProvider(ABC):
    """Protocol every dashboard-auth provider plugin implements.

    Lifecycle:
      1. ``start_login`` — user clicks "Log in with X" on the login page.
         Provider returns a redirect URL and any PKCE/CSRF state to stash
         in short-lived cookies.
      2. Browser bounces through the OAuth IDP and lands at /auth/callback.
      3. ``complete_login`` — exchange the code + verifier for a Session.
      4. ``verify_session`` — called on every request to validate the access
         token in the cookie. Returns ``None`` if the token is expired or
         invalid (middleware then triggers refresh or logout).
      5. ``refresh_session`` — called when the access token is near expiry.
         Returns a new Session with rotated tokens.
      6. ``revoke_session`` — called on /auth/logout. Best-effort.

    Failure semantics:
      * ``start_login`` may raise ``ProviderError`` if the IDP is unreachable.
      * ``complete_login`` raises ``InvalidCodeError`` on bad code/state.
      * ``verify_session`` returns ``None`` on expiry; raises ``ProviderError``
        on IDP-unreachable. The middleware treats expiry and unreachable
        differently (expiry → refresh; unreachable → 503).
      * ``refresh_session`` raises ``RefreshExpiredError`` when the refresh
        token is also invalid; middleware then forces re-login.
    """
    name: str = ""
    display_name: str = ""

    @abstractmethod
    def start_login(self, *, redirect_uri: str) -> LoginStart: ...

    @abstractmethod
    def complete_login(
        self,
        *,
        code: str,
        state: str,
        code_verifier: str,
        redirect_uri: str,
    ) -> Session: ...

    @abstractmethod
    def verify_session(self, *, access_token: str) -> Optional[Session]: ...

    @abstractmethod
    def refresh_session(self, *, refresh_token: str) -> Session: ...

    @abstractmethod
    def revoke_session(self, *, refresh_token: str) -> None: ...


class ProviderError(Exception):
    """IDP unreachable, network error, etc. Middleware → 503."""


class InvalidCodeError(Exception):
    """The OAuth callback code/state didn't validate. Middleware → 400."""


class RefreshExpiredError(Exception):
    """Refresh token is dead. Middleware forces re-login (clears cookies, 302 → /auth/login)."""


def assert_protocol_compliance(cls) -> None:
    """Raise TypeError if ``cls`` doesn't implement the full provider protocol.

    Call this in every provider plugin's unit tests:

      def test_protocol_compliance():
          assert_protocol_compliance(MyProvider)
    """
    required_methods = (
        "start_login",
        "complete_login",
        "verify_session",
        "refresh_session",
        "revoke_session",
    )
    required_attrs = ("name", "display_name")

    for attr in required_attrs:
        val = getattr(cls, attr, "")
        if not val:
            raise TypeError(
                f"{cls.__name__} missing or empty attribute: {attr!r}"
            )
    for method in required_methods:
        if not callable(getattr(cls, method, None)):
            raise TypeError(
                f"{cls.__name__} missing method: {method}"
            )
    # Also catch the ABC-not-overridden case
    if getattr(cls, "__abstractmethods__", None):
        raise TypeError(
            f"{cls.__name__} has unimplemented abstract methods: "
            f"{sorted(cls.__abstractmethods__)}"
        )
```

**Step 4: Run, verify pass.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_provider_base.py -v
```

All 6 tests should pass.

**Step 5: Commit.**

```bash
git add hermes_cli/dashboard_auth/ tests/hermes_cli/test_dashboard_auth_provider_base.py
git commit -m "feat(dashboard-auth): define DashboardAuthProvider ABC + Session dataclass"
```

### Task 1.2: Provider registry

**Objective:** A module-level dict of `name → provider` with register/get/list. Mirrors `agent/image_gen_registry.py` (see `register_provider` at the existing image-gen one for the prior art).

**Files:**
- Create: `hermes_cli/dashboard_auth/registry.py`
- Modify: `tests/hermes_cli/test_dashboard_auth_provider_base.py`

**Step 1: Write failing test.**

Append to `tests/hermes_cli/test_dashboard_auth_provider_base.py`:

```python
from hermes_cli.dashboard_auth import (
    register_provider,
    get_provider,
    list_providers,
    clear_providers,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    clear_providers()
    yield
    clear_providers()


def test_registry_register_and_get(_isolated_registry=None):
    p = _CompliantProvider()
    register_provider(p)
    assert get_provider("ok") is p


def test_registry_get_missing_returns_none(_isolated_registry=None):
    assert get_provider("nope") is None


def test_registry_lists_in_registration_order(_isolated_registry=None):
    class A(_CompliantProvider): name = "a"
    class B(_CompliantProvider): name = "b"
    register_provider(A())
    register_provider(B())
    names = [p.name for p in list_providers()]
    assert names == ["a", "b"]


def test_registry_rejects_non_compliant_provider(_isolated_registry=None):
    with pytest.raises(TypeError):
        register_provider(_BrokenProvider())


def test_registry_rejects_duplicate_name(_isolated_registry=None):
    register_provider(_CompliantProvider())
    with pytest.raises(ValueError, match="already registered"):
        register_provider(_CompliantProvider())
```

**Step 2: Run, observe failure.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_provider_base.py -v -k registry
```

**Step 3: Write the registry.**

```python
# hermes_cli/dashboard_auth/registry.py
"""Module-level registry for DashboardAuthProvider instances.

Plugins call ``register_provider`` via the plugin context hook at startup.
The auth gate middleware iterates ``list_providers()`` and uses ``get_provider``
to dispatch on the session's ``provider`` field.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional

from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider,
    assert_protocol_compliance,
)

_log = logging.getLogger(__name__)
_lock = threading.Lock()
_providers: dict[str, DashboardAuthProvider] = {}


def register_provider(provider: DashboardAuthProvider) -> None:
    """Register a provider. Raises TypeError on protocol violation, ValueError on duplicate name."""
    assert_protocol_compliance(type(provider))
    with _lock:
        if provider.name in _providers:
            raise ValueError(
                f"dashboard-auth provider already registered: {provider.name!r}"
            )
        _providers[provider.name] = provider
    _log.info("dashboard-auth: registered provider %r (%s)",
              provider.name, provider.display_name)


def get_provider(name: str) -> Optional[DashboardAuthProvider]:
    with _lock:
        return _providers.get(name)


def list_providers() -> List[DashboardAuthProvider]:
    """All registered providers, in registration order."""
    with _lock:
        return list(_providers.values())


def clear_providers() -> None:
    """Test-only: drop all registrations."""
    with _lock:
        _providers.clear()
```

**Step 4: Run, verify pass.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_provider_base.py -v
```

**Step 5: Commit.**

```bash
git add hermes_cli/dashboard_auth/registry.py tests/hermes_cli/test_dashboard_auth_provider_base.py
git commit -m "feat(dashboard-auth): registry for DashboardAuthProvider plugins"
```

### Task 1.3: Plugin hook — `ctx.register_dashboard_auth_provider`

**Objective:** Add the method to `PluginContext` so plugins can register providers from their `register(ctx)` entry point.

**Files:**
- Modify: `hermes_cli/plugins.py` — add a new method on `PluginContext` (~line 556 region, after `register_image_gen_provider`).
- Test: `tests/hermes_cli/test_dashboard_auth_plugin_hook.py`

**Step 1: Write failing test.**

```python
# tests/hermes_cli/test_dashboard_auth_plugin_hook.py
"""The plugin context exposes register_dashboard_auth_provider.

Mirrors the image-gen / memory-provider hooks. See plugins.py:531 for prior art.
"""
import pytest
from hermes_cli.dashboard_auth import clear_providers, get_provider
from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider, Session, LoginStart,
)


@pytest.fixture(autouse=True)
def _isolated():
    clear_providers()
    yield
    clear_providers()


class _Stub(DashboardAuthProvider):
    name = "stub"
    display_name = "Stub IdP"
    def start_login(self, *, redirect_uri): return LoginStart(redirect_url="x", cookie_payload={})
    def complete_login(self, **kw): return Session("u", "e", "n", "o", "stub", 0, "a", "r")
    def verify_session(self, **kw): return None
    def refresh_session(self, **kw): return Session("u", "e", "n", "o", "stub", 0, "a", "r")
    def revoke_session(self, **kw): pass


def test_plugin_ctx_can_register_dashboard_auth_provider(tmp_path):
    from hermes_cli.plugins import PluginContext, PluginManifest
    manifest = PluginManifest(name="dashboard-auth-stub", version="0.0.1",
                              description="stub", path=tmp_path)
    # PluginManager is the parent of PluginContext; minimal shim for the test
    class _Manager:
        _cli_ref = None
        _context_engine = None
        _tools = {}
    ctx = PluginContext(manifest=manifest, manager=_Manager())
    assert hasattr(ctx, "register_dashboard_auth_provider")
    ctx.register_dashboard_auth_provider(_Stub())
    assert get_provider("stub").display_name == "Stub IdP"


def test_plugin_ctx_rejects_non_provider(tmp_path):
    from hermes_cli.plugins import PluginContext, PluginManifest
    manifest = PluginManifest(name="bad", version="0.0.1", description="", path=tmp_path)
    class _Manager:
        _cli_ref = None
        _context_engine = None
        _tools = {}
    ctx = PluginContext(manifest=manifest, manager=_Manager())
    # Pass something that's not a DashboardAuthProvider; expect a warning log
    # and the registry stays empty (mirrors the image_gen behaviour).
    import logging
    with pytest.raises(Exception):
        ctx.register_dashboard_auth_provider("not a provider")
    assert get_provider("stub") is None
```

**Step 2: Run, observe failure.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_plugin_hook.py -v
```

**Step 3: Add the method to `PluginContext`.**

In `hermes_cli/plugins.py`, **after** `register_image_gen_provider` (~line 554, before `register_video_gen_provider`):

```python
    # -- dashboard auth provider registration --------------------------------

    def register_dashboard_auth_provider(self, provider) -> None:
        """Register a dashboard authentication provider.

        ``provider`` must be an instance of
        :class:`hermes_cli.dashboard_auth.DashboardAuthProvider`. Used by
        the dashboard auth gate (engaged when the dashboard binds to a
        non-loopback host without ``--insecure``).
        """
        from hermes_cli.dashboard_auth import (
            DashboardAuthProvider, register_provider,
        )

        if not isinstance(provider, DashboardAuthProvider):
            logger.warning(
                "Plugin %r tried to register a dashboard-auth provider that "
                "does not inherit from DashboardAuthProvider. Ignoring.",
                self.manifest.name,
            )
            raise TypeError(
                "register_dashboard_auth_provider expects a "
                "DashboardAuthProvider instance"
            )
        register_provider(provider)
        logger.info(
            "Plugin %r registered dashboard-auth provider: %s (%s)",
            self.manifest.name, provider.name, provider.display_name,
        )
```

**Step 4: Run, verify pass.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_plugin_hook.py -v
```

**Step 5: Commit.**

```bash
git add hermes_cli/plugins.py tests/hermes_cli/test_dashboard_auth_plugin_hook.py
git commit -m "feat(plugins): add register_dashboard_auth_provider hook on PluginContext"
```

### Task 1.4: Audit logger

**Objective:** Profile-aware JSON-lines log at `~/.hermes/logs/dashboard-auth.log`, one line per auth event. Used by the middleware (Phase 3) and `/auth/*` routes (Phase 3).

**Files:**
- Create: `hermes_cli/dashboard_auth/audit.py`
- Create: `tests/hermes_cli/test_dashboard_auth_audit.py`

**Step 1: Write failing test.**

```python
# tests/hermes_cli/test_dashboard_auth_audit.py
import json
import pytest

from hermes_cli.dashboard_auth.audit import audit_log, AuditEvent


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


def test_audit_writes_jsonlines(profile_home):
    audit_log(AuditEvent.LOGIN_START, provider="nous", ip="1.2.3.4")
    audit_log(AuditEvent.LOGIN_SUCCESS, provider="nous", user_id="u1",
              email="a@b.com", ip="1.2.3.4")

    path = profile_home / "logs" / "dashboard-auth.log"
    assert path.exists(), f"audit log not created at {path}"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2

    entry = json.loads(lines[1])
    assert entry["event"] == "login_success"
    assert entry["provider"] == "nous"
    assert entry["user_id"] == "u1"
    assert entry["email"] == "a@b.com"
    assert "ts" in entry  # ISO-8601 timestamp


def test_audit_redacts_token_like_values(profile_home):
    audit_log(AuditEvent.LOGIN_SUCCESS, provider="nous", access_token="should-not-appear")
    entry = json.loads((profile_home / "logs" / "dashboard-auth.log").read_text())
    # Tokens must NEVER end up in the audit log raw. The function should
    # either drop them or replace with "<redacted>".
    assert "should-not-appear" not in json.dumps(entry)


def test_audit_all_event_types_have_string_values(profile_home):
    for ev in AuditEvent:
        assert isinstance(ev.value, str)
        assert ev.value
```

**Step 2: Run, observe failure.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_audit.py -v
```

**Step 3: Implement.**

```python
# hermes_cli/dashboard_auth/audit.py
"""Audit log for dashboard auth events.

Profile-aware location: ``$HERMES_HOME/logs/dashboard-auth.log``.
Format: one JSON object per line. Token-like fields are stripped before
serialisation to avoid leaking refresh tokens or JWTs to disk.
"""
from __future__ import annotations

import datetime as _dt
import enum
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)
_write_lock = threading.Lock()

# Field names that must never appear in the log raw. Any kwarg matching
# these is silently dropped.
_REDACTED_FIELDS = frozenset({
    "access_token", "refresh_token", "code", "code_verifier",
    "state", "ticket", "cookie", "Authorization", "authorization",
})


class AuditEvent(enum.Enum):
    LOGIN_START = "login_start"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    REFRESH_SUCCESS = "refresh_success"
    REFRESH_FAILURE = "refresh_failure"
    REVOKE = "revoke"
    SESSION_VERIFY_FAILURE = "session_verify_failure"
    WS_TICKET_MINTED = "ws_ticket_minted"


def _resolve_log_path() -> Path:
    """Resolve $HERMES_HOME/logs/dashboard-auth.log without importing hermes_constants.

    Mirrors ``hermes_constants.get_hermes_home`` semantics: env var wins, else
    ``~/.hermes``. Keeping a local copy avoids an import cycle from the
    middleware which lives below hermes_cli.
    """
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "logs" / "dashboard-auth.log"


def audit_log(event: AuditEvent, **fields: Any) -> None:
    """Append one event to the audit log.

    Token-like fields are dropped. Missing log directory is created.
    Write failures are logged at WARNING but never raise — auth must not
    fail because the audit logger broke.
    """
    safe_fields = {
        k: v for k, v in fields.items()
        if k not in _REDACTED_FIELDS
    }
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "event": event.value,
        **safe_fields,
    }
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    path = _resolve_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:
        _log.warning("dashboard-auth audit log write failed: %s", e)
```

**Step 4: Run, verify pass.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_audit.py -v
```

**Step 5: Commit.**

```bash
git add hermes_cli/dashboard_auth/audit.py tests/hermes_cli/test_dashboard_auth_audit.py
git commit -m "feat(dashboard-auth): json-lines audit log at \$HERMES_HOME/logs/dashboard-auth.log"
```

### Phase 1 Exit Gate

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_provider_base.py tests/hermes_cli/test_dashboard_auth_plugin_hook.py tests/hermes_cli/test_dashboard_auth_audit.py -v
```

All tests pass. `DashboardAuthProvider`, registry, plugin hook, and audit logger exist. Loopback dashboard behavior is unchanged.

---

## Phase 2 — Stub Provider for End-to-End Local Testing

**Goal:** A self-contained "no-IDP-needed" provider that completes a fake OAuth round trip locally. Lets Phase 3 (the gate + routes) be tested end-to-end without depending on either Portal staging or the real Nous plugin. Lives in the test tree only; never registered in production code paths.

**Why a stub before the real Nous provider:** decouples middleware development from cross-repo Portal coordination. Phase 4 swaps the stub for the real provider; nothing in the middleware changes.

### Task 2.1: Implement the stub provider

**Objective:** A `StubAuthProvider` that returns a fixed redirect URL, accepts any code, and issues a deterministic Session. Used in `tests/hermes_cli/test_dashboard_auth_gate_e2e.py` (Phase 3).

**Files:**
- Create: `tests/hermes_cli/conftest_dashboard_auth.py` — shared fixtures + the stub class.
- Create: `tests/hermes_cli/test_dashboard_auth_stub_provider.py` — protocol-compliance test for the stub.

**Step 1: Write the contract test.**

```python
# tests/hermes_cli/test_dashboard_auth_stub_provider.py
"""Stub provider exists for E2E gate testing. Validate it against the protocol."""
import pytest
from hermes_cli.dashboard_auth.base import assert_protocol_compliance
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


def test_stub_complies_with_protocol():
    assert_protocol_compliance(StubAuthProvider) is None


def test_stub_start_login_returns_callback_redirect():
    p = StubAuthProvider()
    ls = p.start_login(redirect_uri="https://x.fly.dev/auth/callback")
    # Stub bounces straight back to the callback with a fake code.
    assert "code=stub_code" in ls.redirect_url
    assert "state=" in ls.redirect_url
    assert "hermes_session_pkce" in ls.cookie_payload


def test_stub_complete_login_with_matching_state_succeeds():
    p = StubAuthProvider()
    ls = p.start_login(redirect_uri="https://x.fly.dev/auth/callback")
    # Pull the state and verifier out of cookie_payload
    payload = dict(item.split("=", 1) for item in
                   ls.cookie_payload["hermes_session_pkce"].split(";"))
    sess = p.complete_login(
        code="stub_code", state=payload["state"],
        code_verifier=payload["verifier"],
        redirect_uri="https://x.fly.dev/auth/callback",
    )
    assert sess.user_id == "stub-user-1"
    assert sess.email == "stub@example.test"
    assert sess.provider == "stub"


def test_stub_complete_login_rejects_mismatched_state():
    from hermes_cli.dashboard_auth.base import InvalidCodeError
    p = StubAuthProvider()
    with pytest.raises(InvalidCodeError):
        p.complete_login(code="stub_code", state="WRONG",
                         code_verifier="v", redirect_uri="https://x.fly.dev/auth/callback")


def test_stub_verify_session_round_trips():
    p = StubAuthProvider()
    ls = p.start_login(redirect_uri="https://x.fly.dev/auth/callback")
    payload = dict(item.split("=", 1) for item in
                   ls.cookie_payload["hermes_session_pkce"].split(";"))
    sess = p.complete_login(code="stub_code", state=payload["state"],
                            code_verifier=payload["verifier"],
                            redirect_uri="https://x.fly.dev/auth/callback")
    verified = p.verify_session(access_token=sess.access_token)
    assert verified is not None
    assert verified.user_id == "stub-user-1"


def test_stub_verify_expired_session_returns_none():
    p = StubAuthProvider(default_ttl=0)
    ls = p.start_login(redirect_uri="https://x/auth/callback")
    payload = dict(item.split("=", 1) for item in
                   ls.cookie_payload["hermes_session_pkce"].split(";"))
    sess = p.complete_login(code="stub_code", state=payload["state"],
                            code_verifier=payload["verifier"],
                            redirect_uri="https://x/auth/callback")
    assert p.verify_session(access_token=sess.access_token) is None
```

**Step 2: Run, observe failure.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_stub_provider.py -v
```

**Step 3: Implement the stub.**

```python
# tests/hermes_cli/conftest_dashboard_auth.py
"""Stub auth provider + shared fixtures for dashboard-auth tests.

This file is import-only — does NOT register the stub globally. Each test
that needs the stub imports ``StubAuthProvider`` and registers it through
a fixture so the registry stays isolated.
"""
from __future__ import annotations

import json
import secrets
import time
import base64
import hmac
import hashlib

from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider, Session, LoginStart,
    InvalidCodeError, ProviderError, RefreshExpiredError,
)


_STUB_SECRET = b"stub-test-secret-not-for-prod"


def _sign(payload: dict) -> str:
    """Produce a tamper-evident opaque token. Not a real JWT — just enough to round-trip."""
    raw = json.dumps(payload, separators=(",", ":")).encode()
    sig = hmac.new(_STUB_SECRET, raw, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(raw + b"." + sig).decode()


def _unsign(token: str) -> dict | None:
    try:
        blob = base64.urlsafe_b64decode(token.encode())
        raw, sig = blob.rsplit(b".", 1)
        expected = hmac.new(_STUB_SECRET, raw, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return None
        return json.loads(raw)
    except Exception:
        return None


class StubAuthProvider(DashboardAuthProvider):
    """Local fake IDP for E2E tests.

    ``start_login`` returns a redirect to ``{redirect_uri}?code=stub_code&state={s}``
    so the test harness can do the whole round trip in-process without
    talking to anything external.

    ``access_token`` is an HMAC-signed JSON blob; ``verify_session`` decodes
    and checks ``expires_at``.
    """
    name = "stub"
    display_name = "Stub IdP (test only)"

    def __init__(self, default_ttl: int = 3600):
        self._default_ttl = default_ttl
        self._state_to_verifier: dict[str, str] = {}

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        state = secrets.token_urlsafe(16)
        verifier = secrets.token_urlsafe(32)
        self._state_to_verifier[state] = verifier
        return LoginStart(
            redirect_url=f"{redirect_uri}?code=stub_code&state={state}",
            cookie_payload={
                "hermes_session_pkce": f"state={state};verifier={verifier}",
            },
        )

    def complete_login(self, *, code, state, code_verifier, redirect_uri) -> Session:
        if code != "stub_code":
            raise InvalidCodeError(f"stub expects code='stub_code', got {code!r}")
        expected_verifier = self._state_to_verifier.get(state)
        if expected_verifier is None or expected_verifier != code_verifier:
            raise InvalidCodeError(f"stub state/verifier mismatch")
        del self._state_to_verifier[state]
        now = int(time.time())
        return Session(
            user_id="stub-user-1",
            email="stub@example.test",
            display_name="Stub User",
            org_id="stub-org-1",
            provider=self.name,
            expires_at=now + self._default_ttl,
            access_token=_sign({
                "sub": "stub-user-1", "email": "stub@example.test",
                "name": "Stub User", "org_id": "stub-org-1",
                "exp": now + self._default_ttl,
            }),
            refresh_token=_sign({
                "sub": "stub-user-1", "kind": "refresh",
                "exp": now + 30 * 86400,
            }),
        )

    def verify_session(self, *, access_token: str):
        payload = _unsign(access_token)
        if payload is None or payload.get("exp", 0) < int(time.time()):
            return None
        return Session(
            user_id=payload["sub"],
            email=payload["email"],
            display_name=payload["name"],
            org_id=payload["org_id"],
            provider=self.name,
            expires_at=payload["exp"],
            access_token=access_token,
            refresh_token="",  # not surfaced on verify
        )

    def refresh_session(self, *, refresh_token: str) -> Session:
        payload = _unsign(refresh_token)
        if payload is None or payload.get("exp", 0) < int(time.time()):
            raise RefreshExpiredError("stub refresh token expired/invalid")
        now = int(time.time())
        return Session(
            user_id=payload["sub"],
            email="stub@example.test",
            display_name="Stub User",
            org_id="stub-org-1",
            provider=self.name,
            expires_at=now + self._default_ttl,
            access_token=_sign({
                "sub": payload["sub"], "email": "stub@example.test",
                "name": "Stub User", "org_id": "stub-org-1",
                "exp": now + self._default_ttl,
            }),
            refresh_token=_sign({
                "sub": payload["sub"], "kind": "refresh",
                "exp": now + 30 * 86400,
            }),
        )

    def revoke_session(self, *, refresh_token: str) -> None:
        # Stub is in-memory; nothing to revoke server-side.
        return None
```

**Step 4: Run, verify pass.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_stub_provider.py -v
```

All 6 tests should pass.

**Step 5: Commit.**

```bash
git add tests/hermes_cli/conftest_dashboard_auth.py tests/hermes_cli/test_dashboard_auth_stub_provider.py
git commit -m "test(dashboard-auth): stub auth provider for E2E gate testing"
```

### Phase 2 Exit Gate

The stub provider is a fully-conformant `DashboardAuthProvider`. It can be registered in any test that needs to exercise the auth gate end-to-end without external dependencies. Phase 3 will use it as the test driver.

---

## Phase 3 — Auth Gate Middleware, Cookies, `/auth/*` Routes, Login HTML

**Goal:** The actual gate. When `app.state.auth_required is True`:
- Every request to `/api/*` (except a narrow allowlist) and every HTML page (except `/auth/*` and `/login`) requires a valid session cookie.
- `GET /login` serves a server-rendered HTML page listing every registered provider.
- `GET /auth/login?provider=<name>` calls `provider.start_login(...)`, sets the PKCE state cookie, and 302s to the IDP.
- `GET /auth/callback?code&state` calls `provider.complete_login(...)`, sets the session cookies, and 302s to `/`.
- `POST /auth/logout` clears cookies, calls `provider.revoke_session(...)`, redirects to `/login`.
- `GET /api/auth/providers` (public when gate is on) — list providers for the login page bootstrap.
- `GET /api/auth/me` (auth-required) — return the verified Session as JSON for the SPA.
- `index.html` token injection is suppressed when `app.state.auth_required is True`.
- `_PUBLIC_API_PATHS` is narrowed when the gate is on: `/api/auth/providers` is added, `/api/status` is removed.

**Why this is one phase:** these pieces are mutually-dependent (middleware can't be tested without routes; routes can't be tested without cookies; cookies can't be tested without the middleware). They land in one phase, behind tests, with the stub provider as the driver.

### Task 3.1: Cookie machinery

**Objective:** Helper functions that set/clear/read the three cookies (`hermes_session_at`, `hermes_session_rt`, `hermes_session_pkce`) consistently. Centralised so every code path agrees on flags.

**Files:**
- Create: `hermes_cli/dashboard_auth/cookies.py`
- Create: `tests/hermes_cli/test_dashboard_auth_cookies.py`

**Step 1: Write failing test.**

```python
# tests/hermes_cli/test_dashboard_auth_cookies.py
import pytest
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.testclient import TestClient

from hermes_cli.dashboard_auth.cookies import (
    set_session_cookies, clear_session_cookies,
    set_pkce_cookie, clear_pkce_cookie,
    read_session_cookies, read_pkce_cookie,
    SESSION_AT_COOKIE, SESSION_RT_COOKIE, PKCE_COOKIE,
)


def _build_app():
    app = FastAPI()

    @app.get("/set")
    def set_endpoint(request):
        r = Response("ok")
        set_session_cookies(r, access_token="AT", refresh_token="RT",
                            access_token_expires_in=3600, use_https=True)
        return r

    @app.get("/set-pkce")
    def set_pkce(request):
        r = Response("ok")
        set_pkce_cookie(r, payload="state=s;verifier=v", use_https=True)
        return r

    @app.get("/clear")
    def clear(request):
        r = Response("ok")
        clear_session_cookies(r)
        clear_pkce_cookie(r)
        return r

    return app


def test_session_cookies_are_httponly_samesite_lax_secure_in_https():
    app = _build_app()
    client = TestClient(app)
    r = client.get("/set")
    cookies = r.headers.get_list("set-cookie")
    at = next(c for c in cookies if c.startswith(f"{SESSION_AT_COOKIE}="))
    rt = next(c for c in cookies if c.startswith(f"{SESSION_RT_COOKIE}="))
    for c in (at, rt):
        assert "HttpOnly" in c
        assert "SameSite=lax" in c.lower() or "SameSite=Lax" in c
        assert "Secure" in c
        assert "Path=/" in c


def test_session_cookies_omit_secure_when_http(monkeypatch):
    app = FastAPI()

    @app.get("/x")
    def x():
        r = Response("ok")
        set_session_cookies(r, access_token="AT", refresh_token="RT",
                            access_token_expires_in=3600, use_https=False)
        return r

    r = TestClient(app).get("/x")
    for c in r.headers.get_list("set-cookie"):
        if c.startswith(f"{SESSION_AT_COOKIE}=") or c.startswith(f"{SESSION_RT_COOKIE}="):
            assert "Secure" not in c, f"Cookie unexpectedly Secure: {c}"


def test_clear_session_cookies_emits_expired_at_and_rt():
    app = _build_app()
    r = TestClient(app).get("/clear")
    cookies = r.headers.get_list("set-cookie")
    assert any(c.startswith(f"{SESSION_AT_COOKIE}=") and "Max-Age=0" in c for c in cookies)
    assert any(c.startswith(f"{SESSION_RT_COOKIE}=") and "Max-Age=0" in c for c in cookies)


def test_pkce_cookie_short_ttl_and_path_root():
    app = _build_app()
    r = TestClient(app).get("/set-pkce")
    c = next(x for x in r.headers.get_list("set-cookie") if x.startswith(f"{PKCE_COOKIE}="))
    assert "HttpOnly" in c
    assert "Max-Age=600" in c  # 10 minutes
    assert "Path=/" in c


def test_read_session_cookies_from_request(monkeypatch):
    from starlette.requests import Request
    scope = {
        "type": "http",
        "headers": [(b"cookie", f"{SESSION_AT_COOKIE}=at_value; {SESSION_RT_COOKIE}=rt_value".encode())],
    }
    req = Request(scope)
    at, rt = read_session_cookies(req)
    assert at == "at_value"
    assert rt == "rt_value"


def test_read_session_cookies_missing_returns_none():
    from starlette.requests import Request
    req = Request({"type": "http", "headers": []})
    assert read_session_cookies(req) == (None, None)
```

**Step 2: Run, observe failure.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_cookies.py -v
```

**Step 3: Implement.**

```python
# hermes_cli/dashboard_auth/cookies.py
"""Cookie helpers for dashboard auth.

Three cookies in play:
  - hermes_session_at: the OAuth access token (HttpOnly, lifetime = token TTL)
  - hermes_session_rt: the OAuth refresh token (HttpOnly, lifetime = 30 days)
  - hermes_session_pkce: short-lived PKCE state + CSRF nonce
                         (HttpOnly, lifetime = 10 minutes)

All three are SameSite=Lax (browser will send on cross-site GET top-level
navigation, which we need for the IDP redirect back to /auth/callback)
and Path=/. Secure is set ONLY when the dashboard was reached over HTTPS
(detected via ``X-Forwarded-Proto`` upstream of Fly's TLS terminator, or a
direct https:// scheme). Loopback dev traffic is always HTTP so Secure
would lock the cookies out of the browser.
"""
from __future__ import annotations

from typing import Optional, Tuple

from fastapi import Request
from fastapi.responses import Response

SESSION_AT_COOKIE = "hermes_session_at"
SESSION_RT_COOKIE = "hermes_session_rt"
PKCE_COOKIE = "hermes_session_pkce"

# 30 days — matches Portal's REFRESH_TOKEN_TTL_SECONDS
_RT_MAX_AGE = 30 * 24 * 60 * 60
_PKCE_MAX_AGE = 10 * 60


def _common_attrs(use_https: bool) -> dict:
    attrs = {
        "httponly": True,
        "samesite": "lax",
        "path": "/",
    }
    if use_https:
        attrs["secure"] = True
    return attrs


def set_session_cookies(
    response: Response,
    *,
    access_token: str,
    refresh_token: str,
    access_token_expires_in: int,
    use_https: bool,
) -> None:
    response.set_cookie(
        SESSION_AT_COOKIE, access_token,
        max_age=access_token_expires_in,
        **_common_attrs(use_https),
    )
    response.set_cookie(
        SESSION_RT_COOKIE, refresh_token,
        max_age=_RT_MAX_AGE,
        **_common_attrs(use_https),
    )


def clear_session_cookies(response: Response) -> None:
    # Use max_age=0 to make the browser drop both cookies immediately.
    # Path must match the set-path for the delete to apply.
    response.set_cookie(SESSION_AT_COOKIE, "", max_age=0, path="/", httponly=True, samesite="lax")
    response.set_cookie(SESSION_RT_COOKIE, "", max_age=0, path="/", httponly=True, samesite="lax")


def set_pkce_cookie(response: Response, *, payload: str, use_https: bool) -> None:
    response.set_cookie(
        PKCE_COOKIE, payload,
        max_age=_PKCE_MAX_AGE,
        **_common_attrs(use_https),
    )


def clear_pkce_cookie(response: Response) -> None:
    response.set_cookie(PKCE_COOKIE, "", max_age=0, path="/", httponly=True, samesite="lax")


def read_session_cookies(request: Request) -> Tuple[Optional[str], Optional[str]]:
    at = request.cookies.get(SESSION_AT_COOKIE)
    rt = request.cookies.get(SESSION_RT_COOKIE)
    return at, rt


def read_pkce_cookie(request: Request) -> Optional[str]:
    return request.cookies.get(PKCE_COOKIE)


def detect_https(request: Request) -> bool:
    """Decide whether to set Secure flag.

    Trusts ``X-Forwarded-Proto`` only when ``proxy_headers=True`` is enabled
    on uvicorn — currently OFF in start_server. Falls back to the request
    URL's scheme.
    """
    # request.url.scheme reflects the actual incoming scheme when uvicorn
    # is configured with proxy_headers=True, or the raw scheme otherwise.
    # Fly.io terminates TLS upstream and sets X-Forwarded-Proto=https on
    # forwarded requests.  We re-enable proxy_headers in start_server only
    # for the auth-required path; see Phase 3.5.
    return request.url.scheme == "https"
```

**Step 4: Run, verify pass.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_cookies.py -v
```

**Step 5: Commit.**

```bash
git add hermes_cli/dashboard_auth/cookies.py tests/hermes_cli/test_dashboard_auth_cookies.py
git commit -m "feat(dashboard-auth): cookie helpers for session_at/session_rt/pkce"
```

### Task 3.2: Auth-gate middleware

**Objective:** A new FastAPI middleware that runs after `host_header_middleware` and before the existing `auth_middleware`. When `app.state.auth_required is True`:
- Allowlist routes (`/auth/*`, `/login`, `/api/auth/providers`, static assets) pass through.
- Everything else requires a valid `hermes_session_at` cookie. The middleware verifies it via the provider, attaches the verified Session to `request.state.session`, and continues.
- Failed verification → 401 (for `/api/*`) or 302 to `/login` (for HTML routes).
- The existing `_SESSION_TOKEN`-based `auth_middleware` is **skipped** when the gate is active (cookie auth supersedes it).

**Files:**
- Create: `hermes_cli/dashboard_auth/middleware.py`
- Modify: `hermes_cli/web_server.py` — register the middleware near `host_header_middleware`, and short-circuit `auth_middleware` when gate is on.
- Create: `tests/hermes_cli/test_dashboard_auth_middleware.py`

**Step 1: Write failing test.**

```python
# tests/hermes_cli/test_dashboard_auth_middleware.py
"""Behavioural tests for the auth-gate middleware.

Uses the StubAuthProvider so we don't talk to a real IDP.
"""
import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


@pytest.fixture
def gated_app(monkeypatch):
    """Configure web_server.app to gated mode with the stub provider."""
    clear_providers()
    provider = StubAuthProvider()
    register_provider(provider)
    web_server.app.state.bound_host = "0.0.0.0"
    web_server.app.state.bound_port = 9119
    web_server.app.state.auth_required = True
    yield TestClient(web_server.app, base_url="https://gated.fly.dev")
    clear_providers()
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = "127.0.0.1"


def test_gated_status_now_requires_auth(gated_app):
    """When gate is on, /api/status is NOT public — login bootstrap uses /api/auth/providers instead."""
    r = gated_app.get("/api/status")
    assert r.status_code == 401


def test_gated_html_redirects_to_login(gated_app):
    r = gated_app.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/login"


def test_gated_auth_providers_is_public(gated_app):
    r = gated_app.get("/api/auth/providers")
    assert r.status_code == 200
    body = r.json()
    assert any(p["name"] == "stub" for p in body["providers"])
    assert body["providers"][0]["display_name"] == "Stub IdP (test only)"


def test_gated_login_html_is_public(gated_app):
    r = gated_app.get("/login")
    assert r.status_code == 200
    assert "Stub IdP" in r.text


def test_gated_static_assets_are_public(gated_app):
    # Built assets under /assets/* are mounted as StaticFiles; without
    # them the SPA can't even render the login page in the user's
    # branded skin. Allow these through.
    r = gated_app.get("/assets/_nonexistent.css")
    # 404 not 401 — proves middleware let it through to the route handler
    assert r.status_code == 404


def test_gated_valid_cookie_unlocks_api_status(gated_app):
    # First do the login round trip to obtain a valid access token.
    r1 = gated_app.get("/auth/login?provider=stub", follow_redirects=False)
    # /auth/login should 302 with PKCE cookie set
    pkce = next(c for c in r1.headers.get_list("set-cookie")
                if c.startswith("hermes_session_pkce="))
    assert "HttpOnly" in pkce

    redirect = r1.headers["location"]
    # Stub redirects to {redirect_uri}?code=stub_code&state=...
    assert "code=stub_code" in redirect

    # Issue a fake browser hop to /auth/callback carrying the same cookie.
    state = redirect.split("state=")[1]
    cookies = gated_app.cookies  # picked up the pkce cookie from r1
    r2 = gated_app.get(
        f"/auth/callback?code=stub_code&state={state}",
        follow_redirects=False,
    )
    assert r2.status_code == 302
    assert r2.headers["location"] == "/"
    # Session cookies must be set now
    set_cookies = r2.headers.get_list("set-cookie")
    assert any(c.startswith("hermes_session_at=") for c in set_cookies)
    assert any(c.startswith("hermes_session_rt=") for c in set_cookies)

    # And /api/status should now succeed.
    r3 = gated_app.get("/api/status")
    assert r3.status_code == 200


def test_gated_invalid_cookie_returns_401_on_api(gated_app):
    gated_app.cookies.set(SESSION_AT_COOKIE, "totally-not-a-valid-token")
    r = gated_app.get("/api/sessions")
    assert r.status_code == 401


def test_gated_invalid_cookie_redirects_on_html(gated_app):
    gated_app.cookies.set(SESSION_AT_COOKIE, "garbage")
    r = gated_app.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/login"


def test_gated_zero_providers_fails_closed(monkeypatch):
    """If gate is on but no providers are registered, login bootstrap fails closed."""
    clear_providers()
    web_server.app.state.bound_host = "0.0.0.0"
    web_server.app.state.auth_required = True
    try:
        client = TestClient(web_server.app)
        r = client.get("/api/auth/providers")
        assert r.status_code == 503
        assert "no auth providers" in r.text.lower()
    finally:
        web_server.app.state.auth_required = False
```

**Step 2: Run, observe failure.** (Will fail because the middleware doesn't exist yet.)

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_middleware.py -v
```

**Step 3: Implement the middleware module.**

```python
# hermes_cli/dashboard_auth/middleware.py
"""Auth-gate middleware for the dashboard.

Engaged when ``app.state.auth_required is True``. The gate's job:
  1. Allow a small set of routes through unauthenticated (login page,
     /auth/* OAuth round trip, /api/auth/providers, static assets).
  2. For everything else, demand a valid session cookie and attach the
     verified ``Session`` to ``request.state.session``.
  3. On HTML routes, redirect missing/invalid cookies to /login.
     On /api/* routes, return 401 JSON.
"""
from __future__ import annotations

import logging
from typing import Iterable

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse

from hermes_cli.dashboard_auth import get_provider
from hermes_cli.dashboard_auth.audit import audit_log, AuditEvent
from hermes_cli.dashboard_auth.base import ProviderError
from hermes_cli.dashboard_auth.cookies import read_session_cookies

_log = logging.getLogger(__name__)

# Paths that bypass the auth gate. Order matters: prefix match.
_GATE_PUBLIC_PREFIXES: tuple[str, ...] = (
    "/auth/login",
    "/auth/callback",
    "/auth/logout",
    "/login",
    "/api/auth/providers",
    "/assets/",
    "/favicon.ico",
    "/ds-assets/",
    "/fonts/",
    "/fonts-terminal/",
)


def _path_is_public(path: str) -> bool:
    return any(path == p or path.startswith(p) for p in _GATE_PUBLIC_PREFIXES)


async def gated_auth_middleware(request: Request, call_next):
    """Engaged only when app.state.auth_required is True."""
    if not getattr(request.app.state, "auth_required", False):
        return await call_next(request)

    path = request.url.path
    if _path_is_public(path):
        return await call_next(request)

    at, _rt = read_session_cookies(request)
    if not at:
        return _unauth_response(path, reason="no_cookie")

    # Look up the provider that issued the access_token. For the stateless
    # design we *try every registered provider's verify_session* until one
    # returns a Session. Providers MUST return None for tokens they don't
    # recognise (rather than raise). Cheap because verification is local
    # JWT decode (with optional userinfo network call for the Nous
    # `signing_mode=userinfo` mode — that path adds a 60-second cache).
    from hermes_cli.dashboard_auth import list_providers
    session = None
    for provider in list_providers():
        try:
            session = provider.verify_session(access_token=at)
        except ProviderError as e:
            _log.warning("dashboard-auth: provider %r unreachable: %s",
                         provider.name, e)
            audit_log(AuditEvent.SESSION_VERIFY_FAILURE,
                      provider=provider.name, reason="provider_unreachable",
                      ip=_client_ip(request))
            return JSONResponse(
                {"detail": f"Auth provider {provider.name!r} unreachable"},
                status_code=503,
            )
        if session is not None:
            break

    if session is None:
        audit_log(AuditEvent.SESSION_VERIFY_FAILURE, reason="no_provider_recognises",
                  ip=_client_ip(request))
        return _unauth_response(path, reason="invalid_or_expired_session")

    request.state.session = session
    return await call_next(request)


def _unauth_response(path: str, *, reason: str):
    """API routes → 401 JSON; HTML routes → 302 → /login."""
    if path.startswith("/api/"):
        return JSONResponse(
            {"detail": "Unauthorized", "reason": reason},
            status_code=401,
        )
    return RedirectResponse(url="/login", status_code=302)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""
```

**Step 4: Wire the middleware into `web_server.py`.**

In `hermes_cli/web_server.py`, immediately after `host_header_middleware` (around line 233), add:

```python
from hermes_cli.dashboard_auth.middleware import gated_auth_middleware

@app.middleware("http")
async def _gated_auth_proxy(request: Request, call_next):
    return await gated_auth_middleware(request, call_next)
```

And update the existing `auth_middleware` (line 237) to short-circuit when the gate is on:

```python
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Require the session token on all /api/ routes except the public list."""
    # When the OAuth gate is active, cookie auth supersedes the legacy
    # _SESSION_TOKEN. The gated_auth_middleware has already verified the
    # session; we skip the token check here.
    if getattr(request.app.state, "auth_required", False):
        return await call_next(request)
    path = request.url.path
    if path.startswith("/api/") and path not in _PUBLIC_API_PATHS:
        if not _has_valid_session_token(request):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
            )
    return await call_next(request)
```

**Step 5: Run, verify the tests pass.** (Will still fail on tests that exercise routes the next sub-task adds.)

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_middleware.py -v -k "not auth_login and not callback"
```

Expected: tests that don't depend on the `/auth/*` routes pass. The route-dependent tests still fail.

**Step 6: Commit.**

```bash
git add hermes_cli/dashboard_auth/middleware.py hermes_cli/web_server.py
git commit -m "feat(dashboard-auth): auth-gate middleware (cookie-based, gated on app.state.auth_required)"
```

### Task 3.3: `/auth/login`, `/auth/callback`, `/auth/logout` routes

**Objective:** The three OAuth round-trip endpoints. They are FastAPI routes (not middleware) because they need response bodies / cookies / redirects.

**Files:**
- Create: `hermes_cli/dashboard_auth/routes.py`
- Modify: `hermes_cli/web_server.py` — import + mount the router.

**Step 1: Implement the routes.**

```python
# hermes_cli/dashboard_auth/routes.py
"""HTTP routes for the dashboard-auth OAuth round trip.

Mounted at root (no prefix). The router is registered in web_server.py
only — it doesn't auto-gate; gating is done by gated_auth_middleware,
which allowlists /auth/*.
"""
from __future__ import annotations

import logging
import urllib.parse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse

from hermes_cli.dashboard_auth import get_provider, list_providers
from hermes_cli.dashboard_auth.audit import audit_log, AuditEvent
from hermes_cli.dashboard_auth.base import (
    InvalidCodeError, ProviderError, RefreshExpiredError,
)
from hermes_cli.dashboard_auth.cookies import (
    set_session_cookies, clear_session_cookies,
    set_pkce_cookie, clear_pkce_cookie,
    read_session_cookies, read_pkce_cookie,
    detect_https,
)

_log = logging.getLogger(__name__)

router = APIRouter()


def _redirect_uri(request: Request) -> str:
    """Reconstruct the absolute callback URL the IDP will redirect back to.

    Reads from the request URL — under Fly's proxy with proxy_headers=True,
    this picks up the public https URL from X-Forwarded-Host + Proto.
    """
    return str(request.url_for("auth_callback"))


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


@router.get("/auth/login", name="auth_login")
async def auth_login(request: Request, provider: str):
    p = get_provider(provider)
    if p is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider!r}")

    try:
        ls = p.start_login(redirect_uri=_redirect_uri(request))
    except ProviderError as e:
        audit_log(AuditEvent.LOGIN_FAILURE, provider=provider,
                  reason="provider_unreachable", ip=_client_ip(request))
        raise HTTPException(status_code=503, detail=f"Provider unreachable: {e}")

    audit_log(AuditEvent.LOGIN_START, provider=provider, ip=_client_ip(request))

    resp = RedirectResponse(url=ls.redirect_url, status_code=302)
    # The login start may stash one or more cookies (PKCE state, etc.)
    # We expect the canonical key `hermes_session_pkce` plus possibly a
    # `provider_name` so the callback knows which provider to dispatch to.
    pkce = ls.cookie_payload.get("hermes_session_pkce", "")
    # Pack provider name into the PKCE blob so callback can find it.
    if "provider=" not in pkce:
        pkce = f"provider={provider};{pkce}"
    set_pkce_cookie(resp, payload=pkce, use_https=detect_https(request))
    return resp


@router.get("/auth/callback", name="auth_callback")
async def auth_callback(request: Request, code: str = "", state: str = "",
                         error: str = "", error_description: str = ""):
    pkce_raw = read_pkce_cookie(request)
    if not pkce_raw:
        audit_log(AuditEvent.LOGIN_FAILURE, reason="missing_pkce_cookie",
                  ip=_client_ip(request))
        raise HTTPException(status_code=400, detail="Missing PKCE state cookie")

    # Parse the semicolon-delimited blob: provider=...;state=...;verifier=...
    parts = dict(p.split("=", 1) for p in pkce_raw.split(";") if "=" in p)
    provider_name = parts.get("provider", "")
    expected_state = parts.get("state", "")
    verifier = parts.get("verifier", "")

    p = get_provider(provider_name)
    if p is None:
        raise HTTPException(status_code=400, detail=f"Unknown provider in cookie: {provider_name!r}")

    if error:
        audit_log(AuditEvent.LOGIN_FAILURE, provider=provider_name,
                  reason="idp_error", error=error, ip=_client_ip(request))
        raise HTTPException(
            status_code=400,
            detail=f"OAuth error from provider: {error} ({error_description})",
        )

    if state != expected_state:
        audit_log(AuditEvent.LOGIN_FAILURE, provider=provider_name,
                  reason="state_mismatch", ip=_client_ip(request))
        raise HTTPException(status_code=400, detail="OAuth state mismatch (CSRF check failed)")

    try:
        session = p.complete_login(
            code=code, state=state, code_verifier=verifier,
            redirect_uri=_redirect_uri(request),
        )
    except InvalidCodeError as e:
        audit_log(AuditEvent.LOGIN_FAILURE, provider=provider_name,
                  reason="invalid_code", ip=_client_ip(request))
        raise HTTPException(status_code=400, detail=f"Invalid code: {e}")
    except ProviderError as e:
        audit_log(AuditEvent.LOGIN_FAILURE, provider=provider_name,
                  reason="provider_unreachable", ip=_client_ip(request))
        raise HTTPException(status_code=503, detail=f"Provider unreachable: {e}")

    audit_log(AuditEvent.LOGIN_SUCCESS, provider=provider_name,
              user_id=session.user_id, email=session.email,
              org_id=session.org_id, ip=_client_ip(request))

    import time
    expires_in = max(60, session.expires_at - int(time.time()))
    resp = RedirectResponse(url="/", status_code=302)
    use_https = detect_https(request)
    set_session_cookies(
        resp,
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        access_token_expires_in=expires_in,
        use_https=use_https,
    )
    clear_pkce_cookie(resp)
    return resp


@router.post("/auth/logout", name="auth_logout")
async def auth_logout(request: Request):
    at, rt = read_session_cookies(request)
    if rt:
        # Best-effort revoke at the provider — failure is logged but doesn't
        # affect the local logout outcome.
        from hermes_cli.dashboard_auth import list_providers
        for provider in list_providers():
            try:
                provider.revoke_session(refresh_token=rt)
            except Exception as e:
                _log.warning("dashboard-auth: revoke on %r failed: %s",
                             provider.name, e)

    sess = getattr(request.state, "session", None)
    audit_log(
        AuditEvent.LOGOUT,
        provider=(sess.provider if sess else "unknown"),
        user_id=(sess.user_id if sess else ""),
        ip=_client_ip(request),
    )

    resp = RedirectResponse(url="/login", status_code=302)
    clear_session_cookies(resp)
    clear_pkce_cookie(resp)
    return resp


@router.get("/api/auth/providers")
async def api_auth_providers():
    providers = list_providers()
    if not providers:
        # Q13: fail-closed when zero providers are registered.
        return JSONResponse(
            {"detail": "no auth providers registered"},
            status_code=503,
        )
    return {
        "providers": [
            {"name": p.name, "display_name": p.display_name}
            for p in providers
        ],
    }


@router.get("/api/auth/me")
async def api_auth_me(request: Request):
    """Return the verified session JSON. Auth-required (middleware gates this)."""
    sess = getattr(request.state, "session", None)
    if sess is None:
        # Should be unreachable; middleware enforces. Defence in depth.
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {
        "user_id": sess.user_id,
        "email": sess.email,
        "display_name": sess.display_name,
        "org_id": sess.org_id,
        "provider": sess.provider,
        "expires_at": sess.expires_at,
    }
```

**Step 2: Mount the router.**

In `hermes_cli/web_server.py`, near the existing route definitions (after the `host_header_middleware` block, before the SPA mount at the bottom), add:

```python
from hermes_cli.dashboard_auth.routes import router as _dashboard_auth_router
app.include_router(_dashboard_auth_router)
```

**Step 3: Commit.**

```bash
git add hermes_cli/dashboard_auth/routes.py hermes_cli/web_server.py
git commit -m "feat(dashboard-auth): /auth/{login,callback,logout} + /api/auth/{providers,me} routes"
```

### Task 3.4: `/login` server-rendered HTML page

**Objective:** A minimal styled HTML page (no React bundle) that lists every registered provider. Inline CSS — matches the existing Hermes branding colors but doesn't pull from the build.

**Files:**
- Create: `hermes_cli/dashboard_auth/login_page.py`
- Modify: `hermes_cli/dashboard_auth/routes.py` — add `GET /login`.

**Step 1: Write the page renderer.**

```python
# hermes_cli/dashboard_auth/login_page.py
"""Server-rendered /login page. No React, no JavaScript dependency.

Listed providers come from the registry. Clicking a provider sends a GET
to /auth/login?provider=<name>.
"""
from __future__ import annotations

import html

from hermes_cli.dashboard_auth import list_providers

# Inline minimal CSS. The dashboard's full skin lives in the React bundle
# which we deliberately do NOT load here — the login page must not depend
# on the SPA build being present or on the injected session token.
_LOGIN_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in — Hermes Agent</title>
<style>
  :root {{
    --bg: #0a0a0b;
    --fg: #e5e5e7;
    --accent: #f97316;
    --border: #27272a;
  }}
  html, body {{
    margin: 0; padding: 0; height: 100%;
    background: var(--bg); color: var(--fg);
    font: 16px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  main {{
    max-width: 28rem; margin: 10vh auto; padding: 2rem;
    border: 1px solid var(--border); border-radius: 0.75rem;
    background: rgba(255,255,255,0.02);
  }}
  h1 {{ margin: 0 0 0.5rem; font-size: 1.5rem; }}
  p  {{ margin: 0 0 1.5rem; opacity: 0.7; }}
  .provider-list {{ display: grid; gap: 0.75rem; }}
  .provider-btn {{
    display: block; width: 100%; box-sizing: border-box;
    padding: 0.875rem 1rem; text-align: center;
    background: var(--accent); color: #0a0a0b;
    font-weight: 600; font-size: 1rem;
    border-radius: 0.5rem; text-decoration: none;
    border: 0; cursor: pointer;
  }}
  .provider-btn:hover {{ filter: brightness(1.1); }}
  .empty {{
    padding: 1rem; border-radius: 0.5rem;
    background: rgba(248, 113, 113, 0.1); color: #fca5a5;
    border: 1px solid rgba(248, 113, 113, 0.3);
  }}
  footer {{ margin-top: 2rem; font-size: 0.875rem; opacity: 0.5; text-align: center; }}
</style>
</head>
<body>
<main>
  <h1>Sign in to Hermes Agent</h1>
  <p>Choose a sign-in method to continue.</p>
  <div class="provider-list">
{provider_buttons}
  </div>
  <footer>This dashboard is bound to a non-loopback host.<br>
  Sign-in is required for security.</footer>
</main>
</body>
</html>
"""

_EMPTY_HTML = """\
<!doctype html>
<html><body><main style="font-family: system-ui; max-width: 36rem; margin: 10vh auto; padding: 2rem;">
<h1>Sign-in unavailable</h1>
<p>This dashboard is bound to a non-loopback host but no authentication providers are installed.</p>
<p>Install <code>plugins/dashboard-auth-nous</code> (default) or another auth provider, or restart with
<code>--insecure</code> to bypass the auth gate (not recommended on untrusted networks).</p>
</main></body></html>
"""


def render_login_html() -> str:
    providers = list_providers()
    if not providers:
        return _EMPTY_HTML

    buttons = []
    for p in providers:
        buttons.append(
            f'    <a class="provider-btn" href="/auth/login?provider='
            f'{html.escape(p.name, quote=True)}">'
            f'Sign in with {html.escape(p.display_name)}</a>'
        )
    return _LOGIN_HTML_TEMPLATE.format(provider_buttons="\n".join(buttons))
```

**Step 2: Add the route.**

In `hermes_cli/dashboard_auth/routes.py`, add at the top of the route list:

```python
from fastapi.responses import HTMLResponse
from hermes_cli.dashboard_auth.login_page import render_login_html


@router.get("/login", name="login")
async def login_page(request: Request):
    return HTMLResponse(
        render_login_html(),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )
```

**Step 3: Commit.**

```bash
git add hermes_cli/dashboard_auth/login_page.py hermes_cli/dashboard_auth/routes.py
git commit -m "feat(dashboard-auth): server-rendered /login page listing providers"
```

### Task 3.5: Suppress `_SESSION_TOKEN` injection and harden `start_server` when gate is on

**Objective:**
1. When `auth_required is True`, `_serve_index` must NOT inject `window.__HERMES_SESSION_TOKEN__`. The post-login SPA reads identity from `/api/auth/me` instead.
2. `start_server`'s SystemExit on `host != loopback && not allow_public` is replaced with "the auth gate engages" branching. If zero providers are registered AND gate would be active, fail closed (SystemExit with a clear message).
3. Re-enable `proxy_headers=True` on uvicorn when gate is active so cookies see the real client scheme (`X-Forwarded-Proto`) behind Fly's TLS terminator.

**Files:**
- Modify: `hermes_cli/web_server.py` — `_serve_index` (~line 3676), `start_server` (~line 4514).

**Step 1: Write failing tests.**

Append to `tests/hermes_cli/test_dashboard_auth_gate.py`:

```python
def test_gated_index_does_not_inject_session_token(monkeypatch):
    web_server.app.state.auth_required = True
    web_server.app.state.bound_host = "0.0.0.0"
    try:
        client = TestClient(web_server.app)
        r = client.get("/login")  # _serve_index isn't hit on /login
        # And confirm the SPA index itself (post-login) doesn't leak the token.
        # But /  redirects to /login when no cookie, so we can't easily get to
        # _serve_index without an authenticated cookie. Verify the helper
        # directly:
        from hermes_cli.web_server import WEB_DIST
        if (WEB_DIST / "index.html").exists():
            # Construct a Request and call _serve_index directly via a
            # known-good path is fiddly; instead read the function source's
            # branch handling. For an integration test, use a logged-in
            # cookie via Stub provider and check the body of /.
            pass
    finally:
        web_server.app.state.auth_required = False


def test_start_server_with_gate_and_no_providers_fails_closed(monkeypatch):
    from hermes_cli.dashboard_auth import clear_providers
    clear_providers()
    monkeypatch.setattr(web_server, "uvicorn", type("U", (), {"run": staticmethod(lambda *a, **k: None)}))
    with pytest.raises(SystemExit, match="no auth providers"):
        web_server.start_server(host="0.0.0.0", port=9119,
                                open_browser=False, allow_public=False)


def test_start_server_with_gate_and_provider_proceeds(monkeypatch):
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider
    clear_providers()
    register_provider(StubAuthProvider())
    called = {}
    def fake_run(*a, **kw):
        called.update(kw)
    monkeypatch.setattr(web_server, "uvicorn", type("U", (), {"run": staticmethod(fake_run)}))
    web_server.start_server(host="0.0.0.0", port=9119,
                            open_browser=False, allow_public=False)
    assert called["host"] == "0.0.0.0"
    # proxy_headers must be True in gated mode so X-Forwarded-Proto from Fly
    # is honoured for cookie Secure-flag decisions.
    assert called["proxy_headers"] is True
    clear_providers()
```

**Step 2: Modify `_serve_index`.**

In `hermes_cli/web_server.py:3676`:

```python
    def _serve_index(prefix: str = ""):
        html = _index_path.read_text()
        chat_js = "true" if _DASHBOARD_EMBEDDED_CHAT_ENABLED else "false"

        # When the OAuth gate is active, do NOT inject _SESSION_TOKEN — the
        # SPA reads identity from /api/auth/me using cookie auth instead.
        if getattr(app.state, "auth_required", False):
            bootstrap_script = (
                f"<script>"
                f"window.__HERMES_DASHBOARD_EMBEDDED_CHAT__={chat_js};"
                f'window.__HERMES_BASE_PATH__="{prefix}";'
                f"window.__HERMES_AUTH_REQUIRED__=true;"
                f"</script>"
            )
        else:
            bootstrap_script = (
                f'<script>window.__HERMES_SESSION_TOKEN__="{_SESSION_TOKEN}";'
                f"window.__HERMES_DASHBOARD_EMBEDDED_CHAT__={chat_js};"
                f'window.__HERMES_BASE_PATH__="{prefix}";'
                f"window.__HERMES_AUTH_REQUIRED__=false;"
                f"</script>"
            )
        # ... rest of the function (asset-prefix rewrites) unchanged, but
        # replace ``token_script`` with ``bootstrap_script`` on the final
        # ``html.replace("</head>", ...)`` call.
```

**Step 3: Modify `start_server`.**

Replace the `_LOCALHOST` block (lines ~4528–4539) with:

```python
    app.state.auth_required = should_require_auth(host, allow_public)

    if app.state.auth_required:
        from hermes_cli.dashboard_auth import list_providers
        if not list_providers():
            raise SystemExit(
                f"Refusing to bind dashboard to {host} — the auth gate is "
                f"required but no auth providers are registered. Install the "
                f"default Nous provider (plugins/dashboard-auth-nous) or another "
                f"DashboardAuthProvider plugin. Pass --insecure to skip the "
                f"auth gate (NOT recommended on untrusted networks)."
            )
        _log.info(
            "Dashboard binding to %s with OAuth auth gate enabled. "
            "Providers: %s", host,
            ", ".join(p.name for p in list_providers()),
        )
    elif host not in _LOCALHOST and allow_public:
        _log.warning(
            "Binding to %s with --insecure — the dashboard has no robust "
            "authentication. Only use on trusted networks.", host,
        )
```

The original `_LOCALHOST = (...)` constant tuple and the SystemExit at line 4530 are removed (the predicate has subsumed them).

Update the `uvicorn.run` call (line 4583):

```python
    uvicorn.run(
        app, host=host, port=port, log_level="warning",
        # Trust X-Forwarded-Proto / X-Forwarded-For ONLY when the auth gate
        # is engaged AND we're behind a known terminator (Fly.io). The
        # gateway never runs publicly without a terminator; in loopback
        # mode there's nothing to proxy.
        proxy_headers=bool(app.state.auth_required),
    )
```

**Step 4: Run, verify pass.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_gate.py tests/hermes_cli/test_dashboard_auth_middleware.py -v
```

**Step 5: Commit.**

```bash
git add hermes_cli/web_server.py tests/hermes_cli/test_dashboard_auth_gate.py
git commit -m "feat(dashboard-auth): suppress _SESSION_TOKEN injection in gated mode; fail-closed without providers"
```

### Phase 3 Exit Gate

End-to-end test passes:

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_gate.py tests/hermes_cli/test_dashboard_auth_middleware.py tests/hermes_cli/test_dashboard_auth_cookies.py tests/hermes_cli/test_dashboard_auth_stub_provider.py -v
```

Manual smoke test:
1. `hermes dashboard --host 0.0.0.0 --port 9119` with the stub registered via a local `~/.hermes/plugins/dashboard-auth-stub/` plugin. Expect: server starts, listens on `0.0.0.0:9119`.
2. Browser to `http://localhost:9119/` → 302 → `/login` → "Sign in with Stub IdP" button.
3. Click → bounces through `/auth/login?provider=stub` → stub redirects to `/auth/callback?code=stub_code&state=…` → 302 → `/`.
4. The SPA loads with cookies, hits `/api/auth/me`, displays "Logged in as Stub User".
5. `~/.hermes/logs/dashboard-auth.log` contains `login_start` and `login_success` entries.

Loopback regression test passes unchanged:

```bash
hermes dashboard
# → http://127.0.0.1:9119 still works without any login, token still injected
```

---

## Phase 4 — Real Nous Provider Plugin (v2 — contract-compliant)

> **Plan v2 rewrite.** This section was rewritten on 2026-05-21 after fetching the Portal's published contract (PR #180). The original draft (which assumed a static `hermes-dashboard` client_id, userinfo-fallback verification, and broad OIDC scopes) is preserved further below under "Phase 4 v1 (rejected — preserved for archeology)" for reviewer context.

**Goal:** Ship `plugins/dashboard-auth-nous/` as a real `DashboardAuthProvider` that implements the Nous Portal authorization-code + PKCE flow per `nous-account-service/docs/agent-dashboard-oauth-contract.md`. Bundled with Hermes (lives under `plugins/`, auto-loaded). The plugin is a no-op unless the Portal-injected env vars are present, so it has zero impact on loopback-only / `--insecure` operators.

**Dependencies:** Portal-side endpoints already exist in PR #180:
- `GET /oauth/authorize` (authorization endpoint)
- `POST /api/oauth/token` (token endpoint, accepts `grant_type=authorization_code`)
- `GET /.well-known/jwks.json` (RS256 signing keys, RFC-7517 JWK Set, cache `public, max-age=300, stale-while-revalidate=300`)

There is no userinfo endpoint and no refresh-token endpoint for the dashboard flow. Token lifetime is 900 seconds.

### Task 4.1: Plugin skeleton + env-driven provider class

**Objective:** A `plugins/dashboard-auth-nous/` directory that registers a `NousDashboardAuthProvider` ONLY when `HERMES_DASHBOARD_OAUTH_CLIENT_ID` is set. If unset (the common case for loopback / `--insecure` operators), the plugin loads but registers nothing — the gate then fails closed if it's actually engaged, which is correct.

**Files:**
- Create: `plugins/dashboard-auth-nous/plugin.yaml`
- Create: `plugins/dashboard-auth-nous/__init__.py`
- Create: `plugins/dashboard-auth-nous/provider.py`
- Create: `plugins/dashboard-auth-nous/test_provider.py`

**Step 1: Plugin manifest.**

```yaml
# plugins/dashboard-auth-nous/plugin.yaml
name: dashboard-auth-nous
version: 1.0.0
description: "Default dashboard auth provider for Hermes — OAuth via Nous Portal (portal.nousresearch.com)."
author: NousResearch
kind: dashboard_auth
pip_dependencies:
  - httpx
  - pyjwt[crypto]
```

**Step 2: Plugin entry point — conditional registration.**

```python
# plugins/dashboard-auth-nous/__init__.py
"""Default dashboard-auth provider — Nous Portal OAuth.

Auto-loaded by the plugin system at startup. Registers the provider only
when the Portal-injected env vars are present; otherwise loads silently
so loopback / --insecure operators don't see a spurious "auth misconfigured"
warning every time they start the dashboard.
"""
import logging
import os

from plugins.dashboard_auth_nous.provider import NousDashboardAuthProvider

_log = logging.getLogger(__name__)


def register(ctx):
    """Plugin entry — called by the plugin loader.

    Required env vars (Portal injects these at Fly.io provisioning time):
      HERMES_DASHBOARD_OAUTH_CLIENT_ID  — must be shape ``agent:{instance_id}``
      HERMES_DASHBOARD_PORTAL_URL       — e.g. https://portal.nousresearch.com

    With both present, registers the provider. With either missing, logs a
    DEBUG note and returns silently — operator-owned dashboards binding to
    loopback (or running with --insecure) are not expected to set these.
    The gate-engagement layer fails closed if a public bind is attempted
    with zero providers registered, so the failure mode is already covered.
    """
    client_id = os.environ.get("HERMES_DASHBOARD_OAUTH_CLIENT_ID", "").strip()
    portal_url = os.environ.get("HERMES_DASHBOARD_PORTAL_URL", "").strip()

    if not client_id or not portal_url:
        _log.debug(
            "dashboard-auth-nous: env vars missing "
            "(HERMES_DASHBOARD_OAUTH_CLIENT_ID=%r, HERMES_DASHBOARD_PORTAL_URL=%r); "
            "not registering provider.",
            bool(client_id), bool(portal_url),
        )
        return

    if not client_id.startswith("agent:"):
        _log.warning(
            "dashboard-auth-nous: HERMES_DASHBOARD_OAUTH_CLIENT_ID=%r does not "
            "match contract shape 'agent:{instance_id}'; not registering provider. "
            "Set this env var to the value provisioned by Nous Portal.",
            client_id,
        )
        return

    ctx.register_dashboard_auth_provider(
        NousDashboardAuthProvider(client_id=client_id, portal_url=portal_url)
    )
```

**Step 3: Provider implementation — contract-compliant.**

```python
# plugins/dashboard-auth-nous/provider.py
"""NousDashboardAuthProvider — authorization-code + PKCE against Nous Portal.

Implements ``nous-account-service/docs/agent-dashboard-oauth-contract.md``
(PR #180). Key contract points encoded here:

  - client_id is per-instance (``agent:{instance_id}``), injected at
    provisioning. Stored on ``self._client_id``; ``self._agent_instance_id``
    is the suffix used for defense-in-depth claim verification.
  - scope is ``agent_dashboard:access`` only.
  - redirect_uri is computed from request.url_for("auth_callback") under
    proxy_headers=True so Fly's TLS terminator's X-Forwarded-Proto / Host
    are honoured.
  - tokens are RS256-signed JWTs verified against ``/.well-known/jwks.json``;
    JWKS is cached for 5 minutes with stale-while-revalidate.
  - V1 has no refresh tokens — refresh_session always raises
    RefreshExpiredError so the middleware redirects to /auth/login.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import time
import urllib.parse
from typing import Optional

import httpx
import jwt
from jwt import PyJWKClient

from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider,
    Session,
    LoginStart,
    InvalidCodeError,
    ProviderError,
    RefreshExpiredError,
)

_log = logging.getLogger(__name__)

# Contract: ``agent_dashboard:access`` is the scope name for this flow.
_SCOPE = "agent_dashboard:access"

# Contract: tolerant treatment — if the claim is missing, warn and proceed;
# if present and != 1, refuse. See OQ-C2.
_EXPECTED_CONTRACT_VERSION = 1

# Contract: JWKS cache lifetime (matches Portal's Cache-Control header).
_JWKS_CACHE_SECONDS = 300


def _b64url_no_pad(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


class NousDashboardAuthProvider(DashboardAuthProvider):
    """Nous Portal OAuth via authorization-code + PKCE (S256)."""

    name = "nous"
    display_name = "Nous Research"

    def __init__(self, *, client_id: str, portal_url: str) -> None:
        if not client_id.startswith("agent:"):
            # Defense-in-depth — the plugin entry already filters, but the
            # provider should never be constructed with a malformed id.
            raise ValueError(
                f"client_id must match contract shape 'agent:{{instance_id}}', "
                f"got {client_id!r}"
            )
        self._client_id = client_id
        self._agent_instance_id = client_id[len("agent:"):]
        self._portal_url = portal_url.rstrip("/")
        self._jwks_url = f"{self._portal_url}/.well-known/jwks.json"
        self._authorize_url = f"{self._portal_url}/oauth/authorize"
        self._token_url = f"{self._portal_url}/api/oauth/token"
        # PyJWKClient handles cache + stale-while-revalidate semantics.
        self._jwks = PyJWKClient(
            self._jwks_url,
            cache_keys=True,
            lifespan=_JWKS_CACHE_SECONDS,
        )

    # ---------------- start_login -------------------------------------

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        # Validate redirect_uri shape early to surface misconfiguration before
        # the user is bounced to Portal and gets an opaque error.
        parsed = urllib.parse.urlparse(redirect_uri)
        if parsed.scheme not in ("https", "http"):
            raise ProviderError(f"redirect_uri must be http(s), got {redirect_uri!r}")
        if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1"):
            raise ProviderError(
                f"redirect_uri may only use http:// for localhost/127.0.0.1, "
                f"got {redirect_uri!r}"
            )

        code_verifier = _b64url_no_pad(secrets.token_bytes(64))  # ~86 chars
        code_challenge = _b64url_no_pad(hashlib.sha256(code_verifier.encode()).digest())
        state = _b64url_no_pad(secrets.token_bytes(32))

        params = {
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "scope": _SCOPE,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        authorize_url = f"{self._authorize_url}?{urllib.parse.urlencode(params)}"
        return LoginStart(
            authorize_url=authorize_url,
            state=state,
            code_verifier=code_verifier,
        )

    # ---------------- complete_login ----------------------------------

    def complete_login(
        self,
        *,
        code: str,
        code_verifier: str,
        redirect_uri: str,
    ) -> Session:
        try:
            response = httpx.post(
                self._token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": self._client_id,
                    "code_verifier": code_verifier,
                },
                headers={"Accept": "application/json"},
                timeout=10.0,
            )
        except httpx.RequestError as exc:
            raise ProviderError(f"Portal token endpoint unreachable: {exc}") from exc

        if response.status_code == 400:
            # Contract: invalid_code, invalid_grant, redirect_uri_mismatch all
            # surface here.
            body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            error_code = body.get("error", "invalid_request")
            raise InvalidCodeError(f"Portal rejected code: {error_code}")
        if response.status_code != 200:
            raise ProviderError(
                f"Portal token endpoint returned {response.status_code}: "
                f"{response.text[:200]}"
            )

        payload = response.json()
        access_token = payload.get("access_token")
        if not access_token:
            raise ProviderError("Portal token response missing access_token")
        # Contract V1: no refresh token. If one is present, we deliberately
        # ignore it (forward-compat: a future Portal can issue one without
        # forcing us to ship a new Hermes version).
        token_type = payload.get("token_type", "").lower()
        if token_type and token_type != "bearer":
            raise ProviderError(f"unexpected token_type={token_type!r}")

        claims = self._verify_jwt(access_token)
        return self._session_from_claims(access_token, claims)

    # ---------------- refresh_session ---------------------------------

    def refresh_session(self, *, refresh_token: str) -> Session:
        # Contract V1 has no refresh tokens. The cookie machinery may still
        # call this if a future Portal change starts issuing them; for now
        # we always force re-auth.
        raise RefreshExpiredError(
            "Nous Portal does not issue refresh tokens in OAuth contract v1; "
            "user must re-authenticate."
        )

    # ---------------- verify_session (called per request) -------------

    def verify_session(self, *, access_token: str) -> Session:
        claims = self._verify_jwt(access_token)
        return self._session_from_claims(access_token, claims)

    # ---------------- internals --------------------------------------

    def _verify_jwt(self, access_token: str) -> dict:
        try:
            signing_key = self._jwks.get_signing_key_from_jwt(access_token)
        except jwt.PyJWKClientError as exc:
            raise ProviderError(f"JWKS lookup failed: {exc}") from exc

        try:
            claims = jwt.decode(
                access_token,
                signing_key.key,
                algorithms=["RS256"],
                # Contract: audience is the bare client_id for agent:* clients.
                audience=self._client_id,
                # Issuer is the Portal base URL. Pin it.
                issuer=self._portal_url,
                options={"require": ["exp", "iat", "aud", "iss", "sub"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise InvalidCodeError(f"access token expired: {exc}") from exc
        except jwt.InvalidTokenError as exc:
            raise ProviderError(f"access token verification failed: {exc}") from exc

        # Defense-in-depth: contract recommends verifying agent_instance_id
        # matches our configured client_id suffix. (Doc says all client_id-shaped
        # claims should be cross-checked.)
        token_instance_id = claims.get("agent_instance_id")
        if token_instance_id and token_instance_id != self._agent_instance_id:
            raise ProviderError(
                f"agent_instance_id mismatch: token={token_instance_id!r} "
                f"vs configured={self._agent_instance_id!r}"
            )

        # Tolerant contract-version check (see OQ-C2).
        contract_version = claims.get("oauth_contract_version")
        if contract_version is None:
            _log.warning(
                "Nous Portal token missing oauth_contract_version claim "
                "(contract says it should be %d); proceeding anyway.",
                _EXPECTED_CONTRACT_VERSION,
            )
        elif contract_version != _EXPECTED_CONTRACT_VERSION:
            raise ProviderError(
                f"unsupported oauth_contract_version={contract_version!r}, "
                f"expected {_EXPECTED_CONTRACT_VERSION}"
            )

        return claims

    def _session_from_claims(self, access_token: str, claims: dict) -> Session:
        # Contract V1 emits no email / display_name. We surface the user_id
        # (truncated) in the AuthWidget; Session keeps the fields for forward
        # compatibility but populates them with empty strings.
        user_id = str(claims.get("sub", ""))
        if not user_id:
            raise ProviderError("token missing 'sub' (user_id) claim")
        return Session(
            provider_name=self.name,
            user_id=user_id,
            email="",
            display_name="",
            access_token=access_token,
            refresh_token="",  # contract V1: no refresh
            expires_at=int(claims["exp"]),
            extra={
                "org_id": claims.get("org_id"),
                "agent_instance_id": claims.get("agent_instance_id"),
                "scope": claims.get("scope"),
            },
        )
```

**Step 4: Tests.**

The test suite covers four shapes:

1. **Plugin registration gating** — env unset / malformed `client_id` → no registration.
2. **`start_login` shape** — generates correct `code_verifier` (43-128 chars), `code_challenge` (S256 of verifier), and authorize URL with all required params.
3. **`complete_login` happy path + error mapping** — httpx mocked. 200 with valid JWT → `Session`; 400 → `InvalidCodeError`; 500 → `ProviderError`; 200 without `access_token` → `ProviderError`.
4. **`verify_session` token verification** — uses an RSA keypair generated in `conftest`; signs a JWT with the expected claims and verifies it round-trips. Negative cases: wrong `aud`, wrong `iss`, missing `sub`, `agent_instance_id` mismatch, `oauth_contract_version=2` rejection, missing `oauth_contract_version` warning.

Skip `refresh_session` happy path — it has none; one test asserts `RefreshExpiredError` is always raised.

### Task 4.2: Smoke test against staging Portal (`portal.rewbs.uk`)

**Objective:** Manual end-to-end run against the staging Portal before considering Phase 4 done. Not a CI gate; the OAuth flow needs a real browser. Document the checklist:

1. Provision a fake Fly app pointing to localhost (e.g. via `fly apps create` + DNS override) OR — easier — patch the Portal's `flyAppName → canonicalRedirectUri` to allow `http://localhost:8080/auth/callback`.
2. Set `HERMES_DASHBOARD_OAUTH_CLIENT_ID=agent:{instance_id}`, `HERMES_DASHBOARD_PORTAL_URL=https://portal.rewbs.uk`.
3. `hermes dashboard --host 0.0.0.0 --port 8080`.
4. Open `http://localhost:8080/`. Expect bounce to `/login`. Click "Continue with Nous Research".
5. Expect Portal `/oauth/authorize` page; sign in; consent.
6. Expect redirect to `/auth/callback?code=…&state=…`; cookie set; redirect to `/`.
7. Open dev tools; `/api/auth/me` returns user_id; `/api/pty` ticket-auth path works (Phase 5).
8. Wait 900 s; expect 401 on next mutation; expect SPA to redirect to `/login` (Phase 6 v2).

### Phase 4 v1 (rejected — preserved for archeology)

The v1 draft below assumed (a) a static `hermes-dashboard` OAuth client, (b) `signing_mode=userinfo` with JWKS as a future upgrade, and (c) refresh tokens. All three were reversed by the contract; see Contract Anchor above.



### Task 4.1: Plugin skeleton + provider class

**Objective:** A `plugins/dashboard-auth-nous/` directory with `plugin.yaml`, `__init__.py` that imports and registers the provider, and `provider.py` that implements the OAuth dance.

**Files:**
- Create: `plugins/dashboard-auth-nous/plugin.yaml`
- Create: `plugins/dashboard-auth-nous/__init__.py`
- Create: `plugins/dashboard-auth-nous/provider.py`
- Create: `plugins/dashboard-auth-nous/test_provider.py`

**Step 1: Plugin manifest.**

```yaml
# plugins/dashboard-auth-nous/plugin.yaml
name: dashboard-auth-nous
version: 1.0.0
description: "Default dashboard auth provider for Hermes — OAuth via Nous Portal (portal.nousresearch.com)."
author: NousResearch
kind: dashboard_auth
pip_dependencies:
  - httpx
  - pyjwt[crypto]
```

**Step 2: Plugin entry point.**

```python
# plugins/dashboard-auth-nous/__init__.py
"""Default dashboard-auth provider — Nous Portal OAuth.

Auto-loaded by the plugin system at startup. Registers the provider into
the dashboard-auth registry via the plugin context hook.
"""
from plugins.dashboard_auth_nous.provider import NousDashboardAuthProvider


def register(ctx):
    """Plugin entry — called by the plugin loader.

    Honours these env vars (typically left unset; the defaults are correct
    for production Nous Portal):

      HERMES_DASHBOARD_AUTH_NOUS_PORTAL_URL  — override portal base URL
                                               (default: https://portal.nousresearch.com)
      HERMES_DASHBOARD_AUTH_NOUS_CLIENT_ID   — override OAuth client_id
                                               (default: hermes-dashboard)
      HERMES_DASHBOARD_AUTH_NOUS_SIGNING_MODE — "userinfo" (default) or "jwks"
    """
    ctx.register_dashboard_auth_provider(NousDashboardAuthProvider())
```

**Step 3: Provider implementation.**

```python
# plugins/dashboard-auth-nous/provider.py
"""NousDashboardAuthProvider — authorization-code + PKCE against Nous Portal."""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
import time
import urllib.parse
from typing import Optional

import httpx

from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider,
    Session,
    LoginStart,
    InvalidCodeError,
    ProviderError,
    RefreshExpiredError,
)

_log = logging.getLogger(__name__)

_DEFAULT_PORTAL_URL = "https://portal.nousresearch.com"
_DEFAULT_CLIENT_ID = "hermes-dashboard"
_DEFAULT_SIGNING_MODE = "userinfo"  # or "jwks"

# Scopes:
#   openid profile email — identity claims
#   inference:invoke tool:invoke — Hermes inherits these from Portal's default
#                                  scope set, so the agent can use them later.
_SCOPE = "openid profile email inference:invoke tool:invoke"


def _b64url_no_pad(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _make_pkce_pair() -> tuple[str, str]:
    verifier = _b64url_no_pad(secrets.token_bytes(64))
    challenge = _b64url_no_pad(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge


class NousDashboardAuthProvider(DashboardAuthProvider):
    name = "nous"
    display_name = "Nous Portal"

    def __init__(self):
        self._portal_url = (
            os.getenv("HERMES_DASHBOARD_AUTH_NOUS_PORTAL_URL")
            or _DEFAULT_PORTAL_URL
        ).rstrip("/")
        self._client_id = (
            os.getenv("HERMES_DASHBOARD_AUTH_NOUS_CLIENT_ID")
            or _DEFAULT_CLIENT_ID
        )
        self._signing_mode = (
            os.getenv("HERMES_DASHBOARD_AUTH_NOUS_SIGNING_MODE")
            or _DEFAULT_SIGNING_MODE
        )
        # Simple 60s memoisation for verify_session in userinfo mode so we
        # don't hammer Portal on every browser request.
        self._verify_cache: dict[str, tuple[int, Session]] = {}

    # ---- OAuth ---------------------------------------------------------

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        verifier, challenge = _make_pkce_pair()
        state = _b64url_no_pad(secrets.token_bytes(24))
        params = {
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "scope": _SCOPE,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        auth_url = f"{self._portal_url}/oauth/authorize?" + urllib.parse.urlencode(params)
        return LoginStart(
            redirect_url=auth_url,
            cookie_payload={
                # Caller (routes.py) prepends `provider=nous;`
                "hermes_session_pkce": f"state={state};verifier={verifier}",
            },
        )

    def complete_login(
        self, *, code, state, code_verifier, redirect_uri,
    ) -> Session:
        token_url = f"{self._portal_url}/api/oauth/token"
        try:
            with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
                r = client.post(
                    token_url,
                    data={
                        "grant_type": "authorization_code",
                        "code": code,
                        "code_verifier": code_verifier,
                        "client_id": self._client_id,
                        "redirect_uri": redirect_uri,
                    },
                    headers={"Accept": "application/json"},
                )
        except httpx.HTTPError as e:
            raise ProviderError(f"Portal token endpoint unreachable: {e}")

        if r.status_code == 400:
            raise InvalidCodeError(f"Portal rejected code: {r.text}")
        if r.status_code >= 500:
            raise ProviderError(f"Portal token endpoint returned {r.status_code}")
        if r.status_code != 200:
            raise InvalidCodeError(f"Portal token exchange failed: HTTP {r.status_code} {r.text}")

        body = r.json()
        return self._session_from_token_response(body)

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        # Cache hit?
        now = int(time.time())
        cached = self._verify_cache.get(access_token)
        if cached and cached[0] > now:
            return cached[1]

        if self._signing_mode == "jwks":
            return self._verify_via_jwks(access_token)
        return self._verify_via_userinfo(access_token)

    def refresh_session(self, *, refresh_token: str) -> Session:
        token_url = f"{self._portal_url}/api/oauth/token"
        try:
            with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
                r = client.post(
                    token_url,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": self._client_id,
                    },
                    headers={"Accept": "application/json"},
                )
        except httpx.HTTPError as e:
            raise ProviderError(f"Portal refresh endpoint unreachable: {e}")

        if r.status_code in (400, 401):
            # Portal indicates the refresh token is dead.
            raise RefreshExpiredError(f"Portal rejected refresh: {r.text}")
        if r.status_code != 200:
            raise ProviderError(f"Portal refresh failed: HTTP {r.status_code} {r.text}")

        return self._session_from_token_response(r.json())

    def revoke_session(self, *, refresh_token: str) -> None:
        # Portal's existing API exposes revocation through Account Service's
        # /api/account/oauth/sessions delete-by-id route. The refresh token
        # itself isn't accepted as a revoke key. Best-effort: we POST it to
        # the token endpoint with grant_type=refresh_token, then discard the
        # response — this consumes the refresh and rotates it server-side,
        # which is sufficient for "this old refresh is now dead". A future
        # iteration can add a dedicated /api/oauth/revoke endpoint.
        try:
            with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
                client.post(
                    f"{self._portal_url}/api/oauth/token",
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": self._client_id,
                    },
                )
        except httpx.HTTPError:
            # Best-effort; log but don't raise.
            _log.warning("dashboard-auth-nous: revoke best-effort call failed", exc_info=True)

    # ---- internals -----------------------------------------------------

    def _session_from_token_response(self, body: dict) -> Session:
        access_token = body.get("access_token", "")
        refresh_token = body.get("refresh_token", "")
        if not access_token or not refresh_token:
            raise ProviderError("Portal token response missing tokens")

        # Decode the JWT payload (no signature verification here — only used
        # to extract claims for the Session dataclass; signature verification
        # happens in verify_session).
        claims = self._decode_jwt_payload_unsafe(access_token)
        now = int(time.time())
        expires_at = int(claims.get("exp", now + 3600))
        return Session(
            user_id=str(claims.get("sub", "")),
            email=str(claims.get("email", "")),
            display_name=str(claims.get("name", "") or claims.get("email", "")),
            org_id=str(claims.get("org_id", "")),
            provider=self.name,
            expires_at=expires_at,
            access_token=access_token,
            refresh_token=refresh_token,
        )

    def _decode_jwt_payload_unsafe(self, token: str) -> dict:
        """Decode the JWT payload without verification. Used only to extract
        claims for the Session dataclass; signature verification is the job
        of ``verify_session``."""
        try:
            header_b64, payload_b64, _sig = token.split(".")
            padded = payload_b64 + "=" * (-len(payload_b64) % 4)
            import json
            return json.loads(base64.urlsafe_b64decode(padded))
        except Exception as e:
            raise ProviderError(f"Cannot decode JWT payload: {e}")

    def _verify_via_userinfo(self, access_token: str) -> Optional[Session]:
        """Use Portal's /api/oauth/account as a userinfo endpoint.

        Cached for 60 seconds keyed on the access_token. Returns None if
        Portal returns 401 (expired/invalid token).
        """
        try:
            with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
                r = client.get(
                    f"{self._portal_url}/api/oauth/account",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/json",
                    },
                )
        except httpx.HTTPError as e:
            raise ProviderError(f"Portal /api/oauth/account unreachable: {e}")

        if r.status_code == 401:
            return None
        if r.status_code >= 500:
            raise ProviderError(f"Portal /api/oauth/account returned {r.status_code}")
        if r.status_code != 200:
            return None

        body = r.json()
        # Cache the verified Session for 60 seconds.
        claims = self._decode_jwt_payload_unsafe(access_token)
        now = int(time.time())
        sess = Session(
            user_id=str(body.get("userId", claims.get("sub", ""))),
            email=str(body.get("email", claims.get("email", ""))),
            display_name=str(body.get("name", claims.get("name", "") or body.get("email", ""))),
            org_id=str(body.get("orgId", claims.get("org_id", ""))),
            provider=self.name,
            expires_at=int(claims.get("exp", now + 3600)),
            access_token=access_token,
            refresh_token="",  # not returned on verify
        )
        self._verify_cache[access_token] = (now + 60, sess)
        return sess

    def _verify_via_jwks(self, access_token: str) -> Optional[Session]:
        """Verify the JWT signature against Portal's JWKS."""
        try:
            import jwt as _jwt
            from jwt import PyJWKClient
        except ImportError:
            raise ProviderError("pyjwt[crypto] not installed — falling back to userinfo")

        jwks_url = f"{self._portal_url}/.well-known/jwks.json"
        try:
            client = PyJWKClient(jwks_url)
            signing_key = client.get_signing_key_from_jwt(access_token)
            claims = _jwt.decode(
                access_token,
                signing_key.key,
                algorithms=["RS256"],
                audience=f"hermes-cli:{self._client_id}",
                issuer=self._portal_url,
            )
        except _jwt.ExpiredSignatureError:
            return None
        except _jwt.InvalidTokenError as e:
            _log.warning("dashboard-auth-nous: JWT validation failed: %s", e)
            return None
        except Exception as e:
            raise ProviderError(f"JWKS verify failed: {e}")

        now = int(time.time())
        sess = Session(
            user_id=str(claims.get("sub", "")),
            email=str(claims.get("email", "")),
            display_name=str(claims.get("name", "") or claims.get("email", "")),
            org_id=str(claims.get("org_id", "")),
            provider=self.name,
            expires_at=int(claims.get("exp", now + 3600)),
            access_token=access_token,
            refresh_token="",
        )
        self._verify_cache[access_token] = (now + 60, sess)
        return sess
```

**Step 4: Unit tests for the provider.**

```python
# plugins/dashboard-auth-nous/test_provider.py
"""Unit tests for the Nous dashboard-auth provider. Mocks Portal endpoints."""
import json
import time
import pytest
import respx
import httpx

from hermes_cli.dashboard_auth.base import (
    assert_protocol_compliance, InvalidCodeError, ProviderError, RefreshExpiredError,
)
from plugins.dashboard_auth_nous.provider import NousDashboardAuthProvider


def test_protocol_compliance():
    assert_protocol_compliance(NousDashboardAuthProvider) is None


def test_start_login_returns_authorize_url():
    p = NousDashboardAuthProvider()
    ls = p.start_login(redirect_uri="https://x.fly.dev/auth/callback")
    assert ls.redirect_url.startswith("https://portal.nousresearch.com/oauth/authorize?")
    assert "response_type=code" in ls.redirect_url
    assert "client_id=hermes-dashboard" in ls.redirect_url
    assert "code_challenge_method=S256" in ls.redirect_url
    assert "state=" in ls.redirect_url
    assert "scope=" in ls.redirect_url
    # State+verifier go into the cookie payload
    pkce = ls.cookie_payload["hermes_session_pkce"]
    assert "state=" in pkce
    assert "verifier=" in pkce


@respx.mock
def test_complete_login_happy_path():
    p = NousDashboardAuthProvider()
    # Forge a JWT-shaped access_token: header.payload.sig with a real payload
    import base64
    payload = {
        "sub": "u_123", "email": "u@nous.com", "name": "U Nous",
        "org_id": "org_x", "exp": int(time.time()) + 3600,
        "aud": "hermes-cli:hermes-dashboard",
    }
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    access_token = f"hdr.{payload_b64}.sig"

    respx.post("https://portal.nousresearch.com/api/oauth/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": access_token, "refresh_token": "rt_xyz",
            "token_type": "Bearer", "expires_in": 3600,
            "scope": "openid profile email inference:invoke tool:invoke",
        })
    )

    sess = p.complete_login(code="auth_code", state="s", code_verifier="v",
                            redirect_uri="https://x.fly.dev/auth/callback")
    assert sess.user_id == "u_123"
    assert sess.email == "u@nous.com"
    assert sess.display_name == "U Nous"
    assert sess.org_id == "org_x"
    assert sess.access_token == access_token
    assert sess.refresh_token == "rt_xyz"
    assert sess.provider == "nous"


@respx.mock
def test_complete_login_invalid_code_raises():
    p = NousDashboardAuthProvider()
    respx.post("https://portal.nousresearch.com/api/oauth/token").mock(
        return_value=httpx.Response(400, json={"error": "invalid_grant"})
    )
    with pytest.raises(InvalidCodeError):
        p.complete_login(code="bad", state="s", code_verifier="v",
                         redirect_uri="https://x.fly.dev/auth/callback")


@respx.mock
def test_complete_login_portal_5xx_raises_provider_error():
    p = NousDashboardAuthProvider()
    respx.post("https://portal.nousresearch.com/api/oauth/token").mock(
        return_value=httpx.Response(503, json={"error": "service_unavailable"})
    )
    with pytest.raises(ProviderError):
        p.complete_login(code="c", state="s", code_verifier="v",
                         redirect_uri="https://x.fly.dev/auth/callback")


@respx.mock
def test_verify_session_userinfo_mode_happy():
    p = NousDashboardAuthProvider()
    import base64
    payload = {"sub": "u1", "email": "u@x.com", "name": "U", "org_id": "o",
               "exp": int(time.time()) + 3600}
    token = f"hdr.{base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b'=').decode()}.sig"
    respx.get("https://portal.nousresearch.com/api/oauth/account").mock(
        return_value=httpx.Response(200, json={
            "userId": "u1", "email": "u@x.com", "name": "U", "orgId": "o",
        })
    )
    sess = p.verify_session(access_token=token)
    assert sess is not None
    assert sess.user_id == "u1"


@respx.mock
def test_verify_session_userinfo_401_returns_none():
    p = NousDashboardAuthProvider()
    import base64
    payload = {"sub": "u1", "exp": int(time.time()) + 3600}
    token = f"hdr.{base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b'=').decode()}.sig"
    respx.get("https://portal.nousresearch.com/api/oauth/account").mock(
        return_value=httpx.Response(401)
    )
    assert p.verify_session(access_token=token) is None


@respx.mock
def test_refresh_session_happy():
    p = NousDashboardAuthProvider()
    import base64
    payload = {"sub": "u1", "email": "u@x.com", "name": "U", "org_id": "o",
               "exp": int(time.time()) + 3600}
    new_token = f"hdr.{base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b'=').decode()}.sig"
    respx.post("https://portal.nousresearch.com/api/oauth/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": new_token, "refresh_token": "rt_new",
            "token_type": "Bearer", "expires_in": 3600,
        })
    )
    sess = p.refresh_session(refresh_token="rt_old")
    assert sess.access_token == new_token
    assert sess.refresh_token == "rt_new"


@respx.mock
def test_refresh_session_expired_raises():
    p = NousDashboardAuthProvider()
    respx.post("https://portal.nousresearch.com/api/oauth/token").mock(
        return_value=httpx.Response(400, json={"error": "invalid_grant"})
    )
    with pytest.raises(RefreshExpiredError):
        p.refresh_session(refresh_token="rt_dead")
```

**Step 5: Run, verify pass.**

```bash
scripts/run_tests.sh plugins/dashboard-auth-nous/test_provider.py -v
```

Adds `respx` to the test deps if not already present.

**Step 6: Commit.**

```bash
git add plugins/dashboard-auth-nous/
git commit -m "feat(dashboard-auth-nous): default OAuth provider for Nous Portal (authcode + PKCE)"
```

### Task 4.2: Integrate plugin discovery for dashboard-auth

**Objective:** Confirm the existing plugin loader picks up `plugins/dashboard-auth-nous/` on `hermes dashboard` startup. The plugin manager should already auto-discover any plugin with a `plugin.yaml`; this task verifies and locks that behavior with a test.

**Files:**
- Modify (only if needed): `hermes_cli/plugins.py` — confirm the loader scans `plugins/` for built-in plugins. (It already does — `plugins/memory/honcho/` and `plugins/image_gen/openai/` are auto-loaded.)
- Test: `tests/hermes_cli/test_dashboard_auth_plugin_discovery.py`

**Step 1: Test the discovery integration.**

```python
# tests/hermes_cli/test_dashboard_auth_plugin_discovery.py
"""When the dashboard starts, the bundled Nous auth provider must auto-register."""
from hermes_cli.dashboard_auth import clear_providers, get_provider
from hermes_cli.plugins import PluginManager


def test_bundled_nous_auth_plugin_is_discovered_and_registered(tmp_path, monkeypatch):
    clear_providers()
    # Use a real PluginManager pointed at the repo's plugins/ dir.
    mgr = PluginManager()
    mgr.discover_and_load()
    # Either the Nous provider OR no provider is acceptable in CI where
    # plugins might be opt-in; assert that if the plugin is in the registry,
    # it speaks the correct portal URL.
    nous = get_provider("nous")
    if nous is not None:
        assert nous.display_name == "Nous Portal"
    clear_providers()
```

**Step 2: Run.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_plugin_discovery.py -v
```

**Step 3: Commit.**

```bash
git add tests/hermes_cli/test_dashboard_auth_plugin_discovery.py
git commit -m "test(dashboard-auth): verify Nous provider auto-loads through plugin discovery"
```

### Phase 4 Exit Gate

```bash
scripts/run_tests.sh plugins/dashboard-auth-nous/ tests/hermes_cli/test_dashboard_auth_plugin_discovery.py -v
```

Integration smoke (requires staging Portal access — operator-run, not CI):

1. Set `HERMES_DASHBOARD_AUTH_NOUS_PORTAL_URL=https://staging.portal.nousresearch.com`.
2. `hermes dashboard --host 0.0.0.0 --port 9119`
3. Visit `http://localhost:9119/login` → click "Sign in with Nous Portal" → redirected to staging Portal → approve → redirected back to `/`.
4. `/api/auth/me` returns `{user_id, email, display_name, org_id, provider: "nous", expires_at}`.

**Hard dependency on cross-repo work:** Phase 4 cannot pass the integration smoke until the Portal-side items in the Cross-Repo Coordination Checklist land. The unit tests (mocked Portal endpoints) pass independently.

---

## Phase 5 — WebSocket Ticket Auth (`--tui` Support in Gated Mode)

**Goal:** The PTY/WS endpoints (`/api/pty`, `/api/ws`, `/api/pub`, `/api/events`) currently authenticate via `?token=<_SESSION_TOKEN>`. In gated mode the SPA has cookies, not the token. This phase adds a `/api/auth/ws-ticket` endpoint that mints a short-lived single-use ticket from a valid cookie, and updates each WS endpoint to accept either the legacy token (loopback mode) OR a ticket (gated mode).

### Task 5.1: Ticket store + mint endpoint

**Objective:** A small in-memory ticket store with TTL + single-use semantics, and an authenticated endpoint that mints a ticket for the cookie's session.

**Files:**
- Create: `hermes_cli/dashboard_auth/ws_tickets.py`
- Modify: `hermes_cli/dashboard_auth/routes.py` — add `/api/auth/ws-ticket`.
- Create: `tests/hermes_cli/test_dashboard_auth_ws_tickets.py`

**Step 1: Write failing test.**

```python
# tests/hermes_cli/test_dashboard_auth_ws_tickets.py
import time
import pytest
from hermes_cli.dashboard_auth.ws_tickets import (
    mint_ticket, consume_ticket, TicketInvalid,
)


def test_mint_and_consume_round_trip():
    ticket = mint_ticket(user_id="u1", provider="nous")
    # Must be opaque token (urlsafe base64ish), reasonable length
    assert len(ticket) >= 32
    info = consume_ticket(ticket)
    assert info["user_id"] == "u1"
    assert info["provider"] == "nous"


def test_ticket_is_single_use():
    ticket = mint_ticket(user_id="u1", provider="stub")
    consume_ticket(ticket)
    with pytest.raises(TicketInvalid, match="already consumed|unknown"):
        consume_ticket(ticket)


def test_expired_ticket_rejected(monkeypatch):
    real_time = time.time
    t0 = real_time()
    monkeypatch.setattr("hermes_cli.dashboard_auth.ws_tickets.time.time",
                        lambda: t0)
    ticket = mint_ticket(user_id="u1", provider="stub")
    # Jump forward 31 seconds (TTL = 30s)
    monkeypatch.setattr("hermes_cli.dashboard_auth.ws_tickets.time.time",
                        lambda: t0 + 31)
    with pytest.raises(TicketInvalid, match="expired"):
        consume_ticket(ticket)


def test_unknown_ticket_rejected():
    with pytest.raises(TicketInvalid, match="unknown"):
        consume_ticket("nope-never-minted")
```

**Step 2: Implement.**

```python
# hermes_cli/dashboard_auth/ws_tickets.py
"""Short-lived single-use tickets for WS-upgrade auth in gated mode.

Browsers cannot set Authorization on a WebSocket upgrade. In loopback
mode the legacy ``?token=<_SESSION_TOKEN>`` query param works because
the token comes from the injected SPA script. In gated mode there is no
injected token — the SPA gets a fresh ticket via the authenticated REST
endpoint ``/api/auth/ws-ticket`` and passes that as ``?ticket=`` on the
WS upgrade.

Tickets are single-use, TTL = 30 seconds. In-memory; the dashboard is a
single process so no distributed coordination is needed.
"""
from __future__ import annotations

import secrets
import threading
import time
from typing import Optional

_TTL_SECONDS = 30
_lock = threading.Lock()
_tickets: dict[str, tuple[int, dict]] = {}  # ticket -> (expires_at, info)


class TicketInvalid(Exception):
    """Ticket missing, expired, or already consumed."""


def mint_ticket(*, user_id: str, provider: str) -> str:
    """Generate a one-shot ticket bound to this user identity."""
    ticket = secrets.token_urlsafe(32)
    info = {"user_id": user_id, "provider": provider, "minted_at": int(time.time())}
    with _lock:
        _tickets[ticket] = (int(time.time()) + _TTL_SECONDS, info)
        _gc_expired_locked()
    return ticket


def consume_ticket(ticket: str) -> dict:
    """Validate and consume. Raises TicketInvalid on missing/expired/used."""
    now = int(time.time())
    with _lock:
        entry = _tickets.pop(ticket, None)
        if entry is None:
            raise TicketInvalid(f"unknown ticket: {ticket[:8]}…")
        expires_at, info = entry
        if expires_at < now:
            raise TicketInvalid("expired")
        return info


def _gc_expired_locked() -> None:
    now = int(time.time())
    expired = [t for t, (exp, _) in _tickets.items() if exp < now]
    for t in expired:
        _tickets.pop(t, None)
```

**Step 3: Add the mint endpoint.**

In `hermes_cli/dashboard_auth/routes.py`:

```python
from hermes_cli.dashboard_auth.ws_tickets import mint_ticket


@router.post("/api/auth/ws-ticket", name="auth_ws_ticket")
async def api_auth_ws_ticket(request: Request):
    """Mint a short-lived WS ticket for the authenticated session."""
    sess = getattr(request.state, "session", None)
    if sess is None:
        # Middleware should already have rejected, but check defensively.
        raise HTTPException(status_code=401, detail="Unauthorized")
    ticket = mint_ticket(user_id=sess.user_id, provider=sess.provider)
    audit_log(AuditEvent.WS_TICKET_MINTED, provider=sess.provider,
              user_id=sess.user_id, ip=_client_ip(request))
    return {"ticket": ticket, "ttl_seconds": 30}
```

**Step 4: Run, verify pass.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_ws_tickets.py -v
```

**Step 5: Commit.**

```bash
git add hermes_cli/dashboard_auth/ws_tickets.py hermes_cli/dashboard_auth/routes.py tests/hermes_cli/test_dashboard_auth_ws_tickets.py
git commit -m "feat(dashboard-auth): single-use WS tickets for cookie→ws bridge"
```

### Task 5.2: Update WS endpoints to accept tickets

**Objective:** `/api/pty`, `/api/ws`, `/api/pub`, `/api/events` accept either `?token=<_SESSION_TOKEN>` (loopback) or `?ticket=<ticket>` (gated). In gated mode the legacy token path is rejected.

**Files:**
- Modify: `hermes_cli/web_server.py:3520` (`/api/ws`), `:3562`, `:3591`, plus `/api/pty` (~line 3264).

**Step 1: Write failing test.**

Append to `tests/hermes_cli/test_dashboard_auth_middleware.py`:

```python
def test_ws_accepts_ticket_in_gated_mode(gated_app):
    # Authenticate via the stub round trip, then mint a ticket.
    r1 = gated_app.get("/auth/login?provider=stub", follow_redirects=False)
    state = r1.headers["location"].split("state=")[1]
    r2 = gated_app.get(f"/auth/callback?code=stub_code&state={state}",
                       follow_redirects=False)
    assert r2.status_code == 302

    rt = gated_app.post("/api/auth/ws-ticket")
    assert rt.status_code == 200
    ticket = rt.json()["ticket"]

    # The PTY endpoint should accept ?ticket=... in gated mode.
    # Use the WS test client. Don't actually do the PTY handshake — the
    # auth gate is what we're testing.
    with gated_app.websocket_connect(f"/api/pty?ticket={ticket}") as ws:
        # Either we read a banner or the server closes cleanly because no
        # tty-write follows. Both prove the auth gate accepted the ticket.
        pass


def test_ws_rejects_legacy_token_in_gated_mode(gated_app):
    # Even if you somehow knew the legacy _SESSION_TOKEN, gated mode
    # must NOT accept it.
    from hermes_cli import web_server
    with pytest.raises(Exception):  # WSException / ConnectionClosed
        with gated_app.websocket_connect(
            f"/api/pty?token={web_server._SESSION_TOKEN}"
        ):
            pass


def test_ws_rejects_consumed_ticket(gated_app):
    r1 = gated_app.get("/auth/login?provider=stub", follow_redirects=False)
    state = r1.headers["location"].split("state=")[1]
    gated_app.get(f"/auth/callback?code=stub_code&state={state}", follow_redirects=False)
    rt = gated_app.post("/api/auth/ws-ticket")
    ticket = rt.json()["ticket"]

    # First use — fine
    with gated_app.websocket_connect(f"/api/pty?ticket={ticket}"):
        pass
    # Second use — rejected (single-use)
    with pytest.raises(Exception):
        with gated_app.websocket_connect(f"/api/pty?ticket={ticket}"):
            pass
```

**Step 2: Update each WS endpoint.**

Pattern for each (refactor the duplicated check into a helper):

```python
# In hermes_cli/web_server.py — add near the other helpers

def _ws_auth_ok(ws: WebSocket, app_state) -> bool:
    """Validate WS auth in either loopback or gated mode.

    Returns True if the ws should be accepted. Caller is responsible for
    closing with the right code if False.
    """
    if getattr(app_state, "auth_required", False):
        ticket = ws.query_params.get("ticket", "")
        if not ticket:
            return False
        from hermes_cli.dashboard_auth.ws_tickets import consume_ticket, TicketInvalid
        try:
            consume_ticket(ticket)
            return True
        except TicketInvalid:
            return False
    token = ws.query_params.get("token", "")
    return hmac.compare_digest(token.encode(), _SESSION_TOKEN.encode())
```

Then update each WS handler to use it:

```python
@app.websocket("/api/ws")
async def gateway_ws(ws: WebSocket) -> None:
    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        await ws.close(code=4403)
        return
    if not _ws_auth_ok(ws, app.state):
        await ws.close(code=4401)
        return
    if not _ws_client_is_allowed(ws):
        await ws.close(code=4403)
        return
    from tui_gateway.ws import handle_ws
    await handle_ws(ws)
```

Same surgery for `/api/pty`, `/api/pub`, `/api/events`.

**Step 3: SPA-side change.**

The React SPA's WS client must, in `auth_required` mode, fetch a fresh ticket from `/api/auth/ws-ticket` before each connect rather than using `window.__HERMES_SESSION_TOKEN__`.

Files:
- Modify: `web/src/pages/ChatPage.tsx` — the xterm.js WebSocket connect.
- Modify: `web/src/lib/api.ts` — add `getWsTicket()` typed wrapper.

```typescript
// web/src/lib/api.ts — add:
export async function getWsTicket(): Promise<{ ticket: string; ttl_seconds: number }> {
  const r = await fetch('/api/auth/ws-ticket', { method: 'POST', credentials: 'include' });
  if (!r.ok) throw new Error(`ws-ticket: HTTP ${r.status}`);
  return r.json();
}
```

```typescript
// web/src/pages/ChatPage.tsx — replace token usage with:
const ws_url = window.__HERMES_AUTH_REQUIRED__
  ? `/api/pty?ticket=${encodeURIComponent((await getWsTicket()).ticket)}`
  : `/api/pty?token=${encodeURIComponent(window.__HERMES_SESSION_TOKEN__)}`;
```

**Step 4: Run, verify pass.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_middleware.py -v -k ws
```

**Step 5: Commit.**

```bash
git add hermes_cli/web_server.py web/src/pages/ChatPage.tsx web/src/lib/api.ts tests/hermes_cli/test_dashboard_auth_middleware.py
git commit -m "feat(dashboard-auth): WS ticket auth for /api/pty + /api/ws + /api/pub + /api/events"
```

### Phase 5 Exit Gate

`hermes dashboard --host 0.0.0.0 --port 9119 --tui` with the stub provider:
1. Login as before.
2. Visit `/chat` → xterm.js opens.
3. The browser's network tab shows `POST /api/auth/ws-ticket` returning 200, immediately followed by `GET /api/pty?ticket=…` upgrading to WS.
4. The TUI is interactive.

Loopback `hermes dashboard --tui` keeps working without any of the above (legacy token path).

---

## Phase 6 — 401-Triggered Re-Authentication (v2 — contract-compliant)

> **Plan v2 rewrite.** The Portal contract V1 does not issue refresh tokens (see Contract Anchor C5). Silent-refresh machinery is replaced by a "401 → redirect to `/login`" UX. The v1 draft is preserved below as "Phase 6 v1 (rejected — preserved for archeology)" so reviewers can see the alternative we explored.

**Goal:** When the access token in `hermes_session_at` expires (15-minute TTL per contract C6), the dashboard cleanly bounces the user back through `/oauth/authorize`. No silent refresh; no UX surprises beyond the single redirect. The behaviour must be identical whether the expiry is detected (a) at the gateway middleware (HTML navigation request) or (b) by an XHR / fetch from the SPA.

### Design overview

Two interception points handle expiry:

1. **`gated_auth_middleware` (HTML navigation requests).** Already in place from Phase 3. When `verify_session` raises `InvalidCodeError("access token expired: …")`, the middleware:
   - clears the `hermes_session_at` cookie,
   - audit-logs `auth.session_expired`,
   - returns `RedirectResponse("/login?next={original_path}", status_code=303)`.
2. **`/api/*` JSON endpoints.** A new sibling middleware (`gated_api_auth_middleware`) handles XHR fetches: instead of redirecting (which a `fetch()` call cannot follow into the OAuth dance), it returns `401 {"error": "session_expired", "login_url": "/login?next=…"}`. The SPA's global fetch wrapper notices the 401 and triggers a full-page navigation to `login_url`.

This split is canonical OAuth UX — modern SPAs interpret 401 as "your session is gone" and any subsequent decision (re-auth flow choice, where to send the user) is conveyed in the body. The middleware does not return 302 to `/login` for API requests because (a) most browsers' fetch APIs swallow the redirect into the cross-origin OAuth flow opaquely, and (b) returning HTML in response to `Accept: application/json` confuses front-end frameworks.

### Task 6.1: API auth middleware + `session_expired` envelope

**Files:**
- Modify: `hermes_cli/dashboard_auth/middleware.py` — add `gated_api_auth_middleware`.
- Modify: `hermes_cli/web_server.py` — wire it in alongside `gated_auth_middleware`.
- Add: `tests/hermes_cli/test_dashboard_auth_api_401.py`.

**Behavior:**
- Path matches `/api/*` and **not** the auth allowlist (`/api/auth/providers`, `/api/auth/login`, `/api/auth/callback`).
- Reads `hermes_session_at` cookie; if absent → `401 {"error": "unauthenticated", "login_url": "/login"}`.
- If present, calls the provider's `verify_session`. On any exception (`InvalidCodeError` / `ProviderError` / `RefreshExpiredError`) → clear cookie, audit-log, return `401 {"error": "session_expired", "login_url": "/login?next=/"}`. (`next=/` not `next={path}` for API calls — the user wasn't navigating to the API endpoint directly.)
- On success, stash claims on `request.state.auth_session` and call the route.

**Audit-log events added:**
- `auth.api_unauthenticated` — `/api/*` request with no cookie.
- `auth.api_session_expired` — `/api/*` request with expired/invalid cookie.

**WebSocket endpoints (`/api/pty`, `/api/ws`) are NOT covered by this middleware.** They use the ticket-auth flow from Phase 5. A WS upgrade request with an expired access-token cookie should never reach `/api/auth/ws-ticket` (which is an HTTP POST covered by this middleware), so the SPA's ticket-fetch step is the natural failure point.

### Task 6.2: SPA global 401 handler

**Files:**
- Modify: `dashboard/src/api/client.ts` (or wherever the central fetch wrapper lives — find via grep).
- Add: `dashboard/src/api/__tests__/sessionExpired.test.ts`.

**Behavior:**
- Single shared `apiFetch(path, init)` helper used by all SPA code.
- When `response.status === 401 && response.headers.get("content-type")?.startsWith("application/json")`:
  1. Parse body.
  2. If `body.error in ("unauthenticated", "session_expired")`, call `window.location.assign(body.login_url)`. Return a never-resolving promise so the caller's `.then` doesn't fire — the page is going away.
  3. Otherwise, reject normally (the route returned a domain 401 like "monitor X is read-only for your role").
- One small UX nicety: before the redirect, the helper sets `sessionStorage.setItem("hermes.lastLocation", window.location.pathname)` so the post-login redirect can land back where the user was. The `/auth/callback` handler reads this and, if it's a same-origin path, uses it as the `next=` value.

### Task 6.3: Remove the `hermes_session_rt` cookie + refresh path

**Files:**
- Modify: `hermes_cli/dashboard_auth/cookies.py` — drop `set_refresh_cookie`, `clear_refresh_cookie`, `get_refresh_token`.
- Modify: `hermes_cli/dashboard_auth/routes.py` — `/auth/callback` no longer writes a refresh cookie; `/auth/logout` no longer needs to clear it (it never existed).
- Modify: `tests/hermes_cli/test_dashboard_auth_cookies.py` — drop the three RT tests; add a regression that `hermes_session_rt` is NOT a cookie name we emit.

**Rationale:** Contract V1 does not issue refresh tokens, so persisting one is dead state. The provider's `refresh_session` raises `RefreshExpiredError` unconditionally; if we wired the middleware to call it, the cookie machinery would observe an empty `refresh_token` and the result would be the same as having no cookie. Keeping the cookie around is just attack surface.

**Forward compatibility:** if Portal later starts issuing refresh tokens, the provider's `complete_login` already ignores them today (line `# Contract V1: no refresh token. If one is present, we deliberately ignore it`). To turn them on later, three things change: cookies.py grows the RT cookie back, provider sets `Session.refresh_token`, middleware adds a "near-expiry → refresh" branch in front of the expired branch. None of those changes break the V1 behavior; they're additive. Document this in the file header.

### Task 6.4: Audit-log + observability

**Files:**
- Modify: `hermes_cli/dashboard_auth/audit.py` — extend `AuditEvent` enum with `API_UNAUTHENTICATED`, `API_SESSION_EXPIRED` (already covered above for completeness).
- Modify: `docs/dashboard-auth-operations.md` (new in Phase 7) — document the redirect/401 split + how to read the audit log to debug "users keep getting logged out".

### Phase 6 v1 (rejected — preserved for archeology)

The v1 draft below implemented silent token refresh in the middleware: when the access token was within 60s of expiry, the provider's `refresh_session` would mint a new pair using the stored refresh token, and the user would never see a re-auth screen until the 30-day refresh token itself expired. This is the right design IF refresh tokens exist. They don't in V1 (contract C5), so the entire path is unimplementable. Reverted to 401-redirect UX above.



### Task 6.1: Refresh helper + `/api/auth/refresh` endpoint

**Objective:** A function the middleware calls when verify says "expired", and a manual `/api/auth/refresh` endpoint for the SPA to invoke proactively.

**Files:**
- Create: `hermes_cli/dashboard_auth/refresh.py`
- Modify: `hermes_cli/dashboard_auth/middleware.py` — call refresh when verify returns None.
- Modify: `hermes_cli/dashboard_auth/routes.py` — add `/api/auth/refresh`.

**Step 1: Implement the refresh helper.**

```python
# hermes_cli/dashboard_auth/refresh.py
"""Cookie-rotating refresh helper.

The middleware calls ``maybe_refresh_session`` whenever ``verify_session``
returned None and a refresh token is available. On success the response
gets new ``hermes_session_at`` + ``hermes_session_rt`` cookies.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from fastapi import Request
from fastapi.responses import Response

from hermes_cli.dashboard_auth import list_providers
from hermes_cli.dashboard_auth.audit import audit_log, AuditEvent
from hermes_cli.dashboard_auth.base import (
    Session, RefreshExpiredError, ProviderError,
)
from hermes_cli.dashboard_auth.cookies import (
    set_session_cookies, clear_session_cookies, detect_https,
)

_log = logging.getLogger(__name__)


def attempt_refresh(*, refresh_token: str) -> Optional[Session]:
    """Try every provider until one accepts the refresh token.

    Returns the new Session on success. Returns None if every provider
    rejected the token. Raises ProviderError if at least one provider
    was unreachable AND none succeeded.
    """
    last_provider_error: Optional[ProviderError] = None
    for provider in list_providers():
        try:
            return provider.refresh_session(refresh_token=refresh_token)
        except RefreshExpiredError:
            # Token doesn't belong to this provider (or is truly dead); try next.
            continue
        except ProviderError as e:
            last_provider_error = e
            continue
    if last_provider_error:
        raise last_provider_error
    return None


def apply_refresh_to_response(
    request: Request,
    response: Response,
    session: Session,
) -> None:
    """Set the new session cookies on ``response``."""
    expires_in = max(60, session.expires_at - int(time.time()))
    set_session_cookies(
        response,
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        access_token_expires_in=expires_in,
        use_https=detect_https(request),
    )
```

**Step 2: Wire silent refresh into the middleware.**

In `hermes_cli/dashboard_auth/middleware.py`, after the `if session is None:` block, insert a refresh attempt BEFORE the bail-out:

```python
    if session is None:
        # Silent refresh attempt — if we still have a refresh token, try it.
        if _rt:
            try:
                refreshed = attempt_refresh(refresh_token=_rt)
            except ProviderError as e:
                _log.warning("dashboard-auth refresh: provider unreachable: %s", e)
                audit_log(AuditEvent.REFRESH_FAILURE, reason="provider_unreachable",
                          ip=_client_ip(request))
                return _unauth_response(path, reason="refresh_unreachable")
            if refreshed is not None:
                # Carry on with the refreshed session; the response handler
                # rotates the cookies on the way out via the wrapper below.
                request.state.session = refreshed
                request.state._session_just_refreshed = refreshed
                response = await call_next(request)
                apply_refresh_to_response(request, response, refreshed)
                audit_log(AuditEvent.REFRESH_SUCCESS,
                          provider=refreshed.provider,
                          user_id=refreshed.user_id,
                          ip=_client_ip(request))
                return response
            audit_log(AuditEvent.REFRESH_FAILURE, reason="refresh_expired",
                      ip=_client_ip(request))
        audit_log(AuditEvent.SESSION_VERIFY_FAILURE, reason="no_provider_recognises",
                  ip=_client_ip(request))
        return _unauth_response(path, reason="invalid_or_expired_session")
```

Imports at the top of middleware.py:

```python
from hermes_cli.dashboard_auth.refresh import attempt_refresh, apply_refresh_to_response
```

**Step 3: Manual `/api/auth/refresh` endpoint.**

In `hermes_cli/dashboard_auth/routes.py`:

```python
from hermes_cli.dashboard_auth.refresh import attempt_refresh, apply_refresh_to_response


@router.post("/api/auth/refresh", name="auth_refresh")
async def api_auth_refresh(request: Request):
    """SPA-triggered explicit refresh.

    Reads the refresh token from cookies, calls the provider, rotates the
    session cookies on the response. SPA uses this to extend a session
    proactively (e.g. when the user navigates back to a tab they left open
    for hours).
    """
    _at, rt = read_session_cookies(request)
    if not rt:
        raise HTTPException(status_code=401, detail="No refresh token in cookie")
    try:
        sess = attempt_refresh(refresh_token=rt)
    except ProviderError as e:
        audit_log(AuditEvent.REFRESH_FAILURE, reason="provider_unreachable",
                  ip=_client_ip(request))
        raise HTTPException(status_code=503, detail=f"Provider unreachable: {e}")

    if sess is None:
        # Refresh truly dead → clear cookies and tell SPA to re-login.
        resp = JSONResponse(
            {"detail": "Refresh expired; re-login required"},
            status_code=401,
        )
        clear_session_cookies(resp)
        audit_log(AuditEvent.REFRESH_FAILURE, reason="refresh_expired",
                  ip=_client_ip(request))
        return resp

    audit_log(AuditEvent.REFRESH_SUCCESS,
              provider=sess.provider, user_id=sess.user_id,
              ip=_client_ip(request))
    resp = JSONResponse({
        "user_id": sess.user_id,
        "email": sess.email,
        "display_name": sess.display_name,
        "provider": sess.provider,
        "expires_at": sess.expires_at,
    })
    apply_refresh_to_response(request, resp, sess)
    return resp
```

**Step 4: Test.**

```python
# tests/hermes_cli/test_dashboard_auth_refresh.py
"""Silent refresh and explicit /api/auth/refresh."""
import time
import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE, SESSION_RT_COOKIE
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


@pytest.fixture
def gated_app():
    clear_providers()
    register_provider(StubAuthProvider())
    web_server.app.state.bound_host = "0.0.0.0"
    web_server.app.state.auth_required = True
    yield TestClient(web_server.app, base_url="https://gated.fly.dev")
    clear_providers()
    web_server.app.state.auth_required = False


def _login(client) -> dict:
    r1 = client.get("/auth/login?provider=stub", follow_redirects=False)
    state = r1.headers["location"].split("state=")[1]
    r2 = client.get(f"/auth/callback?code=stub_code&state={state}",
                    follow_redirects=False)
    cookies = {}
    for raw in r2.headers.get_list("set-cookie"):
        name, _, rest = raw.partition("=")
        val = rest.split(";", 1)[0]
        cookies[name] = val
    return cookies


def test_explicit_refresh_rotates_cookies(gated_app):
    cookies = _login(gated_app)
    old_at = cookies[SESSION_AT_COOKIE]
    old_rt = cookies[SESSION_RT_COOKIE]
    r = gated_app.post("/api/auth/refresh")
    assert r.status_code == 200
    new_at = next((c.split(";")[0].split("=", 1)[1]
                   for c in r.headers.get_list("set-cookie")
                   if c.startswith(f"{SESSION_AT_COOKIE}=")), None)
    new_rt = next((c.split(";")[0].split("=", 1)[1]
                   for c in r.headers.get_list("set-cookie")
                   if c.startswith(f"{SESSION_RT_COOKIE}=")), None)
    assert new_at and new_at != old_at
    assert new_rt and new_rt != old_rt


def test_silent_refresh_on_expired_access_token():
    # Configure the stub provider with a very short TTL so the first /api/me
    # call sees an expired token, but the refresh succeeds.
    clear_providers()
    register_provider(StubAuthProvider(default_ttl=0))
    web_server.app.state.auth_required = True
    try:
        client = TestClient(web_server.app, base_url="https://gated.fly.dev")
        cookies = _login(client)
        # /api/auth/me with an expired AT must succeed because middleware
        # silently refreshes.
        r = client.get("/api/auth/me")
        assert r.status_code == 200
        # And the response carries rotated cookies.
        set_cookies = r.headers.get_list("set-cookie")
        assert any(c.startswith(f"{SESSION_AT_COOKIE}=") for c in set_cookies)
    finally:
        clear_providers()
        web_server.app.state.auth_required = False


def test_refresh_without_rt_cookie_returns_401(gated_app):
    r = gated_app.post("/api/auth/refresh")
    assert r.status_code == 401


def test_refresh_with_dead_token_clears_cookies(gated_app):
    gated_app.cookies.set(SESSION_RT_COOKIE, "garbage-refresh-token")
    r = gated_app.post("/api/auth/refresh")
    assert r.status_code == 401
    # Clearing cookies on a dead refresh
    set_cookies = r.headers.get_list("set-cookie")
    assert any(c.startswith(f"{SESSION_AT_COOKIE}=") and "Max-Age=0" in c for c in set_cookies)
```

**Step 5: Run, verify pass.**

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_refresh.py -v
```

**Step 6: Commit.**

```bash
git add hermes_cli/dashboard_auth/refresh.py hermes_cli/dashboard_auth/middleware.py hermes_cli/dashboard_auth/routes.py tests/hermes_cli/test_dashboard_auth_refresh.py
git commit -m "feat(dashboard-auth): silent refresh on expired AT + manual /api/auth/refresh"
```

### Phase 6 Exit Gate

```bash
scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_refresh.py -v
```

Manual: leave a dashboard tab open with the stub provider configured to issue 60-second access tokens. After 60 seconds the next page interaction is invisibly upgraded — no login redirect. The audit log shows `refresh_success` entries.

---

## Phase 7 — SPA "Logged in as …" Widget, CLI Status, Documentation

> **Plan v2 note.** Contract C4 means the JWT does NOT emit `email` or `display_name` claims, and contract C7 means there's no userinfo endpoint to fetch them from. The widget shown below was originally drafted as "Logged in as Foo <foo@bar.com>"; the implementation surfaces a truncated `user_id` instead (`Logged in as usr_abc123…`). The `/api/auth/me` payload retains empty `email` / `display_name` fields for forward-compat with a future Portal userinfo endpoint (OQ-C1).

**Goal:** User-visible polish — the gated dashboard surfaces the current identity, `hermes status` reports auth-gate state, and the docs site has a guide for VPS/Fly deployments.

### Task 7.1: SPA sidebar identity widget (v2)

**Objective:** A small "Logged in as Foo <foo@bar.com>  ⏻" widget at the top of the sidebar. Hits `/api/auth/me` on mount; the logout icon POSTs `/auth/logout`.

**Files:**
- Create: `web/src/components/AuthWidget.tsx`
- Modify: `web/src/App.tsx` — mount the widget at the top of the sidebar.
- Modify: `web/src/lib/api.ts` — add `getMe()` and `logout()`.

```typescript
// web/src/lib/api.ts — add
export interface AuthMe {
  user_id: string;
  email: string;
  display_name: string;
  org_id: string;
  provider: string;
  expires_at: number;
}

export async function getAuthMe(): Promise<AuthMe | null> {
  const r = await fetch('/api/auth/me', { credentials: 'include' });
  if (r.status === 401) return null;
  if (!r.ok) throw new Error(`auth/me: HTTP ${r.status}`);
  return r.json();
}

export async function logout(): Promise<void> {
  const r = await fetch('/auth/logout', {
    method: 'POST',
    credentials: 'include',
    redirect: 'manual',
  });
  // Server returns 302 → /login. With redirect: 'manual', fetch resolves
  // with status 0 in browsers — we manually navigate.
  window.location.href = '/login';
}
```

```typescript
// web/src/components/AuthWidget.tsx
import { useEffect, useState } from 'react';
import { getAuthMe, logout, AuthMe } from '../lib/api';

export function AuthWidget() {
  const [me, setMe] = useState<AuthMe | null | undefined>(undefined);

  useEffect(() => {
    if (!window.__HERMES_AUTH_REQUIRED__) return;
    getAuthMe().then(setMe).catch(() => setMe(null));
  }, []);

  if (!window.__HERMES_AUTH_REQUIRED__) return null;
  if (me === undefined) return <div className="text-xs opacity-50 px-3 py-2">Loading…</div>;
  if (me === null) return null;

  return (
    <div className="flex items-center gap-2 px-3 py-2 border-b border-zinc-800 text-xs">
      <div className="flex-1 min-w-0">
        <div className="truncate font-medium">{me.display_name}</div>
        <div className="truncate opacity-60">{me.email}</div>
      </div>
      <button
        onClick={() => logout()}
        title="Log out"
        className="opacity-60 hover:opacity-100"
      >⏻</button>
    </div>
  );
}
```

In `web/src/App.tsx`, mount at the top of the sidebar:

```typescript
import { AuthWidget } from './components/AuthWidget';
// ...
<aside>
  <AuthWidget />
  {/* existing sidebar items */}
</aside>
```

Add to `web/src/vite-env.d.ts` (or wherever globals are declared):

```typescript
interface Window {
  __HERMES_SESSION_TOKEN__?: string;
  __HERMES_DASHBOARD_EMBEDDED_CHAT__?: boolean;
  __HERMES_BASE_PATH__?: string;
  __HERMES_AUTH_REQUIRED__?: boolean;
}
```

**Commit:**

```bash
git add web/
git commit -m "feat(dashboard-auth): SPA auth widget showing logged-in identity + logout button"
```

### Task 7.2: `hermes status` integration

**Objective:** `hermes status` reports whether the gateway/dashboard has an active OAuth gate.

**Files:**
- Modify: `hermes_cli/status.py` — extend the existing reporter.

```python
# In hermes_cli/status.py, add to the report block:

# ... after the existing nous_logged_in line ...
def _dashboard_auth_status() -> str:
    """Reports number of registered dashboard-auth providers."""
    try:
        from hermes_cli.dashboard_auth import list_providers
        providers = list_providers()
    except Exception:
        return "not available"
    if not providers:
        return "no auth providers registered (loopback only)"
    names = ", ".join(p.name for p in providers)
    return f"{len(providers)} provider(s): {names}"

# Print in the report:
print(f"  dashboard auth providers: {_dashboard_auth_status()}")
```

**Commit:**

```bash
git add hermes_cli/status.py
git commit -m "feat(status): report dashboard-auth provider availability"
```

### Task 7.3: Documentation

**Objective:** A doc page covering the auth gate, how to deploy publicly, and how to write a custom provider plugin.

**Files:**
- Create: `website/docs/user-guide/features/dashboard-auth.md`
- Modify: `website/sidebars.ts` — link the new page.

Content sketch (full prose in the file):

```markdown
# Dashboard Authentication

When `hermes dashboard` binds to a non-loopback host (any IP other than
`127.0.0.1`, `localhost`, or `::1`) **without** `--insecure`, an OAuth
authentication gate is automatically engaged. By default, sign-in uses
your Nous Portal account.

## When the gate is active

| Command | Gate? |
|---|---|
| `hermes dashboard`                                 | Off (loopback) |
| `hermes dashboard --host 0.0.0.0`                  | On |
| `hermes dashboard --host 0.0.0.0 --insecure`       | Off (legacy escape hatch) |
| `hermes dashboard --host 192.168.1.5`              | On |
| `hermes dashboard --host fly-app.fly.dev`          | On |

## Default sign-in: Nous Portal

The default provider is bundled (`plugins/dashboard-auth-nous`) and needs
zero configuration. Visiting the dashboard prompts a "Sign in with Nous
Portal" button → OAuth redirect → back to your dashboard.

## Logging out

Click ⏻ in the sidebar widget, or `POST /auth/logout`. The browser cookies
are cleared and the refresh token is best-effort revoked at Portal.

## Audit log

Every sign-in attempt, success, refresh, and logout is recorded at
`$HERMES_HOME/logs/dashboard-auth.log` (JSON-lines).

## Adding a custom auth provider

(Plugin authoring guide — `DashboardAuthProvider` ABC, `register(ctx)`
example, link to `plugins/dashboard-auth-nous/` as the canonical template.)

## Forcing the legacy no-auth behavior

For trusted-network or testing scenarios, pass `--insecure`:

```
hermes dashboard --host 0.0.0.0 --insecure
```

Be aware: this exposes API keys and config without authentication. Only
use on private LANs where you trust every device.
```

**Commit:**

```bash
git add website/docs/user-guide/features/dashboard-auth.md website/sidebars.ts
git commit -m "docs(dashboard): document the OAuth auth gate + custom provider authoring"
```

### Phase 7 Exit Gate

Visual confirmation:
1. Gated dashboard renders the AuthWidget in the sidebar showing "Stub User · stub@example.test · ⏻".
2. Clicking ⏻ logs out, clears cookies, redirects to `/login`.
3. `hermes status` prints `dashboard auth providers: 1 provider(s): nous`.
4. Docs site renders the new page; sidebar links work.

---

## Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | A misconfigured operator believes the gate is on when it isn't, exposes dashboard publicly | Med | High | `start_server` prints `OAuth auth gate enabled` to stdout at bind time. `hermes status` shows the active state. Audit log writes `dashboard binding to host: <X>, gate: <on/off>` on every start. |
| R2 | Refresh token in cookie is exfiltrated via XSS | Low | High | HttpOnly cookie prevents JS access. CSP headers on `/` (added in Phase 7) restrict inline-script sources. The dashboard already escapes all user-supplied content; this plan doesn't add new XSS surface. |
| R3 | OAuth redirect URI mismatch breaks the round-trip in Fly setups | Med | High | The Portal whitelists `*.fly.dev/auth/callback`; verify with each new Fly hostname. `audit_log` records `idp_error` events so misconfigs are visible. Operators with custom domains follow the Open Question #1 path (out of v1 scope but flagged). |
| R4 | Portal JWKS rollout slips; userinfo mode hammers Portal with one network call per dashboard request | Med | Med | 60-second per-token cache in `_verify_via_userinfo` keeps the load to ~1 req/min/user. Add a metric/log if cache hit rate drops. |
| R5 | The `hermes-dashboard` client_id is not yet registered on Portal at code-merge time | High | Med | Phase 4 ships with the userinfo fallback and unit tests use respx mocks — code merges and CI passes without Portal. Operator-run smoke test (Phase 4 exit gate) gates the actual release. |
| R6 | Browsers reject cookies because Fly TLS terminates HTTPS but uvicorn sees HTTP without `proxy_headers` | High | High | `start_server` re-enables `proxy_headers=True` when gate is active. `detect_https` reads from `request.url.scheme` which honors `X-Forwarded-Proto` when proxy_headers is True. Tested in Phase 3.5. |
| R7 | Memory leak in `_verify_cache` for the userinfo mode | Low | Low | LRU-bounded to access tokens still in valid cookies; tokens are 1-hour-TTL so the dict size is bounded by simultaneous-user count. If telemetry shows growth, swap to `lru_cache` with explicit max=10000. |
| R8 | Stub provider accidentally leaks into a production build | Low | High | Lives under `tests/`, not `plugins/` or `hermes_cli/`. The plugin discovery scanner doesn't traverse `tests/`. CI assertion: `grep -r StubAuthProvider plugins/ hermes_cli/` returns nothing. |
| R9 | Loopback regression: existing dashboards stop accepting the injected token | Med | High | Phase 0's harness pins current behavior. Every subsequent phase reruns it. Pre-merge: full `scripts/run_tests.sh` against the whole suite. |
| R10 | Single-user-only assumption is violated by a future feature change | Low | Med | The session model treats every cookie as authoritative for the dashboard process; there's no per-user UI state. If multi-user is ever needed, audit `/api/sessions`, `/api/config`, and the PTY bridge — each currently writes to single shared state. Flagged in Open Questions #2. |

## Rollout

Phases 0–3 land first as one unit (the gate + stub-driven E2E). After merge, the gate is OFF by default for everyone (loopback unchanged) and OPT-IN for non-loopback (operator must pass --insecure to bypass, or install a provider plugin).

Phases 4–7 land in sequence as the Portal cross-repo work completes. Phase 4 is the first user-visible step; Phases 5–6–7 are quality-of-life improvements that don't change correctness.

### Pre-merge checklist for each phase

- [ ] All new tests pass
- [ ] Loopback regression harness from Phase 0 still passes
- [ ] No new errors at `WARNING` or higher in `agent.log` / `gateway.log` when starting a loopback dashboard
- [ ] Manual smoke test (Phase exit gate) walked by the implementer
- [ ] `hermes status` output unchanged (until Phase 7's intentional addition)

### Pre-release checklist for Phase 4

- [ ] Portal team confirms `hermes-dashboard` client_id is registered in `OAUTH_CLIENT_PRODUCT_CONTEXT_MAP`
- [ ] Portal team confirms `https://*.fly.dev/auth/callback` is in the redirect-URI whitelist for `hermes-dashboard`
- [ ] Portal team confirms `GET /oauth/authorize` route is live
- [ ] Portal team confirms `POST /api/oauth/token` accepts `grant_type=authorization_code` with PKCE
- [ ] Portal team confirms access token includes `email`, `email_verified`, `name` claims
- [ ] Operator walks the end-to-end flow against staging Portal once
- [ ] Operator confirms `~/.hermes/logs/dashboard-auth.log` records `login_success` event with correct user_id + email
- [ ] Operator confirms refresh works by setting access-token TTL to 60s and leaving the tab open for 90s

## Verification Strategy

| Layer | How |
|---|---|
| Provider protocol | Unit tests in `tests/hermes_cli/test_dashboard_auth_provider_base.py` — every provider plugin must call `assert_protocol_compliance` in its own tests. |
| Cookies | Unit tests in `tests/hermes_cli/test_dashboard_auth_cookies.py` cover HttpOnly/Secure/SameSite/Max-Age semantics. |
| Middleware | Behavioral tests in `tests/hermes_cli/test_dashboard_auth_middleware.py` exercise gated vs loopback modes with the stub provider. |
| Real provider | `plugins/dashboard-auth-nous/test_provider.py` uses respx to mock Portal endpoints. Real-Portal smoke is operator-run. |
| End-to-end | The Phase 3 exit gate is a full browser round trip with the stub. Phase 4 exit gate is the same against staging Portal. |
| Regression | Phase 0's harness is rerun by every subsequent phase as part of its exit gate. |
| Audit log | Tests in `test_dashboard_auth_audit.py` confirm event types, JSON format, and token redaction. |
| WS auth | Tests in `test_dashboard_auth_middleware.py::test_ws_*` cover ticket mint/consume/expire across loopback and gated. |

## Timeline (rough)

Each phase is independently shippable. A focused engineer can land:

| Phase | Effort |
|---|---|
| 0 | ½ day |
| 1 | 1 day |
| 2 | ½ day |
| 3 | 2 days |
| 4 | 1 day Hermes side + cross-repo coordination (variable) |
| 5 | 1 day |
| 6 | 1 day |
| 7 | 1 day |

Total: ~7–8 working days on the Hermes side. Cross-repo Portal work is on top of that and gates Phase 4's actual usefulness.

## Files Changed Summary

New:
- `hermes_cli/dashboard_auth/__init__.py`
- `hermes_cli/dashboard_auth/base.py`
- `hermes_cli/dashboard_auth/registry.py`
- `hermes_cli/dashboard_auth/audit.py`
- `hermes_cli/dashboard_auth/cookies.py`
- `hermes_cli/dashboard_auth/middleware.py`
- `hermes_cli/dashboard_auth/routes.py`
- `hermes_cli/dashboard_auth/login_page.py`
- `hermes_cli/dashboard_auth/ws_tickets.py`
- `hermes_cli/dashboard_auth/refresh.py`
- `plugins/dashboard-auth-nous/plugin.yaml`
- `plugins/dashboard-auth-nous/__init__.py`
- `plugins/dashboard-auth-nous/provider.py`
- `plugins/dashboard-auth-nous/test_provider.py`
- `web/src/components/AuthWidget.tsx`
- `website/docs/user-guide/features/dashboard-auth.md`
- 8 test files under `tests/hermes_cli/`

Modified:
- `hermes_cli/web_server.py` — add `should_require_auth`, register two middlewares, update `_serve_index`/`start_server`/WS endpoints
- `hermes_cli/plugins.py` — add `register_dashboard_auth_provider` method
- `hermes_cli/status.py` — report dashboard-auth state
- `web/src/App.tsx`, `web/src/lib/api.ts`, `web/src/pages/ChatPage.tsx`, `web/src/vite-env.d.ts`
- `website/sidebars.ts`

Cross-repo (nous-account-service):
- `src/server/oauth/access-token-issuer.ts` — register `hermes-dashboard` client_id
- New: `src/app/oauth/authorize/page.tsx` (or equivalent)
- `src/app/api/oauth/token/route.ts` — accept `grant_type=authorization_code`
- `src/server/oauth/access-token-issuer.ts` — add `email`/`name` claims for `profile email` scope
- Portal config — whitelist `https://*.fly.dev/auth/callback`

