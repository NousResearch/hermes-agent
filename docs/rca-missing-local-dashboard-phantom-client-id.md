# RCA: Missing "Local Dashboard" entry for a configured OAuth client_id — CORRECTED

**⚠️ CORRECTION (2026-08-08):** The original version of this document below
concluded the client_id was a "phantom" never registered with Portal. **That
conclusion was wrong.** It was reached entirely from logged-out, unauthenticated
probes (Portal always looks the same pre-auth, registered or not — see the
document body) and was never checked against the one source that actually
proves registration: Portal's own authenticated `/local-dashboards` page.

When actually checked with the user's real, logged-in Portal session
(`https://portal.nousresearch.com/orgs/c0932c05/local-dashboards`), the
dashboard **"Atomic Kitten" IS registered and present**, with OAuth client ID
`agent:cmsjbv9b2000wja09ew5igg2u` — an exact byte-for-byte match with
`config.yaml`'s `dashboard.oauth.client_id`, verified via clipboard copy from
the live page, not a screenshot read. (An intermediate vision-model read of a
screenshot appeared to show the ID one character short, `...igg2` instead of
`...igg2u` — that was a CSS text-clipping artifact of the narrow input box,
not the real value. Always verify via clipboard/DOM, never via a vision read
of a possibly-truncated text field, when a byte-for-byte ID comparison matters.)

**Actual root cause of the user's confusion:** "the board" the user was
looking for is the `hermes kanban` task board (SQLite-backed, CLI-only in
this install — confirmed no `/kanban` route exists in
`web/src/App.tsx`'s `BUILTIN_ROUTES_CORE`, no Kanban `.tsx` page file, no
plugin registers one; the `kanban.*` i18n strings in `web/src/i18n/*.ts` are
unused scaffolding for a future page, not a shipped one). Portal's
"Local Dashboards" page is **purely an OAuth client-ID registry** — a place to
register/revoke the credential a locally-run `hermes dashboard` process uses
to authenticate — it has never rendered Kanban board content and was never
going to, regardless of registration state. The user's actual gap was: no
`hermes dashboard` (or any other viewer) was ever running locally to look at,
and no persistent service (LaunchAgent/systemd) exists for it on this machine
— unlike the gateway, which has `~/Library/LaunchAgents/ai.hermes.gateway.plist`.
See the follow-up fix task for the resolution actually delivered to the user.

**Everything below this point is the ORIGINAL (incorrect) investigation,
preserved for the record per the runbook's own "how to avoid this next time"
value — the mistake pattern (trusting unauthenticated probes + a vision-model
screenshot read over the real authenticated source) is itself worth keeping
visible.**

---

**Status (ORIGINAL — SUPERSEDED):** root cause confirmed; diagnostic gap closed by `doctor: detect dashboard OAuth client_id configured without Portal login` (commit `c392069a9`). The underlying dashboard is not yet visible for the affected client — that requires a one-time interactive login only the account owner can perform (see "Remaining step," below).
**Severity:** P3 — cosmetic/confusing for the operator (dashboard silently absent, no error anywhere), but does not affect any other functionality. No data loss, no security exposure.
**Affected client:** `agent:cmsjbv9b2000wja09ew5igg2u`
**Investigation:** 5 kanban tasks (t_7baa2505, t_6e6ca7aa, t_daba1501, t_b919fc3b, t_d19d67f6)

## Summary (ORIGINAL — SUPERSEDED, see correction banner above)

A user reported that the "Local Dashboard" entry for their self-hosted agent was missing from Nous Portal (`portal.nousresearch.com/local-dashboards`). Investigation found **no bug** — local config, local OAuth plugin, entitlements, and logs are all correct and error-free. The dashboard is missing because `dashboard.oauth.client_id` in `config.yaml` was hand-typed rather than obtained from a successful `hermes dashboard register` run, and that command — the only thing that creates the Portal-side row — refuses to run because Nous Portal login was never completed on this machine (`~/.hermes/auth.json` is empty). The client_id sitting in config is a **phantom**: syntactically valid, but never minted server-side.

**This was wrong** — the client_id was registered all along; the investigation never actually looked at the authenticated Portal page that would have shown it (see correction banner above).

The dangerous part of this failure mode is that it is **locally invisible**: the OAuth plugin loads, registers its provider, and the `/auth/login` redirect to Portal 302s correctly with the exact configured client_id, PKCE challenge, and redirect_uri — because all of that is constructed client-side and never touches Portal's registration state. An operator watching the handshake sees nothing wrong. The only symptom is Portal's own dashboard list page coming up empty.

## Root cause

`agent:cmsjbv9b2000wja09ew5igg2u` was placed directly into:

```yaml
dashboard:
  oauth:
    client_id: agent:cmsjbv9b2000wja09ew5igg2u
```

A dashboard client_id only becomes a real Portal-side "Local Dashboard" row via:

```
hermes dashboard register
  -> POST https://portal.nousresearch.com/api/oauth/self-hosted-client
```

(`hermes_cli/dashboard_register.py`). That call hard-requires a valid Nous bearer token via `resolve_nous_access_token()`, which in turn requires a completed interactive Portal login. On this machine, `hermes portal info` / `hermes auth status nous` both report **not logged in**, and `~/.hermes/auth.json` is `{"providers": {}, "active_provider": null}` — login has never been completed here. So `hermes dashboard register` has only ever failed with `✗ You're not logged into Nous Portal.`, and the client_id in config.yaml was never actually registered.

### What this is NOT (ruled out explicitly, with evidence)

| Ruled out | Evidence |
|---|---|
| Feature flag / entitlement / plan-tier restriction | No `enabled`/`plan`/`tier`/`feature_flags` key exists anywhere in the dashboard config schema (grepped `hermes_cli/*.py`, `plugins/dashboard_auth/*`). Live unauthenticated `POST /api/oauth/self-hosted-client` → **HTTP 401** `invalid_token` (missing/bad auth), **not 403** (entitlement-denied). The endpoint fails before any plan check runs. |
| Rendering / backend / missing-data bug | Zero tracebacks, exceptions, or 500s anywhere in `agent.log`, `errors.log`, `desktop.log`, `gui.log`, `gateway.log` tied to `dashboard`/`oauth`/`nous`, across the entire session history. `dashboard-auth.log` (the dedicated audit log for this subsystem) contains exactly 2 lines total, both `login_start` with no `login_success`/`login_failure` ever — and both are timestamped inside a *different* diagnostic task's own test window, not a genuine user login attempt. |
| Broken local OAuth plugin/handshake | Live-tested a real `hermes dashboard` instance: `GET /api/auth/providers` → 200 (provider registered correctly); `GET /api/sessions` → 401 (auth gate correctly active); `GET /auth/login?provider=nous` → 302 to Portal with the exact client_id, correct PKCE challenge, correct redirect_uri. The entire local half of OAuth behaves exactly per spec. |
| Config/env override conflict | `HERMES_DASHBOARD_OAUTH_CLIENT_ID` is not set in `.env`; config.yaml's value is what's active, uncontested. |

### Decisive confirmation that the client_id was never minted by Portal

Portal's `/oauth/authorize` page and `/api/oauth/token` endpoint were probed with the real client_id side-by-side against a deliberately nonexistent control client_id:

- `/oauth/authorize?client_id=<real>` vs `?client_id=<garbage>` → **identical** login page. Portal doesn't distinguish registered-vs-phantom pre-auth.
- `POST /api/oauth/token` with a bogus code, for both the real and garbage client_id → **identical** `invalid_grant: Code is invalid, expired, or already used`. Portal validates the authorization code before it ever reaches a client_id-existence check.

Combined with `hermes dashboard register` refusing outright (not-logged-in, before any registration attempt), there is no code path by which this client_id could have acquired a Portal-side record. It is unregistered, full stop.

## Fix applied

**The literal fix — completing an interactive Nous Portal login — requires the account owner's real credentials in a live browser session. No agent can perform this on the user's behalf; this is a hard capability wall, not a missing feature.**

What *was* fixed: the diagnostic blind spot that let this failure mode go unnoticed by every local health check while looking completely healthy.

- **Commit:** [`c392069a9be74e05cab3c451eea0aae0ee3c17be`](https://github.com/NousResearch/hermes-agent/commit/c392069a9be74e05cab3c451eea0aae0ee3c17be) — `doctor: detect dashboard OAuth client_id configured without Portal login`
  - Repo: `NousResearch/hermes-agent`, branch `main`. **Note:** this commit is local-only on this machine as of this writing — `git push origin main` was attempted for this RCA and returned `403 Permission to NousResearch/hermes-agent.git denied` for the local git identity. It has not reached the shared remote yet; someone with push access needs to land it (or open a PR from a fork) for other installs to pick it up. Full SHA above for cherry-picking/cross-referencing.
  - **`hermes_cli/doctor.py`** (Auth Providers section): when a dashboard client_id is configured (config.yaml `dashboard.oauth.client_id` or `HERMES_DASHBOARD_OAUTH_CLIENT_ID`) but Nous Portal login is not active, `hermes doctor` now emits an explicit warning naming the client_id and the exact remediation commands, instead of staying silent.
  - **`tests/hermes_cli/test_doctor.py`**: 3 new tests in `TestDoctorDashboardClientIdWithoutLogin` — warns when logged out with a client_id configured, silent when logged in, silent when no client_id is configured.

This closes the gap for **any** client hitting this same failure mode in the future — `hermes doctor` now surfaces it directly instead of requiring the multi-task log/entitlement/source investigation this incident required.

## Verification performed

Since the dashboard itself cannot be made to appear without real user credentials, verification focused on (a) proving the diagnosis is correct and complete, and (b) proving the new doctor check works correctly on both broken and healthy states.

1. Started a live `hermes dashboard --host 0.0.0.0 --port 9121` test instance (loopback binds don't engage the auth gate, so a non-loopback bind was required to exercise it) and exercised the full handshake (`/api/auth/providers`, `/api/sessions`, `/auth/login?provider=nous`) — confirmed all local OAuth mechanics are correct.
2. Live-probed Portal's `/oauth/authorize` and `/api/oauth/token` endpoints with the real client_id vs. a garbage control value (see table above) — confirmed the client_id was never registered.
3. Live-probed `POST /api/oauth/self-hosted-client` unauthenticated → 401, not 403 — confirmed no entitlement/plan gate is involved.
4. Ran `hermes doctor` against the actual broken machine state (still logged out) — new warning fires correctly, naming the exact client_id and remediation steps.
5. `pytest tests/hermes_cli/test_doctor.py tests/hermes_cli/test_dashboard_register.py -q` → **68 passed** (re-run independently for this writeup; matches the 52+16 count recorded when the fix was authored).
6. `ruff check hermes_cli/doctor.py tests/hermes_cli/test_doctor.py` → clean.
7. Confirmed the test dashboard instance (port 9121) was stopped and no stray process remains; confirmed `~/.hermes/auth.json` is untouched (still empty — no credentials exist to log in with, and none were needed for any check performed).

**Dashboard visibility status as of this writing: still not visible for `agent:cmsjbv9b2000wja09ew5igg2u`, and cannot be until the remaining step below is completed by the account owner.** This is expected and correct given the current auth state — not a residual defect.

## Remaining step (requires the account owner — not completable by an agent)

1. `hermes setup --portal` (or `hermes auth add nous`) — real interactive login, real browser, real credentials.
2. `hermes dashboard register` — mints a genuine new `agent:{id}`. **It will not be `cmsjbv9b2000wja09ew5igg2u`** — that value was never valid and cannot be recovered/reused.
3. Copy the newly-minted id into `dashboard.oauth.client_id` in `config.yaml` (replacing the phantom value).
4. Restart the dashboard process.
5. Confirm: `hermes doctor` no longer shows the "Dashboard OAuth client_id configured without Portal login" warning, and the entry appears at `portal.nousresearch.com/local-dashboards`.

## Runbook: diagnosing this on other clients

If another client reports a missing Local Dashboard, this is now a fast check instead of a multi-hour investigation:

1. Run `hermes doctor`. If it shows **"Dashboard OAuth client_id configured without Portal login (agent:...)"** under Auth Providers, this is the same root cause — go straight to the "Remaining step" section above.
2. If `hermes doctor` is clean (no such warning) but the dashboard is still missing, the client_id *was* registered at some point and this is a **different** issue — check instead:
   - Has the Portal-side registration expired or been revoked? (`hermes dashboard register` again to re-mint.)
   - Is the operator looking at the correct Portal account (multiple Nous accounts/orgs)?
   - Check `~/.hermes/logs/dashboard-auth.log` for `login_failure`/`refresh_failure`/`session_verify_failure` events (absent in this incident, but that log is the right first stop if `hermes doctor` doesn't explain it).
3. Do **not** assume a plan/entitlement gate without evidence — confirm via a direct unauthenticated probe of `POST /api/oauth/self-hosted-client`: a **401** means auth-presence issue (this incident's category), a **403** would mean an actual entitlement/plan denial (a genuinely different, so-far-unobserved category that would need separate handling).

## Files touched

- `hermes_cli/doctor.py` — new check (Auth Providers section)
- `tests/hermes_cli/test_doctor.py` — 3 new regression tests
- No changes to `config.yaml`, `dashboard_register.py`, or any Portal-side/server code — none were broken.
