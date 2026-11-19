# Clerk authentication

Unitalk Desktop uses `@clerk/electron` for sign-in. The main process owns the
Clerk bridge, token storage, and OAuth deep links; the renderer uses
`@clerk/electron/react` and never sees `CLERK_SECRET_KEY`.

## Configuration

Set the public key in `apps/desktop/.env.local`:

```sh
VITE_CLERK_PUBLISHABLE_KEY=pk_live_...
```

The same file is read by Vite and by `scripts/bundle-electron-main.mjs`, which
bakes the key into the main and preload bundles. Never put a secret key in this
file or in any `VITE_*` variable.

When the key is absent the app keeps the existing `file://` renderer and starts
without the Clerk gate. When the key is present, packaged builds serve the
renderer from `unitalk-app://renderer/` and require a Clerk session.

## Clerk Dashboard

For each Clerk instance used by the desktop app:

- Enable the Native API.
- Allow `http://127.0.0.1:5174` and `http://localhost:5174` for development.
- Allow `unitalk-app://renderer` for packaged builds.
- Add `unitalk-app://renderer/` to the SSO redirect allowlist.
- Add `unitalk-app-dev://renderer/` to the SSO redirect allowlist for Windows development.
- Configure the sign-in methods and OAuth providers the app should offer.

The `unitalk-app` scheme is registered by `electron-builder.config.cjs` and by
`protocol.registerSchemesAsPrivileged` in the main process. Packaged builds serve
the renderer from `unitalk-app://renderer/` and complete OAuth through the scheme
deep link.

## OAuth in development

`@clerk/electron` completes OAuth in the system browser and returns through a
custom-scheme deep link, so the scheme must be registered with the operating
system:

- **Windows:** works unpackaged. `pnpm dev` registers `unitalk-app-dev://` at
  runtime with the Electron entry path, which is a separate scheme so a checkout
  never steals the installed app's `unitalk-app://` handler.
- **macOS / Linux:** Launch Services and `.desktop` entries only route custom
  schemes to packaged apps, so OAuth sign-in requires a packaged build. Clerk
  documents this beta limitation; use `pnpm build` and launch the packaged app
  to test OAuth on those platforms.

`HERMES_DESKTOP_SKIP_PROTOCOL_REGISTRATION=1` (used by E2E harnesses) leaves OS
handlers untouched and therefore disables OAuth deep links.

## Token storage

`@clerk/electron/storage` persists tokens with `electron-store` and
`safeStorage`. If the OS keychain is unavailable, tokens stay in memory and the
user signs in again on the next launch; plaintext fallback is not enabled.

## Browser-hosted Webapp

The Webapp uses `@clerk/react` and the same `ClerkSessionBridge` / `host.clerk`
interface as Desktop. Its browser-only instance gate mounts the Hermes viewer
after the signed-in user's organization resolves to this instance.

After sign-in, session restoration, or an organization change, the gate refreshes
Clerk's session cookies and fetches `https://unitalk.ai/api/instance`. Requests to
Unitalk use `credentials: 'include'` and `Accept: application/json`; authentication
uses the shared Clerk cookies, with no `Authorization` header.

- A ready instance whose `webDomain` matches the current hostname keeps the
  existing Hermes session, after checking `/api/auth/me`.
- A different instance closes the current viewer's gateway connections and
  fetches the existing `POST /auth/logout` without navigating. It then fetches
  `https://unitalk.ai/api/auth-init?session=<Clerk session ID>&instanceId=<_id>`
  as JSON, and replaces the current location with the returned instance URL.
  Clerk stays signed in throughout the handoff.
- A missing or inactive instance ends the old Hermes session and shows an error,
  an organization picker, Retry, and a link to Unitalk.
- Signing out of Clerk ends the Hermes browser session and shows Clerk sign-in
  on the loaded page. Signing in again initializes a fresh Hermes session even
  when the resolved hostname is unchanged.

The gate prevents stale lookups from reopening the viewer or navigating after
another account change. Cookie-changing requests finish in order. It also
handles plain-text HTTP-200 errors from `auth-init`. Browser organization changes
do not submit provider tokens to the previous instance.

`/api/instance` response handling:

| Response | Webapp behavior |
| --- | --- |
| `200 { instance }`, ready | Validate the account and organization, then open the matching instance. |
| `200 { instance }`, not ready | Show the activation message, even before a web domain has been assigned. |
| `401 { error: "Unauthorized" }` | End the old Hermes session and offer **Sign in again** or Retry. |
| `403 { error: "Active organization required" }` | End the old Hermes session and show the organization picker. |
| `404 { error: "Instance not found" }` | End the old Hermes session and prompt the user to create an instance. |
| `503 { error: "Instance could not be retrieved" }` | Keep the viewer blocked and show the error with Retry. |

A `401` from `auth-init`, including its plain-text response, also offers sign-in
recovery. Other `auth-init` errors retain their own meaning: a denied launch is
not reported as a missing instance.

The helper endpoints must allow credentialed CORS requests from each Webapp's
origin. Reloading after logout still goes through the backend's existing login
gate; the frontend sign-in screen is retained only on the already-loaded page.

The HTML response's Content Security Policy is generated by
`hermes_cli/webapp_page_policy.py`. Its `script-src` allows
`https://clerk.unitalk.ai` for Clerk's browser SDK, plus
`https://challenges.cloudflare.com` and `https://*.protect.clerk.com` for Clerk's
bot and abuse protection, as required by
[Clerk's CSP guidance](https://clerk.com/docs/guides/secure/best-practices/csp-headers).
The policy's connection, image, frame, style, and worker directives already allow
Clerk's other required resources.

If Clerk sign-in does not load and the console reports `failed_to_load_clerk_js`
or a CSP violation for `clerk.browser.js`, check the HTML response's
`Content-Security-Policy` header. Policy changes require updating the backend
serving the Webapp, then reloading the page; a frontend-only rebuild is insufficient.
Verify that `clerk.browser.js` loads and the Clerk sign-in screen appears.

## User panel and organization switching

The renderer owns the live Clerk client, so organization operations stay in
`src/app/auth/clerk-session-bridge.tsx` and reach plugins through
`host.clerk` in `@hermes/plugin-sdk`. No new preload or main-process IPC is
needed, and plugins never import Clerk React hooks or components.

`host.clerk.snapshot` exposes a sanitized user, the active organization, and
the user's organization memberships. `setActiveOrganization(orgId | null)`
uses `clerk.setActive`, forces a fresh token with
`getToken({ skipCache: true })`, and re-submits that token through the existing
`submitClerkToken` path. Passing `null` selects the personal account.

The initiating window announces the switch over `window.hermesDesktop.windowRelay`;
each other open window calls `setActive` for the same organization, refreshes
its token, and re-submits it for its own active profile. Token submission is
best-effort: the organization switch still succeeds and the snapshot records
`tokenSyncError` so the user can retry.

The standalone User Panel plugin lives in
`desktop-plugins/user-panel/plugin.js`. Copy that folder to
`$HERMES_HOME/desktop-plugins/user-panel` (the folder name must match the plugin
id) and the runtime loader picks it up. It renders an account avatar immediately
left of the titlebar's sidebar toggle and implements the organization list,
personal account, create organization, and sign-out.

## Verification

1. `pnpm --filter unitalk-desktop typecheck`
2. `pnpm --filter unitalk-desktop lint`
3. `pnpm --filter unitalk-desktop test:ui`
4. `pnpm --filter unitalk-desktop build`
5. Install the packaged build and verify Clerk's built-in sign-in/sign-up
   screens, email/password, OAuth/SSO, restart persistence, and sign-out.
6. With the User Panel plugin installed, switch organizations and confirm a
   second open window follows the switch; then sign out and confirm the app
   returns to Clerk's sign-in screen.
7. In the Webapp, switch to an organization with a different ready instance:
   verify `/api/instance`, the current instance's `/auth/logout`, then JSON
   `/api/auth-init`, followed by navigation to the returned `webDomain`.
8. Switch to an organization with no instance, verify the blocked-viewer error,
   then switch back and retry. Sign out and back in on the same hostname and
   confirm the fresh Hermes session opens the viewer again.

The desktop app does not add its own Clerk account page or branded auth screens;
it renders Clerk's prebuilt UI directly. `@clerk/electron` is beta. Re-test the
packaged app after SDK upgrades.
