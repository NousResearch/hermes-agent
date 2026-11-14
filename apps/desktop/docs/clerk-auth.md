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
- Configure the sign-in methods and OAuth providers the app should offer.

The `unitalk-app` scheme is registered by `electron-builder.config.cjs` and by
`protocol.registerSchemesAsPrivileged` in the main process. OAuth deep-link
transport is disabled when the app runs against the Vite dev server.

## Token storage

`@clerk/electron/storage` persists tokens with `electron-store` and
`safeStorage`. If the OS keychain is unavailable, tokens stay in memory and the
user signs in again on the next launch; plaintext fallback is not enabled.

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

The desktop app does not add its own Clerk account page or branded auth screens;
it renders Clerk's prebuilt UI directly. `@clerk/electron` is beta. Re-test the
packaged app after SDK upgrades.
