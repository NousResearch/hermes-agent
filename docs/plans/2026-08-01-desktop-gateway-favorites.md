# Desktop Gateway Favorites Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Let desktop users save, edit, remove, and reorder named remote gateway favorites in Settings → Gateway, then switch the active desktop connection to one from the main UI gateway menu without re-entering credentials.

**Architecture:** Favorites are desktop-installation state, not Hermes backend/profile config. Store them alongside `connection.json` in Electron's `userData` directory, with token credentials encrypted using the existing desktop secret path and OAuth/SSH credentials continuing to use their existing per-gateway/session mechanisms. Expose a typed IPC CRUD/use API; the renderer keeps a cache in a small gateway-favorites store. The Settings → Gateway page owns list management, while the statusbar Gateway menu consumes the same store and calls one switching helper that reuses the existing connection-apply/rehome path.

**Tech Stack:** Electron IPC + preload bridge, TypeScript/React, Nanostores, existing `connection.json` persistence, existing `applyConnectionChange` / `ensureGatewayProfile` switching flow, Vitest + Testing Library.

---

## Scope and decisions

- Favorites are global to one desktop installation (`app.getPath('userData')`), so they survive Hermes profile changes and are visible from every profile.
- A favorite represents a remote connection only: direct URL (`remote`), Hermes Cloud (`cloud`), or SSH (`ssh`). Local gateway and profile-inheritance entries are not favorites.
- Favorite records contain a stable generated id, user-facing name, displayable connection fields, and ordering. Static session tokens are encrypted at rest and never returned to the renderer. OAuth session state remains keyed by gateway URL through the existing OAuth partition; SSH host-key/served-token handling remains in the existing SSH lifecycle.
- Saving a favorite must not change the active gateway. Choosing one must apply it to the currently active gateway/profile scope and then re-home that scope through the existing switching path.
- The quick-switch menu should show the current favorite as selected when the active connection has the favorite id. A connection edited manually should clear that association; an identical URL should not be inferred as the same favorite when two favorites intentionally have different names/authentication.
- Existing global/per-profile connection settings, OAuth login, Cloud discovery, SSH bootstrap, profile switching, soft rehome behavior, and background secondary sockets remain the source of truth for connection behavior. Favorites add selection/persistence; they do not duplicate gateway transport logic.

## Existing seams to preserve

- `apps/desktop/src/app/settings/gateway-settings.tsx:148-149` is the existing settings surface, with scope selection at `:199-203`, current connection loading at `:216-250`, and save/apply at `:445-506`.
- `apps/desktop/src/global.d.ts:503-546` defines renderer-facing connection config types and `:15-112` defines the preload bridge contract.
- `apps/desktop/electron/main.ts:611-621` owns the userData paths; `:6859-6910` reads/writes cached `connection.json`; `:6945-7002` sanitizes config; `:7030-7114` coerces writes; and `:9754-9960` registers connection/profile IPC handlers.
- `apps/desktop/electron/connection-config.ts` is the electron-free pure helper module and test seam. Keep favorite validation/normalization there or in a small adjacent pure module rather than adding more untestable policy to `main.ts`.
- `apps/desktop/src/store/profile.ts:255-305` is the active profile/gateway switch coordinator. `ensureGatewayProfile()` serializes swaps, keeps secondary sockets alive, and synchronizes `$connection` after activation.
- `apps/desktop/src/app/gateway/hooks/use-gateway-boot.ts:262-329` owns primary soft rehome after `applyConnectionConfig`; `apps/desktop/src/store/gateway-switch.ts:31-74` owns the explicit session-list wipe needed for a primary mode change.
- `apps/desktop/src/app/shell/hooks/use-statusbar-items.tsx:235-247` builds the statusbar Gateway menu and `:355-378` builds the current connection item. `apps/desktop/src/app/shell/gateway-menu-panel.tsx` owns the menu body.
- Renderer-local preference persistence goes through `apps/desktop/src/lib/storage.ts` / `persisted.ts`, but favorites must not use localStorage because credentials and authoritative connection state belong to Electron and should be shared across windows.

---

### Task 1: Define and test the favorite data contract and sanitization

**Objective:** Establish one validated representation for favorite records, including safe renderer output and stable ids, before wiring persistence or UI.

**Files:**
- Create: `apps/desktop/electron/gateway-favorites.ts`
- Test: `apps/desktop/electron/gateway-favorites.test.ts`
- Modify: `apps/desktop/src/global.d.ts:503-546` (renderer-facing favorite types)

**Step 1: Write failing pure-helper tests**

Cover these invariants:

- Names are trimmed and required; blank names are rejected.
- URL favorites use the same `normalizeRemoteBaseUrl()` rules as existing remote settings, including scheme-less host/port input and removal of query/hash.
- Only `remote`, `cloud`, and `ssh` modes are accepted.
- Favorite ids are preserved when valid and generated when absent; duplicate ids are not silently merged by the sanitizer.
- The renderer DTO includes `id`, `name`, mode and non-secret connection fields, plus `hasToken`/OAuth connection status as needed, but never includes the raw or encrypted token.
- SSH fields preserve host/user/port/key/path through `normalizeSshConfig()` semantics.
- Cloud favorites preserve the selected organization tag.

Run: `cd apps/desktop && npx vitest run --project electron electron/gateway-favorites.test.ts`

Expected: FAIL because the helper and types do not exist.

**Step 2: Implement the pure contract**

Add explicit types for:

```ts
export type DesktopGatewayFavoriteMode = 'cloud' | 'remote' | 'ssh'

export interface DesktopGatewayFavorite {
  id: string
  name: string
  mode: DesktopGatewayFavoriteMode
  remoteAuthMode: 'oauth' | 'token'
  remoteUrl: string
  cloudOrg: string
  sshHost: string
  sshUser: string
  sshPort: number | null
  sshKeyPath: string
  sshRemoteHermesPath: string
  hasToken: boolean
  oauthConnected: boolean
}
```

Keep the persisted shape separate from the renderer DTO so encrypted token objects never cross IPC. Export pure functions for normalizing a favorite input, sanitizing persisted entries, and producing the renderer DTO. Use the existing URL/SSH helpers instead of reimplementing their rules.

**Step 3: Run the focused tests**

Run: `cd apps/desktop && npx vitest run --project electron electron/gateway-favorites.test.ts`

Expected: PASS.

---

### Task 2: Persist favorites in Electron and expose typed IPC

**Objective:** Make favorites durable, encrypted where applicable, and available to every renderer window through the main process.

**Files:**
- Modify: `apps/desktop/electron/main.ts:611-621,6859-7114,9754-9960`
- Modify: `apps/desktop/electron/preload.ts:74-96`
- Modify: `apps/desktop/src/global.d.ts:15-112`
- Test: `apps/desktop/electron/gateway-favorites-main.test.ts` or the existing Electron main-process test seam used for userData persistence

**Step 1: Add failing persistence/IPC contract tests**

Test against a temporary userData/config fixture, not a mocked return value only:

- Missing or malformed `favorites` data in an older `connection.json` loads as an empty list and preserves the old connection fields.
- Save/upsert writes a favorite with an encrypted token and no plaintext token in the JSON file.
- Read returns favorites in stored order with secret-free DTOs.
- Delete and reorder are id-based, reject unknown ids, and preserve unrelated connection/profile settings.
- A favorite can be applied to the global scope or a named profile scope without exposing its token to the renderer.
- Applying a favorite routes through the same primary-vs-pool decision used by `applyConnectionChange`; primary application emits the existing soft-switch event, while a non-primary scope stops/restarts only that pool backend.

Run: `cd apps/desktop && npx vitest run --project electron electron/gateway-favorites-main.test.ts`

Expected: FAIL because the handlers and storage fields do not exist.

**Step 2: Add a version-tolerant favorites collection to connection persistence**

Extend the parsed connection document with an ordered `favorites` array or equivalent id-keyed records plus order. Preserve the current `{ mode, remote, profiles }` shape for existing users. Use the existing `writeFileAtomic`, connection-config cache invalidation, and `encryptDesktopSecret`/`decryptDesktopSecret` behavior. Do not add a user-facing `HERMES_*` environment variable.

Recommended persisted record shape:

```ts
{
  id: 'generated-stable-id',
  name: 'Production',
  mode: 'remote',
  url: 'https://gateway.example.com/hermes',
  authMode: 'token',
  token: { /* existing encrypted secret shape */ },
  org: '',
  ssh: { /* normalized SSH fields when mode === 'ssh' */ }
}
```

Do not return `token` from `getGatewayFavorites`. For OAuth favorites, `oauthConnected` is derived from the existing URL-scoped OAuth/native-token liveness checks. For SSH favorites, `hasToken` is only a display/status hint; SSH bootstrap remains responsible for minting/reusing its served token.

**Step 3: Add the preload bridge and renderer types**

Expose these narrow methods, with exact return types:

- `getGatewayFavorites()`
- `saveGatewayFavorite(input)` for add/update
- `deleteGatewayFavorite(id)`
- `reorderGatewayFavorites(ids)`
- `applyGatewayFavorite(id, profile?)`

The apply operation must accept only an id and optional connection scope. It must read/decrypt the secret inside Electron, call the existing connection coercion/apply path, and return the sanitized resulting connection config. It must not accept a renderer-supplied token for switching.

**Step 4: Run typecheck and focused Electron tests**

Run:

```bash
cd apps/desktop
npm run typecheck
npx vitest run --project electron electron/gateway-favorites.test.ts electron/gateway-favorites-main.test.ts electron/connection-config.test.ts
```

Expected: PASS.

---

### Task 3: Add the renderer favorite store and one switching action

**Objective:** Give Settings and the main UI one shared cache and one connection-switching function, including active-profile routing and failure recovery.

**Files:**
- Create: `apps/desktop/src/store/gateway-favorites.ts`
- Test: `apps/desktop/src/store/gateway-favorites.test.ts`
- Modify: `apps/desktop/src/store/profile.ts` only if the new helper needs a narrow exported activation primitive

**Step 1: Write failing store tests**

Cover:

- Refresh populates the store from `window.hermesDesktop.getGatewayFavorites()`.
- Add/update/delete/reorder update the cache only from the authoritative IPC response, not from guessed local state.
- `switchToGatewayFavorite(id)` targets `normalizeProfileKey($activeGatewayProfile.get())`, calls `applyGatewayFavorite(id, profile)` once, then ensures that profile's gateway is active and synchronizes `$connection` through the existing `ensureGatewayProfile()` path.
- Repeated clicks serialize through one in-flight switch and do not race the active gateway pointer.
- Failed application leaves the previous favorite/current connection state visible and surfaces an error through the existing notification path.
- A primary apply relies on the existing soft-switch event; a named non-primary profile is reactivated after the pool apply so it does not sit on a closed/stale socket.

Run: `cd apps/desktop && npx vitest run --project ui src/store/gateway-favorites.test.ts`

Expected: FAIL because the store/action does not exist.

**Step 2: Implement the minimal Nanostore and action**

Use an atom for the favorite list and a small status atom for loading/switching id. Keep it global to the desktop window, matching the existing profile rail stores. The store should call `notifyError`/`notify` only at the boundary where an authoritative operation fails.

The switching action should be the only UI-facing route to `applyGatewayFavorite`; Settings “Use” buttons and the statusbar menu must both call it.

**Step 3: Run focused renderer tests**

Run: `cd apps/desktop && npx vitest run --project ui src/store/gateway-favorites.test.ts src/store/profile.test.ts`

Expected: PASS.

---

### Task 4: Add favorite list management to Settings → Gateway

**Objective:** Let users create and manage favorites without disturbing the existing connection-mode controls.

**Files:**
- Modify: `apps/desktop/src/app/settings/gateway-settings.tsx:148-250,432-506,989-1490`
- Modify: `apps/desktop/src/app/settings/primitives.tsx` only if an existing row primitive cannot express the list cleanly
- Modify: `apps/desktop/src/app/settings/gateway-settings.test.tsx`
- Modify: `apps/desktop/src/app/settings/gateway-settings.test.ts`
- Modify: `apps/desktop/src/app/settings/types.ts` only if a shared favorite input type belongs there

**Step 1: Write failing UI tests**

Test through user-visible behavior:

- Settings loads and displays the favorite list in stored order above or beside the connection editor.
- Empty state explains that favorites can be added from a configured remote/cloud/SSH connection.
- Add favorite validates the name and current remote configuration; local/inherit mode cannot be saved as a favorite.
- Token mode allows a newly entered token to be stored, while leaving the token blank preserves the existing saved token for the same configured gateway when appropriate.
- OAuth favorite creation requires a signed-in gateway but never renders or serializes a token.
- Cloud and SSH favorites show their relevant identity fields without pretending they are direct URL remotes.
- Edit, delete, and reorder work and refresh the authoritative list after each write.
- “Use” applies the favorite and reports success/failure without navigating away from Settings. The existing profile scope selector is respected.
- Existing gateway settings tests for profile inheritance and current save/apply payloads continue to pass.

Run: `cd apps/desktop && npx vitest run --project ui src/app/settings/gateway-settings.test.tsx src/app/settings/gateway-settings.test.ts`

Expected: FAIL because the favorite controls and bridge mocks do not exist.

**Step 2: Implement the UI using existing primitives**

Add a flat `GatewayFavoritesSection` within `GatewaySettings` or a focused sibling component. Use `ListRow`, `Button`, `Input`, existing select/dialog primitives, the existing `CONTROL_TEXT`, and current settings styling. Do not create card-in-card chrome or ad hoc button variants.

Use a small add/edit dialog or inline editor with:

- Name
- Connection kind and the relevant fields
- “Save favorite” / “Update favorite”
- “Use” on each row
- Delete with confirmation if the existing dialog convention warrants it
- Reordering via existing reorder primitives if the list needs drag handles; otherwise provide explicit move controls and keep the stored order deterministic

When saving from the current editor, pass the same normalized connection fields already used by `payload()`. Keep remote tokens in Electron; if an existing token is needed to clone a connection, resolve that inside the main handler rather than expanding `DesktopConnectionConfig` to return secrets.

Refresh the list on mount and after a successful add/update/delete/reorder. Do not reload the whole Gateway settings editor unnecessarily when only the favorites list changes.

**Step 3: Run focused UI tests and typecheck**

Run:

```bash
cd apps/desktop
npx vitest run --project ui src/app/settings/gateway-settings.test.tsx src/app/settings/gateway-settings.test.ts
npm run typecheck
```

Expected: PASS.

---

### Task 5: Add quick switching to the main Gateway menu

**Objective:** Make saved favorites reachable from the statusbar Gateway control without opening Settings.

**Files:**
- Modify: `apps/desktop/src/app/shell/gateway-menu-panel.tsx:90-229`
- Modify: `apps/desktop/src/app/shell/hooks/use-statusbar-items.tsx:235-247,355-378,380-512`
- Modify: `apps/desktop/src/store/gateway-favorites.ts` (only for selectors/state already defined in Task 3)
- Create or modify: `apps/desktop/src/app/shell/gateway-menu-panel.test.tsx`

**Step 1: Write failing menu tests**

Cover:

- The menu renders a “Favorite gateways” section with the same order as Settings.
- The current favorite has a selected/connected indicator; current connection status remains visible even when no favorite matches.
- Clicking a favorite calls the shared `switchToGatewayFavorite()` action with the active profile scope, closes the menu only after the switch has been accepted, and does not open Settings or the command center.
- While switching, the clicked item is disabled or shows a bounded loading state; a failed switch leaves the menu usable and shows the existing notification.
- No favorites renders a compact empty state and an action/link to Settings → Gateway for management.
- The existing restart/system/log/platform behavior remains unchanged.

Run: `cd apps/desktop && npx vitest run --project ui src/app/shell/gateway-menu-panel.test.tsx`

Expected: FAIL because the section and action do not exist.

**Step 2: Implement the menu section**

Subscribe to the favorite store in `GatewayMenuPanel`; do not fetch directly from Electron on every render. Keep the current `useGatewayLogTail()` polling and status rows intact. Add a separator/section below the connection summary, with concise labels derived from favorite name and host/remote kind. Use `Star`/existing icon exports only if the icon set already provides one; otherwise use a text affordance rather than adding an icon dependency.

Pass the current active profile from `$activeGatewayProfile` through the menu or read it in the shared store action. Keep `useStatusbarItems` responsible for wiring the menu, not for connection policy.

**Step 3: Run menu and gateway-switch tests**

Run: `cd apps/desktop && npx vitest run --project ui src/app/shell/gateway-menu-panel.test.tsx src/store/gateway-favorites.test.ts src/store/gateway-switch.test.ts`

Expected: PASS.

---

### Task 6: Mark the active favorite and reconcile manual edits

**Objective:** Keep quick-switch selection accurate after boot, reconnects, profile changes, and manual Gateway settings edits.

**Files:**
- Modify: `apps/desktop/src/global.d.ts:455-476,503-528` if the sanitized connection needs `favoriteId`
- Modify: `apps/desktop/electron/main.ts:6945-7114,7151-7239,7515-8120` to preserve/propagate the association through resolution
- Modify: `apps/desktop/src/app/gateway/hooks/use-gateway-boot.ts` or the connection-ready callback path
- Modify: `apps/desktop/src/store/gateway-favorites.ts`
- Test: `apps/desktop/electron/connection-config.test.ts`, renderer boot/menu/settings tests

**Step 1: Write failing association tests**

- Applying favorite A makes the resulting sanitized config/live connection identify A.
- A normal manual save of a different endpoint clears the association.
- Reconnect and boot preserve A instead of losing the selection marker.
- Same URL favorites with different ids do not both appear selected.
- Switching profiles selects the favorite associated with that profile's active connection, not the favorite last used by another scope.

**Step 2: Implement association propagation**

Add an optional `favoriteId` metadata field to the persisted connection block and sanitized connection/config shape, or use one equivalent main-owned association map if that fits the existing config migration better. Keep it non-secret. `applyGatewayFavorite` sets it; ordinary `saveConnectionConfig` clears it unless the save explicitly preserves the same favorite. Ensure every resolution path that builds `HermesConnection` carries the metadata through so boot/reconnect can refresh the renderer store.

**Step 3: Run focused regressions**

Run:

```bash
cd apps/desktop
npx vitest run --project electron electron/connection-config.test.ts
npx vitest run --project ui src/app/gateway/hooks/use-gateway-boot.test.tsx src/app/shell/gateway-menu-panel.test.tsx src/app/settings/gateway-settings.test.tsx
```

Expected: PASS.

---

### Task 7: i18n, documentation, and full verification

**Objective:** Finish the feature to desktop contribution standards and prove the real seams still work.

**Files:**
- Modify: `apps/desktop/src/i18n/types.ts:498-622` and `:1984+` for shell copy
- Modify: `apps/desktop/src/i18n/en.ts:596-733` and `:2407-2420`
- Modify: `apps/desktop/src/i18n/zh.ts`, `zh-hant.ts`, `ja.ts`, `ar.ts` at their corresponding `settings.gateway` and `shell.gatewayMenu` blocks, using the locale fallback conventions already used in the repository
- Modify: `website/docs/user-guide/desktop.md` if the desktop user guide documents Gateway settings/statusbar behavior
- Test: relevant i18n/typecheck/UI suites

**Step 1: Add typed copy**

Add strings for favorite section title/description, add/edit/save/update/delete/use/reorder, empty state, validation/error messages, switching state, selected state, and the menu section/Settings link. Keep the English source complete; partial locales should use the existing fallback mechanism rather than duplicating English strings by hand.

**Step 2: Verify formatting, lint, typecheck, and focused behavior**

Run:

```bash
cd apps/desktop
npm run typecheck
npm run lint
npx vitest run --project ui src/app/settings/gateway-settings.test.tsx src/app/settings/gateway-settings.test.ts src/app/shell/gateway-menu-panel.test.tsx src/store/gateway-favorites.test.ts src/store/profile.test.ts src/store/gateway-switch.test.ts
npx vitest run --project electron electron/gateway-favorites.test.ts electron/connection-config.test.ts electron/connection-apply.test.ts
```

Expected: all commands pass.

**Step 3: Exercise the desktop path**

Run the repository's desktop check appropriate to the environment:

```bash
cd apps/desktop
npm run check:lint
npm run check:test:ui
npm run test:desktop:platforms
```

If a packaged/E2E environment is available, also run `npm run test:e2e` and cover the Gateway menu → favorite → reconnect path against a controlled fixture. Verify a temporary userData directory contains encrypted token material only, that the active gateway changes, and that the previous session list does not leak across a primary mode rehome.

**Step 4: Update the plan/implementation notes with real output**

Record only commands and outcomes that actually ran. Do not claim packaged/E2E coverage if the environment cannot launch Electron or a remote fixture.

---

## Final acceptance criteria

- A user can create a named favorite for each supported remote gateway kind, including auth fields needed by that kind.
- Favorites survive app restart, retain order, and do not leak tokens to renderer state or logs.
- Settings → Gateway owns all list management and respects existing global/profile scope semantics.
- The main Gateway menu lists favorites, identifies the active one, and switches without opening Settings.
- Switching uses the existing primary soft-rehome or non-primary pool lifecycle, clears/reseeds gateway-bound session state where required, and keeps unrelated background sockets alive.
- OAuth reauthentication, SSH host-key/served-token behavior, Cloud org/agent behavior, manual gateway edits, and legacy connection.json files remain compatible.
- Unit, UI, typecheck, lint, and available desktop integration checks pass with real tool output.
