# Session contributions and owned URL viewers

These generic desktop APIs are exported by `@hermes/plugin-sdk`. They do not provide a remote desktop protocol; a plugin owns its viewer HTML, binary transport, authentication, and status model.

## Session contribution areas

```ts
SESSION_AREAS.statusStack // session.statusStack — above the addressed composer
SESSION_AREAS.tileBadge   // session.tileBadge — main and split-session tab lead
SESSION_AREAS.listBadge   // session.listBadge — sidebar session row

interface PluginSessionContext {
  runtimeSessionId: string | null
  storedSessionId: string | null
  profile: string
  connectionId: string
}
```

Register `ctx.register({ id, area, data: { render: Component } })`. The component receives `{ session }`, may use React hooks, and can return null. Use `data.render`, not the top-level `render` used by other areas. Unregistering or disabling the plugin removes its contributions; rendering errors are contained by the existing contribution boundary.

The host resolves the session belonging to each surface, not the globally focused tile. Unresolved or ambiguous ownership suppresses the contribution. Stored-only list rows can have a null runtime ID. Cache plugin state by connection + profile + session identity. Keep list and tile badges compact and put actions in the status row, using SDK UI primitives and theme tokens.

`ctx.rest(path, { scope: session, ...options })` explicitly routes a request to this session's connection/profile; omitted scope retains the legacy ambient routing behavior. No credentials are included in the contribution context.

## Explicit preview action

```ts
const opened = await host.openPreview({ url, label: 'Viewer', session })
```

The existing preview pane opens an absolute HTTP(S) URL. Relative URLs, embedded credentials, unsafe schemes, and stale/mismatched session contexts return false. Viewer tabs are transient and omitted from persisted preview tabs. They use the generic `PreviewTarget.browserContext: 'isolated'` intent: a fresh nonpersistent Electron session, no inherited Browser cookies or gateway auth headers, and no browser-profile import/onboarding offer. `transient` alone remains only a persistence policy; ordinary transient browsers retain normal consent behavior. Viewers do not replace ordinary Browser tabs, and unrelated consent changes do not recreate their guest. The storage-based Browser popout is unavailable for runtime-only tabs; use `ctx.os.openViewer` for a native viewer window instead. The guest has no app preload or Node integration and retains the existing sandbox/context isolation and preview controls. The action does not tunnel a remote service or add gateway credentials: obtain a client-reachable URL with a scoped ticket from the plugin backend. A true result means the pane accepted the target, not that the remote service successfully loaded.

Call from a user action, not a background status update. Native viewer/preview navigation does not inherit `ctx.rest` headers. `ctx.socket` remains JSON-only; binary protocols need their own ticket-authorized WebSocket in the viewer page.

## Owned native viewer

```ts
await ctx.os.openViewer({ id: 'watch-session', url, title: 'Session viewer', session })
await ctx.os.closeViewer('watch-session')
```

IDs use `[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}`; titles are at most 120 characters and cannot contain ASCII control characters. The main-process payload accepts only id, url and title; renderer session validation precedes IPC. There are no caller-provided Electron options, preloads, partitions, dimensions, or app classes.

Windows are keyed by originating app renderer + plugin + id. Reopening the same id refreshes its URL without focusing; changing origin requires closing it first. The native title is locked to `Hermes Viewer [<plugin>/<id>] — <title>`. Creation uses existing browser-popup dimensions and the session-window registry. Initial reveal uses `showInactive`; navigation does not request focus. User interaction can still focus a viewer normally.

Remote content is sandboxed with Node integration off, context isolation on, no app preload, no webviews, and a nonpersistent per-viewer partition. Permissions, downloads, child popups, cross-origin navigation and redirects are denied. IPC only accepts the main frame at the trusted app renderer URL (including an exact protocol check for file builds). Plugin disposal closes owned windows; app-renderer reload, crash, or destruction closes all its viewers. Calls from disposed plugin contexts fail closed.

## Lifecycle-bound keep-alive (optional)

Both `PluginPreviewInput` and `PluginViewerInput` accept the additive field
`onKeepAlive?: () => Promise<void>`. Their open actions still return `Promise<boolean>`.

```ts
// Capture the contribution's owner and the plugin REST closure at the user action.
const owner = { ...session }
const rest = ctx.rest
const onKeepAlive = async () => {
  await rest('/viewer/renew', { method: 'POST', scope: owner, body: { ticket } })
}
await host.openPreview({ url, label: 'Viewer', session: owner, onKeepAlive })
// Or, for an owned native window:
await ctx.os.openViewer({ id: 'watch-session', url, title: 'Viewer', session: owner, onKeepAlive })
```

The host schedules an initial renewal once the preview's actual guest document
is ready / the native viewer successfully opens, then waits 60 seconds after each completed renewal.
It checks liveness before invoking the callback and again after completion.
Renewals never overlap. Callback or liveness-probe failures are retried at
5-second intervals, with at most three consecutive failed attempts; success
resets that budget. Exhaustion stops renewal without reopening, navigating, or
closing the viewer. Callback errors do not reject the open action or become
unhandled rejections. A callback already in flight cannot be cancelled, but its
late result cannot restart a retired loop. The backend remains responsible for
finite expiry, authentication, ownership and revocation.

Preview liveness uses the **original transient tab and ready guest-document identity**,
not the visible/active tab or a cached URL alone. The preview pane publishes a
memory-only document identity at `dom-ready` and invalidates it on full navigation,
renderer loss, destruction and unmount; detached guests fail the live check too.
Reopening/replacing that tab or document, closing it, or navigating its guest away
retires its callback. Genuinely mounted parked tabs and switches to another profile/session do not.
The native viewer checks the originating renderer + plugin + id + exact initial
URL against its registry and the actual WebContents location. Document-location
comparison preserves origin, pathname and query and ignores the fragment, so a
guest can remove a bootstrap ticket fragment. Navigation away retires renewal
even if the guest subsequently returns. Explicit close/replacement or plugin
disposal stops the native renderer loop; native close/navigation is detected by
the next liveness check. Renderer unload ends both kinds of loop naturally.

`isPluginViewerOpen(pluginId, id, initialUrl): Promise<boolean>` is a narrow,
read-only **internal native bridge** (`hermes:window:isPluginViewerOpen`), guarded
by the same trusted main-frame check as open/close. It is not a model tool or a
guest API. Only id/url/title cross the open IPC boundary; the callback and its
captured REST scope remain in the trusted renderer. Older bridges without the
liveness capability still open viewers but do not start native keep-alives.
Never resolve the callback's owner from the foreground profile at renewal time.

### Linux limitation

Electron 40.10.2 does not expose per-window WM_CLASS/app_id. The popup retains the host process identity; only its safe, host-prefixed title distinguishes it. There is **no guarantee of a separate `hermes-plugin-viewer` class**. Changing the global app name would affect unrelated windows, and an X11-only xprop workaround would not cover Wayland. Neither is used. See [Electron issue 45866](https://github.com/electron/electron/issues/45866) and [BaseWindow options](https://www.electronjs.org/docs/latest/api/structures/base-window-options).

A distinct per-viewer class remains an acceptance gap. Actual focus behavior and compositor identity require native E2E on the target desktop; unit tests prove calls/policy, not window-manager behavior.

## Verification

From `apps/desktop`, run the contribution/viewer behavior tests and all three TypeScript projects:

```sh
npx vitest run src/contrib/session.test.tsx src/contrib/plugin-viewer.test.ts \
  src/sdk/preview.test.ts src/sdk/viewer-keep-alive.test.ts src/sdk/preview-lifecycle.test.tsx \
  src/app/chat/composer/status-stack/session-contribution.test.tsx \
  src/app/chat/session-tile-contribution.test.tsx src/app/chat/sidebar/session-row.test.tsx \
  src/api/plugins.test.ts src/contrib/plugin.test.ts \
  electron/plugin-viewer-windows.test.ts electron/viewer-guest-policy.test.ts
npm run typecheck
npm run build
```

These exercise session ownership, stale contexts, first-open consent behavior, native viewer lifecycle, permission denial and guest preference validation. The build checks renderer assets and Electron main/preload bundles with `assert-dist-built`. Native compositor focus and a plugin's actual streaming/input protocol require separate integration tests; a unit-test pass is not a claim of those behaviors.
