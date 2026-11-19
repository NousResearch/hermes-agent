# Capabilities catalog

Data layer for the Capabilities panes (plugins, skills, toolsets). Catalog
fetches go through a shared TanStack Query client (`queryClient` in
`catalog-data.ts`), and non-active catalogs are warmed ahead of tab switches
by `prefetchCatalogWhenIdle`.

## The `requestIdleCallback` guard

`prefetchCatalogWhenIdle` defers its prefetch with `requestIdleCallback` so
the warm-up never competes with the mount it follows. **`requestIdleCallback`
is not available in every renderer the webapp runs in** — notably Safari
before 16.4 and some embedded WKWebView shells (the absence is a
JavaScriptCore `ReferenceError`, surfacing to users as the
"workspace failed to render" error boundary when Capabilities mounts).

The function therefore checks `typeof requestIdleCallback === 'function'`
first and falls back to `setTimeout(..., 0)` on engines without it. The
fallback is behaviour-identical on capable browsers and merely drops the
idle-timing hint on the others — a prefetch is advisory, so firing it on the
next macrotask costs nothing.

This is a house convention, not a one-off: every other `requestIdleCallback`
call site in the app is guarded the same way (`idle-mount.tsx` degrades to an
eager mount; `mermaid-render-cache.ts` falls back to `setTimeout`). A future
call site added without the guard will reintroduce the WebKit crash, so
please guard new ones.
