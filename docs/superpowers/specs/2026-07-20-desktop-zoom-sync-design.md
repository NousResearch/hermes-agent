# Desktop zoom renderer sync design

## Problem

On startup, the renderer initializes `$zoomPercent` with two independent inputs:

1. an asynchronous `zoom.get()` request; and
2. `zoom.onChanged` events from the main process.

If a restored 125% event arrives before an older `zoom.get()` request resolves with 100%, the stale request overwrites the newer event. The window remains correctly scaled while Settings reports 100%.

## Design

Keep the existing store and IPC API. Subscribe to `zoom.onChanged` before starting the initial `zoom.get()` request. Track whether any change event has arrived during initialization. Apply the initial request result only when no change event has arrived; otherwise discard it as stale.

This is local initialization state, not a new exported store or abstraction. `setZoomPercent()` and main-process persistence remain unchanged.

## Alternatives rejected

- Reordering the subscription before `zoom.get()` without a guard does not fix the race: the older promise can still resolve last.
- Removing `zoom.get()` risks leaving Settings at its default until the main process emits an event.
- Adding a general synchronization helper or sequence framework is unnecessary for one initial request.

## Testing

Add `apps/desktop/src/store/zoom.test.ts`. Stub `window.hermesDesktop.zoom` with a deferred `get()` promise and a captured `onChanged` callback, then dynamically import the store after resetting modules.

The regression sequence is:

1. import the store while `get()` remains pending;
2. emit `onChanged({ percent: 125 })`;
3. resolve the older `get()` request with `{ percent: 100 }`;
4. assert `$zoomPercent` remains 125.

Run the focused regression and the desktop store suite with the repository CI runtime:

```bash
PATH=/opt/homebrew/opt/node@22/bin:$PATH npm run test:ui --workspace apps/desktop -- src/store/zoom.test.ts
PATH=/opt/homebrew/opt/node@22/bin:$PATH npm run test:ui --workspace apps/desktop -- src/store
```

## Scope

- Modify `apps/desktop/src/store/zoom.ts`.
- Add `apps/desktop/src/store/zoom.test.ts`.
- Do not change persistence, IPC contracts, Settings components, or main-process zoom behavior.
