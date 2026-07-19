# Desktop Zoom Renderer Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent a stale initial desktop zoom read from overwriting a newer main-process zoom event.

**Architecture:** Keep the existing Nanostore and preload IPC API. During module initialization, subscribe first and record whether `onChanged` has delivered newer state; apply the pending `get()` result only when no event has arrived.

**Tech Stack:** TypeScript, Nanostores, Vitest, jsdom

## Global Constraints

- Modify only `apps/desktop/src/store/zoom.ts` and add `apps/desktop/src/store/zoom.test.ts`.
- Keep persistence, IPC contracts, Settings components, and main-process zoom behavior unchanged.
- Add no dependency or reusable synchronization abstraction.
- Run desktop tests with Node 22, matching `.github/workflows/js-tests.yml`.

---

### Task 1: Guard the initial renderer zoom read

**Files:**
- Modify: `apps/desktop/src/store/zoom.ts:19-22`
- Create: `apps/desktop/src/store/zoom.test.ts`

**Interfaces:**
- Consumes: `window.hermesDesktop.zoom.get(): Promise<{ level: number; percent: number }>` and `onChanged(callback): () => void`.
- Produces: unchanged `$zoomPercent` public store; newer `onChanged` state wins over the initial read.

- [ ] **Step 1: Write the failing race regression**

```ts
import { beforeEach, describe, expect, it, vi } from 'vitest'

describe('zoom store initialization', () => {
  beforeEach(() => {
    vi.resetModules()
  })

  it('does not let a stale initial read overwrite a newer change event', async () => {
    type Payload = { level: number; percent: number }

    let emitChanged: ((payload: Payload) => void) | undefined
    let resolveGet: ((payload: Payload) => void) | undefined
    const get = vi.fn(
      () => new Promise<Payload>(resolve => {
        resolveGet = resolve
      })
    )

    ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = {
      zoom: {
        get,
        setPercent: vi.fn(),
        onChanged: vi.fn((callback: (payload: Payload) => void) => {
          emitChanged = callback
          return vi.fn()
        })
      }
    }

    const { $zoomPercent } = await import('./zoom')
    emitChanged?.({ level: 3, percent: 125 })
    resolveGet?.({ level: 0, percent: 100 })
    await Promise.resolve()

    expect($zoomPercent.get()).toBe(125)
  })
})
```

- [ ] **Step 2: Run the focused test to verify RED**

Run:

```bash
PATH=/opt/homebrew/opt/node@22/bin:$PATH npm run test:ui --workspace apps/desktop -- src/store/zoom.test.ts
```

Expected: FAIL because `$zoomPercent.get()` is 100 after the stale `get()` promise resolves.

- [ ] **Step 3: Implement the minimal initialization guard**

Replace the initialization block body with:

```ts
const zoom = window.hermesDesktop.zoom
let receivedChange = false

zoom.onChanged(({ percent }) => {
  receivedChange = true
  $zoomPercent.set(percent)
})
void zoom.get().then(({ percent }) => {
  if (!receivedChange) $zoomPercent.set(percent)
})
```

- [ ] **Step 4: Run focused and store tests to verify GREEN**

Run:

```bash
PATH=/opt/homebrew/opt/node@22/bin:$PATH npm run test:ui --workspace apps/desktop -- src/store/zoom.test.ts
PATH=/opt/homebrew/opt/node@22/bin:$PATH npm run test:ui --workspace apps/desktop -- src/store
PATH=/opt/homebrew/opt/node@22/bin:$PATH npm run typecheck --workspace apps/desktop
```

Expected: focused regression passes, all 345 existing store tests plus the new regression pass, typecheck exits 0.

- [ ] **Step 5: Review and checkpoint**

Run `git diff --check`, verify only the two scoped implementation files changed, then commit:

```bash
git add apps/desktop/src/store/zoom.ts apps/desktop/src/store/zoom.test.ts
git commit -m "wip: prevent stale desktop zoom initialization"
```
