// jsdom implements no layout and no animation, so component libraries that
// call those APIs unconditionally throw on mount. These install inert
// stand-ins: enough for the component to render, never enough to assert on. A
// test that needs one of them to actually report should install its own.

import type { Mock } from 'vitest'
import { onTestFinished, vi } from 'vitest'

class InertResizeObserver {
  disconnect() {}
  observe() {}
  unobserve() {}
}

/** A ResizeObserver that accepts observers and never calls them back. */
export function stubResizeObserver() {
  vi.stubGlobal('ResizeObserver', InertResizeObserver)
}

/** The pointer-capture and scroll calls Radix and cmdk make while opening a
 *  popover, menu, or combobox — and again on the item they focus. */
export function stubMenuDomApis() {
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => undefined
  Element.prototype.releasePointerCapture ??= () => undefined
  Element.prototype.scrollIntoView ??= () => undefined
}

type StubbedStorageMethod = 'clear' | 'getItem' | 'key' | 'removeItem' | 'setItem'

export type StorageStub = { [K in StubbedStorageMethod]: Mock<Storage[K]> } & {
  /** Put the real binding back. Runs automatically when the test finishes. */
  restore: () => void
}

/** Replace `localStorage` for the current test with an observable stand-in:
 *  every method is a `vi.fn` delegating to the real store, unless `overrides`
 *  supplies one (say a `setItem` that throws).
 *
 *  Swapping the whole object is deliberate, because neither spy reaches this
 *  code. jsdom's `localStorage` is a WebIDL legacy platform object behind a
 *  proxy with a named-property setter, so `vi.spyOn(localStorage, 'setItem')`
 *  never installs: the `defineProperty` is routed into the store, which saves
 *  an entry literally called "setItem" and leaves the real method on
 *  `Storage.prototype` in place. Spying on `Storage.prototype` has the mirror
 *  problem — real against jsdom, inert against the plain object
 *  `vitest.setup.ts` installs when the runtime has no usable localStorage of
 *  its own (Node 26, which `.nvmrc` and CI pin). Both failures are silent, and
 *  a stub that never runs leaves `not.toThrow()` passing on an error that was
 *  never thrown and `mock.calls.every(...)` passing on an empty array.
 *
 *  Methods left un-overridden stay real, so what a test reads back after a
 *  denied write is a fact about the store and not an artifact of the stub. */
export function stubStorage(overrides: Partial<Pick<Storage, StubbedStorageMethod>> = {}): StorageStub {
  const real = window.localStorage

  const methods = {
    clear: vi.fn<Storage['clear']>(overrides.clear ?? (() => real.clear())),
    getItem: vi.fn<Storage['getItem']>(overrides.getItem ?? (key => real.getItem(key))),
    key: vi.fn<Storage['key']>(overrides.key ?? (index => real.key(index))),
    removeItem: vi.fn<Storage['removeItem']>(overrides.removeItem ?? (key => real.removeItem(key))),
    setItem: vi.fn<Storage['setItem']>(overrides.setItem ?? ((key, value) => real.setItem(key, value)))
  }

  const stub: Storage = {
    get length() {
      return real.length
    },
    ...methods
  }

  // `globalThis` and `window` are distinct property slots under vitest's jsdom
  // environment, and code under test reaches for either — override both so a
  // bare `localStorage` can never disagree with `window.localStorage`.
  const targets = [globalThis, globalThis.window].filter(Boolean)
  const originals = targets.map(target => [target, Object.getOwnPropertyDescriptor(target, 'localStorage')] as const)

  for (const target of targets) {
    Object.defineProperty(target, 'localStorage', { configurable: true, value: stub, writable: true })
  }

  let restored = false

  const restore = () => {
    if (restored) {
      return
    }

    restored = true

    for (const [target, descriptor] of originals) {
      if (descriptor) {
        Object.defineProperty(target, 'localStorage', descriptor)
      } else {
        delete (target as { localStorage?: Storage }).localStorage
      }
    }
  }

  // A failed assertion skips a manual restore, and the next test's stub would
  // then capture THIS stub as the original and write it back on its own
  // restore. One dead stub must not outlive its test.
  onTestFinished(restore)

  return { ...methods, restore }
}

/** `localStorage` whose writes fail, the way a browser fails on a full quota or
 *  a blocked origin. Returns the undo. */
export function denyStorageWrites(error: Error): () => void {
  return stubStorage({
    setItem: () => {
      throw error
    }
  }).restore
}
