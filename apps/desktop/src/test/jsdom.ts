// jsdom implements no layout and no animation, so component libraries that
// call those APIs unconditionally throw on mount. These install inert
// stand-ins: enough for the component to render, never enough to assert on. A
// test that needs one of them to actually report should install its own.

import { vi } from 'vitest'

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

/** localStorage whose writes fail, the way a browser fails on a full quota or a
 *  blocked origin. Returns the undo.
 *
 *  Swapping the whole object is deliberate. jsdom's `localStorage` is a WebIDL
 *  legacy platform object behind a proxy with a named-property setter, so
 *  `vi.spyOn(localStorage, 'setItem')` never installs: the `defineProperty` is
 *  routed into the store, which saves an entry literally called "setItem" and
 *  leaves the real method on `Storage.prototype` in place. The write under test
 *  then succeeds while the test believes storage is denied. Replacing the
 *  binding works whichever Storage the runtime gave us — jsdom's, or the plain
 *  object `vitest.setup.ts` installs when the runtime has no usable one. (A
 *  `Storage.prototype` spy has the mirror problem: real against jsdom, inert
 *  against that plain object, which inherits nothing from it.)
 *
 *  Reads and removals stay real, so "nothing was persisted" remains a fact
 *  about the store and not an artifact of the stub. */
export function denyStorageWrites(error: Error): () => void {
  const real = window.localStorage

  const denied: Storage = {
    get length() {
      return real.length
    },
    clear: () => real.clear(),
    getItem: key => real.getItem(key),
    key: index => real.key(index),
    removeItem: key => real.removeItem(key),
    setItem: () => {
      throw error
    }
  }

  // `globalThis` and `window` are distinct property slots under vitest's jsdom
  // environment, and code under test reaches for either — override both so a
  // bare `localStorage` can never disagree with `window.localStorage`.
  const targets = [globalThis, globalThis.window].filter(Boolean)
  const originals = targets.map(target => [target, Object.getOwnPropertyDescriptor(target, 'localStorage')] as const)

  for (const target of targets) {
    Object.defineProperty(target, 'localStorage', { configurable: true, value: denied, writable: true })
  }

  return () => {
    for (const [target, descriptor] of originals) {
      if (descriptor) {
        Object.defineProperty(target, 'localStorage', descriptor)
      } else {
        delete (target as { localStorage?: Storage }).localStorage
      }
    }
  }
}
