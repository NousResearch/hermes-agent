import { configure } from '@testing-library/react'
import { afterAll } from 'vitest'

// Some Node runtimes expose placeholder Web Storage globals. Vitest preserves
// globals that already exist when it populates the jsdom environment, which can
// leave window.localStorage without the browser Storage methods. The jsdom
// instance is initialized before setup files run, so bind its real stores here.
const jsdomWindow = (
  globalThis as typeof globalThis & {
    jsdom?: { window: Window }
  }
).jsdom?.window

if (jsdomWindow) {
  const originalDescriptors = new Map<PropertyKey, PropertyDescriptor | undefined>()
  const installStorage = (name: 'localStorage' | 'sessionStorage', storage: Storage) => {
    originalDescriptors.set(name, Object.getOwnPropertyDescriptor(globalThis, name))
    Object.defineProperty(globalThis, name, {
      configurable: true,
      value: storage,
      writable: true
    })
  }

  installStorage('localStorage', jsdomWindow.localStorage)
  installStorage('sessionStorage', jsdomWindow.sessionStorage)

  afterAll(() => {
    for (const [name, descriptor] of originalDescriptors) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor)
      else Reflect.deleteProperty(globalThis, name)
    }
  })
}

// jsdom reports an error every time its unimplemented canvas context is
// requested, then returns null. Renderer components already treat null as the
// no-canvas fallback, so preserve that behavior without polluting test output.
if (typeof HTMLCanvasElement !== 'undefined') {
  const getContextDescriptor = Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype, 'getContext')
  Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
    configurable: true,
    value: () => null,
    writable: true
  })

  afterAll(() => {
    if (getContextDescriptor) {
      Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', getContextDescriptor)
    }
  })
}

// React 19 + Testing Library 16: opt into the act environment so render(),
// fireEvent(), and findBy* queries automatically flush state updates without
// spurious "not wrapped in act(...)" warnings.
;(globalThis as any).IS_REACT_ACT_ENVIRONMENT = true

// findBy*/waitFor default to a 1000ms deadline — too tight for async-heavy
// panels (radix menus, refetch chains) when the full suite runs under xdist
// CPU contention in CI. Success still resolves the instant the node appears;
// the wider deadline only absorbs a starved runner, killing timing flakes.
configure({ asyncUtilTimeout: 5000 })
