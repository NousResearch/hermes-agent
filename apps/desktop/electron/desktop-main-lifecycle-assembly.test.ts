import { expect, test, vi } from 'vitest'

const calls = vi.hoisted(() => [] as string[])

vi.mock('./desktop-app-lifecycle-runtime', () => ({ createDesktopAppLifecycleRuntime: () => {
  calls.push('lifecycle')

  return { isPrimaryInstance: true, handleDeepLink: vi.fn() }
} }))
vi.mock('./desktop-held-quit-runtime', () => ({ createDesktopHeldQuitRuntime: () => {
  calls.push('quit-guard')

  return vi.fn()
} }))
vi.mock('./desktop-quit-runtime', () => ({ registerDesktopQuitRuntime: () => calls.push('quit-listener') }))

import { installDesktopMainLifecycle } from './desktop-main-lifecycle-assembly'

test('main lifecycle installs the active-work guard before teardown listener', () => {
  const deps = new Proxy({ startup: {}, windows: {}, connections: {}, primaryTeardown: {} }, {
    get(target, key) { return key in target ? target[key] : vi.fn() }
  })

  const lifecycle = installDesktopMainLifecycle(deps as any)
  expect(calls).toEqual(['lifecycle', 'quit-guard', 'quit-listener'])
  expect(lifecycle.isPrimaryInstance).toBe(true)
})
