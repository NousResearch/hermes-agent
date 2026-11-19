// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'

const setClerkBridge = (bridge?: { enabled: boolean; host: string; scheme: string }) => {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: bridge ? { clerk: bridge } : undefined
  })
}

afterEach(() => {
  vi.resetModules()
  setClerkBridge()
})

describe('clerk-config host detection', () => {
  it('enables browser Clerk when there is no Electron bridge', async () => {
    setClerkBridge()

    const config = await import('./clerk-config')

    expect(config.clerkBrowserEnabled).toBe(true)
    expect(config.clerkEnabled).toBe(false)
  })

  it('leaves browser Clerk off when the Electron bridge is present', async () => {
    setClerkBridge({ enabled: true, host: 'renderer', scheme: 'unitalk-app' })

    const config = await import('./clerk-config')

    expect(config.clerkEnabled).toBe(true)
    expect(config.clerkBrowserEnabled).toBe(false)
  })
})
