import { afterEach, describe, expect, it, vi } from 'vitest'

const { client } = vi.hoisted(() => ({
  client: {
    autoDownload: true, autoInstallOnAppQuit: true, autoRunAppAfterInstall: false,
    channel: '', allowPrerelease: true, allowDowngrade: true,
    on: vi.fn(), setFeedURL: vi.fn(), checkForUpdates: vi.fn(async () => null)
  }
}))

vi.mock('electron', () => ({ autoUpdater: {} }))
vi.mock('electron-updater', () => ({ default: { MacUpdater: class { constructor() { return client } } } }))

import { createMacStrategy } from './mac-client'

afterEach(() => vi.clearAllMocks())

function deps(feedBaseUrl = '', light = false, channel: 'stable' | 'canary' = 'stable') {
  return { channel, light, feedBaseUrl, appVersion: '0.28.0', log: vi.fn(), emitProgress: vi.fn(), beforeInstall: vi.fn(), onInstallFailure: vi.fn() }
}

describe('macOS client wiring', () => {
  it('uses the generated provider by default and forbids implicit installs or downgrades', async () => {
    const strategy = createMacStrategy(deps())
    expect(client.setFeedURL).not.toHaveBeenCalled()
    await expect(strategy.check()).rejects.toThrow('not active')
    expect(client.autoDownload).toBe(false)
    expect(client.autoInstallOnAppQuit).toBe(false)
    expect(client.autoRunAppAfterInstall).toBe(true)
    expect(client.allowDowngrade).toBe(false)
    expect(client.channel).toBe('stable')
    expect(client.allowPrerelease).toBe(false)
  })

  it('overrides the provider with the same variant/channel path as the publisher', () => {
    createMacStrategy(deps('https://updates.example/', true, 'canary'))
    expect(client.setFeedURL).toHaveBeenCalledWith({ provider: 'generic', url: 'https://updates.example/releases/darwin/light/canary/', channel: 'canary' })
    expect(client.allowPrerelease).toBe(true)
    expect(() => createMacStrategy(deps('http://untrusted.example'))).toThrow('HTTPS')
  })
})
