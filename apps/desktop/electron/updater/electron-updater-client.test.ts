import { describe, expect, it, vi } from 'vitest'

const { mac } = vi.hoisted(() => ({ mac: { mechanism: 'electron-updater', check: vi.fn(), apply: vi.fn() } }))

vi.mock('./mac-client', () => ({
  createMacStrategy: vi.fn(() => mac),
  createChannelMacStrategy: vi.fn(() => mac)
}))
vi.mock('./appimage-client', () => ({ createAppImageStrategy: vi.fn() }))

import type { AppImageStrategy } from './appimage'
import { createElectronUpdaterStrategy, type ElectronUpdaterClientDeps } from './electron-updater-client'

const deps = { channel: 'stable', light: false, feedBaseUrl: '' } as ElectronUpdaterClientDeps
const mount = '/tmp/.mount_HermesAb12'
const image = '/home/u/Applications/Hermes-0.99.0-linux-arm64.AppImage'

function select(platform: NodeJS.Platform, execPath: string, env: NodeJS.ProcessEnv) {
  const appImage = { mechanism: 'electron-updater' } as AppImageStrategy
  const createAppImage = vi.fn(() => appImage)
  const strategy = createElectronUpdaterStrategy(deps, undefined, { platform, execPath, env }, createAppImage)

  return { strategy, appImage, createAppImage }
}

describe('electron-updater strategy selection', () => {
  it('updates a Linux bundle in place when it runs from its own AppImage', () => {
    const { strategy, appImage, createAppImage } = select('linux', `${mount}/hermes`, {
      APPIMAGE: image,
      APPDIR: mount
    })

    expect(strategy).toBe(appImage)
    expect(createAppImage).toHaveBeenCalledWith(deps, image)
  })

  it.each([
    ['run unpacked', '/opt/hermes/linux-arm64-unpacked/hermes', {}],
    ['started from another AppImage', '/opt/hermes/hermes', { APPIMAGE: '/home/u/Editor.AppImage', APPDIR: mount }]
  ])('leaves a Linux bundle %s to its package owner', async (_case, execPath, env) => {
    const { strategy, createAppImage } = select('linux', execPath, env)
    expect(createAppImage).not.toHaveBeenCalled()
    expect(strategy.mechanism).toBe('external')
    await expect(strategy.check()).resolves.toMatchObject({ supported: false, mechanism: 'external' })
  })

  it('keeps macOS on its own updater', () => {
    const { strategy, createAppImage } = select('darwin', '/Applications/Hermes.app/Contents/MacOS/Hermes', {})
    expect(strategy).toBe(mac)
    expect(createAppImage).not.toHaveBeenCalled()
  })
})
