import { afterEach, describe, expect, it, vi } from 'vitest'

import { discoverRuntimePlugins } from './runtime-loader'

describe('desktop runtime plugin discovery', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('scans the Electron-owned local plugin directory', async () => {
    const getDesktopPluginsDir = vi.fn(async () => '/Users/test/.hermes/desktop-plugins')
    const readDir = vi.fn(async () => ({ entries: [] }))

    vi.stubGlobal('window', {
      hermesDesktop: {
        getDesktopPluginsDir,
        readDir
      }
    })

    await discoverRuntimePlugins()

    expect(getDesktopPluginsDir).toHaveBeenCalledOnce()
    expect(readDir).toHaveBeenCalledWith('/Users/test/.hermes/desktop-plugins')
  })
})
