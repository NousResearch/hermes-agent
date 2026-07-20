import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  getLocalHermesHome: vi.fn(),
  loadText: vi.fn(),
  readDir: vi.fn(),
  stopWatch: vi.fn(),
  watch: vi.fn()
}))

vi.mock('@/sdk/runtime', () => ({
  installPluginSdk: vi.fn(),
  sdkImportMap: () => ({})
}))

vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))
vi.mock('./plugin', () => ({ createPluginContext: vi.fn() }))
vi.mock('./plugins-store', () => ({
  dropPlugin: vi.fn(),
  pluginActive: vi.fn(() => true),
  publishPlugin: vi.fn()
}))

import { discoverRuntimePlugins } from './runtime-loader'

describe('runtime disk plugins', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.getLocalHermesHome.mockResolvedValue('/Users/test/.hermes')
    mocks.readDir.mockResolvedValue({
      entries: [{ isDirectory: true, name: 'example', path: '/Users/test/.hermes/desktop-plugins/example' }],
      path: '/Users/test/.hermes/desktop-plugins'
    })
    mocks.loadText
      .mockResolvedValueOnce({ path: '/Users/test/.hermes/desktop-plugins/example/plugin.js', text: 'plugin' })
      .mockRejectedValueOnce(new Error('stop before evaluation'))
    mocks.watch.mockResolvedValue({ id: 'watch-1', url: 'file:///plugin.js' })

    vi.stubGlobal('window', {
      hermesDesktop: {
        getLocalHermesHome: mocks.getLocalHermesHome,
        onPreviewFileChanged: vi.fn(),
        readDir: mocks.readDir,
        readFileText: mocks.loadText,
        stopPreviewFileWatch: mocks.stopWatch,
        watchPreviewFile: mocks.watch
      }
    })
  })

  afterEach(() => vi.unstubAllGlobals())

  it('discovers trusted plugins from the desktop machine home', async () => {
    await discoverRuntimePlugins()

    expect(mocks.getLocalHermesHome).toHaveBeenCalledOnce()
    expect(mocks.readDir).toHaveBeenCalledWith('/Users/test/.hermes/desktop-plugins')
    expect(mocks.loadText).toHaveBeenCalledWith('/Users/test/.hermes/desktop-plugins/example/plugin.js')
  })
})
