import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  getStatus: vi.fn(),
  installPluginSdk: vi.fn(),
  sdkImportMap: vi.fn(() => ({}))
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal()),
  getStatus: mocks.getStatus
}))

vi.mock('@/sdk/runtime', () => ({
  installPluginSdk: mocks.installPluginSdk,
  sdkImportMap: mocks.sdkImportMap
}))

import { $connection } from '@/store/session'

import { discoverRuntimePlugins, watchRuntimePlugins } from './runtime-loader'

describe('runtime disk-plugin discovery', () => {
  const readDir = vi.fn()
  const readFileText = vi.fn()
  const watchPreviewFile = vi.fn()
  const stopPreviewFileWatch = vi.fn()
  const listeners: ((payload: { id: string }) => void)[] = []

  const onPreviewFileChanged = vi.fn(listener => {
    listeners.push(listener)

    return vi.fn()
  })

  beforeEach(() => {
    vi.clearAllMocks()
    listeners.length = 0
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      onPreviewFileChanged,
      readDir,
      readFileText,
      stopPreviewFileWatch,
      watchPreviewFile
    }
    $connection.set(null)
  })

  afterEach(() => {
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    $connection.set(null)
  })

  it('does not probe a remote gateway home through the local Electron filesystem', async () => {
    mocks.getStatus.mockResolvedValue({ hermes_home: '/remote/hermes' })
    $connection.set({ mode: 'remote' } as never)

    await discoverRuntimePlugins()

    expect(mocks.getStatus).not.toHaveBeenCalled()
    expect(readDir).not.toHaveBeenCalled()
  })

  it('keeps scanning the local Hermes home for local gateways', async () => {
    mocks.getStatus.mockResolvedValue({ hermes_home: '/local/hermes' })
    readDir.mockResolvedValue({ entries: [] })
    $connection.set({ mode: 'local' } as never)

    await discoverRuntimePlugins()

    expect(readDir).toHaveBeenCalledWith('/local/hermes/desktop-plugins')
  })

  it('does not scan a local home when the connection becomes remote during the status request', async () => {
    let resolveStatus!: (value: { hermes_home: string }) => void
    mocks.getStatus.mockReturnValue(new Promise(resolve => (resolveStatus = resolve)))
    $connection.set({ mode: 'local' } as never)

    const scan = discoverRuntimePlugins()
    $connection.set({ mode: 'remote' } as never)
    resolveStatus({ hermes_home: '/local/hermes' })
    await scan

    expect(readDir).not.toHaveBeenCalled()
  })

  it('disposes local disk watches before a remote watched-file change can reload them', async () => {
    mocks.getStatus.mockResolvedValue({ hermes_home: '/local/hermes' })
    readDir.mockResolvedValue({ entries: [{ isDirectory: true, name: 'example', path: '/local/hermes/desktop-plugins/example' }] })
    readFileText.mockResolvedValue({ text: 'export default { id: "example", register() {} }' })
    watchPreviewFile.mockResolvedValue({ id: 'watch-example' })
    $connection.set({ mode: 'local' } as never)

    watchRuntimePlugins()
    await vi.waitFor(() => expect(watchPreviewFile).toHaveBeenCalledWith('/local/hermes/desktop-plugins/example/plugin.js'))

    $connection.set({ mode: 'remote' } as never)
    expect(stopPreviewFileWatch).toHaveBeenCalledWith('watch-example')

    readFileText.mockClear()
    listeners[0]?.({ id: 'watch-example' })
    await Promise.resolve()

    expect(readFileText).not.toHaveBeenCalled()
  })
})
