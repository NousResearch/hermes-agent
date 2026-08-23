import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import type * as HermesModule from '@/hermes'

import { $pluginRecords, publishPlugin, setPluginEnabled } from './plugins-store'
import { discoverRuntimePlugins, loadRuntimePlugin, watchRuntimePlugins } from './runtime-loader'

// getStatus would supply the connected backend's hermes_home — a REMOTE path in
// remote mode. The disk scanner must NOT derive the plugin root from it (#66899).
const getStatus = vi.fn(async () => ({ hermes_home: '/remote/box/.hermes' }))

vi.mock('@/hermes', async importActual => ({
  ...(await importActual<typeof HermesModule>()),
  getStatus: () => getStatus()
}))

const desktopPluginsRoot = vi.fn<() => Promise<string>>()
const agentPluginsRoot = vi.fn<() => Promise<string>>()
const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()
const readFileText = vi.fn<(path: string) => Promise<{ text: string; truncated?: boolean }>>()
const readPluginSource = vi.fn<(path: string) => Promise<{ text: string; truncated?: boolean }>>()
const watchDirectory = vi.fn<(path: string) => Promise<{ id: string }>>()
const watchPreviewFile = vi.fn<(path: string) => Promise<{ id: string }>>()
const onPreviewFileChanged = vi.fn()

beforeEach(() => {
  desktopPluginsRoot.mockReset()
  agentPluginsRoot.mockReset()
  readDir.mockReset()
  readFileText.mockReset()
  readPluginSource.mockReset()
  watchDirectory.mockReset()
  watchPreviewFile.mockReset()
  onPreviewFileChanged.mockReset()
  getStatus.mockClear()
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    agentPluginsRoot,
    desktopPluginsRoot,
    onPreviewFileChanged,
    readDir,
    readFileText,
    watchDirectory,
    watchPreviewFile
  }
})

afterEach(() => {
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('scanDiskPlugins (#66899)', () => {
  it('scans the Electron-resolved local roots, never the backend hermes_home', async () => {
    desktopPluginsRoot.mockResolvedValue('/local/.hermes/desktop-plugins')
    agentPluginsRoot.mockResolvedValue('/local/.hermes/plugins')
    readDir.mockResolvedValue({ entries: [] })

    await discoverRuntimePlugins()

    expect(desktopPluginsRoot).toHaveBeenCalled()
    expect(readDir).toHaveBeenCalledWith('/local/.hermes/desktop-plugins')
    expect(readDir).toHaveBeenCalledWith('/local/.hermes/plugins')
    // The remote backend's hermes_home must never feed the local plugin scan.
    expect(getStatus).not.toHaveBeenCalled()
    expect(readDir).not.toHaveBeenCalledWith('/remote/box/.hermes/desktop-plugins')
  })

  it('no-ops when the resolvers yield no local root', async () => {
    desktopPluginsRoot.mockResolvedValue('')
    agentPluginsRoot.mockResolvedValue('')

    await discoverRuntimePlugins()

    expect(readDir).not.toHaveBeenCalled()
  })

  it('probes desktop/plugin.js inside agent-plugin packages (unified packaging)', async () => {
    desktopPluginsRoot.mockResolvedValue('/local/.hermes/desktop-plugins')
    agentPluginsRoot.mockResolvedValue('/local/.hermes/plugins')
    readDir.mockImplementation(async dir =>
      dir === '/local/.hermes/plugins'
        ? { entries: [{ isDirectory: true, name: 'my-feature', path: '/local/.hermes/plugins/my-feature' }] }
        : { entries: [] }
    )
    // No desktop half in this package — probe must target desktop/plugin.js.
    readFileText.mockRejectedValue(new Error('ENOENT'))

    await discoverRuntimePlugins()

    expect(readFileText).toHaveBeenCalledWith('/local/.hermes/plugins/my-feature/desktop/plugin.js')
    // The Python half's files must never be probed as a desktop entry.
    expect(readFileText).not.toHaveBeenCalledWith('/local/.hermes/plugins/my-feature/plugin.js')
  })

  it('still scans the standalone root when agentPluginsRoot is absent (older shell)', async () => {
    delete (window.hermesDesktop as unknown as { agentPluginsRoot?: unknown }).agentPluginsRoot
    desktopPluginsRoot.mockResolvedValue('/local/.hermes/desktop-plugins')
    readDir.mockResolvedValue({ entries: [] })

    await discoverRuntimePlugins()

    expect(readDir).toHaveBeenCalledWith('/local/.hermes/desktop-plugins')
    expect(readDir).toHaveBeenCalledTimes(1)
  })

  it('loads a unified desktop half OPT-IN: inventoried but not activated by default', async () => {
    desktopPluginsRoot.mockResolvedValue('/local/.hermes/desktop-plugins')
    agentPluginsRoot.mockResolvedValue('/local/.hermes/plugins')
    readDir.mockImplementation(async dir =>
      dir === '/local/.hermes/plugins'
        ? { entries: [{ isDirectory: true, name: 'uni', path: '/local/.hermes/plugins/uni' }] }
        : { entries: [] }
    )

    const register = vi.fn()

    ;(globalThis as unknown as { __uniRegister: unknown }).__uniRegister = register
    readFileText.mockResolvedValue({
      text: 'export default { id: "uni", register: globalThis.__uniRegister }'
    })
    watchPreviewFile.mockResolvedValue({ id: 'w-uni' })

    // The loader evaluates plugins via blob-URL import(), which vite's module
    // runner can't resolve in tests — reroute to a data: URL, which node's
    // native ESM loader handles.
    const createObjectURL = vi
      .spyOn(URL, 'createObjectURL')
      .mockImplementation(
        blob =>
          `data:text/javascript;base64,${Buffer.from((blob as unknown as { parts: string[] }).parts.join('')).toString('base64')}`
      )

    const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    const RealBlob = globalThis.Blob
    vi.stubGlobal(
      'Blob',
      class {
        parts: string[]
        constructor(parts: string[]) {
          this.parts = parts
        }
      }
    )

    try {
      await discoverRuntimePlugins()

      // Inventoried for Settings → Plugins, but the root's opt-in posture wins:
      // ~/.hermes/plugins stays installed-but-inert until the user toggles it.
      expect($pluginRecords.get().uni).toMatchObject({ kind: 'disk', status: 'disabled' })
      expect(register).not.toHaveBeenCalled()

      // The user's explicit enable still activates it.
      await setPluginEnabled('uni', true)
      expect(register).toHaveBeenCalledTimes(1)
      expect($pluginRecords.get().uni.status).toBe('loaded')
    } finally {
      createObjectURL.mockRestore()
      revokeObjectURL.mockRestore()
      vi.stubGlobal('Blob', RealBlob)
      delete (globalThis as unknown as { __uniRegister?: unknown }).__uniRegister
    }
  })
})

describe('watchRuntimePlugins dir watch (#66899)', () => {
  it('watches both Electron-resolved local roots, never the backend hermes_home', async () => {
    desktopPluginsRoot.mockResolvedValue('/local/.hermes/desktop-plugins')
    agentPluginsRoot.mockResolvedValue('/local/.hermes/plugins')
    readDir.mockResolvedValue({ entries: [] })
    watchDirectory.mockResolvedValue({ id: 'watch-1' })

    watchRuntimePlugins()
    // Drain the async scan + startDirWatches chains.
    await vi.waitFor(() => expect(watchDirectory).toHaveBeenCalledTimes(2))

    expect(watchDirectory).toHaveBeenCalledWith('/local/.hermes/desktop-plugins')
    expect(watchDirectory).toHaveBeenCalledWith('/local/.hermes/plugins')
    expect(watchDirectory).not.toHaveBeenCalledWith('/remote/box/.hermes/desktop-plugins')
    expect(getStatus).not.toHaveBeenCalled()
  })
})

describe('plugin source reads (512 KiB preview-cap bug)', () => {
  const blobToDataUrl = () => {
    const createObjectURL = vi
      .spyOn(URL, 'createObjectURL')
      .mockImplementation(
        blob =>
          `data:text/javascript;base64,${Buffer.from((blob as unknown as { parts: string[] }).parts.join('')).toString('base64')}`
      )

    const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    const RealBlob = globalThis.Blob
    vi.stubGlobal(
      'Blob',
      class {
        parts: string[]
        constructor(parts: string[]) {
          this.parts = parts
        }
      }
    )

    return () => {
      createObjectURL.mockRestore()
      revokeObjectURL.mockRestore()
      vi.stubGlobal('Blob', RealBlob)
    }
  }

  it('loads the full source via readPluginSource when the shell offers it', async () => {
    ;(window.hermesDesktop as unknown as { readPluginSource: unknown }).readPluginSource = readPluginSource
    desktopPluginsRoot.mockResolvedValue('/local/.hermes/desktop-plugins')
    agentPluginsRoot.mockResolvedValue('')
    readDir.mockResolvedValue({
      entries: [{ isDirectory: true, name: 'big', path: '/local/.hermes/desktop-plugins/big' }]
    })
    // The existence probe still uses the preview read (metadata is enough
    // there) — it may report truncation without failing the scan.
    readFileText.mockResolvedValue({ text: '// first 512 KiB only', truncated: true })

    const register = vi.fn()

    ;(globalThis as unknown as { __bigRegister: unknown }).__bigRegister = register
    readPluginSource.mockResolvedValue({
      text: 'export default { id: "big", register: globalThis.__bigRegister }'
    })
    watchPreviewFile.mockResolvedValue({ id: 'w-big' })

    const restore = blobToDataUrl()

    try {
      await discoverRuntimePlugins()

      // The EVALUATED source came from the full read, not the truncated preview.
      expect(readPluginSource).toHaveBeenCalledWith('/local/.hermes/desktop-plugins/big/plugin.js')
      expect(register).toHaveBeenCalledTimes(1)
      expect($pluginRecords.get().big).toMatchObject({ kind: 'disk', status: 'loaded' })
    } finally {
      restore()
      delete (globalThis as unknown as { __bigRegister?: unknown }).__bigRegister
    }
  })

  it('older shell without readPluginSource: a truncated preview read fails LOUDLY, never evaluates', async () => {
    desktopPluginsRoot.mockResolvedValue('/local/.hermes/desktop-plugins')
    agentPluginsRoot.mockResolvedValue('')
    readDir.mockResolvedValue({
      entries: [{ isDirectory: true, name: 'huge', path: '/local/.hermes/desktop-plugins/huge' }]
    })
    // 512 KiB window of a larger file — parses fine, but is NOT the plugin.
    readFileText.mockResolvedValue({
      text: 'export default { id: "huge", register: () => { throw new Error("must never evaluate") } }',
      truncated: true
    })
    watchPreviewFile.mockResolvedValue({ id: 'w-huge' })

    const restore = blobToDataUrl()

    try {
      await discoverRuntimePlugins()

      // No live plugin — an error inventory row names the folder instead.
      expect($pluginRecords.get().huge).toMatchObject({
        kind: 'disk',
        status: 'error',
        file: '/local/.hermes/desktop-plugins/huge/plugin.js'
      })
      expect($pluginRecords.get().huge.error).toMatch(/512 KiB/)
    } finally {
      restore()
    }
  })

  it('older shell, small plugin (not truncated): still loads through readFileText', async () => {
    desktopPluginsRoot.mockResolvedValue('/local/.hermes/desktop-plugins')
    agentPluginsRoot.mockResolvedValue('')
    readDir.mockResolvedValue({
      entries: [{ isDirectory: true, name: 'small', path: '/local/.hermes/desktop-plugins/small' }]
    })

    const register = vi.fn()

    ;(globalThis as unknown as { __smallRegister: unknown }).__smallRegister = register
    readFileText.mockResolvedValue({
      text: 'export default { id: "small", register: globalThis.__smallRegister }'
    })
    watchPreviewFile.mockResolvedValue({ id: 'w-small' })

    const restore = blobToDataUrl()

    try {
      await discoverRuntimePlugins()

      expect(register).toHaveBeenCalledTimes(1)
      expect($pluginRecords.get().small).toMatchObject({ kind: 'disk', status: 'loaded' })
    } finally {
      restore()
      delete (globalThis as unknown as { __smallRegister?: unknown }).__smallRegister
    }
  })
})

describe('bundled-shadowed disk copies', () => {
  it('skips a disk copy of a bundled plugin but publishes a visible inventory row', async () => {
    // The bundled twin is already registered (build-time glob).
    publishPlugin({ id: 'hermes-bots', name: 'Bot Mode', kind: 'bundled', status: 'loaded' })

    // Same blob→data: URL reroute as the opt-in test above.
    const createObjectURL = vi
      .spyOn(URL, 'createObjectURL')
      .mockImplementation(
        blob =>
          `data:text/javascript;base64,${Buffer.from((blob as unknown as { parts: string[] }).parts.join('')).toString('base64')}`
      )

    const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    const RealBlob = globalThis.Blob
    vi.stubGlobal(
      'Blob',
      class {
        parts: string[]
        constructor(parts: string[]) {
          this.parts = parts
        }
      }
    )

    try {
      const id = await loadRuntimePlugin(
        'export default { id: "hermes-bots", name: "Bot Mode", register() {} }',
        'hermes-bots',
        { file: '/local/.hermes/desktop-plugins/hermes-bots/plugin.js' }
      )

      // Skipped — the bundled copy stays the only live registration...
      expect(id).toBeNull()
      expect($pluginRecords.get()['hermes-bots']).toMatchObject({ kind: 'bundled', status: 'loaded' })

      // ...but the stale folder is DISCOVERABLE: an inventory row names it,
      // carries its path (reveal/delete affordance), and can never activate.
      expect($pluginRecords.get()['hermes-bots:disk-shadowed']).toMatchObject({
        kind: 'disk',
        status: 'disabled',
        file: '/local/.hermes/desktop-plugins/hermes-bots/plugin.js'
      })
    } finally {
      createObjectURL.mockRestore()
      revokeObjectURL.mockRestore()
      vi.stubGlobal('Blob', RealBlob)
    }
  })
})
