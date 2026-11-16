import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'

import { $pluginRecords, enableCatalogDesktopPlugin, publishPlugin } from './plugins-store'
import { discoverRuntimePlugins } from './runtime-loader'
import { routeBlobImportsToDataUrls } from './runtime-loader.test-helpers'

// A standalone desktop plugin installed from the catalog carries `.hermes-origin.json`
// (electron/desktop-plugins-root.ts INSTALL_ORIGIN). The loader puts that entry on the plugin's
// record, which is what the catalog card matches; the folder stays standalone (no package).
const root = '/local/.hermes/desktop-plugins'
const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()
const readFileText = vi.fn<(path: string) => Promise<{ text: string }>>()

const ORIGIN = {
  catalogName: 'accent-picker',
  repo: 'https://github.com/NousResearch/hermes-official-plugins',
  sha: 'a'.repeat(40)
}

function folder(name: string, files: Record<string, string>) {
  return { name, files }
}

function stubDisk(folders: ReturnType<typeof folder>[]) {
  readDir.mockImplementation(async dir => {
    if (dir === root) {
      return { entries: folders.map(f => ({ isDirectory: true, name: f.name, path: `${root}/${f.name}` })) }
    }

    const match = folders.find(f => dir === `${root}/${f.name}`)

    return {
      entries: Object.keys(match?.files ?? {}).map(name => ({ isDirectory: false, name, path: `${dir}/${name}` }))
    }
  })
  readFileText.mockImplementation(async file => {
    const f = folders.find(entry => file.startsWith(`${root}/${entry.name}/`))

    return { text: f?.files[file.slice(`${root}/${f.name}/`.length)] ?? '' }
  })
}

let restore: () => void

beforeEach(() => {
  $pluginRecords.set({})
  readDir.mockReset()
  readFileText.mockReset()
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    desktopPluginsRoot: vi.fn(async () => root),
    onPreviewFileChanged: vi.fn(),
    readDir,
    readFileText,
    stopPreviewFileWatch: vi.fn(async () => true),
    watchDirectory: vi.fn(async () => ({ id: 'd' })),
    watchPreviewFile: vi.fn(async () => ({ id: 'w' }))
  }

  restore = routeBlobImportsToDataUrls()
})

afterEach(() => {
  restore()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

const OFF_BY_DEFAULT = 'export default { id: "accent", defaultEnabled: false, register() {} }'

describe('a standalone plugin installed from the catalog', () => {
  it('carries its catalog entry on its record and stays standalone', async () => {
    stubDisk([folder('accent-picker', { 'plugin.js': OFF_BY_DEFAULT, '.hermes-origin.json': JSON.stringify(ORIGIN) })])

    await discoverRuntimePlugins()

    const record = $pluginRecords.get().accent
    expect(record).toMatchObject({ kind: 'disk', packageOrigin: ORIGIN })
    expect(record.packageName).toBeUndefined()
  })

  it('ignores an origin file with no catalog entry, and a folder with none', async () => {
    stubDisk([
      folder('a', { 'plugin.js': 'export default { id: "a", register() {} }', '.hermes-origin.json': '{"repo":"x"}' }),
      folder('b', { 'plugin.js': 'export default { id: "b", register() {} }' })
    ])

    await discoverRuntimePlugins()

    expect($pluginRecords.get().a.packageOrigin).toBeUndefined()
    expect($pluginRecords.get().b.packageOrigin).toBeUndefined()
  })

  it('a reload picks up an origin recorded after the plugin loaded (an install adopting an old copy)', async () => {
    const files: Record<string, string> = { 'plugin.js': OFF_BY_DEFAULT }
    stubDisk([folder('accent-picker', files)])
    await discoverRuntimePlugins()
    expect($pluginRecords.get().accent.packageOrigin).toBeUndefined()

    files['.hermes-origin.json'] = JSON.stringify(ORIGIN)
    await discoverRuntimePlugins()

    expect($pluginRecords.get().accent.packageOrigin).toEqual(ORIGIN)
  })

  it('a reload reads again what a folder is (a Force reinstall made it a package half)', async () => {
    const files: Record<string, string> = { 'plugin.js': OFF_BY_DEFAULT, '.hermes-origin.json': JSON.stringify(ORIGIN) }
    stubDisk([folder('accent-picker', files)])
    await discoverRuntimePlugins()
    expect($pluginRecords.get().accent.packageName).toBeUndefined()

    files['.hermes-package.json'] = JSON.stringify({ package: 'accent-pkg', repo: ORIGIN.repo })
    await discoverRuntimePlugins()

    expect($pluginRecords.get().accent).toMatchObject({
      packageName: 'accent-pkg',
      packageOrigin: { repo: ORIGIN.repo }
    })
    expect($pluginRecords.get().accent.packageOrigin?.catalogName).toBeUndefined()
  })

  it('a reload that finds a linked package half watches its source, as a fresh load does', async () => {
    const files: Record<string, string> = { 'plugin.js': OFF_BY_DEFAULT }
    stubDisk([folder('accent-picker', files)])
    await discoverRuntimePlugins()
    const watch = vi.mocked(window.hermesDesktop.watchPreviewFile)
    expect(watch).not.toHaveBeenCalledWith('/dev/accent/desktop/plugin.js')

    files['.hermes-package.json'] = JSON.stringify({
      linked: true,
      package: 'accent-pkg',
      source: '/dev/accent/desktop'
    })
    await discoverRuntimePlugins()

    expect(watch).toHaveBeenCalledWith('/dev/accent/desktop/plugin.js')
  })

  it('a reload asked for during a scan waits for it and then reads the disk again', async () => {
    // An install lands its folder, the root watch starts a scan, then the install asks for a reload and
    // enables what it finds: dropping that reload left a catalog plugin off and its card on Install.
    const files: Record<string, string> = { 'plugin.js': OFF_BY_DEFAULT }
    stubDisk([folder('accent-picker', files)])
    await discoverRuntimePlugins()
    // Hold the first scan once it has read the folder (it is loading plugin.js) ...
    let release!: () => void
    let loading!: () => void
    const gate = new Promise<void>(resolve => (release = resolve))
    const reached = new Promise<void>(resolve => (loading = resolve))
    const read = readFileText.getMockImplementation()!
    readFileText.mockImplementation(async file => {
      if (file.endsWith('/plugin.js')) {
        loading()
        await gate
      }

      return read(file)
    })
    const first = discoverRuntimePlugins()
    await reached
    // ... then the install records the origin and asks for its reload.
    files['.hermes-origin.json'] = JSON.stringify(ORIGIN)
    readFileText.mockImplementation(read)
    const second = discoverRuntimePlugins()
    release()
    await Promise.all([first, second])

    expect($pluginRecords.get().accent.packageOrigin).toEqual(ORIGIN)
  })

  it('enableCatalogDesktopPlugin turns on the plugin that entry installed, and only that one', async () => {
    stubDisk([folder('accent-picker', { 'plugin.js': OFF_BY_DEFAULT, '.hermes-origin.json': JSON.stringify(ORIGIN) })])
    await discoverRuntimePlugins()
    publishPlugin({ id: 'other', kind: 'disk', name: 'Other', status: 'disabled' })
    expect($pluginRecords.get().accent.status).toBe('disabled')

    await enableCatalogDesktopPlugin('accent-picker')

    expect($pluginRecords.get().accent.status).toBe('loaded')
    expect($pluginRecords.get().other.status).toBe('disabled')
  })
})
