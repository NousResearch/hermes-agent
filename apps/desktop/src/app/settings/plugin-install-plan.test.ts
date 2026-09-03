import { atom } from 'nanostores'
import { describe, expect, it, vi } from 'vitest'

import type { PluginRecord } from '@/contrib/plugins-store'

import {
  desktopHalfMayShareLocalRoot,
  findStandaloneDesktopEntry,
  findUnifiedDesktopEntry,
  findUnifiedDesktopPluginId,
  type HybridInstallPlanInput,
  type PluginRootsFs,
  settleUnifiedDesktopPluginId,
  waitForUnifiedDesktopPluginId
} from './plugin-install-plan'

const hybridOnLocal: HybridInstallPlanInput = {
  connectionMode: 'local',
  probeAgent: true,
  probeDesktop: true,
  desktopSourceSubdir: 'desktop',
  standaloneCopy: false,
  installAgent: true,
  installDesktop: true
}

const ROOT = '/home/u/.hermes/plugins'
const DESKTOP_ROOT = '/home/u/.hermes/desktop-plugins'
const FILE = `${ROOT}/word-count/desktop/plugin.js`

const record = (id: string, file: string | undefined, patch: Partial<PluginRecord> = {}): PluginRecord => ({
  id,
  name: id,
  kind: 'disk',
  status: 'disabled',
  file,
  ...patch
})

/** In-memory `readDir` over a posix tree: dir path -> child names ('x/' = directory). */
function fakeFs(
  tree: Record<string, string[]>,
  roots: Partial<Record<'agent' | 'desktop', string>> = {}
): PluginRootsFs {
  return {
    agentPluginsRoot: async () => roots.agent ?? ROOT,
    desktopPluginsRoot: async () => roots.desktop ?? DESKTOP_ROOT,
    readDir: async dirPath => {
      const children = tree[dirPath]

      if (!children) {
        throw new Error(`ENOENT ${dirPath}`)
      }

      return {
        entries: children.map(child => {
          const isDirectory = child.endsWith('/')
          const name = isDirectory ? child.slice(0, -1) : child

          return { name, path: `${dirPath}/${name}`, isDirectory }
        })
      }
    }
  }
}

describe('desktopHalfMayShareLocalRoot (#100412)', () => {
  it('rules in a hybrid repo with a nested desktop half installed into a local backend', () => {
    expect(desktopHalfMayShareLocalRoot(hybridOnLocal)).toBe(true)
  })

  it('still copies when the agent half targets a remote backend', () => {
    // The remote backend's plugins/ tree is not on this machine, so the
    // desktop-plugins/ copy is the only local one (#97005 dual-target flow).
    expect(desktopHalfMayShareLocalRoot({ ...hybridOnLocal, connectionMode: 'remote' })).toBe(false)
  })

  it('fails toward copying when the connection mode is unknown', () => {
    expect(desktopHalfMayShareLocalRoot({ ...hybridOnLocal, connectionMode: undefined })).toBe(false)
  })

  it('copies a root-level plugin.js desktop half (the unified door only serves desktop/plugin.js)', () => {
    expect(desktopHalfMayShareLocalRoot({ ...hybridOnLocal, desktopSourceSubdir: '.' })).toBe(false)
    expect(desktopHalfMayShareLocalRoot({ ...hybridOnLocal, desktopSourceSubdir: null })).toBe(false)
  })

  it('keeps refreshing an existing standalone copy (the one the loader serves)', () => {
    expect(desktopHalfMayShareLocalRoot({ ...hybridOnLocal, standaloneCopy: true })).toBe(false)
  })

  it('copies a desktop-only package (nothing lands in plugins/)', () => {
    expect(desktopHalfMayShareLocalRoot({ ...hybridOnLocal, probeAgent: false })).toBe(false)
  })

  it('copies when the user unticked the agent half', () => {
    expect(desktopHalfMayShareLocalRoot({ ...hybridOnLocal, installAgent: false })).toBe(false)
  })

  it('is moot when the desktop half is unticked', () => {
    expect(desktopHalfMayShareLocalRoot({ ...hybridOnLocal, installDesktop: false })).toBe(false)
  })
})

describe('findUnifiedDesktopEntry (#100412)', () => {
  it('returns the entry path when plugins/<name>/desktop/plugin.js exists', async () => {
    const fs = fakeFs({
      [ROOT]: ['word-count/', 'other/'],
      [`${ROOT}/word-count`]: ['plugin.yaml', '__init__.py', 'desktop/'],
      [`${ROOT}/word-count/desktop`]: ['plugin.js']
    })

    await expect(findUnifiedDesktopEntry(fs, 'word-count')).resolves.toBe(FILE)
  })

  it('is null for an agent-only install (no desktop folder) or a missing plugin', async () => {
    const fs = fakeFs({
      [ROOT]: ['word-count/'],
      [`${ROOT}/word-count`]: ['plugin.yaml', '__init__.py']
    })

    await expect(findUnifiedDesktopEntry(fs, 'word-count')).resolves.toBeNull()
    await expect(findUnifiedDesktopEntry(fs, 'absent')).resolves.toBeNull()
  })

  it('is null when a segment has the wrong kind (a "desktop" file, a "plugin.js" folder)', async () => {
    const fileNotDir = fakeFs({
      [ROOT]: ['word-count/'],
      [`${ROOT}/word-count`]: ['desktop']
    })

    const dirNotFile = fakeFs({
      [ROOT]: ['word-count/'],
      [`${ROOT}/word-count`]: ['desktop/'],
      [`${ROOT}/word-count/desktop`]: ['plugin.js/']
    })

    await expect(findUnifiedDesktopEntry(fileNotDir, 'word-count')).resolves.toBeNull()
    await expect(findUnifiedDesktopEntry(dirNotFile, 'word-count')).resolves.toBeNull()
  })

  it('is null (copy) when the root is unreadable, the bridge is missing, or the shell predates agentPluginsRoot', async () => {
    await expect(findUnifiedDesktopEntry(fakeFs({}), 'word-count')).resolves.toBeNull()
    await expect(findUnifiedDesktopEntry(undefined, 'word-count')).resolves.toBeNull()
    await expect(
      findUnifiedDesktopEntry({ readDir: vi.fn(async () => ({ entries: [] })) }, 'word-count')
    ).resolves.toBeNull()
  })
})

describe('findStandaloneDesktopEntry (#100412)', () => {
  it('returns desktop-plugins/<name>/plugin.js from an earlier install', async () => {
    const fs = fakeFs({
      [DESKTOP_ROOT]: ['word-count/'],
      [`${DESKTOP_ROOT}/word-count`]: ['plugin.js', 'README.md']
    })

    await expect(findStandaloneDesktopEntry(fs, 'word-count')).resolves.toBe(`${DESKTOP_ROOT}/word-count/plugin.js`)
  })

  it('is null when absent, when the folder has no plugin.js, or on an older shell', async () => {
    const fs = fakeFs({ [DESKTOP_ROOT]: ['other/'], [`${DESKTOP_ROOT}/other`]: ['plugin.js'] })

    await expect(findStandaloneDesktopEntry(fs, 'word-count')).resolves.toBeNull()
    await expect(
      findStandaloneDesktopEntry(
        fakeFs({ [DESKTOP_ROOT]: ['word-count/'], [`${DESKTOP_ROOT}/word-count`]: [] }),
        'word-count'
      )
    ).resolves.toBeNull()
    await expect(
      findStandaloneDesktopEntry({ readDir: vi.fn(async () => ({ entries: [] })) }, 'word-count')
    ).resolves.toBeNull()
  })
})

describe('findUnifiedDesktopPluginId (#100412)', () => {
  it('finds the disk record published for the entry file', () => {
    const records = {
      other: record('other', '/home/u/.hermes/plugins/other/desktop/plugin.js'),
      'word-count': record('word-count', FILE)
    }

    expect(findUnifiedDesktopPluginId(records, FILE)).toEqual({ id: 'word-count' })
  })

  it('never matches the standalone desktop-plugins/<name>/plugin.js copy', () => {
    const records = {
      'word-count': record('word-count', '/home/u/.hermes/desktop-plugins/word-count/plugin.js')
    }

    expect(findUnifiedDesktopPluginId(records, FILE)).toBeNull()
  })

  it('reports a broken load as a terminal error, not "not yet"', () => {
    const records = { 'word-count': record('word-count', FILE, { status: 'error', error: 'syntax' }) }

    expect(findUnifiedDesktopPluginId(records, FILE)).toEqual({ error: 'syntax' })
  })

  it('reports a bundled-shadowed row as terminal instead of enabling the phantom id', () => {
    const records = {
      'word-count:disk-shadowed': record('word-count:disk-shadowed', FILE, {
        name: 'Word count (stale disk copy)',
        description: 'Shadowed by the bundled "word-count" plugin'
      })
    }

    expect(findUnifiedDesktopPluginId(records, FILE)).toEqual({ error: 'Shadowed by the bundled "word-count" plugin' })
  })

  it('ignores bundled records and records without a file', () => {
    const records = {
      bundled: record('bundled', FILE, { kind: 'bundled' }),
      nofile: record('nofile', undefined)
    }

    expect(findUnifiedDesktopPluginId(records, FILE)).toBeNull()
  })
})

describe('waitForUnifiedDesktopPluginId (#100412)', () => {
  it('resolves at once when the record is already published, and releases its listener', async () => {
    const records = atom<Record<string, PluginRecord>>({ 'word-count': record('word-count', FILE) })

    await expect(waitForUnifiedDesktopPluginId(records, FILE, 50)).resolves.toEqual({ id: 'word-count' })
    expect(records.lc).toBe(0)
  })

  it('resolves when an in-flight scan publishes the record later', async () => {
    const records = atom<Record<string, PluginRecord>>({})
    const pending = waitForUnifiedDesktopPluginId(records, FILE, 500)

    setTimeout(() => records.set({ 'word-count': record('word-count', FILE) }), 20)

    await expect(pending).resolves.toEqual({ id: 'word-count' })
    expect(records.lc).toBe(0)
  })

  it('gives up with null after the timeout and stops listening', async () => {
    const records = atom<Record<string, PluginRecord>>({})

    await expect(waitForUnifiedDesktopPluginId(records, FILE, 30)).resolves.toBeNull()

    records.set({ 'word-count': record('word-count', FILE) })

    expect(records.lc).toBe(0)
  })
})

describe('settleUnifiedDesktopPluginId (#100412)', () => {
  it('rescans again when the first rescan was dropped by the loader lock', async () => {
    // Round 1: scanDiskPlugins() returns at once because a watch/poll scan
    // holds its lock and nothing else publishes. Round 2 lands.
    const records = atom<Record<string, PluginRecord>>({})

    const rescan = vi
      .fn<() => Promise<void>>()
      .mockResolvedValueOnce(undefined)
      .mockImplementationOnce(async () => {
        records.set({ 'word-count': record('word-count', FILE) })
      })

    await expect(settleUnifiedDesktopPluginId(rescan, records, FILE, 3, 30)).resolves.toEqual({ id: 'word-count' })
    expect(rescan).toHaveBeenCalledTimes(2)
  })

  it('stops at once on a terminal outcome (broken load) instead of burning every round', async () => {
    const records = atom<Record<string, PluginRecord>>({})

    const rescan = vi.fn<() => Promise<void>>(async () => {
      records.set({ 'word-count': record('word-count', FILE, { status: 'error', error: 'syntax' }) })
    })

    await expect(settleUnifiedDesktopPluginId(rescan, records, FILE, 3, 30)).resolves.toEqual({ error: 'syntax' })
    expect(rescan).toHaveBeenCalledTimes(1)
  })

  it('stops after the last round with null', async () => {
    const records = atom<Record<string, PluginRecord>>({})
    const rescan = vi.fn<() => Promise<void>>(async () => undefined)

    await expect(settleUnifiedDesktopPluginId(rescan, records, FILE, 3, 10)).resolves.toBeNull()
    expect(rescan).toHaveBeenCalledTimes(3)
    expect(records.lc).toBe(0)
  })
})
