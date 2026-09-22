/// <reference types="node" />

import { Buffer } from 'node:buffer'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirEntry, HermesReadDirResult } from '@/global'

import { clearProjectDirCache, readProjectDir } from './ipc'
import { $showIgnoredRoots, setShowIgnoredFiles } from './prefs'

const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()
const readFileDataUrl = vi.fn<(path: string) => Promise<string>>()
const gitRoot = vi.fn<(path: string) => Promise<string | null>>()

function ok(entries: HermesReadDirEntry[]): HermesReadDirResult {
  return { entries }
}

function dataUrl(text: string) {
  return `data:text/plain;base64,${Buffer.from(text, 'utf8').toString('base64')}`
}

function installBridge() {
  ;(
    window as unknown as {
      hermesDesktop: {
        gitRoot: typeof gitRoot
        readDir: typeof readDir
        readFileDataUrl: typeof readFileDataUrl
      }
    }
  ).hermesDesktop = { gitRoot, readDir, readFileDataUrl }
}

describe('readProjectDir', () => {
  beforeEach(() => {
    clearProjectDirCache()
    readDir.mockReset()
    readFileDataUrl.mockReset()
    gitRoot.mockReset()
    installBridge()
  })

  afterEach(() => {
    clearProjectDirCache()
    $showIgnoredRoots.set([])
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('returns no-bridge when the desktop bridge is unavailable', async () => {
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop

    await expect(readProjectDir('/repo')).resolves.toEqual({ entries: [], error: 'no-bridge' })
  })

  it('filters gitignored entries when readDir returns Windows-style paths', async () => {
    gitRoot.mockResolvedValue('C:\\repo')
    readDir.mockImplementation(async path => {
      if (path === 'C:\\repo\\src') {
        return ok([
          { name: 'debug.log', path: 'C:\\repo\\src\\debug.log', isDirectory: false },
          { name: '临时.txt', path: 'C:\\repo\\src\\临时.txt', isDirectory: false },
          { name: 'keep.ts', path: 'C:\\repo\\src\\keep.ts', isDirectory: false }
        ])
      }

      if (path === 'C:/repo') {
        return ok([{ name: '.gitignore', path: 'C:/repo/.gitignore', isDirectory: false }])
      }

      if (path === 'C:/repo/src') {
        return ok([])
      }

      return ok([])
    })
    readFileDataUrl.mockResolvedValue(dataUrl('# Unicode 路径规则\nsrc/*.log\nsrc/临时.txt\n'))

    const result = await readProjectDir('C:\\repo\\src', 'C:\\repo')

    expect(result.entries.map(entry => entry.name)).toEqual(['keep.ts'])
    expect(gitRoot).toHaveBeenCalledWith('C:/repo')
    expect(readFileDataUrl).toHaveBeenCalledWith('C:/repo/.gitignore')
  })

  it('filters gitignored entries when Windows path casing differs across IPC results', async () => {
    gitRoot.mockResolvedValue('C:\\Repo')
    readDir.mockImplementation(async path => {
      if (path === 'c:\\repo\\src') {
        return ok([
          { name: 'debug.log', path: 'c:\\repo\\src\\debug.log', isDirectory: false },
          { name: 'keep.ts', path: 'c:\\repo\\src\\keep.ts', isDirectory: false }
        ])
      }

      if (path === 'C:/Repo') {
        return ok([{ name: '.gitignore', path: 'C:/Repo/.gitignore', isDirectory: false }])
      }

      if (path === 'C:/Repo/src') {
        return ok([])
      }

      return ok([])
    })
    readFileDataUrl.mockResolvedValue(dataUrl('src/*.log\n'))

    const result = await readProjectDir('c:\\repo\\src', 'c:\\repo')

    expect(result.entries.map(entry => entry.name)).toEqual(['keep.ts'])
  })

  it('does not fetch .gitignore contents when listings do not contain .gitignore', async () => {
    gitRoot.mockResolvedValue('/repo')
    readDir.mockImplementation(async path => {
      if (path === '/repo/src') {
        return ok([{ name: 'debug.log', path: '/repo/src/debug.log', isDirectory: false }])
      }

      return ok([])
    })

    const result = await readProjectDir('/repo/src', '/repo')

    expect(result.entries.map(entry => entry.name)).toEqual(['debug.log'])
    expect(readFileDataUrl).not.toHaveBeenCalled()
  })

  it('keeps gitignored entries — and skips the gitignore reads — when the root opted in', async () => {
    setShowIgnoredFiles('/repo', true)
    gitRoot.mockResolvedValue('/repo')
    readDir.mockImplementation(async path => {
      if (path === '/repo/src') {
        return ok([
          { name: 'debug.log', path: '/repo/src/debug.log', isDirectory: false },
          { name: 'keep.ts', path: '/repo/src/keep.ts', isDirectory: false }
        ])
      }

      if (path === '/repo') {
        return ok([{ name: '.gitignore', path: '/repo/.gitignore', isDirectory: false }])
      }

      return ok([])
    })
    readFileDataUrl.mockResolvedValue(dataUrl('src/*.log\n'))

    const result = await readProjectDir('/repo/src', '/repo')

    expect(result.entries.map(entry => entry.name)).toEqual(['debug.log', 'keep.ts'])
    expect(gitRoot).not.toHaveBeenCalled()
    expect(readFileDataUrl).not.toHaveBeenCalled()
  })

  it('still excludes ALWAYS_EXCLUDED entries when the root opted in', async () => {
    setShowIgnoredFiles('/repo', true)
    readDir.mockResolvedValue(
      ok([
        { name: '.git', path: '/repo/.git', isDirectory: true },
        { name: 'node_modules', path: '/repo/node_modules', isDirectory: true },
        { name: 'src', path: '/repo/src', isDirectory: true }
      ])
    )

    const result = await readProjectDir('/repo', '/repo')

    expect(result.entries.map(entry => entry.name)).toEqual(['src'])
  })

  it('opting one root in leaves other roots filtered', async () => {
    setShowIgnoredFiles('/repo', true)
    gitRoot.mockResolvedValue('/other')
    readDir.mockImplementation(async path => {
      if (path === '/other/src') {
        return ok([
          { name: 'debug.log', path: '/other/src/debug.log', isDirectory: false },
          { name: 'keep.ts', path: '/other/src/keep.ts', isDirectory: false }
        ])
      }

      if (path === '/other') {
        return ok([{ name: '.gitignore', path: '/other/.gitignore', isDirectory: false }])
      }

      return ok([])
    })
    readFileDataUrl.mockResolvedValue(dataUrl('src/*.log\n'))

    const result = await readProjectDir('/other/src', '/other')

    expect(result.entries.map(entry => entry.name)).toEqual(['keep.ts'])
  })
})
