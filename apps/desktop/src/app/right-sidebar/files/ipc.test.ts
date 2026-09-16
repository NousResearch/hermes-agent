/// <reference types="node" />

import { Buffer } from 'node:buffer'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirEntry, HermesReadDirResult } from '@/global'

import { clearProjectDirCache, readProjectDir } from './ipc'

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

  it('keeps nested repository roots visible when the parent gitignores them', async () => {
    gitRoot.mockResolvedValue('/repo')
    readDir.mockImplementation(async path => {
      if (path === '/repo') {
        return ok([{ name: '.gitignore', path: '/repo/.gitignore', isDirectory: false }])
      }

      if (path === '/repo/dev') {
        return ok([
          { name: 'ownward-studio', path: '/repo/dev/ownward-studio', isDirectory: true },
          { name: 'scratch', path: '/repo/dev/scratch', isDirectory: true },
          { name: 'notes.txt', path: '/repo/dev/notes.txt', isDirectory: false },
          { name: 'README.md', path: '/repo/dev/README.md', isDirectory: false }
        ])
      }

      if (path === '/repo/dev/ownward-studio') {
        return ok([{ name: '.git', path: '/repo/dev/ownward-studio/.git', isDirectory: true }])
      }

      return ok([])
    })
    readFileDataUrl.mockResolvedValue(dataUrl('dev/*\n!dev/README.md\n'))

    const result = await readProjectDir('/repo/dev', '/repo')

    expect(result.entries.map(entry => entry.name)).toEqual(['ownward-studio', 'README.md'])
  })

  it('keeps worktree roots with a .git file visible', async () => {
    gitRoot.mockResolvedValue('/repo')
    readDir.mockImplementation(async path => {
      if (path === '/repo') {
        return ok([{ name: '.gitignore', path: '/repo/.gitignore', isDirectory: false }])
      }

      if (path === '/repo/dev') {
        return ok([{ name: 'wt-feature', path: '/repo/dev/wt-feature', isDirectory: true }])
      }

      if (path === '/repo/dev/wt-feature') {
        return ok([{ name: '.git', path: '/repo/dev/wt-feature/.git', isDirectory: false }])
      }

      return ok([])
    })
    readFileDataUrl.mockResolvedValue(dataUrl('dev/*\n'))

    const result = await readProjectDir('/repo/dev', '/repo')

    expect(result.entries.map(entry => entry.name)).toEqual(['wt-feature'])
  })

  it('invalidates cached nested repository probes below a refreshed root', async () => {
    let nestedRepoExists = false

    gitRoot.mockResolvedValue('/repo')
    readDir.mockImplementation(async path => {
      if (path === '/repo') {
        return ok([{ name: '.gitignore', path: '/repo/.gitignore', isDirectory: false }])
      }

      if (path === '/repo/dev') {
        return ok([{ name: 'nested', path: '/repo/dev/nested', isDirectory: true }])
      }

      if (path === '/repo/dev/nested') {
        return nestedRepoExists
          ? ok([{ name: '.git', path: '/repo/dev/nested/.git', isDirectory: true }])
          : ok([])
      }

      return ok([])
    })
    readFileDataUrl.mockResolvedValue(dataUrl('dev/*\n'))

    const before = await readProjectDir('/repo/dev', '/repo')
    expect(before.entries).toEqual([])

    nestedRepoExists = true
    clearProjectDirCache('/repo')

    const after = await readProjectDir('/repo/dev', '/repo')
    expect(after.entries.map(entry => entry.name)).toEqual(['nested'])
  })
})
