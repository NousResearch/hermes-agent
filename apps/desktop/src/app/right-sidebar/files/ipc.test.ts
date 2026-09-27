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
    // The rule lookup anchors at the nearest git root of the LISTED directory,
    // so the bridge is asked about dirPath first (the mock answers with the repo root).
    expect(gitRoot).toHaveBeenCalledWith('C:/repo/src')
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

  it('keeps nested repository roots visible when the parent gitignores them', async () => {
    // The bridge strips .git from listings, so repo-ness is detected via gitRoot:
    // a nested repo resolves to itself, an ordinary ignored dir to the parent's root.
    gitRoot.mockImplementation(async path => {
      if (path === '/repo/dev/ownward-studio') {
        return '/repo/dev/ownward-studio'
      }

      return '/repo'
    })
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

      return ok([])
    })
    readFileDataUrl.mockResolvedValue(dataUrl('dev/*\n!dev/README.md\n'))

    const result = await readProjectDir('/repo/dev', '/repo')

    expect(result.entries.map(entry => entry.name)).toEqual(['ownward-studio', 'README.md'])
  })

  it('keeps worktree roots (a .git file, not a directory) visible', async () => {
    // A linked worktree is its own toplevel too, so the same git check covers it.
    gitRoot.mockImplementation(async path => {
      if (path === '/repo/dev/wt-feature') {
        return '/repo/dev/wt-feature'
      }

      return '/repo'
    })
    readDir.mockImplementation(async path => {
      if (path === '/repo') {
        return ok([{ name: '.gitignore', path: '/repo/.gitignore', isDirectory: false }])
      }

      if (path === '/repo/dev') {
        return ok([{ name: 'wt-feature', path: '/repo/dev/wt-feature', isDirectory: true }])
      }

      return ok([])
    })
    readFileDataUrl.mockResolvedValue(dataUrl('dev/*\n'))

    const result = await readProjectDir('/repo/dev', '/repo')

    expect(result.entries.map(entry => entry.name)).toEqual(['wt-feature'])
  })

  it('inside a nested repo, only that repo’s own .gitignore governs', async () => {
    // The parent's dev/* must NOT empty the nested repo: the rule chain re-anchors
    // at the nested repo's own root, so only its own .gitignore applies.
    gitRoot.mockImplementation(async path => {
      if (path === '/repo/dev/ownward-studio') {
        return '/repo/dev/ownward-studio'
      }

      return '/repo'
    })
    readDir.mockImplementation(async path => {
      if (path === '/repo') {
        return ok([{ name: '.gitignore', path: '/repo/.gitignore', isDirectory: false }])
      }

      if (path === '/repo/dev/ownward-studio') {
        return ok([
          { name: '.gitignore', path: '/repo/dev/ownward-studio/.gitignore', isDirectory: false },
          { name: 'company-memory', path: '/repo/dev/ownward-studio/company-memory', isDirectory: true },
          { name: 'coverage', path: '/repo/dev/ownward-studio/coverage', isDirectory: true },
          { name: 'README.md', path: '/repo/dev/ownward-studio/README.md', isDirectory: false }
        ])
      }

      return ok([])
    })
    readFileDataUrl.mockImplementation(async path => {
      if (path === '/repo/.gitignore') {
        return dataUrl('dev/*\n')
      }

      return dataUrl('coverage/\n')
    })

    const result = await readProjectDir('/repo/dev/ownward-studio', '/repo')

    expect(result.entries.map(entry => entry.name)).toEqual(['.gitignore', 'company-memory', 'README.md'])
  })

  it('a scoped clearProjectDirCache evicts that root’s subtree and leaves sibling roots cached', async () => {
    // `nested` is the set of directories that are their own repository root;
    // everything else resolves to its parent repo. Growing the set between
    // reads simulates the user running `git init` underneath the project.
    const gitRootOf =
      (nested: string[]) =>
      async (path: string): Promise<string | null> =>
        nested.includes(path) ? path : path.startsWith('/repo2') ? '/repo2' : '/repo'

    const countCalls = (mock: typeof gitRoot | typeof readFileDataUrl, arg: string) =>
      mock.mock.calls.filter(([callArg]) => callArg === arg).length

    readDir.mockImplementation(async path => {
      if (path === '/repo') {
        return ok([{ name: '.gitignore', path: '/repo/.gitignore', isDirectory: false }])
      }

      if (path === '/repo/dev') {
        return ok([
          { name: 'ownward-studio', path: '/repo/dev/ownward-studio', isDirectory: true },
          { name: 'scratch', path: '/repo/dev/scratch', isDirectory: true },
          { name: 'README.md', path: '/repo/dev/README.md', isDirectory: false }
        ])
      }

      if (path === '/repo2') {
        return ok([{ name: '.gitignore', path: '/repo2/.gitignore', isDirectory: false }])
      }

      if (path === '/repo2/src') {
        return ok([
          { name: 'debug.log', path: '/repo2/src/debug.log', isDirectory: false },
          { name: 'keep.ts', path: '/repo2/src/keep.ts', isDirectory: false }
        ])
      }

      return ok([])
    })
    readFileDataUrl.mockImplementation(async path =>
      path === '/repo/.gitignore' ? dataUrl('dev/*\n!dev/README.md\n') : dataUrl('src/*.log\n')
    )

    // First pass: scratch is an ordinary ignored directory, so it is hidden.
    gitRoot.mockImplementation(gitRootOf(['/repo/dev/ownward-studio']))

    await expect(readProjectDir('/repo/dev', '/repo')).resolves.toMatchObject({
      entries: [{ name: 'ownward-studio' }, { name: 'README.md' }]
    })
    await expect(readProjectDir('/repo2/src', '/repo2')).resolves.toMatchObject({
      entries: [{ name: 'keep.ts' }]
    })

    // The user runs `git init` in /repo/dev/scratch and hits refresh: the
    // scoped clear must evict everything cached at or under /repo — the git
    // roots, the .gitignore chain, the nested-repo answers — so the re-read
    // sees the new repository instead of the stale `false`.
    gitRoot.mockImplementation(gitRootOf(['/repo/dev/ownward-studio', '/repo/dev/scratch']))
    clearProjectDirCache('/repo')

    await expect(readProjectDir('/repo/dev', '/repo')).resolves.toMatchObject({
      entries: [{ name: 'ownward-studio' }, { name: 'scratch' }, { name: 'README.md' }]
    })
    // The cleared root was genuinely re-probed, not served from cache.
    expect(countCalls(gitRoot, '/repo/dev')).toBe(2)
    expect(countCalls(readFileDataUrl, '/repo/.gitignore')).toBe(2)

    // The sibling root /repo2 is NOT under /repo — a bare string prefix would
    // have evicted it — so its answers stay cached and nothing re-probes.
    await expect(readProjectDir('/repo2/src', '/repo2')).resolves.toMatchObject({
      entries: [{ name: 'keep.ts' }]
    })
    expect(countCalls(gitRoot, '/repo2/src')).toBe(1)
    expect(countCalls(readFileDataUrl, '/repo2/.gitignore')).toBe(1)
  })
})
