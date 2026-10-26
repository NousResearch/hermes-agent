import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import { normalizeRepoScanPath, repoScanPathIsWithin, scanGitRepos } from './git-repo-scan'

const tempDirs: string[] = []

function tempDir(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-repo-scan-'))
  tempDirs.push(dir)

  return dir
}

function makeRepo(root: string, valid = true): void {
  fs.mkdirSync(path.join(root, '.git'), { recursive: true })

  if (valid) {
    fs.writeFileSync(path.join(root, '.git', 'HEAD'), 'ref: refs/heads/main\n')
  }
}

function makeRepoAt(root: string, ...segments: string[]): string {
  const repo = path.join(root, ...segments)
  makeRepo(repo)

  return repo
}

function foundRoots(results: { root: string }[]): string[] {
  return results.map(entry => entry.root).sort()
}

afterEach(() => {
  vi.restoreAllMocks()

  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

describe('scanGitRepos', () => {
  it('does not read the filesystem when discovery is disabled', async () => {
    const read = vi.spyOn(fs.promises, 'readdir')

    await expect(scanGitRepos([], { enabled: false })).resolves.toEqual([])
    expect(read).not.toHaveBeenCalled()
  })

  it('does not fall back to scanning the home directory when roots are empty (#53328)', async () => {
    const read = vi.spyOn(fs.promises, 'readdir')

    await expect(scanGitRepos([], { enabled: true })).resolves.toEqual([])
    expect(read).not.toHaveBeenCalled()
  })

  it('scans only configured roots and excludes complete subtrees', async () => {
    const root = tempDir()
    const included = path.join(root, 'included')
    const excluded = path.join(root, 'excluded')
    const invalid = path.join(root, 'invalid')
    makeRepo(included)
    makeRepo(excluded)
    makeRepo(invalid, false)

    await expect(scanGitRepos([root], { enabled: true, excludePaths: [excluded], maxDepth: 2 })).resolves.toEqual([
      { label: 'included', root: included }
    ])
  })

  it('bounds readdir concurrency across the entire branching traversal', async () => {
    const root = path.resolve(path.sep, 'synthetic-repo-scan-root')
    const branching = 32
    const maxDepth = 3
    const concurrencyLimit = 32

    const directories = Array.from({ length: branching }, (_, index) => ({
      name: `branch-${index}`,
      isDirectory: () => true
    })) as fs.Dirent[]

    let activeReads = 0
    let peakActiveReads = 0

    const syntheticReaddir = async (dir: fs.PathLike): Promise<fs.Dirent[]> => {
      activeReads += 1
      peakActiveReads = Math.max(peakActiveReads, activeReads)
      await Promise.resolve()
      activeReads -= 1

      const relative = path.relative(root, String(dir))
      const depth = relative === '' ? 0 : relative.split(path.sep).length

      return depth < maxDepth ? directories : []
    }

    vi.spyOn(fs.promises, 'readdir').mockImplementation(
      syntheticReaddir as unknown as typeof fs.promises.readdir
    )

    await expect(scanGitRepos([root], { enabled: true, maxDepth })).resolves.toEqual([])
    expect(peakActiveReads).toBeLessThanOrEqual(concurrencyLimit)
  })

  it('returns repositories in deterministic path order despite read completion order', async () => {
    const root = tempDir()
    const first = path.join(root, 'a-repo')
    const second = path.join(root, 'z-repo')
    makeRepo(first)
    makeRepo(second)

    const readdir = fs.promises.readdir

    const delayedReaddir = async (dir: fs.PathLike, options: { withFileTypes: true }): Promise<fs.Dirent[]> => {
      if (String(dir) === first) {
        await new Promise(resolve => setTimeout(resolve, 10))
      }

      return readdir(dir, options)
    }

    vi.spyOn(fs.promises, 'readdir').mockImplementation(delayedReaddir as unknown as typeof fs.promises.readdir)

    await expect(scanGitRepos([first, second], { enabled: true })).resolves.toEqual([
      { label: 'a-repo', root: first },
      { label: 'z-repo', root: second }
    ])
  })

  it('deduplicates overlapping roots', async () => {
    const root = tempDir()
    const repo = path.join(root, 'repo')
    makeRepo(repo)

    const result = await scanGitRepos([root, repo], { enabled: true })
    expect(result).toEqual([{ label: 'repo', root: repo }])
  })
})

describe.runIf(process.platform !== 'win32')('macOS TCC-protected media exclusions (issue #57611 salvage)', () => {
  it('finds a normal repo but skips root-level media folders on darwin', async () => {
    const root = tempDir()
    const dev = makeRepoAt(root, 'dev', 'proj')
    makeRepoAt(root, 'Pictures', 'wallpapers')
    makeRepoAt(root, 'Music', 'samples')
    makeRepoAt(root, 'Movies', 'clips')
    makeRepoAt(root, 'Public', 'shared')

    expect(foundRoots(await scanGitRepos([root], { enabled: true, platform: 'darwin' }))).toEqual([dev])
  })

  it('still scans a media-named directory below the search root on darwin', async () => {
    const root = tempDir()
    const nested = makeRepoAt(root, 'dev', 'Music', 'app')

    expect(foundRoots(await scanGitRepos([root], { enabled: true, platform: 'darwin' }))).toEqual([nested])
  })

  it('skips Apple media-library packages at any depth on darwin', async () => {
    const root = tempDir()
    const keeper = makeRepoAt(root, 'code', 'site')
    makeRepoAt(root, 'code', 'Photos Library.photoslibrary', 'inner')
    makeRepoAt(root, 'backups', 'Music Library.MUSICLIBRARY', 'inner')
    makeRepoAt(root, 'backups', 'TV Library.tvlibrary', 'inner')
    makeRepoAt(root, 'backups', 'Old.aplibrary', 'inner')

    expect(foundRoots(await scanGitRepos([root], { enabled: true, platform: 'darwin' }))).toEqual([keeper])
  })

  it('walks an explicitly passed media root on darwin', async () => {
    const root = tempDir()
    const musicRoot = path.join(root, 'Music')
    const repo = makeRepoAt(musicRoot, 'samples')

    expect(foundRoots(await scanGitRepos([musicRoot], { enabled: true, platform: 'darwin' }))).toEqual([repo])
  })

  it('does not exclude media-named folders on linux', async () => {
    const root = tempDir()
    const dev = makeRepoAt(root, 'dev', 'proj')
    const music = makeRepoAt(root, 'Music', 'samples')

    expect(foundRoots(await scanGitRepos([root], { enabled: true, platform: 'linux' }))).toEqual([dev, music].sort())
  })
})

describe('repository scan path normalization', () => {
  it('expands tilde and resolves relative paths from home', () => {
    expect(normalizeRepoScanPath('~/src', { homeDir: '/Users/rudi', platform: 'darwin' })?.value).toBe(
      '/Users/rudi/src'
    )
    expect(normalizeRepoScanPath('src', { homeDir: '/Users/rudi', platform: 'linux' })?.value).toBe('/Users/rudi/src')
  })

  it('uses segment-aware, case-insensitive containment on Windows', () => {
    const options = { homeDir: 'C:\\Users\\Rudi', platform: 'win32' as const }
    expect(repoScanPathIsWithin('c:\\SRC\\Fever\\repo', 'C:\\src\\fever', options)).toBe(true)
    expect(repoScanPathIsWithin('C:\\src\\feverish', 'C:\\src\\fever', options)).toBe(false)
  })
})
