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

  it('deduplicates overlapping roots', async () => {
    const root = tempDir()
    const repo = path.join(root, 'repo')
    makeRepo(repo)

    const result = await scanGitRepos([root, repo], { enabled: true })
    expect(result).toEqual([{ label: 'repo', root: repo }])
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
