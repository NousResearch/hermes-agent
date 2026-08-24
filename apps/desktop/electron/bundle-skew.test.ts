import { execFileSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'

import { afterAll, describe, expect, it } from 'vitest'

import { detectBundleSkew, isFallbackCommit, type RunGit, RUNTIME_PATHS } from './bundle-skew'

const REPO = '/repo'
const STAMP = { commit: 'a'.repeat(40), source: 'ci' }

function gitReturning(stdout: string, code = 0): RunGit {
  return async () => ({ code, stderr: '', stdout })
}

/**
 * A git fake that answers per subcommand, so a test can say "ancestry fails,
 * but the count would have claimed skew" — which is the shape of #92233.
 */
function gitAnswering(answers: Record<string, { code?: number; stderr?: string; stdout?: string }>): {
  calls: string[][]
  git: RunGit
} {
  const calls: string[][] = []

  const git: RunGit = async args => {
    calls.push(args)

    const answer = answers[args[0]] ?? {}

    return {
      code: answer.code ?? 0,
      stderr: answer.stderr ?? '',
      stdout: answer.stdout ?? ''
    }
  }

  return { calls, git }
}

/** Every subcommand succeeds; rev-list reports `count`. */
function gitCounting(count: string): RunGit {
  return gitAnswering({ 'merge-base': { code: 0 }, 'rev-list': { stdout: count } }).git
}

describe('isFallbackCommit', () => {
  it('matches the all-zero placeholder at any stamp length', () => {
    expect(isFallbackCommit('0'.repeat(40))).toBe(true)
    expect(isFallbackCommit('0'.repeat(7))).toBe(true)
    expect(isFallbackCommit('a'.repeat(40))).toBe(false)
  })
})

describe('detectBundleSkew', () => {
  it('reports stale when desktop commits landed after the stamp', async () => {
    const result = await detectBundleSkew(STAMP, gitCounting('3\n'), REPO)

    expect(result).toEqual({ desktopCommitsBehind: 3, outOfSync: true })
  })

  it('counts only commits that touch runtime desktop paths', async () => {
    const { calls, git } = gitAnswering({ 'merge-base': { code: 0 }, 'rev-list': { stdout: '0' } })

    await detectBundleSkew(STAMP, git, REPO)

    expect(calls[1]).toEqual(['rev-list', '--count', `${STAMP.commit}..HEAD`, '--', ...RUNTIME_PATHS])
  })

  it('is quiet when no desktop commits follow the stamp', async () => {
    const result = await detectBundleSkew(STAMP, gitCounting('0\n'), REPO)

    expect(result).toEqual({ desktopCommitsBehind: 0, outOfSync: false })
  })

  it('is quiet without a stamp (dev runs)', async () => {
    expect(await detectBundleSkew(null, gitReturning('9'), REPO)).toEqual({
      desktopCommitsBehind: null,
      outOfSync: false
    })
  })

  it('is quiet on a fallback stamp (non-git build)', async () => {
    const fallback = { commit: '0'.repeat(40), source: 'fallback' }

    expect(await detectBundleSkew(fallback, gitReturning('9'), REPO)).toEqual({
      desktopCommitsBehind: null,
      outOfSync: false
    })
  })

  it('is quiet when git fails (unknown commit, shallow clone, no git)', async () => {
    expect(await detectBundleSkew(STAMP, gitReturning('', 128), REPO)).toEqual({
      desktopCommitsBehind: null,
      outOfSync: false
    })
  })

  it('is quiet when git throws', async () => {
    const git: RunGit = async () => {
      throw new Error('spawn ENOENT')
    }

    expect(await detectBundleSkew(STAMP, git, REPO)).toEqual({
      desktopCommitsBehind: null,
      outOfSync: false
    })
  })

  it('is quiet on unparsable rev-list output', async () => {
    expect(await detectBundleSkew(STAMP, gitCounting('fatal: bad object'), REPO)).toEqual({
      desktopCommitsBehind: null,
      outOfSync: false
    })
  })

  // #92233: a ZIP-fallback update rewrites the tree into a synthetic root, so
  // the stamp commit still RESOLVES but is unreachable from HEAD. `A..HEAD`
  // then counts HEAD's own history instead of measuring skew, and reports a
  // permanent 1 even though apps/desktop is byte-identical. The user gets an
  // "App build out of date" warning that cannot go off, so no remedy clears it.
  it('is quiet when the stamp is not an ancestor of HEAD', async () => {
    const { git } = gitAnswering({
      'merge-base': { code: 1 },
      'rev-list': { stdout: '1\n' }
    })

    expect(await detectBundleSkew(STAMP, git, REPO)).toEqual({
      desktopCommitsBehind: null,
      outOfSync: false
    })
  })

  it('does not consult the commit count once ancestry is refused', async () => {
    const { calls, git } = gitAnswering({
      'merge-base': { code: 1 },
      'rev-list': { stdout: '9999\n' }
    })

    await detectBundleSkew(STAMP, git, REPO)

    expect(calls.map(args => args[0])).toEqual(['merge-base'])
  })

  it('asks about ancestry before counting, against the same stamp', async () => {
    const { calls, git } = gitAnswering({
      'merge-base': { code: 0 },
      'rev-list': { stdout: '2\n' }
    })

    const result = await detectBundleSkew(STAMP, git, REPO)

    expect(calls[0]).toEqual(['merge-base', '--is-ancestor', STAMP.commit, 'HEAD'])
    expect(calls[1]?.[0]).toBe('rev-list')
    expect(result).toEqual({ desktopCommitsBehind: 2, outOfSync: true })
  })

  it('is quiet when git cannot answer the ancestry question at all', async () => {
    const { git } = gitAnswering({
      'merge-base': { code: 128 },
      'rev-list': { stdout: '4\n' }
    })

    expect(await detectBundleSkew(STAMP, git, REPO)).toEqual({
      desktopCommitsBehind: null,
      outOfSync: false
    })
  })

  // Shallow clones, measured against git 2.55 rather than assumed. A stamp
  // commit from BEFORE the graft boundary is not an object the clone has, so
  // `--is-ancestor` exits 128 with "Not a valid object name" — the same
  // unknowable bucket as any other missing commit, not a shallow-specific
  // failure. A stamp INSIDE the shallow graph is answered normally, so
  // `--fetch-depth`-limited CI checkouts do not lose skew detection wholesale;
  // only builds stamped deeper than the checkout goes do.
  it('is quiet on a shallow clone whose stamp predates the graft boundary', async () => {
    const { calls, git } = gitAnswering({
      'merge-base': {
        code: 128,
        stderr: `fatal: Not a valid object name ${STAMP.commit}`
      },
      'rev-list': { stdout: '7\n' }
    })

    expect(await detectBundleSkew(STAMP, git, REPO)).toEqual({
      desktopCommitsBehind: null,
      outOfSync: false
    })
    expect(calls).toHaveLength(1)
  })

  it('still detects skew on a shallow clone when the stamp is in the graph', async () => {
    const { git } = gitAnswering({
      'merge-base': { code: 0 },
      'rev-list': { stdout: '2\n' }
    })

    expect(await detectBundleSkew(STAMP, git, REPO)).toEqual({
      desktopCommitsBehind: 2,
      outOfSync: true
    })
  })
})

// Real-git integration: proves the pathspec discriminates docs/e2e-only
// commits from runtime commits, and that a disconnected stamp goes quiet, in
// an actual repository rather than against a hand-written fake.
const scratchRepos: string[] = []

afterAll(() => {
  for (const dir of scratchRepos) {
    rmSync(dir, { force: true, recursive: true })
  }
})

function scratchGit(repoRoot: string) {
  return (...args: string[]) =>
    execFileSync('git', ['-c', 'user.email=skew@test', '-c', 'user.name=skew', ...args], {
      cwd: repoRoot,
      stdio: ['ignore', 'pipe', 'pipe']
    })
      .toString()
      .trim()
}

function makeScratchRepo(): { base: string; repoRoot: string } {
  const repoRoot = mkdtempSync(join(tmpdir(), 'bundle-skew-'))
  scratchRepos.push(repoRoot)

  const git = scratchGit(repoRoot)

  git('init', '-q', '-b', 'main')
  git('commit', '-q', '--allow-empty', '-m', 'base')

  return { base: git('rev-parse', 'HEAD'), repoRoot }
}

function writeFiles(repoRoot: string, files: string[]) {
  for (const file of files) {
    const target = join(repoRoot, file)

    mkdirSync(dirname(target), { recursive: true })
    writeFileSync(target, '')
  }
}

function realGitRun(root: string): RunGit {
  return async (args, options) => {
    try {
      const stdout = execFileSync('git', args, {
        cwd: options.cwd || root,
        stdio: ['ignore', 'pipe', 'pipe']
      }).toString()

      return { code: 0, stderr: '', stdout }
    } catch (error) {
      const e = error as { status?: number; stderr?: Buffer; stdout?: Buffer }

      return {
        code: e.status ?? 1,
        stderr: e.stderr?.toString() ?? '',
        stdout: e.stdout?.toString() ?? ''
      }
    }
  }
}

describe('detectBundleSkew against a real git repo', () => {
  it('is quiet when only docs and e2e specs changed under apps/desktop', async () => {
    const { base, repoRoot } = makeScratchRepo()
    const git = scratchGit(repoRoot)

    writeFiles(repoRoot, ['apps/desktop/AGENTS.md', 'apps/desktop/e2e/boot.spec.ts'])
    git('add', '.')
    git('commit', '-q', '-m', 'docs and e2e only')

    const result = await detectBundleSkew({ commit: base, source: 'local' }, realGitRun(repoRoot), repoRoot)

    expect(result).toEqual({ desktopCommitsBehind: 0, outOfSync: false })
  })

  it('warns when a renderer file changed under apps/desktop', async () => {
    const { base, repoRoot } = makeScratchRepo()
    const git = scratchGit(repoRoot)

    writeFiles(repoRoot, ['apps/desktop/src/app/new-feature.tsx', 'apps/desktop/README.md'])
    git('add', '.')
    git('commit', '-q', '-m', 'renderer change')

    const result = await detectBundleSkew({ commit: base, source: 'local' }, realGitRun(repoRoot), repoRoot)

    expect(result).toEqual({ desktopCommitsBehind: 1, outOfSync: true })
  })

  // The #92233 install, reproduced: the update rewrote the tree onto a fresh
  // orphan root, so the stamp resolves but is unreachable. Real git answers
  // `rev-list` with a positive count here — ancestry is the only thing that
  // keeps the banner off.
  it('is quiet when the stamp sits on a disconnected root', async () => {
    const { base, repoRoot } = makeScratchRepo()
    const git = scratchGit(repoRoot)

    git('checkout', '-q', '--orphan', 'rewritten')
    writeFiles(repoRoot, ['apps/desktop/src/app/shell.tsx'])
    git('add', '.')
    git('commit', '-q', '-m', 'synthetic root after a ZIP-fallback update')

    const runGit = realGitRun(repoRoot)

    // Precondition: the raw count this function used to trust is nonzero.
    const raw = await runGit(['rev-list', '--count', `${base}..HEAD`, '--', ...RUNTIME_PATHS], { cwd: repoRoot })

    expect(Number.parseInt(raw.stdout.trim(), 10)).toBeGreaterThan(0)

    const result = await detectBundleSkew({ commit: base, source: 'local' }, runGit, repoRoot)

    expect(result).toEqual({ desktopCommitsBehind: null, outOfSync: false })
  })
})
