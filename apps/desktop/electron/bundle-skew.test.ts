import { execFileSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'

import { afterAll, describe, expect, it } from 'vitest'

import { createBundleSkewProbe, detectBundleSkew, isFallbackCommit, type RunGit, RUNTIME_PATHS } from './bundle-skew'

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

  // Fix 3: the standalone caller discards `cacheable`, so resolving the
  // shallow flag here would spawn a rev-parse whose answer is thrown away.
  // The not-an-ancestor answer is still returned; nothing but merge-base runs.
  it('spawns no rev-parse when merge-base reports not-an-ancestor', async () => {
    const { calls, git } = gitAnswering({
      'merge-base': { code: 1 },
      'rev-list': { stdout: '1' }
    })

    expect(await detectBundleSkew(STAMP, git, REPO)).toEqual({
      desktopCommitsBehind: null,
      outOfSync: false
    })
    expect(calls.map(args => args[0])).toEqual(['merge-base'])
  })

  // Fix 2: only a resolved object id may fill a git argument. A stamp that is
  // an option-like or non-hex string never reaches git, so it cannot be
  // interpreted as an option or an error message.
  it('spawns no git for a stamp that is not a resolved sha', async () => {
    const { calls, git } = gitAnswering({
      'merge-base': { code: 0 },
      'rev-list': { stdout: '2' }
    })

    expect(await detectBundleSkew({ commit: '--all', source: 'ci' }, git, REPO)).toEqual({
      desktopCommitsBehind: null,
      outOfSync: false
    })
    expect(await detectBundleSkew({ commit: 'A'.repeat(40), source: 'ci' }, git, REPO)).toEqual({
      desktopCommitsBehind: null,
      outOfSync: false
    })
    expect(await detectBundleSkew({ commit: 'a'.repeat(39), source: 'ci' }, git, REPO)).toEqual({
      desktopCommitsBehind: null,
      outOfSync: false
    })
    expect(calls).toEqual([])
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

const NOT_STALE = { desktopCommitsBehind: null, outOfSync: false }

/**
 * A git fake that records every subcommand, so a probe test can count the
 * expensive spawns (merge-base, rev-list) separately from the cheap HEAD
 * resolution (rev-parse).
 */
function gitScripted(options: {
  count?: string
  head?: string | (() => string)
  lineEnding?: string
  mergeBaseCode?: number | (() => number)
  revListCode?: number | (() => number)
  revParseCode?: number | (() => number)
  shallow?: boolean | (() => boolean)
}): { calls: string[][]; git: RunGit } {
  const calls: string[][] = []

  const git: RunGit = async args => {
    calls.push(args)

    if (args[0] === 'rev-parse') {
      const code = typeof options.revParseCode === 'function' ? options.revParseCode() : (options.revParseCode ?? 0)

      if (code !== 0) {
        return { code, stderr: 'fatal: bad revision', stdout: '' }
      }

      const head = typeof options.head === 'function' ? options.head() : (options.head ?? 'a'.repeat(40))
      const shallow = typeof options.shallow === 'function' ? options.shallow() : (options.shallow ?? false)
      const eol = options.lineEnding ?? '\n'

      // `git rev-parse --is-shallow-repository HEAD` prints the flag, then the
      // sha, on two lines. Measured against git 2.54, not assumed.
      return { code, stderr: '', stdout: `${shallow ? 'true' : 'false'}${eol}${head}${eol}` }
    }

    if (args[0] === 'merge-base') {
      const code = typeof options.mergeBaseCode === 'function' ? options.mergeBaseCode() : (options.mergeBaseCode ?? 0)

      return { code, stderr: code === 0 ? '' : 'fatal: git could not answer', stdout: '' }
    }

    if (args[0] === 'rev-list') {
      const code = typeof options.revListCode === 'function' ? options.revListCode() : (options.revListCode ?? 0)

      return { code, stderr: code === 0 ? '' : 'fatal: git could not answer', stdout: options.count ?? '2\n' }
    }

    return { code: 1, stderr: '', stdout: '' }
  }

  return { calls, git }
}

function spawnsOf(calls: string[][], verb: string): number {
  return calls.filter(args => args[0] === verb).length
}

/** Let a pending continuation (a settled git promise) drain its microtasks. */
function tick(): Promise<void> {
  return new Promise(resolve => {
    setTimeout(resolve, 5)
  })
}

describe('createBundleSkewProbe', () => {
  // The piling-up process bug: focus, the update poller and checkUpdates can
  // all fire before the first probe answers, and each used to spawn its own
  // merge-base/rev-list pair.
  it('shares a single probe across concurrent callers', async () => {
    const { calls, git } = gitScripted({ count: '3\n' })
    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: REPO })

    const [first, second] = await Promise.all([probe(), probe()])

    expect(first).toEqual({ desktopCommitsBehind: 3, outOfSync: true })
    expect(second).toEqual(first)
    expect(spawnsOf(calls, 'rev-list')).toBe(1)
    expect(spawnsOf(calls, 'merge-base')).toBe(1)
  })

  it('caches the result for an unchanged HEAD and respawns nothing but rev-parse', async () => {
    const { calls, git } = gitScripted({ count: '2\n' })
    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: REPO })

    const first = await probe()
    const afterFirst = calls.length
    const second = await probe()

    expect(second).toEqual(first)
    expect(calls.slice(afterFirst).map(args => args[0])).toEqual(['rev-parse'])
    expect(spawnsOf(calls, 'rev-list')).toBe(1)
  })

  it('reruns the probe when HEAD moves', async () => {
    let head = 'a'.repeat(40)
    const { calls, git } = gitScripted({ count: '1\n', head: () => head })
    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: REPO })

    await probe()

    expect(spawnsOf(calls, 'rev-list')).toBe(1)

    head = 'b'.repeat(40)

    expect(await probe()).toEqual({ desktopCommitsBehind: 1, outOfSync: true })
    expect(spawnsOf(calls, 'rev-list')).toBe(2)
  })

  // The cache key is read from rev-parse, but the expensive calls used to
  // resolve the literal 'HEAD' again. HEAD can move between them, so a commit
  // landing mid-probe made the answer describe a different commit than its key —
  // and merge-base and rev-list could even disagree with each other. Both must
  // run against the one sha the key was built from.
  it('pins merge-base and rev-list to the resolved HEAD sha, not the symbolic HEAD', async () => {
    const head = 'f'.repeat(40)
    const { calls, git } = gitScripted({ count: '2\n', head })
    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: REPO })

    expect(await probe()).toEqual({ desktopCommitsBehind: 2, outOfSync: true })

    expect(calls.find(args => args[0] === 'merge-base')).toEqual(['merge-base', '--is-ancestor', STAMP.commit, head])
    expect(calls.find(args => args[0] === 'rev-list')).toEqual([
      'rev-list',
      '--count',
      `${STAMP.commit}..${head}`,
      '--',
      ...RUNTIME_PATHS
    ])
  })

  it('gives up at the timeout, resolves not-stale and aborts the git call', async () => {
    let signal: AbortSignal | undefined

    const git: RunGit = (_args, options) => {
      signal = options.signal

      return new Promise(() => {})
    }

    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: REPO, timeoutMs: 25 })

    expect(await probe()).toEqual(NOT_STALE)
    expect(signal?.aborted).toBe(true)
  })

  it('reruns the full probe when an aborted git exits non-zero after the timeout', async () => {
    const pending: { resolve?: (value: { code: number; stderr: string; stdout: string }) => void } = {}
    const calls: string[][] = []
    let revListCalls = 0

    const git: RunGit = async (args, options) => {
      calls.push(args)

      if (args[0] === 'rev-parse') {
        return { code: 0, stderr: '', stdout: `false\n${'a'.repeat(40)}\n` }
      }

      if (args[0] === 'merge-base') {
        return { code: 0, stderr: '', stdout: '' }
      }

      if (args[0] === 'rev-list') {
        revListCalls += 1

        if (options.signal?.aborted || revListCalls > 1) {
          return { code: 130, stderr: 'fatal: killed by the timeout', stdout: '' }
        }

        return new Promise(resolve => {
          pending.resolve = resolve
        })
      }

      return { code: 1, stderr: '', stdout: '' }
    }

    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: REPO, timeoutMs: 25 })

    expect(await probe()).toEqual(NOT_STALE)
    expect(spawnsOf(calls, 'rev-list')).toBe(1)

    // The killed git only reports its failure once the probe already gave up.
    pending.resolve?.({ code: 130, stderr: 'fatal: killed by the timeout', stdout: '' })
    await tick()

    // A fail-quiet answer that arrived after the timeout is not proof, so the
    // next call must ask git again instead of trusting a pinned result.
    await probe()

    expect(spawnsOf(calls, 'rev-list')).toBe(2)
  })

  it('caches a proven not-an-ancestor answer for an unchanged HEAD', async () => {
    const { calls, git } = gitScripted({ count: '3\n', mergeBaseCode: 1 })
    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: REPO })

    expect(await probe()).toEqual(NOT_STALE)
    expect(await probe()).toEqual(NOT_STALE)

    // "the stamp is not an ancestor" is a real answer, not an unknowable one,
    // so it is reused instead of respawning the expensive merge-base/rev-list.
    expect(spawnsOf(calls, 'merge-base')).toBe(1)
    expect(spawnsOf(calls, 'rev-list')).toBe(0)
  })

  // In a SHALLOW clone exit 1 can mean "the history that would prove ancestry
  // is not here yet", not "unrelated". Deepening or fetching changes that
  // answer without moving HEAD, so the cached exit-1 would hide real skew
  // until HEAD moved. A shallow answer is returned, never remembered.
  it('does not cache a not-an-ancestor answer from a shallow clone', async () => {
    const { calls, git } = gitScripted({ count: '3\n', mergeBaseCode: 1, shallow: true })
    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: REPO })

    expect(await probe()).toEqual(NOT_STALE)
    expect(await probe()).toEqual(NOT_STALE)

    expect(spawnsOf(calls, 'merge-base')).toBe(2)
    expect(spawnsOf(calls, 'rev-list')).toBe(0)
  })

  it('still caches a proven ancestry answer from a shallow clone', async () => {
    const { calls, git } = gitScripted({ count: '2\n', shallow: true })
    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: REPO })

    expect(await probe()).toEqual({ desktopCommitsBehind: 2, outOfSync: true })
    expect(await probe()).toEqual({ desktopCommitsBehind: 2, outOfSync: true })

    // A proven ancestor stays an ancestor when the clone is deepened, so the
    // count is reusable even in a shallow clone.
    expect(spawnsOf(calls, 'merge-base')).toBe(1)
    expect(spawnsOf(calls, 'rev-list')).toBe(1)
  })

  it('does not cache a transient git failure and asks git again on the next call', async () => {
    let mergeBaseCode = 128
    const { calls, git } = gitScripted({ count: '2\n', mergeBaseCode: () => mergeBaseCode })
    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: REPO })

    expect(await probe()).toEqual(NOT_STALE)
    expect(spawnsOf(calls, 'merge-base')).toBe(1)

    mergeBaseCode = 0

    // A git that could not answer is not proof of anything; the next call
    // must retry rather than trust the pinned not-stale.
    expect(await probe()).toEqual({ desktopCommitsBehind: 2, outOfSync: true })
    expect(spawnsOf(calls, 'merge-base')).toBe(2)
  })

  it('reruns the probe when the repo root changes without HEAD moving', async () => {
    const { calls, git } = gitScripted({ count: '2\n' })
    let root = '/repo-a'
    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: () => root })

    expect(await probe()).toEqual({ desktopCommitsBehind: 2, outOfSync: true })
    expect(spawnsOf(calls, 'rev-list')).toBe(1)

    // The source tree can be retargeted at runtime; the same HEAD under a
    // different root is a different tree and must not reuse the old answer.
    root = '/repo-b'

    expect(await probe()).toEqual({ desktopCommitsBehind: 2, outOfSync: true })
    expect(spawnsOf(calls, 'rev-list')).toBe(2)
  })

  // The root is resolved per call, so a probe already in flight for the old
  // root must not serve a caller whose root has just changed: joining it would
  // hand back another tree's answer.
  it('starts a second run when the root changes mid-flight and answers each caller for its own root', async () => {
    const pending: { resolve?: (value: { code: number; stderr: string; stdout: string }) => void } = {}
    const calls: Array<{ args: string[]; cwd: string }> = []
    let root = '/repo-a'

    const git: RunGit = async (args, options) => {
      calls.push({ args, cwd: options.cwd })

      if (args[0] === 'rev-parse') {
        return { code: 0, stderr: '', stdout: `false\n${'a'.repeat(40)}\n` }
      }

      if (args[0] === 'merge-base') {
        return { code: 0, stderr: '', stdout: '' }
      }

      if (args[0] === 'rev-list') {
        if (options.cwd === '/repo-a') {
          return new Promise(resolve => {
            pending.resolve = resolve
          })
        }

        return { code: 0, stderr: '', stdout: '4\n' }
      }

      return { code: 1, stderr: '', stdout: '' }
    }

    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: () => root })

    const first = probe()
    await tick()

    root = '/repo-b'

    expect(await probe()).toEqual({ desktopCommitsBehind: 4, outOfSync: true })
    expect(calls.filter(call => call.args[0] === 'rev-list').map(call => call.cwd)).toEqual(['/repo-a', '/repo-b'])

    // The abandoned run still answers its own caller with its own root's count.
    pending.resolve?.({ code: 0, stderr: '', stdout: '9\n' })
    await tick()

    expect(await first).toEqual({ desktopCommitsBehind: 9, outOfSync: true })
  })

  it('does not let a late old run overwrite a newer answer', async () => {
    const pending: { resolve?: (value: { code: number; stderr: string; stdout: string }) => void } = {}
    const calls: string[][] = []
    let revListCalls = 0

    const git: RunGit = async args => {
      calls.push(args)

      if (args[0] === 'rev-parse') {
        return { code: 0, stderr: '', stdout: `false\n${'a'.repeat(40)}\n` }
      }

      if (args[0] === 'merge-base') {
        return { code: 0, stderr: '', stdout: '' }
      }

      if (args[0] === 'rev-list') {
        revListCalls += 1

        if (revListCalls === 1) {
          return new Promise(resolve => {
            pending.resolve = resolve
          })
        }

        return { code: 0, stderr: '', stdout: '5\n' }
      }

      return { code: 1, stderr: '', stdout: '' }
    }

    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: REPO, timeoutMs: 25 })

    // The first run hangs, gives up, and is aborted — but keeps working.
    expect(await probe()).toEqual(NOT_STALE)

    // A newer run answers for the same HEAD.
    expect(await probe()).toEqual({ desktopCommitsBehind: 5, outOfSync: true })

    // The abandoned run finally answers, with a different count: a stale run
    // must not write over the newer, trustworthy result.
    pending.resolve?.({ code: 0, stderr: '', stdout: '9\n' })
    await tick()

    expect(await probe()).toEqual({ desktopCommitsBehind: 5, outOfSync: true })
    expect(spawnsOf(calls, 'rev-list')).toBe(2)
  })

  it('settles a work rejection that arrives after the timeout without an unhandled rejection', async () => {
    const pending: { reject?: (error: Error) => void } = {}

    const git: RunGit = async args => {
      if (args[0] === 'rev-parse') {
        return new Promise((_resolve, reject) => {
          pending.reject = reject
        })
      }

      return { code: 0, stderr: '', stdout: '' }
    }

    const seen: unknown[] = []

    const onUnhandled = (reason: unknown): void => {
      seen.push(reason)
    }

    process.on('unhandledRejection', onUnhandled)

    try {
      const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: REPO, timeoutMs: 25 })

      expect(await probe()).toEqual(NOT_STALE)

      // The probe already gave up; nothing awaited this git any more.
      pending.reject?.(new Error('git died after the timeout'))
      await tick()

      expect(seen).toEqual([])
    } finally {
      process.off('unhandledRejection', onUnhandled)
    }
  })

  it('reports not-stale on an unanswerable HEAD without caching the failure', async () => {
    let revParseCode = 128
    const { calls, git } = gitScripted({ count: '2\n', revParseCode: () => revParseCode })
    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: REPO })

    expect(await probe()).toEqual(NOT_STALE)
    expect(spawnsOf(calls, 'rev-list')).toBe(0)

    revParseCode = 0

    expect(await probe()).toEqual({ desktopCommitsBehind: 2, outOfSync: true })
    expect(spawnsOf(calls, 'rev-list')).toBe(1)
  })

  // A rev-parse line that is not 40 or 64 lowercase hex cannot be trusted in
  // the cache key or in the git arguments; the probe fails quiet instead.
  it('reports not-stale and spawns no merge-base when rev-parse returns a HEAD that is not a sha', async () => {
    const { calls, git } = gitScripted({ head: 'A'.repeat(40) })
    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: REPO })

    expect(await probe()).toEqual(NOT_STALE)
    expect(await probe()).toEqual(NOT_STALE)

    expect(spawnsOf(calls, 'merge-base')).toBe(0)
    expect(spawnsOf(calls, 'rev-list')).toBe(0)
    // Not cached, so the next call reads HEAD again.
    expect(spawnsOf(calls, 'rev-parse')).toBe(2)
  })

  // Fix 1: a Windows-hosted git can print CRLF. Splitting the two-line
  // rev-parse output on '\n' alone leaves a CR on the flag, which used to fail
  // the shape check and skip the cache. The answer must still be reused.
  it('caches a result when rev-parse prints CRLF line endings', async () => {
    const { calls, git } = gitScripted({ count: '2', lineEnding: '\r\n' })
    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: REPO })

    const first = await probe()

    expect(first).toEqual({ desktopCommitsBehind: 2, outOfSync: true })

    const afterFirst = calls.length

    expect(await probe()).toEqual(first)

    // A cached answer, so the second call spawns nothing but rev-parse.
    expect(calls.slice(afterFirst).map(args => args[0])).toEqual(['rev-parse'])
    expect(spawnsOf(calls, 'rev-list')).toBe(1)
  })

  // Fix 2 on the probe path: an invalid stamp is rejected before the HEAD
  // read, so no git process is spawned and nothing is cached.
  it('spawns no git for a stamp that is not a resolved sha', async () => {
    const { calls, git } = gitScripted({ count: '2' })
    const probe = createBundleSkewProbe({ stamp: { commit: '--all', source: 'ci' }, runGit: git, repoRoot: REPO })

    expect(await probe()).toEqual(NOT_STALE)
    expect(await probe()).toEqual(NOT_STALE)
    expect(calls).toEqual([])
  })

  it('accepts a 64-character lowercase HEAD sha', async () => {
    const head = 'b'.repeat(64)
    const { calls, git } = gitScripted({ count: '1\n', head })
    const probe = createBundleSkewProbe({ stamp: STAMP, runGit: git, repoRoot: REPO })

    expect(await probe()).toEqual({ desktopCommitsBehind: 1, outOfSync: true })

    expect(calls.find(args => args[0] === 'merge-base')).toEqual(['merge-base', '--is-ancestor', STAMP.commit, head])
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

/** Wrap a real RunGit so a test can count the expensive spawns. */
function countingGit(inner: RunGit): { calls: string[][]; git: RunGit } {
  const calls: string[][] = []

  const git: RunGit = async (args, options) => {
    calls.push(args)

    return inner(args, options)
  }

  return { calls, git }
}

/** A depth-1 clone of `sourceRoot`, every branch fetched. */
function makeShallowClone(sourceRoot: string): string {
  const cloneRoot = mkdtempSync(join(tmpdir(), 'bundle-skew-shallow-'))
  scratchRepos.push(cloneRoot)

  execFileSync('git', ['clone', '-q', '--depth', '1', '--no-single-branch', `file://${sourceRoot}`, cloneRoot], {
    stdio: ['ignore', 'pipe', 'pipe']
  })

  return cloneRoot
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

  // Finding: in a SHALLOW clone, exit 1 can mean the history that would prove
  // ancestry is not fetched yet. Real git, a real depth-1 clone: the probe
  // must answer not-stale but must not remember it, so the next same-HEAD call
  // asks again (a fetch or a deepen can change the answer without moving HEAD).
  it('answers not-stale in a real shallow clone and does not cache a not-an-ancestor answer', async () => {
    const { base, repoRoot: origin } = makeScratchRepo()
    const originGit = scratchGit(origin)

    originGit('checkout', '-q', '--orphan', 'rewritten')
    writeFiles(origin, ['apps/desktop/src/app/shell.tsx'])
    originGit('add', '.')
    originGit('commit', '-q', '-m', 'synthetic root after a ZIP-fallback update')

    const clone = makeShallowClone(origin)

    execFileSync('git', ['checkout', '-q', 'rewritten'], { cwd: clone, stdio: ['ignore', 'pipe', 'pipe'] })

    const { calls, git } = countingGit(realGitRun(clone))

    // Preconditions on real git, not on a fake: the clone is shallow and the
    // stamp commit (main's tip) is present but NOT an ancestor of HEAD.
    const shallow = await git(['rev-parse', '--is-shallow-repository'], { cwd: clone })

    expect(shallow.stdout.trim()).toBe('true')

    const ancestry = await git(['merge-base', '--is-ancestor', base, 'HEAD'], { cwd: clone })

    expect(ancestry.code).toBe(1)

    const headSha = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: clone }).toString().trim()

    const before = calls.filter(args => args[0] === 'merge-base').length
    const probe = createBundleSkewProbe({ stamp: { commit: base, source: 'local' }, runGit: git, repoRoot: clone })

    expect(await probe()).toEqual({ desktopCommitsBehind: null, outOfSync: false })
    expect(await probe()).toEqual({ desktopCommitsBehind: null, outOfSync: false })

    const mergeBases = calls.filter(args => args[0] === 'merge-base')

    // The probe asked about the sha it read from the two-line rev-parse, and
    // asked TWICE: the shallow answer was returned but never remembered.
    expect(mergeBases.length - before).toBe(2)
    expect(mergeBases.slice(before)).toEqual([
      ['merge-base', '--is-ancestor', base, headSha],
      ['merge-base', '--is-ancestor', base, headSha]
    ])
  })
})
