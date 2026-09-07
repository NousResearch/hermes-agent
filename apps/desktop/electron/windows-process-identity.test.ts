import assert from 'node:assert/strict'

import { describe, it } from 'vitest'

import {
  createCachedWindowsProcessCreateTimeProbe,
  type ProcessIdentityCacheEntry,
  pruneProcessIdentityCache,
  queryDarwinProcessCreatedAt,
  queryProcessCreatedAt,
  queryWindowsProcessCreatedAt,
  readLinuxProcessCreatedAt
} from './windows-process-identity'

describe('queryWindowsProcessCreatedAt', () => {
  it.runIf(process.platform === 'win32')(
    'proves the real Electron test process creation identity',
    async () => {
      const expected = Math.floor(Date.now() / 1_000 - process.uptime())
      // Real `Get-Process` via `execFile`, not a stub: on a loaded host (many
      // live process-tree specs running in the same job) the default 3s probe
      // budget flaked once. This raises only this arm's budget through the
      // options object `queryWindowsProcessCreatedAt` already exposes to
      // callers -- production's own default timeout is untouched -- and
      // asserts the exact same identity match as before.
      const createdAt = await queryWindowsProcessCreatedAt(process.pid, { timeoutMs: 12_000 })

      assert.ok(createdAt, 'the current process must be queryable on native Windows')
      assert.ok(Math.abs(createdAt - expected) <= 2, 'the OS identity must match this exact test process generation')
    },
    15_000
  )

  it('returns exact integer epoch seconds from the bounded hidden query', async () => {
    const calls: Array<{ args: string[]; timeoutMs: number }> = []

    const result = await queryWindowsProcessCreatedAt(42, {
      platform: 'win32',
      run: async (_command, args, timeoutMs) => {
        calls.push({ args, timeoutMs })

        return '1723330000\r\n'
      }
    })

    assert.equal(result, 1_723_330_000)
    assert.equal(calls.length, 1)
    assert.match(calls[0].args.at(-1) ?? '', /Get-Process -Id 42/)
    assert.ok(calls[0].timeoutMs > 0)
  })

  it('fails closed on unsupported platforms, errors, and malformed output', async () => {
    assert.equal(await queryWindowsProcessCreatedAt(42, { platform: 'linux' }), null)
    assert.equal(
      await queryWindowsProcessCreatedAt(42, { platform: 'win32', run: async () => '1723330000suffix' }),
      null
    )
    assert.equal(
      await queryWindowsProcessCreatedAt(42, {
        platform: 'win32',
        run: async () => {
          throw new Error('access denied')
        }
      }),
      null
    )
  })
})

describe('createCachedWindowsProcessCreateTimeProbe', () => {
  it('returns unknown while querying, then exposes a short-lived exact result', async () => {
    let now = 1_000
    let resolveQuery!: (value: number | null) => void
    let calls = 0

    const probe = createCachedWindowsProcessCreateTimeProbe({
      cacheMs: 250,
      now: () => now,
      query: async () => {
        calls += 1

        return new Promise(resolve => {
          resolveQuery = resolve
        })
      }
    })

    assert.equal(probe(42), null)
    assert.equal(probe(42), null)
    assert.equal(calls, 1)
    resolveQuery(1_723_330_000)
    await new Promise(resolve => setImmediate(resolve))
    assert.equal(probe(42), 1_723_330_000)

    now += 251
    assert.equal(probe(42), null)
    assert.equal(calls, 2)
  })

  it('retains a resolved identity across the production one-second marker poll', async () => {
    let now = 1_000

    const probe = createCachedWindowsProcessCreateTimeProbe({
      now: () => now,
      query: async () => 1_723_330_000
    })

    assert.equal(probe(42), null)
    await new Promise(resolve => setImmediate(resolve))
    now += 1_000
    assert.equal(probe(42), 1_723_330_000)
  })

  // The cache used to be an unbounded Map: every PID a marker poll, a scan, or
  // a holder list ever named stayed resident for the life of the app.
  it('never retains more identities than the cap, evicting the oldest first', async () => {
    const calls = new Map<number, number>()

    const probe = createCachedWindowsProcessCreateTimeProbe({
      cacheMs: 1_000_000,
      now: () => 1_000,
      query: async pid => {
        calls.set(pid, (calls.get(pid) ?? 0) + 1)

        return 1_723_330_000 + pid
      }
    })

    // 300 > the 256 production cap, so the earliest PIDs must have been evicted
    // even though none of them expired.
    for (let pid = 1; pid <= 300; pid += 1) {
      assert.equal(probe(pid), null)
    }

    await new Promise(resolve => setImmediate(resolve))

    assert.equal(probe(300), 1_723_330_300, 'the newest identity stays cached')
    assert.equal(calls.get(300), 1, 'a retained identity is never re-queried')

    assert.equal(probe(1), null, 'the oldest identity was evicted, so it reads unknown again')
    assert.equal(calls.get(1), 2, 'an evicted identity is re-queried from the OS')
  })
})

describe('pruneProcessIdentityCache', () => {
  const settled = (validUntil: number): ProcessIdentityCacheEntry => ({
    pending: false,
    validUntil,
    value: 1_723_330_000
  })

  it('drops settled entries whose validity has passed', () => {
    const entries = new Map<number, ProcessIdentityCacheEntry>([
      [1, settled(500)],
      [2, settled(1_000)],
      [3, settled(1_500)]
    ])

    pruneProcessIdentityCache(entries, 1_000, 64)

    assert.deepEqual([...entries.keys()], [2, 3], 'only the entry past its validUntil is dropped')
  })

  it('keeps an in-flight query so a second query for the same PID never starts', () => {
    const entries = new Map<number, ProcessIdentityCacheEntry>([
      [1, { pending: true, validUntil: 0, value: null }],
      [2, settled(0)]
    ])

    pruneProcessIdentityCache(entries, 10_000, 64)

    assert.deepEqual([...entries.keys()], [1])
  })

  it('trims to the cap, oldest write first, even when nothing has expired', () => {
    const entries = new Map<number, ProcessIdentityCacheEntry>()

    for (let pid = 1; pid <= 10; pid += 1) {
      entries.set(pid, settled(10_000))
      pruneProcessIdentityCache(entries, 1_000, 4)
      assert.ok(entries.size <= 4, `size stayed within the cap after inserting ${pid}`)
    }

    assert.deepEqual([...entries.keys()], [7, 8, 9, 10])
  })
})


// ---------------------------------------------------------------------------
// POSIX creation time (#B7)
//
// Off Windows the probe returned null unconditionally, so every live PID
// classified as `unknown`: a recycled PID kept the marker alive forever and
// the MCP bridge stayed disabled on macOS and Linux -- a platform the PR title
// does not even claim to touch. Both sources here are stdlib/OS only, because
// this runs while deciding whether the venv is safe to touch.
// ---------------------------------------------------------------------------

describe('readLinuxProcessCreatedAt', () => {
  // Real shape: comm is parenthesised and may itself contain spaces and ')'.
  // Fields 1..9, twelve filler fields (10..21), then field 22 = starttime.
  const stat = (starttime: number, comm = '(my )proc)') =>
    `1234 ${comm} S 1 1234 1234 0 -1 4194304 ` +
    Array.from({ length: 12 }, () => '0').join(' ') +
    ` ${starttime} 12345678 900 ...`

  const proc = (files: Record<string, string>) => (file: string) => {
    if (!(file in files)) {
      throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' })
    }

    return files[file]
  }

  it('combines /proc/<pid>/stat field 22 with /proc/stat btime', () => {
    const createdAt = readLinuxProcessCreatedAt(1234, proc({
      '/proc/1234/stat': stat(250),          // 250 ticks at 100 USER_HZ = 2.5s
      '/proc/stat': 'cpu 1 2 3\nbtime 1723330000\nprocesses 9\n'
    }))

    assert.equal(createdAt, 1_723_330_002.5)
  })

  it('parses past a comm containing spaces and close parens', () => {
    assert.equal(
      readLinuxProcessCreatedAt(1234, proc({
        '/proc/1234/stat': stat(100, '(weird ) name)'),
        '/proc/stat': 'btime 1723330000\n'
      })),
      1_723_330_001
    )
  })

  it('fails closed on a dead pid, a missing btime and a bad field', () => {
    assert.equal(readLinuxProcessCreatedAt(1234, proc({})), null)
    assert.equal(
      readLinuxProcessCreatedAt(1234, proc({ '/proc/1234/stat': stat(250), '/proc/stat': 'cpu 1\n' })),
      null
    )
    assert.equal(
      readLinuxProcessCreatedAt(1234, proc({ '/proc/1234/stat': '1234 (p) S', '/proc/stat': 'btime 1\n' })),
      null
    )
    assert.equal(readLinuxProcessCreatedAt(0, proc({})), null)
  })
})

describe('queryDarwinProcessCreatedAt', () => {
  it('parses the ctime form ps prints', async () => {
    const expected = Math.floor(new Date(2026, 8, 6, 12, 34, 56).getTime() / 1000)

    assert.equal(
      await queryDarwinProcessCreatedAt(1234, { run: async () => 'Sun Sep  6 12:34:56 2026\n' }),
      expected
    )
  })

  it('fails closed on empty and unparseable output', async () => {
    assert.equal(await queryDarwinProcessCreatedAt(1234, { run: async () => '\n' }), null)
    assert.equal(await queryDarwinProcessCreatedAt(1234, { run: async () => 'not a date' }), null)
    assert.equal(
      await queryDarwinProcessCreatedAt(1234, { run: async () => { throw new Error('no such process') } }),
      null
    )
  })

  it('asks ps for exactly one pid', async () => {
    const calls: string[][] = []
    await queryDarwinProcessCreatedAt(1234, {
      run: async (_command, args) => {
        calls.push(args)

        return 'Sun Sep  6 12:34:56 2026'
      }
    })
    assert.deepEqual(calls, [['-o', 'lstart=', '-p', '1234']])
  })
})

describe('queryProcessCreatedAt', () => {
  it('proves this process on whichever platform the suite runs on', async () => {
    const expected = Math.floor(Date.now() / 1_000 - process.uptime())
    const createdAt = await queryProcessCreatedAt(process.pid)

    assert.ok(createdAt, `${process.platform} must prove its own creation time`)
    assert.ok(
      Math.abs(Number(createdAt) - expected) <= 2,
      'the OS identity must match this exact test process generation'
    )
  })

  it('fails closed rather than guessing on an unsupported platform', async () => {
    assert.equal(await queryProcessCreatedAt(1234, { platform: 'aix' as NodeJS.Platform }), null)
  })
})
