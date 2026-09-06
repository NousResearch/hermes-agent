import assert from 'node:assert/strict'

import { describe, it } from 'vitest'

import {
  createCachedWindowsProcessCreateTimeProbe,
  queryWindowsProcessCreatedAt
} from './windows-process-identity'

describe('queryWindowsProcessCreatedAt', () => {
  it.runIf(process.platform === 'win32')('proves the real Electron test process creation identity', async () => {
    const expected = Math.floor(Date.now() / 1_000 - process.uptime())
    const createdAt = await queryWindowsProcessCreatedAt(process.pid)

    assert.ok(createdAt, 'the current process must be queryable on native Windows')
    assert.ok(Math.abs(createdAt - expected) <= 2, 'the OS identity must match this exact test process generation')
  })

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
})
