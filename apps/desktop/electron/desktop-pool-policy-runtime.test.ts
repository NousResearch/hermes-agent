import assert from 'node:assert/strict'

import { afterEach, test, vi } from 'vitest'

import { createDesktopPoolPolicyRuntime } from './desktop-pool-policy-runtime'

afterEach(() => vi.useRealTimers())

test('pool policy shares foreground intent, live limits and the shutdown-owned idle timer', async () => {
  vi.useFakeTimers()

  const logs: string[] = []
  const writes: string[] = []
  const retired: string[] = []
  const promoted: string[] = []

  const entry = {
    process: null,
    lastActiveAt: 0,
    activeTurn: false,
    spawnPriority: 'background',
    localBackendSpawnRequest: { promote: (priority: string) => promoted.push(priority) }
  }

  const backendPool = new Map([['named', entry]])
  let poolIdleReaper: ReturnType<typeof setInterval> | null = null

  const policy = createDesktopPoolPolicyRuntime({
    app: { getPath: () => 'C:/temp/pool-policy-test' },
    backendPool,
    fs: {
      readFileSync: () => {
        throw new Error('not saved')
      },
      mkdirSync: () => undefined,
      writeFileSync: (_path: string, body: string) => writes.push(body),
      renameSync: () => undefined
    },
    getPoolRetirer: () => ({
      assertCanOpen: () => undefined,
      evictTo: async (keep: number) => {
        retired.push(`cap:${keep}`)

        return []
      },
      retireIdle: async (key: string) => {
        retired.push(`idle:${key}`)

        return false
      }
    }),
    getPoolIdleReaper: () => poolIdleReaper,
    setPoolIdleReaper: value => {
      poolIdleReaper = value
    },
    rememberLog: message => logs.push(String(message)),
    stopPoolBackend: async key => {
      retired.push(key)
      backendPool.delete(key)
    }
  } as any)

  assert.equal(policy.getPoolLimits().maxBackends > 0, true)
  assert.equal(policy.spawnPriorityFrom('foreground'), 'foreground')
  const clear = policy.applySpawnPriority('future', 'foreground')

  assert.equal(policy.takeForegroundSpawn('future'), true)
  assert.equal(policy.takeForegroundSpawn('future'), false)
  clear()

  policy.applySpawnPriority('named', 'foreground')()
  assert.equal(entry.spawnPriority, 'foreground')
  assert.deepEqual(promoted, ['foreground'])

  policy.touchPoolBackend('named', { activeTurn: true })
  assert.equal(entry.activeTurn, true)
  assert.equal(entry.lastActiveAt, Date.now())

  const applied = policy.setPoolLimits({ maxBackends: 2, idleMs: 60_000 })

  assert.equal(applied.maxBackends, 2)
  assert.equal(policy.localBackendSpawnCoordinator.limit, 2)
  assert.ok(poolIdleReaper)
  const originalTimer = poolIdleReaper

  policy.startPoolIdleReaper()
  assert.equal(poolIdleReaper, originalTimer)
  assert.equal(writes.length, 1)
  await Promise.resolve()
  assert.deepEqual(retired, ['cap:2'])

  await vi.advanceTimersByTimeAsync(120_000)
  assert.deepEqual(retired, ['cap:2', 'named'])
  assert.equal(backendPool.size, 0)
  assert.equal(poolIdleReaper, null)
  assert.equal(vi.getTimerCount(), 0)

  backendPool.set('local', { ...entry, process: {}, lastActiveAt: 0 } as any)
  policy.startPoolIdleReaper()
  await vi.advanceTimersByTimeAsync(60_000)
  assert.ok(retired.includes('idle:local'))
  assert.ok(poolIdleReaper)

  backendPool.clear()
  await vi.advanceTimersByTimeAsync(60_000)
  assert.equal(poolIdleReaper, null)
  assert.equal(vi.getTimerCount(), 0)
  assert.ok(logs.some(line => line.includes('no saved file')))
})
