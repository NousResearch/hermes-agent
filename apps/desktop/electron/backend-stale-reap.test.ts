import assert from 'node:assert/strict'

import { test } from 'vitest'

import type { ReapDeps } from './backend-stale-reap'
import {
  findStaleBackendPids,
  reapStaleBackendPids,
  reapStaleBackendsForProfile,
} from './backend-stale-reap'

const noopSleep = (_ms: number) => Promise.resolve()

interface FakeDeps extends ReapDeps {
  __killed: Array<{ pid: number; signal?: string }>
  __treeKilled: number[]
  __alive: Set<number>
}

function fakeDeps(overrides: Partial<ReapDeps> = {}): FakeDeps {
  const killed: Array<{ pid: number; signal?: string }> = []
  const treeKilled: number[] = []
  const alive = new Set<number>()
  const deps: FakeDeps = {
    isWindows: false,
    listCandidatePids: () => [],
    forceKillProcessTree: (pid: number) => {
      treeKilled.push(pid)
    },
    terminatePid: ((pid: number, signal?: string) => {
      killed.push({ pid, signal: signal as string | undefined })
      alive.delete(pid)
    }) as ReapDeps['terminatePid'],
    sleep: noopSleep,
    terminationWaitMs: 0,
    pidAlive: ((pid: number) => alive.has(pid)) as ReapDeps['pidAlive'],
    __killed: killed,
    __treeKilled: treeKilled,
    __alive: alive,
    ...overrides,
  }
  return deps
}

test('findStaleBackendPids excludes non-positive, non-integer, and self pid', () => {
  const deps = fakeDeps({
    listCandidatePids: () => [0, -1, 1.5, 99999, process.pid, 1001, 1001 /* dup */],
  })
  const out = findStaleBackendPids('worker', deps)
  assert.deepEqual(out, [99999, 1001])
})

test('findStaleBackendPids returns empty when profile is empty', () => {
  const deps = fakeDeps({ listCandidatePids: () => [10, 20] })
  assert.deepEqual(findStaleBackendPids('', deps), [])
})

test('reapStaleBackendPids is a no-op on empty input', async () => {
  const deps = fakeDeps()
  const survivors = await reapStaleBackendPids([], deps)
  assert.deepEqual(survivors, [])
  assert.deepEqual(deps.__treeKilled, [])
})

test('reapStaleBackendPids calls forceKillProcessTree on Windows, terminatePid on POSIX', async () => {
  const winDeps = fakeDeps({ isWindows: true })
  await reapStaleBackendPids([101, 102], winDeps)
  assert.deepEqual(winDeps.__treeKilled, [101, 102])
  assert.deepEqual(winDeps.__killed, [])

  const posixDeps = fakeDeps({ isWindows: false })
  await reapStaleBackendPids([201, 202], posixDeps)
  assert.deepEqual(posixDeps.__treeKilled, [])
  // production terminatePid signature is `(pid: number) => void` — no signal arg;
  // the POSIX kill mechanism picks `SIGTERM` itself in the production path.
  // fakeDeps records the pid argument verbatim.
  assert.deepEqual(
    posixDeps.__killed.map((k: { pid: number }) => k.pid),
    [201, 202],
  )
})

test('reapStaleBackendPids returns survivors that did not exit within the bounded wait', async () => {
  // 999 simulates a process that ignores our SIGTERM entirely (a
  // respawning-supervised back-end from #67026's repro). 1000 is a normal
  // holder that exits cleanly. Build deps inline so pidAlive sees the
  // post-kill truth (fakeDeps closes over its own internal alive set).
  const deps: ReapDeps = {
    isWindows: false,
    listCandidatePids: () => [999, 1000],
    forceKillProcessTree: () => undefined,
    terminatePid: () => {
      // Terminate is fire-and-forget: real outcome is observed via pidAlive.
    },
    sleep: noopSleep,
    terminationWaitMs: 0,
    pidAlive: (pid: number) => pid === 999,
  }
  const survivors = await reapStaleBackendPids([999, 1000], deps)
  assert.deepEqual(survivors, [999])
})

test('reapStaleBackendPids silently skips non-integer pid entries (defensive)', async () => {
  const deps = fakeDeps({ isWindows: false })
  await reapStaleBackendPids([0, -1, Number.NaN as unknown as number, 42], deps)
  // Only 42 is a valid PID and should be acted on
  assert.deepEqual(deps.__killed.map((k: { pid: number }) => k.pid), [42])
})

test('reapStaleBackendPids continues past a throwing kill (no aborts cascade)', async () => {
  let killCount = 0
  const deps = {
    isWindows: false,
    listCandidatePids: () => [1, 2, 3],
    forceKillProcessTree: () => undefined,
    terminatePid: ((pid: number) => {
      killCount++
      if (pid === 2) {
        throw new Error('boom')
      }
    }) as ReapDeps['terminatePid'],
    sleep: noopSleep,
    terminationWaitMs: 0,
    pidAlive: () => false,
  } as unknown as ReapDeps
  await reapStaleBackendPids([1, 2, 3], deps)
  assert.equal(killCount, 3)
})

test('reapStaleBackendsForProfile combines find + reap', async () => {
  const killed: number[] = []
  const deps = {
    isWindows: false,
    listCandidatePids: (profile: string) => (profile === 'worker' ? [501] : []),
    forceKillProcessTree: () => undefined,
    terminatePid: ((pid: number) => {
      killed.push(pid)
    }) as ReapDeps['terminatePid'],
    sleep: noopSleep,
    terminationWaitMs: 0,
    pidAlive: () => false,
  } as unknown as ReapDeps
  const survivors = await reapStaleBackendsForProfile('worker', deps)
  assert.deepEqual(survivors, [])
  assert.deepEqual(killed, [501])

  const otherProfile = await reapStaleBackendsForProfile('default', deps)
  assert.deepEqual(otherProfile, [])
  assert.deepEqual(killed, [501])
})

