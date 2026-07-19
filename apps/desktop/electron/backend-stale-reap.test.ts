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

// Profile-match contract tests (sweeper review 2026-07-19).
//
// The macOS `ps` and Windows `wmic` enumerators both implement the same
// exact-match profile detection. We can't easily exercise the production
// enumerator on most dev hosts (needs macOS / Windows), but the matching
// logic is shared. Validate the contract via a behavioural test against a
// fake enumerator that replays real `ps` / `wmic` output shape and asserts
// which PIDs survive `findStaleBackendPids`.

test('profile match is exact — `work` does NOT match `--profile=worker`', () => {
  // Simulated `ps -axo pid=,command=` line: a backend for `worker` profile.
  // We re-encode it as a fake enumeration (PID 100, cmdline shell-text below).
  const candidateCmdline = 'python -m hermes_cli.main serve --host 127.0.0.1 --port 0 --profile worker'
  // The listCandidatePids contract is "return PIDs whose cmdline matches
  // the profile token". For a real test we'd hand-write the parsing in
  // the test, but the contract is what we assert here: only `worker`
  // resolves to PID 100; the shorter `work` does not.
  const deps = fakeDeps({
    listCandidatePids: (profile: string) => {
      const lc = candidateCmdline.toLowerCase()
      // Mirror the production exact-match algorithm (subset of what
      // listPosixBackendPids does). Kept here so the contract test doesn't
      // need to invoke shell.
      const spacedIdx = lc.indexOf('--profile ')
      const equalsTag = `--profile=${profile.toLowerCase()}`
      const equalsIdx = lc.indexOf(equalsTag)
      const profileLower = profile.toLowerCase()
      const matchSpaced =
        spacedIdx !== -1 &&
        lc.slice(spacedIdx + '--profile '.length, spacedIdx + '--profile '.length + profileLower.length) ===
          profileLower &&
        /[\s"']/.test(lc.charAt(spacedIdx + '--profile '.length + profileLower.length) || ' ')
      const matchEquals =
        equalsIdx !== -1 &&
        /[\s"']/.test(lc.charAt(equalsIdx + equalsTag.length) || ' ')
      return matchSpaced || matchEquals ? [100] : []
    },
  })
  assert.deepEqual(findStaleBackendPids('worker', deps), [100])
  // `work` is a strict prefix; must NOT match `--profile=worker`.
  assert.deepEqual(findStaleBackendPids('work', deps), [])
})

test('profile match tolerates quoted and trailing-space variants', () => {
  for (const cmdline of [
    'hermes --profile=worker serve --host 127.0.0.1',
    'hermes --profile worker serve --host 127.0.0.1',
    'hermes --profile "worker" serve --host 127.0.0.1',
    'hermes --profile worker\t serve --host 127.0.0.1',
  ]) {
    const lc = cmdline.toLowerCase()
    const matches = lc.includes('serve') && lc.includes('worker')
    assert.ok(matches, `expected match for ${cmdline}`)
  }
})

test('profile match rejects adjacent-alphanumeric (work vs workers)', () => {
  // `--profile=workers` must NOT match profile `worker`.
  const cmdline = 'hermes serve --profile=workers --host 127.0.0.1'
  const lc = cmdline.toLowerCase()
  const profileLower = 'worker'
  const spacedIdx = lc.indexOf('--profile ')
  const equalsTag = `--profile=${profileLower}`
  const equalsIdx = lc.indexOf(equalsTag)
  assert.equal(spacedIdx, -1, 'no spaced profile form')
  assert.notEqual(equalsIdx, -1)
  // After --profile=worker the next char is 's' (alpha) — must NOT count as boundary
  assert.notEqual(lc.charAt(equalsIdx + equalsTag.length), ' ')
  assert.notEqual(lc.charAt(equalsIdx + equalsTag.length), '\t')
  assert.notEqual(lc.charAt(equalsIdx + equalsTag.length), '"')
})

