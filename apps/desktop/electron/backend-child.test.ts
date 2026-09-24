import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  describeAbruptBackendExit,
  describeRecentTreeKills,
  formatTreeKillLine,
  isAbruptWindowsKillExit,
  noteTreeKill,
  stopBackendChild,
  stopBackendTreesForUpdate,
} from './backend-child'
import { formatBackendExitLine } from './backend-claim'

// Issue #119440: Windows backend exits 4294967295 (-1) with empty stderr and
// the supervisor respawns blind — no log says WHO tree-killed the child
// (our own taskkill /T /F vs an external killer), so multi-instance races
// are unattributable.

function windowsChild(pid: number) {
  return { pid, killed: false, kill: () => {} }
}

test('stopBackendChild threads the kill reason to the tree-kill', () => {
  const kills: Array<{ pid: number; reason?: string }> = []
  stopBackendChild(windowsChild(1234), {
    forceKillProcessTree: (pid: number, reason?: string) => {
      kills.push({ pid, reason })
    },
    isWindows: true,
  }, { reason: 'pool-stop' })
  assert.deepEqual(kills, [{ pid: 1234, reason: 'pool-stop' }])
  // The supervisor logs this line before taskkill /T /F runs, so a later
  // -1/4294967295 exit is attributable to the killer.
  assert.equal(formatTreeKillLine(1234, 'pool-stop'), '[backend-child] tree-kill pid=1234 reason=pool-stop')
})

test('stopBackendTreesForUpdate attributes the primary kill to update-handoff', () => {
  const kills: Array<{ pid: number; reason?: string }> = []
  stopBackendTreesForUpdate({ pid: 555 }, {
    forceKillProcessTree: (pid: number, reason?: string) => {
      kills.push({ pid, reason })
    },
    stopAllPoolBackends: () => {},
  })
  assert.deepEqual(kills, [{ pid: 555, reason: 'update-handoff' }])
})

test('isAbruptWindowsKillExit flags only -1/4294967295 with no signal', () => {
  assert.equal(isAbruptWindowsKillExit(-1, null), true)
  assert.equal(isAbruptWindowsKillExit(4294967295, null), true)
  assert.equal(isAbruptWindowsKillExit(1, null), false)
  assert.equal(isAbruptWindowsKillExit(0, null), false)
  assert.equal(isAbruptWindowsKillExit(null, 'SIGTERM'), false)
  assert.equal(isAbruptWindowsKillExit(4294967295, 'SIGKILL'), false)
})

test('recent tree-kill description names our kill, or rules us out', () => {
  const now = 1_700_000_000_000
  noteTreeKill(7777, 'pool-stop', now - 2_000)
  const named = describeRecentTreeKills(now)
  assert.match(named, /pid=7777/)
  assert.match(named, /pool-stop/)
  // A kill older than the window must not implicate this process.
  const stale = describeRecentTreeKills(now + 60_000)
  assert.match(stale, /no tree-kill by this process/)
})

test('exit line carries the abrupt-death suffix only when present', () => {
  assert.equal(
    formatBackendExitLine('Hermes backend exited', 4294967295, null, null),
    'Hermes backend exited (4294967295)'
  )
  assert.equal(
    formatBackendExitLine('Hermes backend exited', 4294967295, null, null, ' [abrupt suffix]'),
    'Hermes backend exited (4294967295) [abrupt suffix]'
  )
})

test('abrupt-exit suffix stays silent on normal exits, annotates -1', () => {
  assert.equal(describeAbruptBackendExit({ code: 1, ownerText: 'x', recentText: 'y' }), '')
  assert.equal(describeAbruptBackendExit({ code: 0, ownerText: 'x', recentText: 'y' }), '')
  const now = 1_700_000_000_000
  noteTreeKill(8888, 'orphan-reap', now - 1_000)
  const suffix = describeAbruptBackendExit({
    code: 4294967295,
    ownerText: 'owner(parentPid=9999)',
    recentText: describeRecentTreeKills(now),
  })
  assert.match(suffix, /pid=8888/)
  assert.match(suffix, /owner\(parentPid=9999\)/)
})
