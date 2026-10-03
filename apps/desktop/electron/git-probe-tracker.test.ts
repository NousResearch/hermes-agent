import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { test } from 'vitest'

import { createGitProbeTracker, type TrackedGitChild } from './git-probe-tracker'

function makeChild(pid: number): TrackedGitChild & { emit: (event: string, arg?: unknown) => void } {
  const emitter = new EventEmitter()

  return {
    emit: (event, arg) => {
      emitter.emit(event, arg)
    },
    kill: () => {},
    killed: false,
    once: (event, listener) => emitter.once(event, listener),
    pid
  }
}

test('killAll group-kills a still-running tracked probe on POSIX (whole process group, not just the top PID)', () => {
  const child = makeChild(4242)
  const groupKills: Array<[number, string]> = []

  const tracker = createGitProbeTracker({
    forceKillProcessTree: () => {
      throw new Error('must not be used off Windows')
    },
    isWindows: false,
    killGroup: (pgid, signal) => groupKills.push([pgid, signal])
  })

  tracker.track(child)
  tracker.killAll()

  assert.deepEqual(groupKills, [[-4242, 'SIGTERM']], 'must signal the negative pgid so git-spawned descendants die too')
})

test('killAll tree-kills by PID ancestry on Windows instead of group-signaling', () => {
  const child = makeChild(4242)
  const treeKills: number[] = []

  const tracker = createGitProbeTracker({
    forceKillProcessTree: pid => treeKills.push(pid),
    isWindows: true
  })

  tracker.track(child)
  tracker.killAll()

  assert.deepEqual(treeKills, [4242])
})

test('a probe that already closed on its own is not killed again at quit', () => {
  const child = makeChild(4242)
  const groupKills: Array<[number, string]> = []

  const tracker = createGitProbeTracker({
    forceKillProcessTree: () => {},
    isWindows: false,
    killGroup: (pgid, signal) => groupKills.push([pgid, signal])
  })

  tracker.track(child)
  child.emit('close', 0)
  assert.equal(tracker.size(), 0)

  tracker.killAll()

  assert.deepEqual(groupKills, [], 'a probe that finished on its own must not be signaled at quit')
})

test('a probe whose spawn errored stops being tracked', () => {
  const child = makeChild(4242)
  const tracker = createGitProbeTracker({ forceKillProcessTree: () => {}, isWindows: false })

  tracker.track(child)
  child.emit('error', new Error('spawn failed'))

  assert.equal(tracker.size(), 0)
})

test('killAll is a no-op with nothing tracked', () => {
  const groupKills: Array<[number, string]> = []

  const tracker = createGitProbeTracker({
    forceKillProcessTree: () => {},
    isWindows: false,
    killGroup: (pgid, signal) => groupKills.push([pgid, signal])
  })

  tracker.killAll()

  assert.equal(tracker.size(), 0)
  assert.deepEqual(groupKills, [])
})
