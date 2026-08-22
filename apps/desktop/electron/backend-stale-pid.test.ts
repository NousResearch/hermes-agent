import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { test } from 'vitest'

import {
  assertNoPendingForcedBackendStops,
  BackendStopError,
  forceStopBackendChild,
  isLiveProcessRoot,
  requestBackendForceStop,
  requestBackendGracefulStop,
  STOP_ALREADY_EXITED,
  STOP_NO_AUTHORITY,
  STOP_PERMISSION_DENIED,
  STOP_REQUESTED,
  STOP_TIMED_OUT,
  stopBackendChild,
  stopBackendTreesForUpdate
} from './backend-child'

const live = (pid: number, kill: (signal: NodeJS.Signals) => unknown = () => true) => ({
  exitCode: null,
  kill,
  killed: false,
  pid,
  signalCode: null
})

const exited = (pid: number, kill: (signal: NodeJS.Signals) => unknown = () => true, code = 0) => ({
  exitCode: code,
  kill,
  killed: false,
  pid,
  signalCode: null
})

const signalled = (
  pid: number,
  kill: (signal: NodeJS.Signals) => unknown = () => true,
  signalCode = 'SIGTERM'
) => ({
  exitCode: null,
  kill,
  killed: true,
  pid,
  signalCode
})

test('only an explicit live retained owner is actionable', () => {
  assert.equal(isLiveProcessRoot(live(4242)), true)
  assert.equal(isLiveProcessRoot(exited(4242)), false)
  assert.equal(isLiveProcessRoot(signalled(4242)), false)
  assert.equal(isLiveProcessRoot({ pid: 4242 }), false)
  assert.equal(isLiveProcessRoot({ exitCode: null, pid: 0, signalCode: null }), false)
  assert.equal(isLiveProcessRoot({ exitCode: null, pid: -991, signalCode: null }), false)
})

test('a reaped owner with a populated PID is never signalled', () => {
  const signals: string[] = []
  const child = exited(4242, signal => signals.push(signal))

  assert.equal(stopBackendChild(child).kind, STOP_ALREADY_EXITED)
  assert.equal(requestBackendGracefulStop(child).kind, STOP_ALREADY_EXITED)
  assert.deepEqual(signals, [])
})

test('PID-only residue raises instead of silently discarding no authority', () => {
  const pidOnly = { exitCode: null, pid: 9999, signalCode: null }

  assert.throws(
    () => stopBackendChild(pidOnly as any),
    error => error instanceof BackendStopError && error.result.kind === STOP_NO_AUTHORITY
  )
  assert.equal(requestBackendGracefulStop(pidOnly as any).kind, STOP_NO_AUTHORITY)
})

test('signal submission is StopRequested, never a fabricated exit observation', () => {
  const signals: string[] = []
  const child = live(1234, signal => signals.push(signal))

  assert.equal(stopBackendChild(child).kind, STOP_REQUESTED)
  assert.equal(requestBackendGracefulStop(live(1235)).kind, STOP_REQUESTED)
  assert.equal(child.exitCode, null)
  assert.equal(child.signalCode, null)
  assert.deepEqual(signals, ['SIGTERM'])
})

test('child.killed records signal submission, not exit, so force escalation remains actionable', () => {
  const signals: string[] = []
  const child = { ...live(1236, signal => signals.push(signal)), killed: true }

  assert.equal(requestBackendForceStop(child, 'linux').kind, STOP_REQUESTED)
  assert.deepEqual(signals, ['SIGUSR2'])
})

test('a missing retained child is an already-complete no-op', () => {
  assert.equal(stopBackendChild(null).kind, STOP_ALREADY_EXITED)
})

test('forced stop uses the retained platform authority command', () => {
  const windowsSignals: string[] = []
  const posixSignals: string[] = []

  assert.equal(
    forceStopBackendChild(live(1, signal => windowsSignals.push(signal)), 'win32').kind,
    STOP_REQUESTED
  )
  assert.equal(
    forceStopBackendChild(live(2, signal => posixSignals.push(signal)), 'linux').kind,
    STOP_REQUESTED
  )
  assert.equal(requestBackendForceStop(live(3), 'darwin').kind, STOP_REQUESTED)
  assert.deepEqual(windowsSignals, ['SIGKILL'])
  assert.deepEqual(posixSignals, ['SIGUSR2'])
})

test('forced stop blocks replacement generations until exact owner exit is observed', () => {
  const events = new EventEmitter()
  const child = {
    ...live(4040),
    once: events.once.bind(events),
    removeListener: events.removeListener.bind(events)
  }

  assert.equal(forceStopBackendChild(child, 'linux').kind, STOP_REQUESTED)
  assert.throws(
    () => assertNoPendingForcedBackendStops(),
    error => error instanceof BackendStopError && error.result.kind === STOP_TIMED_OUT
  )

  events.emit('exit', 0, null)
  assert.doesNotThrow(() => assertNoPendingForcedBackendStops())
})

test('child.kill failures remain typed and compatibility calls fail loudly', () => {
  const child = live(9, () => {
    throw new Error('EPERM')
  })

  assert.throws(
    () => stopBackendChild(child),
    error => error instanceof BackendStopError && error.result.kind === STOP_PERMISSION_DENIED
  )
  assert.equal(requestBackendGracefulStop(child).kind, STOP_PERMISSION_DENIED)
  assert.match(requestBackendGracefulStop(child).detail || '', /EPERM/)
})

test('update teardown submits the retained primary stop before stopping the pool', async () => {
  const calls: string[] = []
  const primaryChild = live(2002, signal => calls.push(`primary:${signal}`))

  const result = await stopBackendTreesForUpdate(primaryChild, {
    stopAllPoolBackends: () => {
      calls.push('pool-stopped')
    }
  })

  assert.equal(result.kind, STOP_REQUESTED)
  assert.deepEqual(calls, ['primary:SIGTERM', 'pool-stopped'])
})
