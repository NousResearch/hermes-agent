import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  BackendStopError,
  forceStopBackendChild,
  type KillableChild,
  requestBackendForceStop,
  requestBackendGracefulStop,
  STOP_ALREADY_EXITED,
  STOP_NO_AUTHORITY,
  STOP_REQUESTED,
  stopBackendChild,
  stopBackendTreesForUpdate
} from './backend-child'
import { hiddenWindowsChildOptions } from './windows-child-options'

test('hiddenWindowsChildOptions adds windowsHide:true on Windows when unset', () => {
  assert.deepEqual(hiddenWindowsChildOptions({}, true), { windowsHide: true })
})

test('hiddenWindowsChildOptions preserves an existing windowsHide:false on Windows', () => {
  assert.deepEqual(hiddenWindowsChildOptions({ windowsHide: false }, true), { windowsHide: false })
})

test('hiddenWindowsChildOptions preserves an existing windowsHide:true on Windows', () => {
  assert.deepEqual(hiddenWindowsChildOptions({ windowsHide: true }, true), { windowsHide: true })
})

test('hiddenWindowsChildOptions leaves options unchanged off Windows', () => {
  assert.deepEqual(hiddenWindowsChildOptions({}, false), {})
  assert.deepEqual(hiddenWindowsChildOptions({ stdio: 'ignore' }, false), { stdio: 'ignore' })
})

test('hiddenWindowsChildOptions merges windowsHide alongside other options on Windows', () => {
  assert.deepEqual(hiddenWindowsChildOptions({ encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }, true), {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
    windowsHide: true
  })
})

test('hiddenWindowsChildOptions defaults isWindows from process.platform when omitted', () => {
  const result = hiddenWindowsChildOptions({})
  const expectedHide = process.platform === 'win32'

  assert.equal(Boolean(result.windowsHide), expectedHide)
})

function makeChild(overrides: Partial<{ pid: number | null; killed: boolean }> = {}) {
  const calls: string[] = []
  const child: KillableChild = {
    exitCode: null,
    kill: signal => {
      calls.push(String(signal))
    },
    killed: overrides.killed ?? false,
    pid: 'pid' in overrides ? overrides.pid : 1234,
    signalCode: null
  }

  return { calls, child }
}

test('graceful stop reports accepted submission, not fabricated exit', () => {
  const { child, calls } = makeChild({ pid: 4242 })

  assert.equal(stopBackendChild(child).kind, STOP_REQUESTED)
  assert.equal(requestBackendGracefulStop(makeChild().child).kind, STOP_REQUESTED)
  assert.deepEqual(calls, ['SIGTERM'])
})

test('force stop selects the retained Windows Job or POSIX supervisor command', () => {
  const windows = makeChild({ pid: 100 })
  const posix = makeChild({ pid: 200 })

  assert.equal(forceStopBackendChild(windows.child, 'win32').kind, STOP_REQUESTED)
  assert.equal(forceStopBackendChild(posix.child, 'linux').kind, STOP_REQUESTED)
  assert.equal(requestBackendForceStop(makeChild().child, 'win32').kind, STOP_REQUESTED)
  assert.deepEqual(windows.calls, ['SIGKILL'])
  assert.deepEqual(posix.calls, ['SIGUSR2'])
})

test('PID-only records fail loudly while already-terminal children are truthful', () => {
  const pidOnly = { pid: 99 } as KillableChild
  assert.equal(requestBackendGracefulStop(pidOnly).kind, STOP_NO_AUTHORITY)
  assert.throws(
    () => stopBackendChild(pidOnly),
    error => error instanceof BackendStopError && error.result.kind === STOP_NO_AUTHORITY
  )

  const { child, calls } = makeChild({ killed: true })
  child.signalCode = 'SIGTERM'
  assert.equal(stopBackendChild(child).kind, STOP_ALREADY_EXITED)
  assert.equal(requestBackendGracefulStop(child).kind, STOP_ALREADY_EXITED)
  assert.deepEqual(calls, [])
})

test('update teardown preserves typed submission and has no bare-PID tree-kill path', async () => {
  const events: string[] = []
  const primary = makeChild({ pid: 101 })

  const result = await stopBackendTreesForUpdate(primary.child, {
    stopAllPoolBackends: () => {
      events.push('pool-stop')
    }
  })

  assert.equal(result.kind, STOP_REQUESTED)
  assert.deepEqual(events, ['pool-stop'])
  assert.deepEqual(primary.calls, ['SIGTERM'])
})
