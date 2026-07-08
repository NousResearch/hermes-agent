import assert from 'node:assert/strict'
import test from 'node:test'

import { stopBackendChild } from './backend-child'
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

  return {
    calls,
    child: {
      kill: (signal: string) => {
        calls.push(signal)
      },
      killed: overrides.killed ?? false,
      pid: 'pid' in overrides ? overrides.pid : 1234
    }
  }
}

test('stopBackendChild tree-kills on Windows when the child has a pid', () => {
  const { child, calls } = makeChild({ pid: 4242 })
  const treeKillCalls: number[] = []

  stopBackendChild(child, {
    forceKillProcessTree: (pid: number) => treeKillCalls.push(pid),
    isWindows: true
  })

  assert.deepEqual(treeKillCalls, [4242])
  assert.deepEqual(calls, [], 'SIGTERM must not be sent when the Windows tree-kill path is taken')
})

test('stopBackendChild sends SIGTERM on non-Windows platforms', () => {
  const { child, calls } = makeChild({ pid: 4242 })
  const treeKillCalls: number[] = []

  stopBackendChild(child, {
    forceKillProcessTree: (pid: number) => treeKillCalls.push(pid),
    isWindows: false
  })

  assert.deepEqual(calls, ['SIGTERM'])
  assert.deepEqual(treeKillCalls, [], 'tree-kill must not run off Windows')
})

test('stopBackendChild falls back to SIGTERM on Windows when the pid is not an integer', () => {
  const { child, calls } = makeChild({ pid: null })
  const treeKillCalls: number[] = []

  stopBackendChild(child, {
    forceKillProcessTree: (pid: number) => treeKillCalls.push(pid),
    isWindows: true
  })

  assert.deepEqual(calls, ['SIGTERM'])
  assert.deepEqual(treeKillCalls, [])
})

test('stopBackendChild is a no-op for an already-killed child', () => {
  const { child, calls } = makeChild({ killed: true })
  const treeKillCalls: number[] = []

  stopBackendChild(child, {
    forceKillProcessTree: (pid: number) => treeKillCalls.push(pid),
    isWindows: true
  })

  assert.deepEqual(calls, [])
  assert.deepEqual(treeKillCalls, [])
})

test('stopBackendChild is a no-op for a null/undefined child', () => {
  const treeKillCalls: number[] = []

  assert.doesNotThrow(() => {
    stopBackendChild(null, { forceKillProcessTree: (pid: number) => treeKillCalls.push(pid), isWindows: true })
    stopBackendChild(undefined, { forceKillProcessTree: (pid: number) => treeKillCalls.push(pid), isWindows: true })
  })
  assert.deepEqual(treeKillCalls, [])
})

test('stopBackendChild swallows errors thrown by the kill strategy', () => {
  const child = {
    kill: () => {
      throw new Error('ESRCH: no such process')
    },
    killed: false,
    pid: 99
  }

  assert.doesNotThrow(() => {
    stopBackendChild(child, {
      forceKillProcessTree: () => {},
      isWindows: false
    })
  })
})
