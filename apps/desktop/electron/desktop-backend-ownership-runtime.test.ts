import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { createDesktopBackendOwnershipRuntime } from './desktop-backend-ownership-runtime'

function fixture() {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-ownership-'))
  const ownershipPath = path.join(home, 'backend-ownership.json')
  const childPid = 4242
  const events: string[] = []
  let parentMarker = 'parent-start'
  let childMarker = 'child-start'
  let commandLine = 'hermes serve --port 0'

  const processStartMarker = async (pid: number) => (pid === process.pid ? parentMarker : childMarker)
  const runtime = createDesktopBackendOwnershipRuntime({
    ownershipPath,
    isWindows: true,
    execText: async (_command: string, args: string[]) => {
      assert.ok(args.some(arg => arg.includes(String(childPid))))

      return commandLine
    },
    processStartMarker,
    probeStartMarker: async (pid: number) => ({ ok: true as const, startMarker: await processStartMarker(pid) }),
    forceKillProcessTree: (pid: number) => {
      events.push(`stop:${pid}`)
      childMarker = 'gone'
    },
    stopBackendChild: () => events.push('stop-child'),
    waitForBackendExit: async () => events.push('wait-child'),
    rememberLog: (message: string) => events.push(message)
  })

  return {
    home,
    ownershipPath,
    childPid,
    events,
    runtime,
    setParentMarker: (value: string) => {
      parentMarker = value
    },
    setCommandLine: (value: string) => {
      commandLine = value
    },
    readEntries: () => JSON.parse(fs.readFileSync(ownershipPath, 'utf8')).backends
  }
}

test('claim persists the backend and parent identity; release removes only that claim', async () => {
  const f = fixture()

  try {
    const child: any = { pid: f.childPid, exitCode: null, killed: false }
    const identity = await f.runtime.claimBackendChild(child, 'hermes serve --port 0', 'default', 'nonce-1')

    assert.deepEqual(f.readEntries(), [
      {
        nonce: 'nonce-1',
        pid: f.childPid,
        profile: 'default',
        startMarker: 'child-start',
        command: 'hermes serve --port 0',
        parentPid: process.pid,
        parentStartMarker: 'parent-start'
      }
    ])
    assert.equal(child.hermesBackendIdentity, identity)
    f.runtime.releaseBackendChild(child)
    assert.deepEqual(f.readEntries(), [])
    assert.deepEqual(f.events, [])
  } finally {
    fs.rmSync(f.home, { recursive: true, force: true })
  }
})

test('orphan sweep preserves a backend whose recorded parent is still alive', async () => {
  const f = fixture()

  try {
    await f.runtime.claimBackendChild(
      { pid: f.childPid, exitCode: null, killed: false },
      'hermes serve',
      'default',
      'nonce-2'
    )
    await f.runtime.reapOrphanedBackendsOnce()

    assert.equal(f.readEntries().length, 1)
    assert.equal(
      f.events.some(event => event.startsWith('stop:')),
      false
    )
  } finally {
    fs.rmSync(f.home, { recursive: true, force: true })
  }
})

test('orphan sweep stops only the matching backend after parent PID reuse', async () => {
  const f = fixture()

  try {
    await f.runtime.claimBackendChild(
      { pid: f.childPid, exitCode: null, killed: false },
      'hermes serve',
      'default',
      'nonce-3'
    )
    f.setParentMarker('reused-parent-pid')
    await f.runtime.reapOrphanedBackendsOnce()

    assert.deepEqual(f.readEntries(), [])
    assert.ok(f.events.includes(`stop:${f.childPid}`))
    assert.ok(f.events.some(event => event.includes(`Reaped orphaned desktop backend PID(s): ${f.childPid}`)))
  } finally {
    fs.rmSync(f.home, { recursive: true, force: true })
  }
})

test('orphan sweep never stops a PID whose command no longer belongs to Hermes', async () => {
  const f = fixture()

  try {
    await f.runtime.claimBackendChild(
      { pid: f.childPid, exitCode: null, killed: false },
      'hermes serve',
      'default',
      'nonce-4'
    )
    f.setParentMarker('reused-parent-pid')
    f.setCommandLine('unrelated-tool --serve')
    await f.runtime.reapOrphanedBackendsOnce()

    assert.deepEqual(f.readEntries(), [])
    assert.equal(
      f.events.some(event => event.startsWith('stop:')),
      false
    )
  } finally {
    fs.rmSync(f.home, { recursive: true, force: true })
  }
})
