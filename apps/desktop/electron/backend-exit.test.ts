import assert from 'node:assert/strict'
import { type ChildProcess, spawn } from 'node:child_process'
import { once } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { waitForBackendExit } from './backend-child'

async function makeChild(): Promise<ChildProcess> {
  const child = spawn(process.execPath, ['-e', 'process.stdout.write("ready"); setInterval(() => {}, 1000)'], {
    stdio: ['ignore', 'pipe', 'ignore'],
    windowsHide: true
  })

  await once(child.stdout!, 'data')

  return child
}

test('backend exit waits through escalation and removes its listener', async () => {
  const child = await makeChild()
  let escalated = 0
  const before = child.listenerCount('exit')

  try {
    await waitForBackendExit(
      child,
      process => {
        escalated++
        process.kill()
      },
      0
    )
    assert.equal(escalated, 1)
    assert.ok(child.exitCode !== null || child.signalCode !== null)
    assert.equal(child.listenerCount('exit'), before)
    await waitForBackendExit(child, () => {
      throw new Error('an exited process must not be killed again')
    })
  } finally {
    if (child.exitCode === null && child.signalCode === null) {
      const exited = once(child, 'close')
      child.kill()
      await exited
    }
  }
}, 15_000)

test('backend exit refuses a live child rather than reporting a completed shutdown', async () => {
  const child = await makeChild()
  const before = child.listenerCount('exit')

  try {
    await assert.rejects(
      waitForBackendExit(child, () => {}, 0),
      /did not exit/
    )
    assert.equal(child.exitCode, null)
    assert.equal(child.signalCode, null)
    assert.equal(child.listenerCount('exit'), before)
  } finally {
    const exited = once(child, 'close')
    child.kill()
    await exited
  }
}, 15_000)

test('a failed spawn has no process to escalate', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'backend-no-process-'))
  const child = spawn(path.join(root, 'absent-executable'), [], { stdio: 'ignore' })

  try {
    await once(child, 'error')
    assert.equal(child.pid, undefined)
    await waitForBackendExit(
      child,
      () => {
        throw new Error('a failed spawn must not be signalled')
      },
      0
    )
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
}, 15_000)
