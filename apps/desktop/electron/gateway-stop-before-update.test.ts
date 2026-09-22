import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  GATEWAY_STOP_TIMEOUT_MS,
  startGatewaysAfterUpdateAbort,
  stopGatewayBeforeUpdate
} from './gateway-stop-before-update'

const CLI = 'C:\\Users\\x\\hermes\\hermes-agent\\venv\\Scripts\\hermes.exe'
const HOME = 'C:\\Users\\x\\hermes'

function fakeExec(ok: boolean) {
  return (_command: string, _args: string[], _options: unknown, callback: (error: Error | null) => void) => {
    callback(ok ? null : new Error('spawn ENOENT'))
  }
}

test('non-Windows is a no-op and never invokes the CLI', async () => {
  const calls: Array<[string, string[]]> = []

  const ran = await stopGatewayBeforeUpdate(CLI, HOME, {
    isWindows: false,
    existsSync: () => true,
    execFile: fakeExec(true),
    spy: (c, a) => calls.push([c, a])
  })

  assert.equal(ran, false)
  assert.deepEqual(calls, [])
})

test('Windows with missing CLI shim returns false and does not exec', async () => {
  const calls: Array<[string, string[]]> = []

  const ran = await stopGatewayBeforeUpdate(CLI, HOME, {
    isWindows: true,
    existsSync: () => false,
    execFile: fakeExec(true),
    spy: (c, a) => calls.push([c, a])
  })

  assert.equal(ran, false)
  assert.deepEqual(calls, [[CLI, ['gateway', 'stop', '--all']]])
})

test('Windows with live CLI invokes "gateway stop --all" and returns true', async () => {
  let seenCommand = ''
  let seenArgs: string[] = []

  const ran = await stopGatewayBeforeUpdate(CLI, HOME, {
    isWindows: true,
    existsSync: () => true,
    execFile: ((command: string, args: string[], _options: unknown, callback: (error: Error | null) => void) => {
      seenCommand = command
      seenArgs = args
      callback(null)
    })
  })

  assert.equal(ran, true)
  assert.equal(seenCommand, CLI)
  assert.deepEqual(seenArgs, ['gateway', 'stop', '--all'])
})

test('Windows with failing CLI returns false (best-effort, never throws)', async () => {
  const ran = await stopGatewayBeforeUpdate(CLI, HOME, {
    isWindows: true,
    existsSync: () => true,
    execFile: fakeExec(false)
  })

  assert.equal(ran, false)
})

test('Windows gateway-stop timeout returns false without throwing', async () => {
  const timeout = Object.assign(new Error('Command failed: timed out'), { code: 'ETIMEDOUT' })

  const ran = await stopGatewayBeforeUpdate(CLI, HOME, {
    isWindows: true,
    existsSync: () => true,
    execFile: (_command, _args, _options, callback) => {
      callback(timeout)
    }
  })

  assert.equal(ran, false)
})

test('passes a generous timeout with hidden console (taskkill window suppression)', async () => {
  let seenOptions: unknown
  await stopGatewayBeforeUpdate(CLI, HOME, {
    isWindows: true,
    existsSync: () => true,
    execFile: ((_c: string, _a: string[], options: unknown, callback: (error: Error | null) => void) => {
      seenOptions = options
      callback(null)
    })
  })
  assert.deepEqual(seenOptions, {
    timeout: GATEWAY_STOP_TIMEOUT_MS,
    windowsHide: true,
    encoding: 'utf8'
  })
})

test('does not block the event loop while the gateway stop drains', async () => {
  let release: ((error: Error | null) => void) | undefined
  let invoked = false

  const pending = stopGatewayBeforeUpdate(CLI, HOME, {
    isWindows: true,
    existsSync: () => true,
    execFile: ((_c: string, _a: string[], _options: unknown, callback: (error: Error | null) => void) => {
      invoked = true
      release = callback
    })
  })

  assert.equal(invoked, true)
  let settled = false
  void pending.then(() => {
    settled = true
  })
  await Promise.resolve()
  assert.equal(settled, false)
  release!(null)
  assert.equal(await pending, true)
})

test('abort-path counterpart invokes "gateway start --all" (drain-semantics restore)', async () => {
  let seenArgs: string[] = []

  const ran = await startGatewaysAfterUpdateAbort(CLI, {
    isWindows: true,
    existsSync: () => true,
    execFile: ((_c: string, args: string[], _options: unknown, callback: (error: Error | null) => void) => {
      seenArgs = args
      callback(null)
    })
  })

  assert.equal(ran, true)
  assert.deepEqual(seenArgs, ['gateway', 'start', '--all'])
})

test('abort-path counterpart is a no-op off Windows', async () => {
  const calls: Array<[string, string[]]> = []

  const ran = await startGatewaysAfterUpdateAbort(CLI, {
    isWindows: false,
    existsSync: () => true,
    execFile: fakeExec(true),
    spy: (c, a) => calls.push([c, a])
  })

  assert.equal(ran, false)
  assert.deepEqual(calls, [])
})
