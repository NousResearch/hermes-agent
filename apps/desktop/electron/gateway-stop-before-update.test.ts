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
  return (_command: string, _args: string[], _options: unknown) => {
    if (!ok) {
      throw new Error('spawn ENOENT')
    }

    return Buffer.from('')
  }
}

test('non-Windows is a no-op and never invokes the CLI', () => {
  const calls: Array<[string, string[]]> = []

  const ran = stopGatewayBeforeUpdate(CLI, HOME, {
    isWindows: false,
    existsSync: () => true,
    execFileSync: fakeExec(true) as never,
    spy: (c, a) => calls.push([c, a])
  })

  assert.equal(ran, false)
  assert.deepEqual(calls, [])
})

test('Windows with missing CLI shim returns false and does not exec', () => {
  const calls: Array<[string, string[]]> = []

  const ran = stopGatewayBeforeUpdate(CLI, HOME, {
    isWindows: true,
    existsSync: () => false,
    execFileSync: fakeExec(true) as never,
    spy: (c, a) => calls.push([c, a])
  })

  assert.equal(ran, false)
  assert.deepEqual(calls, [[CLI, ['gateway', 'stop', '--all']]])
})

test('Windows with live CLI invokes "gateway stop --all" and returns true', () => {
  let seenCommand = ''
  let seenArgs: string[] = []

  const ran = stopGatewayBeforeUpdate(CLI, HOME, {
    isWindows: true,
    existsSync: () => true,
    execFileSync: ((command: string, args: string[]) => {
      seenCommand = command
      seenArgs = args

      return Buffer.from('')
    }) as never
  })

  assert.equal(ran, true)
  assert.equal(seenCommand, CLI)
  assert.deepEqual(seenArgs, ['gateway', 'stop', '--all'])
})

test('Windows with failing CLI returns false (best-effort, never throws)', () => {
  const ran = stopGatewayBeforeUpdate(CLI, HOME, {
    isWindows: true,
    existsSync: () => true,
    execFileSync: fakeExec(false) as never
  })

  assert.equal(ran, false)
})

test('passes a generous timeout with hidden console (taskkill window suppression)', () => {
  let seenOptions: unknown
  stopGatewayBeforeUpdate(CLI, HOME, {
    isWindows: true,
    existsSync: () => true,
    execFileSync: ((_c: string, _a: string[], options: unknown) => {
      seenOptions = options

      return Buffer.from('')
    }) as never
  })
  assert.deepEqual(seenOptions, {
    timeout: GATEWAY_STOP_TIMEOUT_MS,
    windowsHide: true,
    stdio: 'ignore',
    encoding: 'utf8'
  })
})

test('abort-path counterpart invokes "gateway start --all" (drain-semantics restore)', () => {
  let seenArgs: string[] = []

  const ran = startGatewaysAfterUpdateAbort(CLI, {
    isWindows: true,
    existsSync: () => true,
    execFileSync: ((_c: string, args: string[]) => {
      seenArgs = args

      return Buffer.from('')
    }) as never
  })

  assert.equal(ran, true)
  assert.deepEqual(seenArgs, ['gateway', 'start', '--all'])
})

test('abort-path counterpart is a no-op off Windows', () => {
  const calls: Array<[string, string[]]> = []

  const ran = startGatewaysAfterUpdateAbort(CLI, {
    isWindows: false,
    existsSync: () => true,
    execFileSync: fakeExec(true) as never,
    spy: (c, a) => calls.push([c, a])
  })

  assert.equal(ran, false)
  assert.deepEqual(calls, [])
})
