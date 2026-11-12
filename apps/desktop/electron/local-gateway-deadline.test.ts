import { EventEmitter } from 'node:events'
import fs from 'node:fs'
import path from 'node:path'

import { afterEach, expect, test, vi } from 'vitest'

afterEach(() => { vi.useRealTimers(); vi.doUnmock('node:child_process'); vi.resetModules() })

const repo: string = path.resolve(import.meta.dirname, '../../..')

/** A module-level numeric constant of the CLI that owns the ensure deadline, in milliseconds. */
function cliSeconds(file: string, name: string): number {
  const source: string = fs.readFileSync(path.join(repo, 'hermes_cli', file), 'utf8')
  const match: RegExpExecArray | null = new RegExp(`^${name}\\s*=\\s*([0-9.]+)\\s*$`, 'm').exec(source)

  if (!match) { throw new Error(`hermes_cli/${file} no longer defines ${name}`) }

  return Number(match[1]) * 1000
}

// The CLI's own wait the client must not cut short, read from the CLI rather than restated here.
const CLI_ADAPTIVE_CEILING_MS: number = cliSeconds('gateway_runtime_progress.py', 'MAX_ADAPTIVE_TIMEOUT')
const CLI_BASE_DEADLINE_MS: number = cliSeconds('gateway_runtime.py', 'DEFAULT_ENSURE_TIMEOUT')

test('an ensure client that ignores termination is killed and its promise rejects within a second deadline', async () => {
  vi.useFakeTimers()

  const child = Object.assign(new EventEmitter(), {
    stdout: new EventEmitter(), stderr: new EventEmitter(), kill: vi.fn(() => true)
  })

  vi.doMock('node:child_process', () => ({ spawn: () => child }))
  const { runGatewayEnsure } = await import('./local-gateway')
  const result = runGatewayEnsure({ command: 'owned-client', args: [], env: {}, shell: false }, '.', 'profile', {}, { timeoutMs: 10 })
  const rejected = expect(result).rejects.toThrow('timed out')
  await vi.advanceTimersByTimeAsync(10)
  expect(child.kill.mock.calls).toEqual([[]])
  await vi.advanceTimersByTimeAsync(1000)
  await rejected
  expect(child.kill.mock.calls).toEqual([[], ['SIGKILL']])
})

// restartLocalGatewayOwner (main.ts) logs a non-zero restart exit and lets the re-ensure report
// what answers. A restart client that times out but exits on SIGTERM must keep that contract:
// resolve exit 7 naming the deadline, then re-ensure and attach — not throw past the restart.
test('a stale-owner restart client that times out but obeys SIGTERM is logged and the re-ensure attaches', async () => {
  vi.useFakeTimers()

  const child = Object.assign(new EventEmitter(), {
    stdout: new EventEmitter(), stderr: new EventEmitter(),
    kill: vi.fn((signal?: string) => {
      if (!signal) { queueMicrotask(() => child.emit('close', null)) }

      return true
    })
  })

  vi.doMock('node:child_process', () => ({ spawn: () => child }))
  const { createStaleGatewayRestarter, ensureLocalGateway, runGatewayEnsure } = await import('./local-gateway')
  const logs: string[] = []

  const restartOwner = async (owner: string) => {
    const result = await runGatewayEnsure({ command: 'owned-client', args: [], env: {}, shell: false }, '.', owner, {}, { timeoutMs: 10, label: 'hermes gateway restart' })
    const reason = `${result.stdout}\n${result.stderr}`.trim().split(/\r?\n/).filter(Boolean).pop()
    logs.push(`gateway restart exited ${result.code}${reason ? `: ${reason}` : ''}`)
  }

  const endpoint = { profile_id: '/h', instance_id: 'old', authority_epoch: 1, runtime_protocol: 1, api_origin: 'http://127.0.0.1:1234', capabilities: ['session-authority-v1'], supervisor: 'none' }
  const answers = [{ ...endpoint, code_sha: 'old' }, { ...endpoint, instance_id: 'new', code_sha: 'new' }]
  let ensures = 0

  const attached = ensureLocalGateway(async () => {
    ensures += 1

    return { code: 0, stdout: JSON.stringify({ state: 'ready', endpoint: answers.shift(), client_code_sha: 'new' }) }
  }, undefined, createStaleGatewayRestarter(restartOwner, () => undefined))

  await vi.advanceTimersByTimeAsync(10)
  await expect(attached).resolves.toMatchObject({ gatewayEndpoint: { instance_id: 'new' } })
  expect(child.kill.mock.calls).toEqual([[]])
  expect(ensures).toBe(2)
  expect(logs).toEqual(['gateway restart exited 7: hermes gateway restart timed out'])
})

// `hermes gateway ensure` owns a 60 s startup deadline (hermes_cli/gateway_runtime.py
// DEFAULT_ENSURE_TIMEOUT) and always answers with a protocol verdict by then. Killing the client
// earlier turned a slow-but-valid cold start into "produced no result … Update Hermes".
test('the default ensure client outlives the protocol deadline and reads a late verdict', async () => {
  vi.useFakeTimers()

  const child = Object.assign(new EventEmitter(), {
    stdout: new EventEmitter(), stderr: new EventEmitter(), kill: vi.fn(() => true)
  })

  vi.doMock('node:child_process', () => ({ spawn: () => child }))
  const { runGatewayEnsure } = await import('./local-gateway')
  const result = runGatewayEnsure({ command: 'owned-client', args: [], env: {}, shell: false }, '.', 'profile', {})
  await vi.advanceTimersByTimeAsync(60_000)
  child.stdout.emit('data', Buffer.from('{"state":"starting","reason_code":"deadline"}'))
  child.emit('close', 5)
  await expect(result).resolves.toEqual({ code: 5, stdout: '{"state":"starting","reason_code":"deadline"}', stderr: '' })
  expect(child.kill).not.toHaveBeenCalled()
})

// A multiplexed cold start after logon (#142296: 15 profiles on one host gateway) legitimately
// outlives 75 s: the CLI's default ensure extends its own deadline while the start's owner keeps
// accruing CPU, up to MAX_ADAPTIVE_TIMEOUT. The client killed it first, so Desktop discarded a
// verdict that was about to arrive and kept "Hermes couldn't start" over a gateway that reached
// READY seconds later and then served for hours.
test('a starved cold start that answers late in the CLI\'s adaptive window is attached, not killed', async () => {
  vi.useFakeTimers()

  const child = Object.assign(new EventEmitter(), {
    stdout: new EventEmitter(), stderr: new EventEmitter(), kill: vi.fn(() => true)
  })

  vi.doMock('node:child_process', () => ({ spawn: () => child }))
  const { ensureLocalGateway, runGatewayEnsure } = await import('./local-gateway')
  const endpoint = { profile_id: '/h', instance_id: 'cold', authority_epoch: 1, runtime_protocol: 1, api_origin: 'http://127.0.0.1:1234', capabilities: ['session-authority-v1'], supervisor: 'none' }
  const attached = ensureLocalGateway(() => runGatewayEnsure({ command: 'owned-client', args: [], env: {}, shell: false }, '.', 'profile', {}))

  // Past even the base deadline the CLI reports for a wedged owner, this start is still working:
  // every profile it has yet to connect is a wait it is entitled to finish.
  await vi.advanceTimersByTimeAsync(CLI_BASE_DEADLINE_MS)
  expect(child.kill).not.toHaveBeenCalled()
  await vi.advanceTimersByTimeAsync(CLI_ADAPTIVE_CEILING_MS - CLI_BASE_DEADLINE_MS - 5_000)
  expect(child.kill).not.toHaveBeenCalled()

  child.stdout.emit('data', Buffer.from(JSON.stringify({ state: 'ready', endpoint, client_code_sha: 'cold' })))
  child.emit('close', 0)
  await expect(attached).resolves.toMatchObject({ baseUrl: 'http://127.0.0.1:1234', gatewayEndpoint: { instance_id: 'cold' } })
})

// The fail-fast decision lives in the CLI now (a wedged owner answers `starting: deadline` at its
// base deadline; an explicit --timeout stays hard). The client's own kill covers only a client that
// answers nothing at all, so it must not fire inside the window the CLI still owns — and it must
// still be bounded, SIGTERM then SIGKILL, past it.
test('the ensure client stays a bounded backstop past the CLI\'s adaptive window', async () => {
  vi.useFakeTimers()

  const child = Object.assign(new EventEmitter(), {
    stdout: new EventEmitter(), stderr: new EventEmitter(), kill: vi.fn(() => true)
  })

  vi.doMock('node:child_process', () => ({ spawn: () => child }))
  const { runGatewayEnsure } = await import('./local-gateway')
  const result = runGatewayEnsure({ command: 'owned-client', args: [], env: {}, shell: false }, '.', 'profile', {})
  const settled = result.then(() => new Error('resolved instead of killed'), (error: Error) => error)

  await vi.advanceTimersByTimeAsync(CLI_ADAPTIVE_CEILING_MS - 5_000)
  expect(child.kill).not.toHaveBeenCalled()
  // Past the ceiling: graceful termination first, force-kill only when it is ignored, then reject.
  await vi.advanceTimersByTimeAsync(60_000)
  expect(child.kill.mock.calls).toEqual([[], ['SIGKILL']])
  expect((await settled).message).toBe('hermes gateway ensure timed out')
})
