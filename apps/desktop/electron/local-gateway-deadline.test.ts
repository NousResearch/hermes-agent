import { EventEmitter } from 'node:events'

import { afterEach, expect, test, vi } from 'vitest'

afterEach(() => { vi.useRealTimers(); vi.doUnmock('node:child_process'); vi.resetModules() })

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
