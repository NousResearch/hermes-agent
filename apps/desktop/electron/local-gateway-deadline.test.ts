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
