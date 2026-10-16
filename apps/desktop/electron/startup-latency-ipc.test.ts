import { expect, test, vi } from 'vitest'

const handlers = new Map<string, (...args: unknown[]) => unknown>()

vi.mock('electron', () => ({
  ipcMain: {
    handle: (channel: string, fn: (...args: unknown[]) => unknown) => {
      handlers.set(channel, fn)
    }
  }
}))

const { registerStartupLatencyIpc } = await import('./startup-latency-ipc')

test('the launch latency is claimable once per app launch, whichever window asks', () => {
  registerStartupLatencyIpc()

  const claim = handlers.get('hermes:startup-latency:claim')

  expect(claim).toBeDefined()

  const first = claim!({})

  expect(typeof first).toBe('number')
  expect(first as number).toBeGreaterThanOrEqual(0)
  // A renderer reload, a second window or a post-reconnect boot all land here.
  expect(claim!({})).toBeNull()
  expect(claim!({})).toBeNull()
})
