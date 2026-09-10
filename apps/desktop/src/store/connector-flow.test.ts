import { describe, expect, it, vi } from 'vitest'

import { createConnectorFlow } from './connector-flow'

const gmail = { connector: 'gmail', enabled: true, connected: false }

function harness() {
  const request = vi.fn().mockResolvedValue({ available: true, connectors: [gmail] })
  const open = vi.fn().mockResolvedValue(undefined)
  const flow = createConnectorFlow('session-a', [gmail], { request, open, delay: async () => undefined })

  return { flow, request, open }
}

describe('connector lifecycle', () => {
  it('refreshes a replayed offer without minting or opening authorization', async () => {
    const h = harness()
    await h.flow.refresh()
    expect(h.request).toHaveBeenCalledExactlyOnceWith('connectors.list', { session_id: 'session-a' })
    expect(h.open).not.toHaveBeenCalled()
    expect(h.flow.state.get().rows[0].phase).toBe('idle')
  })

  it('requires live confirmation, not an initiated result, to show connected', async () => {
    const h = harness()
    await h.flow.refresh()
    h.request
      .mockResolvedValueOnce({
        results: [{ connector: 'gmail', status: 'initiated', connect_url: 'https://connect.example.test/one-use' }]
      })
      .mockResolvedValueOnce({ available: true, connectors: [{ ...gmail, connected: true }] })
    await h.flow.connect('gmail')
    expect(h.open).toHaveBeenCalledExactlyOnceWith('https://connect.example.test/one-use')
    expect(h.flow.state.get().rows[0].phase).toBe('connected')
    expect(h.request.mock.calls.at(-1)).toEqual(['connectors.list', { session_id: 'session-a' }])
  })

  it('ignores a late connect response after cancellation without opening a browser', async () => {
    const h = harness()
    await h.flow.refresh()
    let resolve!: (value: unknown) => void
    h.request.mockImplementationOnce(
      () =>
        new Promise(r => {
          resolve = r
        })
    )
    const pending = h.flow.connect('gmail')
    h.flow.skip('gmail')
    resolve({
      results: [{ connector: 'gmail', status: 'initiated', connect_url: 'https://connect.example.test/one-use' }]
    })
    await pending
    expect(h.open).not.toHaveBeenCalled()
    expect(h.flow.state.get().rows[0].phase).toBe('skipped')
  })

  it('does not turn a gateway failure into an empty catalog', async () => {
    const h = harness()
    h.request.mockRejectedValue(new Error('offline'))
    await h.flow.refresh()
    expect(h.flow.state.get().error).toBe('status')
    expect(h.flow.state.get().rows).toHaveLength(1)
    expect(h.flow.state.get().available).toBe(false)
  })

  it('offers keep-waiting after a bounded authorization timeout', async () => {
    let time = 0
    const request = vi.fn().mockResolvedValue({ available: true, connectors: [gmail] })

    const flow = createConnectorFlow('session-a', [gmail], {
      request,
      open: vi.fn(),
      now: () => time,
      delay: async () => {
        time += 60000
      }
    })

    await flow.refresh()
    request.mockResolvedValueOnce({
      results: [{ connector: 'gmail', status: 'initiated', connect_url: 'https://connect.example.test/link' }]
    })
    await flow.connect('gmail')
    expect(flow.state.get().rows[0].phase).toBe('timeout')
    request.mockClear()
    await flow.keepWaiting('gmail')
    expect(flow.state.get().rows[0].phase).toBe('timeout')
    expect(request.mock.calls.every(([method]) => method === 'connectors.list')).toBe(true)
  })
})

it('ignores stale refresh after a user skips and restores status after disposal without reconnecting', async () => {
  const h = harness()
  await h.flow.refresh()
  let resolve!: (value: unknown) => void
  h.request.mockImplementationOnce(
    () =>
      new Promise(r => {
        resolve = r
      })
  )
  const refresh = h.flow.refresh()
  h.flow.skip('gmail')
  resolve({ available: true, connectors: [{ ...gmail, connected: true }] })
  await refresh
  expect(h.flow.state.get().rows[0].phase).toBe('skipped')
  h.flow.dispose()
  await h.flow.refresh()
  expect(h.open).not.toHaveBeenCalled()
  expect(h.request.mock.calls.every(([method]) => method === 'connectors.list')).toBe(true)
})
