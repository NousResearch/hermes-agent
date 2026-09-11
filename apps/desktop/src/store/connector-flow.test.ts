import { describe, expect, it, vi } from 'vitest'

import { createConnectorFlow } from './connector-flow'

const listed = (connected: boolean) => ({
  available: true,
  connectors: [{ connector: 'gmail', connected, enabled: true }]
})

function flowWith(responses: { list: () => unknown; connect?: () => unknown }, onWaiting = vi.fn()) {
  const open = vi.fn(async () => {})

  const request = vi.fn(async (method: string) => {
    if (method === 'connectors.list') {
      return responses.list()
    }

    return responses.connect?.() ?? { results: [] }
  })

  const flow = createConnectorFlow('session', [{ connector: 'gmail' }], {
    request: request as never,
    open,
    onWaiting,
    delay: async () => {},
    now: () => 0
  })

  return { flow, open, onWaiting, request }
}

describe('the moment the browser has the sign-in', () => {
  it('reports waiting once the link is open, before the poll settles', async () => {
    let connected = false
    const order: string[] = []

    const { flow, onWaiting, open } = flowWith(
      {
        list: () => listed(connected),
        connect: () => ({ results: [{ connector: 'gmail', status: 'initiated', connect_url: 'https://auth.test/x' }] })
      },
      vi.fn(() => {
        order.push('waiting')
        // The user authorizes in the browser; the next poll sees it.
        connected = true
      })
    )

    open.mockImplementation(async () => {
      order.push('open')
    })

    await flow.refresh()
    await flow.connect('gmail')

    expect(order).toEqual(['open', 'waiting'])
    expect(onWaiting).toHaveBeenCalledWith('gmail')
    expect(flow.state.get().rows[0].phase).toBe('connected')
  })

  it('says nothing when the mint fails, so the agent is not sent to wait on nothing', async () => {
    const { flow, onWaiting } = flowWith({
      list: () => listed(false),
      connect: () => ({ results: [{ connector: 'gmail', status: 'error' }] })
    })

    await flow.refresh()
    await flow.connect('gmail')

    expect(onWaiting).not.toHaveBeenCalled()
    expect(flow.state.get().rows[0].phase).toBe('error')
  })

  it('says nothing when the app was already active — there is no browser step', async () => {
    const { flow, onWaiting } = flowWith({
      list: () => listed(true),
      connect: () => ({ results: [{ connector: 'gmail', status: 'active' }] })
    })

    await flow.refresh()
    await flow.connect('gmail')

    expect(onWaiting).not.toHaveBeenCalled()
  })

  it('sends the agent back in when the user keeps waiting after a timeout', async () => {
    let clock = 0
    const open = vi.fn(async () => {})
    const onWaiting = vi.fn()

    const request = vi.fn(async (method: string) =>
      method === 'connectors.list'
        ? listed(false)
        : { results: [{ connector: 'gmail', status: 'initiated', connect_url: 'https://auth.test/x' }] }
    )

    const flow = createConnectorFlow('session', [{ connector: 'gmail' }], {
      request: request as never,
      open,
      onWaiting,
      // Each poll tick burns the whole window so the first wait times out.
      delay: async () => {
        clock += 120001
      },
      now: () => clock
    })

    await flow.refresh()
    await flow.connect('gmail')

    expect(flow.state.get().rows[0].phase).toBe('timeout')
    expect(onWaiting).toHaveBeenCalledTimes(1)

    await flow.keepWaiting('gmail')

    expect(onWaiting).toHaveBeenCalledTimes(2)
  })

  it('asks the gateway to reconnect, not connect, when the grant is expired or revoked', async () => {
    const { flow, request } = flowWith({
      list: () => ({
        available: true,
        connectors: [{ connector: 'gmail', connected: false, enabled: true, connectionStatus: 'expired' }]
      }),
      connect: () => ({ results: [{ connector: 'gmail', status: 'active' }] })
    })

    await flow.refresh()
    await flow.connect('gmail')

    expect(request).toHaveBeenCalledWith('connectors.connect', expect.objectContaining({ reconnect: true }))
  })
})
