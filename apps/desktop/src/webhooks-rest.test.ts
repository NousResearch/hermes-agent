import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createWebhook, deleteWebhook, getWebhooks } from './hermes'

describe('Webhook REST parity helpers', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    api = vi.fn().mockResolvedValue({})
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it.each(['bot-chat', 'bot-chat:recipient'])('keeps %s distinct from the execution and store profiles', async deliver => {
    const body = { name: 'fixture', deliver, route_profile: 'producer', deliver_only: true }

    await createWebhook(body, 'operator')
    expect(api).toHaveBeenCalledWith(expect.objectContaining({ method: 'POST', path: '/api/webhooks',
      profile: 'operator', body }))
    await getWebhooks('operator')
    expect(api).toHaveBeenLastCalledWith(expect.objectContaining({ path: '/api/webhooks', profile: 'operator' }))
  })

  it('encodes the name when deleting a subscription', async () => {
    await deleteWebhook('my hook')

    expect(api).toHaveBeenCalledWith(expect.objectContaining({ method: 'DELETE', path: '/api/webhooks/my%20hook' }))
  })

  it('routes an explicitly owned delete to that profile, not the ambient scope', async () => {
    await deleteWebhook('my hook', 'coder')

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ method: 'DELETE', path: '/api/webhooks/my%20hook', profile: 'coder' })
    )
  })
})
