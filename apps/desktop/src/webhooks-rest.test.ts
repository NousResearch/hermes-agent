import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { deleteWebhook } from './rabbit'

describe('Webhook REST parity helpers', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    api = vi.fn().mockResolvedValue({})
    Object.defineProperty(window, 'rabbitDesktop', {
      configurable: true,
      value: { api }
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'rabbitDesktop')
  })

  it('encodes the name when deleting a subscription', async () => {
    await deleteWebhook('my hook')

    expect(api).toHaveBeenCalledWith(expect.objectContaining({ method: 'DELETE', path: '/api/webhooks/my%20hook' }))
  })
})
