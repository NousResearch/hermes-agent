import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createWebhook, deleteWebhook, enableWebhooks, getWebhooks, setWebhookEnabled } from './hermes'

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

  it('lists webhooks from the admin endpoint', async () => {
    await getWebhooks('worker')

    expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/webhooks', profile: 'worker' }))
  })

  it('enables the webhook platform with POST', async () => {
    await enableWebhooks('worker')

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ method: 'POST', path: '/api/webhooks/enable', profile: 'worker' })
    )
  })

  it('creates a subscription with the full payload', async () => {
    const body = {
      deliver: 'telegram',
      deliver_only: true,
      description: 'push events',
      events: ['push'],
      name: 'github-push',
      prompt: 'summarize the push'
    }

    await createWebhook(body, 'worker')

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ body, method: 'POST', path: '/api/webhooks', profile: 'worker' })
    )
  })

  it('encodes the name when deleting a subscription', async () => {
    await deleteWebhook('my hook', 'worker')

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ method: 'DELETE', path: '/api/webhooks/my%20hook', profile: 'worker' })
    )
  })

  it('toggles a subscription enabled state via PUT', async () => {
    await setWebhookEnabled('github-push', false, 'worker')

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        body: { enabled: false },
        method: 'PUT',
        path: '/api/webhooks/github-push/enabled',
        profile: 'worker'
      })
    )
  })
})
