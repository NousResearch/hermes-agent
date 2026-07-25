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
    await getWebhooks('coder')

    expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/webhooks', profile: 'coder' }))
  })

  it('enables the webhook platform with POST', async () => {
    await enableWebhooks('coder')

    expect(api).toHaveBeenCalledWith(expect.objectContaining({ method: 'POST', path: '/api/webhooks/enable', profile: 'coder' }))
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

    await createWebhook(body, 'coder')

    expect(api).toHaveBeenCalledWith(expect.objectContaining({ body, method: 'POST', path: '/api/webhooks', profile: 'coder' }))
  })

  it('encodes the name when deleting a subscription', async () => {
    await deleteWebhook('my hook', 'coder')

    expect(api).toHaveBeenCalledWith(expect.objectContaining({ method: 'DELETE', path: '/api/webhooks/my%20hook', profile: 'coder' }))
  })

  it('toggles a subscription enabled state via PUT', async () => {
    await setWebhookEnabled('github-push', false, 'coder')

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        body: { enabled: false },
        method: 'PUT',
        profile: 'coder',
        path: '/api/webhooks/github-push/enabled'
      })
    )
  })
})
