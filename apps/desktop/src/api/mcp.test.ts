import { afterEach, describe, expect, it, vi } from 'vitest'

import { addMcpServer, cancelMcpOAuthFlow, setMcpServerEnabled } from './mcp'

const originalDesktop = window.hermesDesktop

afterEach(() => {
  window.hermesDesktop = originalDesktop
  vi.restoreAllMocks()
})

describe('source-scoped MCP mutations', () => {
  it('routes setup writes to the backend that raised the request', async () => {
    const api = vi.fn().mockResolvedValue({ ok: true })
    window.hermesDesktop = { api } as unknown as Window['hermesDesktop']
    const scope = { connectionId: 'homelab', profile: 'research' }

    await addMcpServer({ name: 'docs', url: 'https://docs.invalid/mcp' }, scope)
    await setMcpServerEnabled('docs', true, scope)
    await cancelMcpOAuthFlow('flow-1', scope)

    expect(api).toHaveBeenNthCalledWith(1, {
      body: { name: 'docs', url: 'https://docs.invalid/mcp' },
      connectionId: 'homelab',
      method: 'POST',
      path: '/api/mcp/servers',
      profile: 'research'
    })
    expect(api).toHaveBeenNthCalledWith(2, {
      body: { enabled: true },
      connectionId: 'homelab',
      method: 'PUT',
      path: '/api/mcp/servers/docs/enabled',
      profile: 'research'
    })
    expect(api).toHaveBeenNthCalledWith(3, {
      connectionId: 'homelab',
      method: 'DELETE',
      path: '/api/mcp/oauth/flows/flow-1',
      profile: 'research'
    })
  })
})
