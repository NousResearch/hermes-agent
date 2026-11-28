import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { requestGatewayForAgent } from '@/store/gateway'

import { McpOAuthCancelled } from './mcp-dashboard-oauth'
import { completeMcpOAuth } from './mcp-oauth-host'

vi.mock('@/store/gateway', () => ({ requestGatewayForAgent: vi.fn() }))

// The browser-hosted Desktop (Webapp) runs on a machine other than the gateway's, so the
// provider must redirect to the gateway's public callback, never to a loopback listener.
const publicCallback = 'https://chat.example.com/api/mcp/oauth/callback/reports'
const authUrl = `https://idp.example/authorize?state=s1&redirect_uri=${encodeURIComponent(publicCallback)}`
const tools = [{ name: 'list_reports', description: 'List reports' }]

function harness(statuses: string[]) {
  const popup = { closed: false, close: vi.fn(), location: { href: '' }, opener: {} as unknown }
  const open = vi.spyOn(window, 'open').mockReturnValue(popup as unknown as Window)
  const openExternal = vi.fn(async () => {})
  let polls = 0

  const api = vi.fn(async (request: { path: string; method?: string }) => {
    if (request.path === '/api/mcp/servers/reports/auth') {
      return { flow_id: 'f1', server_name: 'reports', status: 'authorization_required', authorization_url: authUrl, error: null }
    }

    if (request.path === '/api/mcp/oauth/flows/f1' && request.method === 'DELETE') {
      return { ok: true }
    }

    if (request.path === '/api/mcp/oauth/flows/f1') {
      const status = statuses[Math.min(polls++, statuses.length - 1)]

      return { flow_id: 'f1', server_name: 'reports', status, authorization_url: authUrl, error: status === 'error' ? 'denied' : null, tools }
    }

    throw new Error(`unexpected ${request.method ?? 'GET'} ${request.path}`)
  })

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api, openExternal } })

  return { api, open, openExternal, popup }
}

beforeEach(() => {
  document.documentElement.dataset.hermesDesktopHost = 'browser'
})

afterEach(() => {
  delete document.documentElement.dataset.hermesDesktopHost
  vi.restoreAllMocks()
  vi.resetAllMocks()
})

describe('browser-hosted MCP OAuth', () => {
  it('uses the gateway public callback flow, not a loopback listener', async () => {
    const { api, open, popup } = harness(['authorization_required', 'approved'])

    const result = await completeMcpOAuth({ serverName: 'reports', profile: 'work', sleep: async () => {} })

    expect(result).toMatchObject({ status: 'approved', flow_id: 'f1', tools })
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/api/mcp/servers/reports/auth', method: 'POST', profile: 'work' })
    )
    // Opened synchronously (inside the click's activation) and pointed at the provider afterwards.
    expect(open).toHaveBeenCalledWith('', '_blank')
    expect(popup.location.href).toBe(authUrl)
    expect(popup.opener).toBeNull()
    expect(vi.mocked(requestGatewayForAgent)).not.toHaveBeenCalled()
    expect(api).not.toHaveBeenCalledWith(expect.objectContaining({ method: 'DELETE' }))
  })

  it('surfaces a provider error and cancels the flow', async () => {
    const { api } = harness(['error'])

    await expect(completeMcpOAuth({ serverName: 'reports', sleep: async () => {} })).rejects.toThrow('denied')
    expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/mcp/oauth/flows/f1', method: 'DELETE' }))
  })

  it('cancels the server flow and closes the tab when the user cancels', async () => {
    const { api, popup } = harness(['authorization_required'])
    let cancelled = false

    const action = completeMcpOAuth({
      serverName: 'reports',
      cancelled: () => cancelled,
      sleep: async () => {
        cancelled = true
      }
    })

    await expect(action).rejects.toBeInstanceOf(McpOAuthCancelled)
    expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/mcp/oauth/flows/f1', method: 'DELETE' }))
    expect(popup.close).toHaveBeenCalled()
  })

  it('falls back to openExternal when the popup is blocked', async () => {
    const { open, openExternal } = harness(['approved'])
    open.mockReturnValue(null)

    await completeMcpOAuth({ serverName: 'reports', sleep: async () => {} })

    expect(openExternal).toHaveBeenCalledWith(authUrl)
  })
})
