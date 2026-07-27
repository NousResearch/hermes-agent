import { describe, expect, it, vi } from 'vitest'

import { completeMcpDesktopOAuth } from './mcp-dashboard-oauth'

describe('completeMcpDesktopOAuth', () => {
  it('opens the returned authorization URL and polls through approval', async () => {
    const openExternal = vi.fn().mockResolvedValue(undefined)

    const status = vi
      .fn()
      .mockResolvedValueOnce({
        flow_id: 'flow-1',
        server_name: 'reports',
        status: 'authorization_required',
        authorization_url: 'https://idp.example/authorize',
        error: null
      })
      .mockResolvedValueOnce({
        flow_id: 'flow-1',
        server_name: 'reports',
        status: 'approved',
        authorization_url: 'https://idp.example/authorize',
        error: null,
        tools: [{ name: 'list_reports', description: 'List reports' }]
      })

    const result = await completeMcpDesktopOAuth({
      serverName: 'reports',
      start: vi.fn().mockResolvedValue({
        flow_id: 'flow-1',
        server_name: 'reports',
        status: 'authorization_required',
        authorization_url: 'https://idp.example/authorize',
        error: null
      }),
      status,
      openExternal,
      sleep: async () => {}
    })

    expect(openExternal).toHaveBeenCalledWith('https://idp.example/authorize')
    expect(result.status).toBe('approved')
  })

  it('retries a transient status failure', async () => {
    const status = vi.fn().mockRejectedValueOnce(new Error('temporary network failure')).mockResolvedValueOnce({
      flow_id: 'flow-2',
      server_name: 'reports',
      status: 'approved',
      authorization_url: 'https://idp.example/authorize',
      error: null,
      tools: []
    })

    const result = await completeMcpDesktopOAuth({
      serverName: 'reports',
      start: vi.fn().mockResolvedValue({
        flow_id: 'flow-2',
        server_name: 'reports',
        status: 'authorization_required',
        authorization_url: 'https://idp.example/authorize',
        error: null
      }),
      status,
      openExternal: vi.fn().mockResolvedValue(undefined),
      sleep: async () => {}
    })

    expect(result.status).toBe('approved')
    expect(status).toHaveBeenCalledTimes(2)
  })
})
