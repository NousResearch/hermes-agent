import { describe, expect, it, vi } from 'vitest'

import { completeMcpDesktopOAuth, isValidAuthorizationUrl } from './mcp-dashboard-oauth'

describe('isValidAuthorizationUrl', () => {
  it('accepts http(s) URLs with a host', () => {
    expect(isValidAuthorizationUrl('https://idp.example/authorize')).toBe(true)
    expect(isValidAuthorizationUrl('http://127.0.0.1:8080/oauth')).toBe(true)
  })

  it('rejects non-http schemes', () => {
    expect(isValidAuthorizationUrl('javascript:alert(1)')).toBe(false)
    expect(isValidAuthorizationUrl('file:///etc/passwd')).toBe(false)
    expect(isValidAuthorizationUrl('not a url')).toBe(false)
  })
})

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

  it('rejects non-http(s) authorization URLs before openExternal', async () => {
    const openExternal = vi.fn()

    await expect(
      completeMcpDesktopOAuth({
        serverName: 'reports',
        start: async () => ({
          flow_id: 'flow-bad-scheme',
          server_name: 'reports',
          status: 'authorization_required',
          authorization_url: 'file:///etc/passwd',
          error: null
        }),
        status: vi.fn(),
        openExternal,
        sleep: async () => {}
      })
    ).rejects.toThrow(/http\(s\) URL/)

    expect(openExternal).not.toHaveBeenCalled()
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
