import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({ request: vi.fn(), open: vi.fn(), navigate: vi.fn(), close: vi.fn(), scope: { connectionId: 'browser', profile: 'coder' } }))
vi.mock('../src/api/client', () => ({ capabilityScoped: () => ({ ...mocks.scope }) }))
vi.mock('./transport', () => ({ requestJson: mocks.request, connection: () => ({ profile: 'default' }) }))

import { completeMcpDesktopOAuth, McpOAuthCancelled } from './mcp-oauth'

const pending = { flow_id: 'flow/1', server_name: 'reports', status: 'authorization_required', authorization_url: 'https://provider.example/authorize', error: null }
beforeEach(() => {
  mocks.request.mockReset()
  mocks.navigate.mockReset()
  mocks.close.mockReset()
  mocks.open.mockReset().mockReturnValue({ opener: null, closed: false, location: { replace: mocks.navigate }, close: mocks.close })
  vi.spyOn(window, 'open').mockImplementation(mocks.open)
  mocks.scope.profile = 'coder'
})

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

describe('browser MCP OAuth using existing backend callback', () => {
  it('starts and polls the dashboard flow with a stable owner even after switching profiles', async () => {
    mocks.request.mockResolvedValueOnce(pending).mockResolvedValueOnce({ ...pending, status: 'approved', tools: [] })
    const result = await completeMcpDesktopOAuth({ serverName: 'reports/a', sleep: async () => { mocks.scope.profile = 'other' } })
    expect(result.status).toBe('approved')
    expect(mocks.open).toHaveBeenCalledWith('about:blank', '_blank')
    expect(mocks.navigate).toHaveBeenCalledWith(pending.authorization_url)
    expect(mocks.open.mock.invocationCallOrder[0]).toBeLessThan(mocks.request.mock.invocationCallOrder[0])
    expect(mocks.request.mock.calls.map(([request]) => request)).toEqual([
      { connectionId: 'browser', profile: 'coder', path: '/api/mcp/servers/reports%2Fa/auth', method: 'POST' },
      { connectionId: 'browser', profile: 'coder', path: '/api/mcp/oauth/flows/flow%2F1' }
    ])
  })

  it('cancels a backend flow created while cancellation was requested without opening authorization', async () => {
    let cancelled = false
    mocks.request.mockImplementationOnce(async () => { cancelled = true;

 return pending }).mockResolvedValue({ ok: true })
    await expect(completeMcpDesktopOAuth({ serverName: 'reports', cancelled: () => cancelled })).rejects.toBeInstanceOf(McpOAuthCancelled)
    expect(mocks.navigate).not.toHaveBeenCalled()
    expect(mocks.close).toHaveBeenCalledTimes(1)
    expect(mocks.request).toHaveBeenLastCalledWith({ connectionId: 'browser', profile: 'coder', path: '/api/mcp/oauth/flows/flow%2F1', method: 'DELETE' })
  })

  it('releases the server flow after bounded polling failures', async () => {
    mocks.request.mockResolvedValueOnce(pending).mockRejectedValueOnce(new Error('network')).mockRejectedValueOnce(new Error('network')).mockResolvedValueOnce({ ok: true })
    await expect(completeMcpDesktopOAuth({ serverName: 'reports', maxPollFailures: 2, sleep: async () => {} })).rejects.toThrow('network')
    expect(mocks.open).toHaveBeenCalledTimes(1)
    expect(mocks.request.mock.calls.at(-1)?.[0].method).toBe('DELETE')
  })

  it('waits for a delayed authorization URL and cleans up a provider rejection', async () => {
    mocks.request.mockResolvedValueOnce({ ...pending, status: 'starting', authorization_url: null })
      .mockResolvedValueOnce(pending).mockResolvedValueOnce({ ...pending, status: 'error', error: 'Provider denied access' }).mockResolvedValueOnce({ ok: true })
    await expect(completeMcpDesktopOAuth({ serverName: 'reports', sleep: async () => {} })).rejects.toThrow('Provider denied access')
    expect(mocks.open).toHaveBeenCalledTimes(1)
    expect(mocks.request.mock.calls.at(-1)?.[0].method).toBe('DELETE')
  })
  it('falls back to the current tab when popups are blocked and preserves backend callback ownership', async () => {
    const assign = vi.fn()
    vi.stubGlobal('window', { open: () => null, location: { assign } })
    mocks.request.mockResolvedValueOnce(pending).mockRejectedValueOnce(new Error('navigation'))
    await expect(completeMcpDesktopOAuth({ serverName: 'reports', maxPollFailures: 1, sleep: async () => {} })).rejects.toThrow('navigation')
    expect(assign).toHaveBeenCalledWith(pending.authorization_url)
    expect(mocks.request.mock.calls.some(([request]) => request.method === 'DELETE')).toBe(false)
  })

  it('rejects executable authorization URLs and closes the reserved context', async () => {
    mocks.request.mockResolvedValueOnce({ ...pending, authorization_url: 'javascript:alert(1)' }).mockResolvedValueOnce({ ok: true })
    await expect(completeMcpDesktopOAuth({ serverName: 'reports' })).rejects.toThrow('Only HTTP(S)')
    expect(mocks.navigate).not.toHaveBeenCalled()
    expect(mocks.close).toHaveBeenCalledTimes(1)
    expect(mocks.request.mock.calls.at(-1)?.[0].method).toBe('DELETE')
  })

})
