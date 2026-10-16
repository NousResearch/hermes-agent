import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ComposerSuggestion } from '@/store/composer-suggestions'

import { mcpServerFromToolName, REPAIR_RE, reportMcpToolResult } from './repair'

const mocks = vi.hoisted(() => ({
  notify: vi.fn(),
  notifyError: vi.fn(),
  offers: [] as ComposerSuggestion[][],
  request: vi.fn()
}))

vi.mock('@/hermes', () => ({ listMcpServers: async () => ({ servers: [{ name: 'figma' }] }) }))
vi.mock('@/lib/mcp-dashboard-oauth', () => ({
  completeMcpDesktopOAuth: async () => ({ tools: [] }),
  McpOAuthCancelled: class extends Error {}
}))
vi.mock('@/store/composer-suggestions', () => ({
  offerSuggestions: (_sessionId: unknown, _provider: string, suggestions: ComposerSuggestion[]) => {
    mocks.offers.push(suggestions)
  }
}))
vi.mock('@/store/gateway', () => ({ $gateway: { get: () => ({ request: mocks.request }) } }))
vi.mock('@/store/notifications', () => ({ notify: mocks.notify, notifyError: mocks.notifyError }))

describe('mcpServerFromToolName', () => {
  it('extracts the server from a namespaced MCP tool', () => {
    expect(mcpServerFromToolName('mcp__figma__get_design_context')).toBe('figma')
    expect(mcpServerFromToolName('mcp__browsermcp__browser_navigate')).toBe('browsermcp')
  })

  it('handles multi-underscore server names', () => {
    expect(mcpServerFromToolName('mcp__comfy_cloud__generate')).toBe('comfy_cloud')
  })

  it('returns null for non-MCP tools', () => {
    expect(mcpServerFromToolName('terminal')).toBeNull()
    expect(mcpServerFromToolName('read_file')).toBeNull()
    expect(mcpServerFromToolName('mcp_not_namespaced')).toBeNull()
  })
})

describe('REPAIR_RE', () => {
  it.each([
    'HTTP 401 Unauthorized',
    'token has expired, please re-authenticate',
    'OAuth flow required',
    'authentication failed for server',
    'connection refused by remote host',
    'ECONNREFUSED 127.0.0.1:3845',
    'server disconnected before response'
  ])('flags auth/connection failures: %s', text => {
    expect(REPAIR_RE.test(text)).toBe(true)
  })

  it.each([
    'issue PROJ-123 not found',
    'no design nodes matched the query',
    'rate limit exceeded, retry after 60s',
    'invalid argument: node_id is required',
    'permission denied editing this file'
  ])('ignores tool-semantics failures: %s', text => {
    expect(REPAIR_RE.test(text)).toBe(false)
  })
})

describe('reconnect reads what reload.mcp did for the chat', () => {
  let session = 0

  // A fresh chat with a failed figma call: the offer it raises.
  const offered = async (): Promise<{ sessionId: string; suggestion: ComposerSuggestion }> => {
    const sessionId = `s${++session}`
    reportMcpToolResult(sessionId, 'mcp__figma__get_design_context', true, 'HTTP 401 Unauthorized')
    await vi.waitFor(() => expect(mocks.offers.at(-1)).toHaveLength(1))

    return { sessionId, suggestion: mocks.offers.at(-1)![0]! }
  }

  const invoke = ({ sessionId, suggestion }: { sessionId: string; suggestion: ComposerSuggestion }) =>
    suggestion.invoke({ cancelled: () => false, sessionId })

  beforeEach(() => {
    mocks.offers.length = 0
    mocks.notify.mockReset()
    mocks.notifyError.mockReset()
    mocks.request.mockReset()
  })

  it('withdraws the offer once the chat runs the reconnected server', async () => {
    const pill = await offered()
    mocks.request.mockResolvedValue({ session: 'applied', status: 'reloaded' })

    await invoke(pill)

    expect(mocks.request).toHaveBeenCalledWith('reload.mcp', { confirm: true, session_id: pill.sessionId })
    expect(mocks.offers.at(-1)).toEqual([])
  })

  it('keeps the offer and says so when the chat is busy and the change is deferred', async () => {
    const pill = await offered()
    const before = mocks.offers.length
    mocks.request.mockResolvedValue({ session: 'busy', status: 'reloaded' })

    await expect(invoke(pill)).rejects.toThrow()

    expect(mocks.offers).toHaveLength(before)
    expect(mocks.notify).toHaveBeenCalledWith(expect.objectContaining({ message: expect.stringMatching(/busy/i) }))
    expect(mocks.notifyError).not.toHaveBeenCalled()
  })

  it('keeps the offer when the server still did not connect', async () => {
    const pill = await offered()
    const before = mocks.offers.length
    mocks.request.mockResolvedValue({ session: 'applied', status: 'reloaded', unavailable: ['figma'] })

    await expect(invoke(pill)).rejects.toThrow(/figma/)

    expect(mocks.offers).toHaveLength(before)
    expect(mocks.notifyError).toHaveBeenCalled()
  })

  it('keeps the offer when the reload itself is refused', async () => {
    const pill = await offered()
    const before = mocks.offers.length
    mocks.request.mockRejectedValue(new Error('mcp_servers_unavailable: did not connect: figma'))

    await expect(invoke(pill)).rejects.toThrow()

    expect(mocks.offers).toHaveLength(before)
  })

  it('withdraws its own offer but names another server that did not connect', async () => {
    const pill = await offered()
    mocks.request.mockResolvedValue({ session: 'unchanged', status: 'reloaded', unavailable: ['linear'] })

    await invoke(pill)

    expect(mocks.offers.at(-1)).toEqual([])
    expect(mocks.notify).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'warning', message: expect.stringContaining('linear') })
    )
  })
})
