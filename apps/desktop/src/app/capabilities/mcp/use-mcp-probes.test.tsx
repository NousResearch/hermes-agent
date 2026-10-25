import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { probeCache, probeKey } from '@/lib/mcp-probe-cache'

import { useMcpProbes } from './use-mcp-probes'

const mocks = vi.hoisted(() => ({
  probe: vi.fn(),
  usage: vi.fn(async () => ({ tools: [] }))
}))

vi.mock('@/hermes', () => ({ testMcpServer: mocks.probe, getUsageAnalytics: mocks.usage }))

afterEach(() => {
  vi.clearAllMocks()
  probeCache.clear()
})

it('keeps a successful OAuth result when an older unauthenticated probe finishes later', async () => {
  let finish!: (result: { ok: boolean; error: string; tools: [] }) => void
  mocks.probe.mockReturnValueOnce(new Promise(resolve => { finish = resolve }))

  const servers = { notion: { url: 'https://mcp.notion.com/mcp', auth: 'oauth' } }

  const { result } = renderHook(() => useMcpProbes({
    appProfile: null,
    profile: null,
    profileEpoch: { current: 0 },
    scopeProfileKey: 'oauth-retry-test',
    servers
  }))

  await waitFor(() => expect(mocks.probe).toHaveBeenCalledTimes(1))
  act(() => result.current.setProbe('notion', { ok: true, tools: [{ name: 'list-pages', description: 'List pages' }] }))
  await act(async () => {
    finish({ ok: false, error: 'OAuth authentication required', tools: [] })
    await Promise.resolve()
  })
  expect(result.current.probes.notion).toMatchObject({ ok: true })
})

it('publishes successful OAuth tools to the exact profile and server probe cache', () => {
  const server = { url: 'https://mcp.notion.com/mcp', auth: 'oauth', enabled: false }
  const tools = [{ name: 'list-pages', description: 'List pages' }]

  const { result } = renderHook(() => useMcpProbes({
    appProfile: null,
    profile: null,
    profileEpoch: { current: 0 },
    scopeProfileKey: 'profile-a',
    servers: { notion: server }
  }))

  act(() => result.current.publishAuthorized('notion', tools, server))
  expect(result.current.probes.notion).toMatchObject({ ok: true, tools })
  expect(probeCache.get(probeKey('notion', server, 'profile-a'))?.result).toMatchObject({ ok: true, tools })
  expect(probeCache.has(probeKey('notion', server, 'profile-b'))).toBe(false)
})
