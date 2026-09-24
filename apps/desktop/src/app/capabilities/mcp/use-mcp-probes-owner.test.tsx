// @vitest-environment jsdom
import { cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as HermesModule from '@/hermes'
import { probeCache, probeKey } from '@/lib/mcp-probe-cache'

import type * as McpStatusModule from './mcp-status'
import { useMcpProbes } from './use-mcp-probes'

const testMcpServer = vi.hoisted(() => vi.fn())

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesModule>()),
  testMcpServer
}))

vi.mock('./mcp-status', async importOriginal => ({
  ...(await importOriginal<typeof McpStatusModule>()),
  loadMcpUsage: vi.fn().mockResolvedValue({})
}))

describe('MCP probe owner changes', () => {
  afterEach(() => {
    cleanup()
    probeCache.clear()
    testMcpServer.mockReset()
  })

  it('probes the same server again when its connection changes', async () => {
    const server = { enabled: true, url: 'https://mcp.example.test/shared' }
    const servers = { shared: server }
    const cachedA = { ok: true, tools: [{ name: 'a-tool', description: '' }] }
    const probedB = { ok: true, tools: [{ name: 'b-tool', description: '' }] }
    probeCache.set(probeKey('shared', server, 'connection-a::default'), { at: Date.now(), result: cachedA })
    testMcpServer.mockResolvedValue(probedB)
    const profileEpoch = { current: 0 }

    const view = renderHook(
      ({ owner }) =>
        useMcpProbes({
          appProfile: 'default',
          profile: 'default',
          profileEpoch,
          scopeProfileKey: owner,
          servers
        }),
      { initialProps: { owner: 'connection-a::default' } }
    )

    await waitFor(() => expect(view.result.current.probes.shared).toEqual(cachedA))
    expect(testMcpServer).not.toHaveBeenCalled()

    profileEpoch.current += 1
    view.rerender({ owner: 'connection-b::default' })

    await waitFor(() => expect(testMcpServer).toHaveBeenCalledWith('shared', 'default'))
    await waitFor(() => expect(view.result.current.probes.shared).toEqual(probedB))
    expect(probeCache.get(probeKey('shared', server, 'connection-b::default'))?.result).toEqual(probedB)
  })
})
