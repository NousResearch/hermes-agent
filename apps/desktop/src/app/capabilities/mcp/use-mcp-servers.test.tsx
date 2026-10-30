import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, renderHook } from '@testing-library/react'
import { type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { getHermesConfigRecord, getMcpCatalog, getUsageAnalytics, type McpTestResult, testMcpServer } from '@/hermes'
import { probeCache, probeKey } from '@/lib/mcp-probe-cache'
import { queryClient } from '@/lib/query-client'

import { hermesConfigKey } from '../../hooks/use-config-record'

import { useMcpServers } from './use-mcp-servers'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getHermesConfigRecord: vi.fn(),
  getMcpCatalog: vi.fn(),
  getUsageAnalytics: vi.fn(),
  testMcpServer: vi.fn()
}))

const rea = { command: 'rea', args: ['mcp'] }
const learn = { url: 'https://learn.microsoft.com/api/mcp' }
const config = { mcp_servers: { rea, learn } }
const success = (tool: string): McpTestResult => ({ ok: true, tools: [{ name: tool, description: tool }] })

function deferred() {
  let resolve!: (value: McpTestResult) => void
  let reject!: (reason: Error) => void

  const promise = new Promise<McpTestResult>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, resolve, reject }
}

const wrapper = ({ children }: { children: ReactNode }) => (
  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
)

beforeEach(() => {
  probeCache.clear()
  queryClient.clear()
  vi.mocked(getHermesConfigRecord).mockResolvedValue(config)
  vi.mocked(getMcpCatalog).mockResolvedValue({ entries: [], diagnostics: [] })
  vi.mocked(getUsageAnalytics).mockRejectedValue(new Error('Usage analytics unavailable'))
  // Settings can mount with cached config but no probe results. The requests
  // then start before StrictMode replays the controller's epoch cleanup.
  queryClient.setQueryData(hermesConfigKey('default'), config)
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  probeCache.clear()
  vi.resetAllMocks()
})

describe('MCP server controller in StrictMode', () => {
  it.each(['resolve', 'reject'] as const)(
    'finishes REA and Learn probes after effect replay and ignores retired requests (%s)',
    async settle => {
      const oldRea = deferred()
      const oldLearn = deferred()
      const currentRea = deferred()
      const currentLearn = deferred()
      const requests = { rea: [oldRea, currentRea], learn: [oldLearn, currentLearn] }
      vi.mocked(testMcpServer).mockImplementation(name => requests[name as keyof typeof requests].shift()!.promise)

      const { result } = renderHook(() => useMcpServers({ gateway: null, profile: 'default' }), {
        wrapper,
        // React 19 only replays effects when StrictMode is at the root.
        reactStrictMode: true
      })

      expect(result.current.statuses).toEqual({ rea: 'probing', learn: 'probing' })

      const reaResult: McpTestResult = { ok: false, error: 'MCP server test timed out after 60s', tools: [] }
      const learnResult = success('learn-search')
      await act(async () => {
        currentRea.resolve(reaResult)
        currentLearn.resolve(learnResult)
      })

      expect(result.current.statuses).toEqual({ rea: 'error', learn: 'ok' })
      expect(result.current.probes).toEqual({ rea: reaResult, learn: learnResult })
      expect(result.current.toolCounts).toEqual({ learn: { on: 1, total: 1 } })
      expect(vi.mocked(testMcpServer).mock.calls).toEqual([
        ['rea', 'default'],
        ['learn', 'default'],
        ['rea', 'default'],
        ['learn', 'default']
      ])

      await act(async () => {
        if (settle === 'resolve') {
          oldRea.resolve(success('obsolete-inspect'))
          oldLearn.resolve({ ok: false, error: 'HTTP 401 Unauthorized', tools: [] })
        } else {
          oldRea.reject(new Error('obsolete REA failure'))
          oldLearn.reject(new Error('HTTP 401 Unauthorized'))
        }
      })

      expect(result.current.statuses).toEqual({ rea: 'error', learn: 'ok' })
      expect(result.current.probes).toEqual({ rea: reaResult, learn: learnResult })
      expect(probeCache.get(probeKey('rea', rea, 'default'))?.result).toBe(reaResult)
      expect(probeCache.get(probeKey('learn', learn, 'default'))?.result).toBe(learnResult)
    }
  )

  it('retains fresh completed results across effect replay without probing again', async () => {
    const at = Date.now() - 1_000
    const reaResult = success('inspect')
    const learnResult = success('learn-search')
    probeCache.set(probeKey('rea', rea, 'default'), { at, result: reaResult })
    probeCache.set(probeKey('learn', learn, 'default'), { at, result: learnResult })

    const { result } = renderHook(() => useMcpServers({ gateway: null, profile: 'default' }), {
      wrapper,
      reactStrictMode: true
    })

    await act(async () => {})

    expect(result.current.statuses).toEqual({ rea: 'ok', learn: 'ok' })
    expect(result.current.probes).toEqual({ rea: reaResult, learn: learnResult })
    expect(testMcpServer).not.toHaveBeenCalled()
    expect(probeCache.get(probeKey('rea', rea, 'default'))).toEqual({ at, result: reaResult })
    expect(probeCache.get(probeKey('learn', learn, 'default'))).toEqual({ at, result: learnResult })
  })
})
