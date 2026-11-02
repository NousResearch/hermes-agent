import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { McpTestResult } from '@/hermes'
import { PROBE_TTL_MS, probeCache, probeKey } from '@/lib/mcp-probe-cache'
import type { McpServers } from '@/lib/mcp-servers'

import { statusOf } from './mcp-status'
import { useMcpProbes, type UseMcpProbesOptions } from './use-mcp-probes'

const mocks = vi.hoisted(() => ({ testMcpServer: vi.fn(), getUsageAnalytics: vi.fn() }))

vi.mock('@/hermes', () => mocks)

const success = (tool = 'inspect'): McpTestResult => ({ ok: true, tools: [{ name: tool, description: tool }] })
const failure = (error: string): McpTestResult => ({ ok: false, error, tools: [] })
const rea = { command: 'rea', args: ['mcp'] }
const learn = { url: 'https://learn.microsoft.com/api/mcp' }

function deferred() {
  let resolve!: (value: McpTestResult) => void
  let reject!: (reason: Error) => void

  const promise = new Promise<McpTestResult>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, resolve, reject }
}

const options = (servers: McpServers): UseMcpProbesOptions => ({
  servers,
  appProfile: null,
  scopeProfileKey: 'default',
  profileEpoch: { current: 0 }
})

beforeEach(() => {
  probeCache.clear()
  mocks.testMcpServer.mockReset()
  mocks.getUsageAnalytics.mockReset().mockResolvedValue({ tools: [] })
})

afterEach(() => {
  cleanup()
  probeCache.clear()
  vi.restoreAllMocks()
})

describe('mounted MCP probes', () => {
  it('refreshes changed connection config while retaining the other server result', async () => {
    const oldRea = success('old-inspect')
    const newRea = success('new-inspect')
    const learnResult = success('learn-search')
    const pending = deferred()
    const initial = options({ rea, learn })
    mocks.testMcpServer.mockImplementation((name: string) => Promise.resolve(name === 'rea' ? oldRea : learnResult))
    const { result, rerender } = renderHook(useMcpProbes, { initialProps: initial })
    await act(async () => {})
    expect(result.current.probes).toEqual({ rea: oldRea, learn: learnResult })

    mocks.testMcpServer.mockReturnValueOnce(pending.promise)
    const editedRea = { ...rea, args: ['mcp', '--new'] }
    rerender({ ...initial, servers: { rea: editedRea, learn } })

    expect(mocks.testMcpServer).toHaveBeenCalledTimes(3)
    expect(result.current.probes).toEqual({ rea: 'probing', learn: learnResult })
    expect(result.current.toolCounts.rea).toBeUndefined()
    await act(async () => pending.resolve(newRea))
    expect(result.current.probes).toEqual({ rea: newRea, learn: learnResult })
    expect(probeCache.get(probeKey('rea', editedRea, 'default'))?.result).toBe(newRea)
  })

  it('uses a fresh cache entry for changed config without probing again', async () => {
    const original = success('old')
    const editedRea = { ...rea, command: 'rea-new' }
    const cached = success('cached')
    mocks.testMcpServer.mockResolvedValue(original)
    const initial = options({ rea })
    const { result, rerender } = renderHook(useMcpProbes, { initialProps: initial })
    await act(async () => {})
    probeCache.set(probeKey('rea', editedRea, 'default'), { at: Date.now(), result: cached })

    rerender({ ...initial, servers: { rea: editedRea } })

    expect(result.current.probes.rea).toBe(cached)
    expect(mocks.testMcpServer).toHaveBeenCalledTimes(1)
  })

  it('rechecks TTL on config refresh but does not probe for fresh cosmetic edits', async () => {
    let now = 1_000
    vi.spyOn(Date, 'now').mockImplementation(() => now)
    const original = success('old')
    const updated = success('updated')
    mocks.testMcpServer.mockResolvedValueOnce(original).mockResolvedValueOnce(updated)
    const initial = options({ rea })
    const { result, rerender } = renderHook(useMcpProbes, { initialProps: initial })
    await act(async () => {})
    rerender({ ...initial, servers: { rea: { ...rea, description: 'cosmetic' } } })
    expect(mocks.testMcpServer).toHaveBeenCalledTimes(1)

    now += PROBE_TTL_MS
    rerender({ ...initial, servers: { rea: { ...rea } } })
    await act(async () => {})

    expect(mocks.testMcpServer).toHaveBeenCalledTimes(2)
    expect(result.current.probes.rea).toBe(updated)
  })

  it.each(['resolve', 'reject'] as const)('rejects an old config completion (%s)', async settle => {
    const old = deferred()
    const current = deferred()
    mocks.testMcpServer.mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise)
    const initial = options({ rea })
    const { result, rerender } = renderHook(useMcpProbes, { initialProps: initial })
    const editedRea = { ...rea, command: 'rea-new' }
    rerender({ ...initial, servers: { rea: editedRea } })
    const newest = success('newest')
    await act(async () => current.resolve(newest))
    await act(async () => {
      if (settle === 'resolve') {
        old.resolve(failure('authentication failed'))
      } else {
        old.reject(new Error('authentication failed'))
      }
    })

    expect(mocks.testMcpServer).toHaveBeenCalledTimes(2)
    expect(result.current.probes.rea).toBe(newest)
    expect(probeCache.has(probeKey('rea', rea, 'default'))).toBe(false)
    expect(probeCache.get(probeKey('rea', editedRea, 'default'))?.result).toBe(newest)
  })

  it.each(['resolve', 'reject'] as const)('rejects a superseded same-config completion (%s)', async settle => {
    const old = deferred()
    const current = deferred()
    mocks.testMcpServer.mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise)
    const { result } = renderHook(useMcpProbes, { initialProps: options({ rea }) })
    let refresh!: Promise<void>
    act(() => {
      refresh = result.current.runProbe('rea')
    })
    const newest = success('newest')
    await act(async () => {
      current.resolve(newest)
      await refresh
    })
    await act(async () => {
      if (settle === 'resolve') {
        old.resolve(failure('HTTP 401 Unauthorized'))
      } else {
        old.reject(new Error('HTTP 401 Unauthorized'))
      }
    })

    expect(result.current.probes.rea).toBe(newest)
    expect(probeCache.get(probeKey('rea', rea, 'default'))?.result).toBe(newest)
  })

  it('does not overwrite a manual auth result or re-probe during authentication', async () => {
    const old = deferred()
    mocks.testMcpServer.mockReturnValue(old.promise)
    const initial = options({ learn })
    const { result, rerender } = renderHook(useMcpProbes, { initialProps: initial })
    act(() => result.current.setProbe('learn', 'probing'))
    rerender({ ...initial, servers: { learn: { ...learn } } })
    expect(mocks.testMcpServer).toHaveBeenCalledTimes(1)
    const authenticated = success('learn-search')
    act(() => result.current.setProbe('learn', authenticated))
    await act(async () => old.resolve(failure('HTTP 401 Unauthorized')))

    expect(result.current.probes.learn).toBe(authenticated)
    expect(probeCache.has(probeKey('learn', learn, 'default'))).toBe(false)

    const authenticatedConfig = { ...learn, auth: 'oauth' }
    probeCache.set(probeKey('learn', authenticatedConfig, 'default'), { at: Date.now(), result: authenticated })
    rerender({ ...initial, servers: { learn: authenticatedConfig } })
    expect(result.current.probes.learn).toBe(authenticated)
    expect(mocks.testMcpServer).toHaveBeenCalledTimes(1)
  })

  it.each(['rea-first', 'learn-first'] as const)(
    'keeps REA timeout and Learn success independent (%s)',
    async order => {
      const reaRequest = deferred()
      const learnRequest = deferred()
      mocks.testMcpServer.mockImplementation((name: string) =>
        name === 'rea' ? reaRequest.promise : learnRequest.promise
      )
      const { result } = renderHook(useMcpProbes, { initialProps: options({ rea, learn }) })
      const learnResult = success('learn-search')

      const reaResult = failure(
        "MCP server 'rea' test timed out after 60s (bounded by connect_timeout; an OAuth login also by oauth.timeout)"
      )

      if (order === 'learn-first') {
        await act(async () => learnRequest.resolve(learnResult))
        expect(result.current.probes).toEqual({ rea: 'probing', learn: learnResult })
        await act(async () => reaRequest.resolve(reaResult))
      } else {
        await act(async () => reaRequest.resolve(reaResult))
        expect(result.current.probes).toEqual({ rea: reaResult, learn: 'probing' })
        await act(async () => learnRequest.resolve(learnResult))
      }

      expect(statusOf(rea, result.current.probes.rea)).toBe('error')
      expect(statusOf(learn, result.current.probes.learn)).toBe('ok')
      expect(result.current.toolCounts).toEqual({ learn: { on: 1, total: 1 } })
      expect(probeCache.get(probeKey('rea', rea, 'default'))?.result).toBe(reaResult)
      expect(probeCache.get(probeKey('learn', learn, 'default'))?.result).toBe(learnResult)
    }
  )

  it('does not extend a shared cache result TTL when mounting it', async () => {
    let now = 1_000
    vi.spyOn(Date, 'now').mockImplementation(() => now)
    const cached = success('cached')
    probeCache.set(probeKey('rea', rea, 'default'), { at: now, result: cached })
    now += PROBE_TTL_MS - 1
    const initial = options({ rea })
    const { result, rerender } = renderHook(useMcpProbes, { initialProps: initial })
    expect(result.current.probes.rea).toBe(cached)
    expect(mocks.testMcpServer).not.toHaveBeenCalled()

    const refreshed = success('refreshed')
    mocks.testMcpServer.mockResolvedValue(refreshed)
    now += 1
    rerender({ ...initial, servers: { rea: { ...rea } } })
    await act(async () => {})

    expect(mocks.testMcpServer).toHaveBeenCalledTimes(1)
    expect(result.current.probes.rea).toBe(refreshed)
  })

  it.each(['remove', 'disable'] as const)('retires an in-flight request on %s before re-adding it', async change => {
    const old = deferred()
    const current = deferred()
    mocks.testMcpServer.mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise)
    const initial = options({ rea })
    const { result, rerender } = renderHook(useMcpProbes, { initialProps: initial })
    rerender({ ...initial, servers: change === 'remove' ? {} : { rea: { ...rea, enabled: false } } })
    rerender({ ...initial, servers: { rea } })
    await act(async () => old.resolve(success('obsolete')))

    expect(result.current.probes.rea).toBe('probing')
    expect(probeCache.size).toBe(0)
    const newest = success('newest')
    await act(async () => current.resolve(newest))
    expect(result.current.probes.rea).toBe(newest)
    expect(mocks.testMcpServer).toHaveBeenCalledTimes(2)
  })

  it('invalidates changed in-flight rows through retainProbes while preserving unchanged ones', async () => {
    const old = deferred()
    mocks.testMcpServer.mockImplementation((name: string) =>
      name === 'rea' ? old.promise : Promise.resolve(success('search'))
    )
    const initial = options({ rea, learn })
    const { result } = renderHook(useMcpProbes, { initialProps: initial })
    await act(async () => {})
    const learnResult = result.current.probes.learn
    act(() => result.current.retainProbes({ rea: { ...rea, command: 'new' }, learn }, initial.servers))
    await act(async () => old.resolve(success('obsolete')))

    expect(result.current.probes).toEqual({ learn: learnResult })
    expect(probeCache.has(probeKey('rea', rea, 'default'))).toBe(false)
  })

  it('refreshes on profile key changes and ignores the previous profile completion', async () => {
    const old = deferred()
    const current = deferred()
    mocks.testMcpServer.mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise)
    const initial = options({ rea })
    const { result, rerender } = renderHook(useMcpProbes, { initialProps: initial })
    rerender({ ...initial, scopeProfileKey: 'work', profile: 'work' })
    const newest = success('work-tool')
    await act(async () => current.resolve(newest))
    await act(async () => old.resolve(success('default-tool')))

    expect(result.current.probes.rea).toBe(newest)
    expect(mocks.testMcpServer).toHaveBeenLastCalledWith('rea', 'work')
    expect(probeCache.has(probeKey('rea', rea, 'default'))).toBe(false)
    expect(probeCache.get(probeKey('rea', rea, 'work'))?.result).toBe(newest)
  })

  it('publishes and caches the active request rejection as a failure', async () => {
    mocks.testMcpServer.mockRejectedValue(new Error('ECONNREFUSED'))
    const { result } = renderHook(useMcpProbes, { initialProps: options({ rea }) })
    await act(async () => {})

    expect(result.current.probes.rea).toEqual(failure('ECONNREFUSED'))
    expect(probeCache.get(probeKey('rea', rea, 'default'))?.result).toEqual(failure('ECONNREFUSED'))
  })

  it('ignores completion after a profile epoch change and explicit reset', async () => {
    const old = deferred()
    mocks.testMcpServer.mockReturnValue(old.promise)
    const initial = options({ rea })
    const { result } = renderHook(useMcpProbes, { initialProps: initial })
    act(() => {
      initial.profileEpoch.current += 1
      result.current.resetForProfileSwitch()
    })
    await act(async () => old.resolve(success()))

    expect(result.current.probes).toEqual({})
    expect(probeCache.size).toBe(0)
  })
})
