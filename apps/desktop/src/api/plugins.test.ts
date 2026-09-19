import { afterEach, describe, expect, it, vi } from 'vitest'

import { getApiRequestConnection, getApiRequestProfile, setApiRequestConnection, setApiRequestProfile } from './client'
import { activeConnection, pluginRest } from './plugins'

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

// A wedged main-process round-trip must not hang pluginSocket forever (#93454).
describe('activeConnection connection timeout (#93454)', () => {
  it('rejects instead of hanging forever when getConnection() wedges', async () => {
    vi.useFakeTimers()
    setApiRequestProfile('coder')
    vi.stubGlobal('hermesDesktop', { getConnection: vi.fn(() => new Promise(() => undefined)) })

    const pending = expect(activeConnection()).rejects.toThrow('Timed out connecting to profile "coder"')

    await vi.advanceTimersByTimeAsync(20_000)
    await pending
  })

  it('rejects instead of hanging forever when getConnectionFor() wedges', async () => {
    vi.useFakeTimers()
    setApiRequestConnection('gw-tailscale')
    setApiRequestProfile('research')
    vi.stubGlobal('hermesDesktop', {
      getConnection: vi.fn(() => new Promise(() => undefined)),
      getConnectionFor: vi.fn(() => new Promise(() => undefined))
    })

    const pending = expect(activeConnection()).rejects.toThrow('Timed out connecting to profile "research"')

    await vi.advanceTimersByTimeAsync(20_000)
    await pending
  })
})

describe('plugin REST source ownership', () => {
  it('pins concurrent reads while foreground scope changes, including explicit local and legacy callers', async () => {
    const requests: Array<Record<string, unknown>> = []

    const api = vi.fn(async (request: Record<string, unknown>) => {
      requests.push(request)
      await Promise.resolve()

      return request.connectionId
    })

    vi.stubGlobal('hermesDesktop', { api })
    setApiRequestConnection('remote-a')
    setApiRequestProfile('chat')
    const source = { connectionId: 'remote-b', profile: 'worker' }
    const remote = pluginRest('kanban', '/board?board=shared', { source })
    source.connectionId = 'removed'
    const local = pluginRest('kanban', '/board?board=shared', { source: { connectionId: 'local', profile: 'worker' } })
    setApiRequestConnection('remote-c')
    setApiRequestProfile('another-chat')
    const legacy = pluginRest('kanban', '/boards')
    expect(await Promise.all([remote, local, legacy])).toEqual(['remote-b', 'local', 'remote-c'])
    expect(requests.map(r => [r.connectionId, r.profile])).toEqual([
      ['remote-b', 'worker'],
      ['local', 'worker'],
      ['remote-c', 'another-chat']
    ])
    expect(getApiRequestConnection()).toBe('remote-c')
    expect(getApiRequestProfile()).toBe('another-chat')
  })

  it('fails closed for invalid selectors and namespace escapes and propagates registry/auth denials without fallback', async () => {
    const api = vi.fn().mockRejectedValue(new Error('No connection with id "removed".'))
    vi.stubGlobal('hermesDesktop', { api })

    for (const path of ['/../other', '/%2e%2e/other', '/%252e%252e/other', '/..\\other', '/%2e%2e%2fother']) {
      await expect(
        pluginRest('kanban', path, { source: { connectionId: 'local', profile: 'worker' } })
      ).rejects.toThrow(/path/i)
    }

    for (const source of [
      { connectionId: '', profile: 'worker' },
      { connectionId: 'local', profile: '' }
    ]) {
      await expect(pluginRest('kanban', '/boards', { source })).rejects.toThrow(/source/i)
    }

    expect(api).not.toHaveBeenCalled()
    await expect(
      pluginRest('kanban', '/boards', { source: { connectionId: 'removed', profile: 'worker' } })
    ).rejects.toThrow(/No connection/)
    api.mockRejectedValueOnce(new Error('403: forbidden'))
    await expect(
      pluginRest('kanban', '/boards', { source: { connectionId: 'denied', profile: 'worker' } })
    ).rejects.toThrow('403')
    expect(api).toHaveBeenCalledTimes(2)
  })
})
