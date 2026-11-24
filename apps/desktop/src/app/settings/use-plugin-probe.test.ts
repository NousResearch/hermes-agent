import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ProbeResult } from './plugin-install-run'
import { usePluginProbe } from './use-plugin-probe'

const found = (agentName: string): ProbeResult => ({ ok: true, agent: true, agentName, desktop: false, warnings: [] })

const deferred = <T>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => (resolve = done))

  return { promise, resolve }
}

afterEach(() => vi.unstubAllGlobals())

describe('usePluginProbe', () => {
  it('answers unavailable when neither Electron nor the gateway can inspect', async () => {
    const { result } = renderHook(() => usePluginProbe('unavailable here'))

    act(() => result.current.start('owner/repo'))

    await waitFor(() =>
      expect(result.current.probe).toMatchObject({ ok: false, unavailable: true, error: 'unavailable here' })
    )
  })

  it('keeps the newest inspection when an older gateway answer lands late', async () => {
    const first = deferred<ProbeResult>()
    const second = deferred<ProbeResult>()
    const gateway = vi.fn((repo: string) => (repo === 'a' ? first.promise : second.promise))
    const { result } = renderHook(() => usePluginProbe('unavailable', gateway))

    act(() => result.current.start('a'))
    act(() => result.current.start('b'))
    const settled = result.current.settled()
    await act(async () => second.resolve(found('b')))
    await act(async () => first.resolve(found('a')))

    expect(result.current.probe?.agentName).toBe('b')
    expect((await settled)?.agentName).toBe('b')
  })

  it('drops an answer that lands after cancel', async () => {
    const late = deferred<ProbeResult>()
    const { result } = renderHook(() => usePluginProbe('unavailable', () => late.promise))

    act(() => result.current.start('a'))
    act(() => result.current.cancel())
    await act(async () => late.resolve(found('a')))

    expect(result.current.probe).toBeNull()
    expect(await result.current.settled()).toBeNull()
  })

  it('prefers Electron inspection when the preload provides it', async () => {
    const probePluginRepo = vi.fn(async () => found('native'))
    const gateway = vi.fn(async () => found('gateway'))
    vi.stubGlobal('hermesDesktop', { probePluginRepo })
    const { result } = renderHook(() => usePluginProbe('unavailable', gateway))

    act(() => result.current.start('owner/repo'))

    await waitFor(() => expect(result.current.probe?.agentName).toBe('native'))
    expect(probePluginRepo).toHaveBeenCalledWith({ identifier: 'owner/repo' })
    expect(gateway).not.toHaveBeenCalled()
  })

  it('does not restart an inspection when the caller passes a fresh gateway closure each render', async () => {
    const gateway = vi.fn(async (_repo: string) => found('one'))
    const { result, rerender } = renderHook(() => usePluginProbe('unavailable', repo => gateway(repo)))

    act(() => result.current.start('owner/repo'))
    await waitFor(() => expect(result.current.probe?.agentName).toBe('one'))
    const start = result.current.start
    rerender()

    expect(result.current.start).toBe(start)
    expect(gateway).toHaveBeenCalledTimes(1)
  })
})
