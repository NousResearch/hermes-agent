import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getStatus } from '@/hermes'
import { $setupReadyTick, notifySetupReady } from '@/store/live-sync'

import { deferred } from '../../../test/deferred'

import { useStatusSnapshot } from './use-status-snapshot'

const reportBackendCodeSkewSpy = vi.fn()

vi.mock('@/hermes', () => ({
  getStatus: vi.fn()
}))

// The store under test reaches the real notification/store graph; mock the one
// export the hook calls so the assertion is about the hook's wiring.
vi.mock('@/store/updates', () => ({
  reportBackendCodeSkew: (...args: unknown[]) => reportBackendCodeSkewSpy(...args)
}))

type GatewayRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

async function flushAsync() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.spyOn(document, 'hasFocus').mockReturnValue(true)
  vi.mocked(getStatus)
    .mockReset()
    .mockResolvedValue({} as never)
  $setupReadyTick.set(0)
  reportBackendCodeSkewSpy.mockClear()
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('useStatusSnapshot', () => {
  it('pauses status RPCs while visible but unfocused, then catches up on focus', async () => {
    vi.mocked(document.hasFocus).mockReturnValue(false)
    const requestGateway = vi.fn().mockResolvedValue({}) as unknown as GatewayRequester

    renderHook(() => useStatusSnapshot('open', requestGateway))
    await flushAsync()

    expect(getStatus).not.toHaveBeenCalled()
    expect(requestGateway).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000)
    })
    expect(getStatus).not.toHaveBeenCalled()

    vi.mocked(document.hasFocus).mockReturnValue(true)
    window.dispatchEvent(new Event('focus'))
    await flushAsync()

    expect(getStatus).toHaveBeenCalledOnce()
    // One refresh round = setup.status + setup.runtime_check + free_tier.status.
    expect(requestGateway).toHaveBeenCalledTimes(3)
  })

  it('keeps the last authoritative readiness through a transient RPC failure', async () => {
    let refresh = 0

    const requestGatewayMock = vi.fn(async (method: string) => {
      const cycle = Math.floor(refresh / 2)
      refresh += 1

      if (cycle > 0) {
        throw new Error(`${method} timed out`)
      }

      return (method === 'setup.runtime_check' ? { ok: true } : { provider_configured: true }) as never
    })

    const requestGateway = requestGatewayMock as unknown as GatewayRequester

    const { result } = renderHook(() => useStatusSnapshot('open', requestGateway))

    await flushAsync()
    expect(result.current.inferenceStatus).toMatchObject({ ready: true, source: 'runtime_check' })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000)
    })

    expect(result.current.inferenceStatus).toMatchObject({ ready: true, source: 'runtime_check' })
  })

  it('does not present an initial transport failure as inference not ready', async () => {
    const requestGatewayMock = vi.fn(async (method: string) => {
      throw new Error(`${method} connection closed`)
    })

    const requestGateway = requestGatewayMock as unknown as GatewayRequester

    const { result } = renderHook(() => useStatusSnapshot('open', requestGateway))

    await flushAsync()

    expect(result.current.inferenceStatus).toBeNull()
  })

  it('still publishes an authoritative runtime failure', async () => {
    const requestGatewayMock = vi.fn(
      async (method: string) =>
        (method === 'setup.runtime_check'
          ? { error: 'No usable credentials found for nous.', ok: false }
          : { provider_configured: true }) as never
    )

    const requestGateway = requestGatewayMock as unknown as GatewayRequester

    const { result } = renderHook(() => useStatusSnapshot('open', requestGateway))

    await flushAsync()

    expect(result.current.inferenceStatus).toMatchObject({
      ready: false,
      reason: expect.stringContaining('No usable credentials found for nous.'),
      source: 'runtime_check'
    })
  })

  it('clears readiness immediately when the gateway disconnects', async () => {
    const pendingStatus = deferred<never>()

    vi.mocked(getStatus)
      .mockResolvedValueOnce({} as never)
      .mockReturnValueOnce(pendingStatus.promise)

    const requestGateway = vi.fn(
      async (method: string) =>
        (method === 'setup.runtime_check' ? { ok: true } : { provider_configured: true }) as never
    ) as unknown as GatewayRequester

    const { rerender, result } = renderHook(({ gatewayState }) => useStatusSnapshot(gatewayState, requestGateway), {
      initialProps: { gatewayState: 'open' }
    })

    await flushAsync()
    expect(result.current.inferenceStatus).toMatchObject({ ready: true, source: 'runtime_check' })

    rerender({ gatewayState: 'connecting' })

    expect(getStatus).toHaveBeenCalledTimes(2)
    expect(result.current.inferenceStatus).toBeNull()
  })

  it('refreshes readiness by source and ignores the previous backend response', async () => {
    const workRuntime = deferred<unknown>()
    const workSetup = deferred<unknown>()
    const homeRuntime = deferred<unknown>()
    const homeSetup = deferred<unknown>()
    let source = 'work'

    const requestGateway = vi.fn((method: string) => {
      if (source === 'work') {
        return method === 'setup.runtime_check' ? workRuntime.promise : workSetup.promise
      }

      return method === 'setup.runtime_check' ? homeRuntime.promise : homeSetup.promise
    }) as unknown as GatewayRequester

    const { rerender, result } = renderHook(({ scope }) => useStatusSnapshot('open', requestGateway, scope), {
      initialProps: { scope: 'work\0default' }
    })

    await flushAsync()
    source = 'home'
    rerender({ scope: 'home\0default' })
    await flushAsync()

    expect(result.current.inferenceStatus).toBeNull()

    await act(async () => {
      homeRuntime.resolve({ ok: true })
      homeSetup.resolve({ provider_configured: true })
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(result.current.inferenceStatus).toMatchObject({ ready: true, source: 'runtime_check' })

    await act(async () => {
      workRuntime.resolve({ error: 'stale backend', ok: false })
      workSetup.resolve({ provider_configured: false })
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(result.current.inferenceStatus).toMatchObject({ ready: true, source: 'runtime_check' })
  })

  it('waits for a slow refresh to settle before scheduling another one', async () => {
    const setup = deferred<unknown>()
    const runtime = deferred<unknown>()

    const requestGatewayMock = vi.fn(
      (method: string) => (method === 'setup.runtime_check' ? runtime.promise : setup.promise) as never
    )

    const requestGateway = requestGatewayMock as unknown as GatewayRequester

    renderHook(() => useStatusSnapshot('open', requestGateway))
    await flushAsync()

    // Open runs the readiness legs once: setup.status, setup.runtime_check, free_tier.status.
    expect(getStatus).toHaveBeenCalledOnce()
    expect(requestGatewayMock).toHaveBeenCalledTimes(3)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000)
    })

    expect(getStatus).toHaveBeenCalledOnce()
    expect(requestGatewayMock).toHaveBeenCalledTimes(3)

    await act(async () => {
      setup.resolve({ provider_configured: true })
      runtime.resolve({ ok: true })
      await vi.advanceTimersByTimeAsync(0)
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(59_999)
    })
    expect(getStatus).toHaveBeenCalledOnce()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })

    // The periodic tick is status-only: readiness and the free-tier verdict
    // arrive by `setup.ready` push plus the one-shots on open and on return.
    expect(getStatus).toHaveBeenCalledTimes(2)
    expect(requestGatewayMock).toHaveBeenCalledTimes(3)
  })

  it('re-reads readiness and the free-tier verdict once per setup.ready, off the status tick', async () => {
    const requestGatewayMock = vi.fn(
      async (method: string) =>
        (method === 'setup.runtime_check' ? { ok: true } : { provider_configured: true }) as never
    )

    const requestGateway = requestGatewayMock as unknown as GatewayRequester

    renderHook(() => useStatusSnapshot('open', requestGateway))
    await flushAsync()
    requestGatewayMock.mockClear()
    vi.mocked(getStatus).mockClear()

    await act(async () => {
      notifySetupReady()
      await vi.advanceTimersByTimeAsync(0)
    })

    const methods = requestGatewayMock.mock.calls.map(([method]) => method)
    expect(methods.filter(method => method === 'free_tier.status')).toHaveLength(1)
    expect(methods.filter(method => method === 'setup.runtime_check')).toHaveLength(1)
    expect(methods.filter(method => method === 'setup.status')).toHaveLength(1)
    expect(getStatus).not.toHaveBeenCalled()
  })

  it('ignores setup.ready while the gateway is not open', async () => {
    const requestGatewayMock = vi.fn(async () => ({}) as never)
    const requestGateway = requestGatewayMock as unknown as GatewayRequester

    renderHook(() => useStatusSnapshot('connecting', requestGateway))
    await flushAsync()

    await act(async () => {
      notifySetupReady()
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(requestGatewayMock).not.toHaveBeenCalled()
  })

  it('keeps the same snapshot reference across a content-equal 60s re-read; a real change publishes', async () => {
    vi.mocked(getStatus).mockImplementation(async () => ({ version: '1.0.0' }) as never)
    const requestGateway = vi.fn().mockResolvedValue({}) as unknown as GatewayRequester

    const { result } = renderHook(() => useStatusSnapshot('open', requestGateway))
    await flushAsync()

    const first = result.current.statusSnapshot
    expect(first).toEqual({ version: '1.0.0' })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000)
    })
    expect(getStatus).toHaveBeenCalledTimes(2)
    // Identity preserved on a no-op: consumers keyed on the snapshot must not re-render.
    expect(result.current.statusSnapshot).toBe(first)

    vi.mocked(getStatus).mockImplementation(async () => ({ version: '1.0.1' }) as never)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000)
    })
    expect(result.current.statusSnapshot).toEqual({ version: '1.0.1' })
  })
})

// Regression for #118998: after an external `hermes update` the app-owned backend
// keeps serving pre-update code, and the only signal used to be a guarded endpoint
// refusing with 503 — invisible to a user who never opens Settings -> Model. The
// backend publishes the skew on /api/status, which this hook already polls, so the
// proactive toast must be reported from here.
describe('useStatusSnapshot code skew', () => {
  it('reports skew as soon as a status poll carries it', async () => {
    vi.mocked(getStatus).mockImplementation(
      async () => ({ version: '1.0.0', code_skew: { boot_rev: 'abc1234567', disk_rev: 'def4567890' } }) as never
    )
    const requestGateway = vi.fn().mockResolvedValue({}) as unknown as GatewayRequester

    renderHook(() => useStatusSnapshot('open', requestGateway))
    await flushAsync()

    expect(reportBackendCodeSkewSpy).toHaveBeenCalledWith({
      boot_rev: 'abc1234567',
      disk_rev: 'def4567890'
    })
  })

  it('reports no skew when the backend is current (no false positive)', async () => {
    vi.mocked(getStatus).mockResolvedValue({ version: '1.0.0' } as never)
    const requestGateway = vi.fn().mockResolvedValue({}) as unknown as GatewayRequester

    renderHook(() => useStatusSnapshot('open', requestGateway))
    await flushAsync()

    expect(reportBackendCodeSkewSpy).toHaveBeenCalledWith(undefined)
  })

  it('stops reporting once the backend catches up after a recycle', async () => {
    vi.mocked(getStatus).mockImplementation(
      async () => ({ version: '1.0.0', code_skew: { boot_rev: 'abc1234567', disk_rev: 'def4567890' } }) as never
    )
    const requestGateway = vi.fn().mockResolvedValue({}) as unknown as GatewayRequester

    renderHook(() => useStatusSnapshot('open', requestGateway))
    await flushAsync()

    // Backend recycled onto the new code: the field is absent again.
    vi.mocked(getStatus).mockImplementation(async () => ({ version: '1.0.1' }) as never)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000)
    })

    expect(reportBackendCodeSkewSpy).toHaveBeenLastCalledWith(undefined)
  })
})
