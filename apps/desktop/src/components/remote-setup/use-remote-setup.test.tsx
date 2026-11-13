import { act, cleanup, fireEvent, render, renderHook, type RenderHookResult, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionProbeResult } from '@/global'
import { en } from '@/i18n/en'
import { deferred } from '@/test/deferred'

import { RemoteSetupFields } from './fields'
import { useRemoteSetup } from './use-remote-setup'
import type { RemoteSetupHost } from './use-remote-setup'

function probeResult(authMode: 'oauth' | 'token', label: string): DesktopConnectionProbeResult {
  return {
    authMode,
    baseUrl: `https://${label}.example`,
    reachable: true,
    error: null,
    version: null,
    providers: [{ name: label, displayName: label }]
  }
}

beforeEach(() => {
  vi.useFakeTimers()
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
  Reflect.deleteProperty(window, 'hermesDesktop')
})

describe('remote setup owner', () => {
  it.each<RemoteSetupHost>(['first-run', 'settings', 'registry'])(
    'rejects stale probe results in %s despite fresh host callbacks',
    async (host: RemoteSetupHost): Promise<void> => {
      const oldProbe = deferred<DesktopConnectionProbeResult>()
      const newProbe = deferred<DesktopConnectionProbeResult>()
      const probeConnectionConfig = vi.fn().mockReturnValueOnce(oldProbe.promise).mockReturnValueOnce(newProbe.promise)

      Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { probeConnectionConfig } })
      const { result, rerender } = renderHook(() => useRemoteSetup({ host, onNotice: () => {} }))
      act(() => {
        result.current.setAuthMode('oauth')
        result.current.setUrl('https://a.example')
      })
      await act(async () => {
        await vi.advanceTimersByTimeAsync(500)
      })
      act(() => result.current.setUrl('https://b.example'))
      rerender()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(500)
      })
      expect(probeConnectionConfig.mock.calls).toEqual([['https://a.example'], ['https://b.example']])
      await act(async (): Promise<void> => {
        newProbe.resolve(probeResult('oauth', 'new'))
        oldProbe.resolve(probeResult('token', 'old'))
      })
      expect(result.current.payload).toEqual({
        mode: 'remote',
        remoteUrl: 'https://b.example',
        remoteAuthMode: 'oauth',
        remoteToken: undefined
      })
      expect(result.current.providerLabel).toBe('new')
    }
  )

  it('does not publish login completion after the editor unmounts without a probe bridge', async () => {
    const onNotice = vi.fn()
    const pendingLogin = deferred<{ connected: boolean }>()
    const oauthLoginConnectionConfig = vi.fn().mockReturnValue(pendingLogin.promise)

    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { oauthLoginConnectionConfig } })
    const { result, unmount } = renderHook(() => useRemoteSetup({ host: 'registry', onNotice }))
    act(() => {
      result.current.setAuthMode('oauth')
      result.current.setUrl('https://a.example')
    })
    let login!: Promise<void>
    await act(async () => {
      login = result.current.signIn()
    })
    expect(oauthLoginConnectionConfig).toHaveBeenCalledExactlyOnceWith('https://a.example')
    unmount()
    await act(async (): Promise<void> => {
      pendingLogin.resolve({ connected: true })
      await login
    })
    expect(onNotice).not.toHaveBeenCalled()
  })
})

it.each<RemoteSetupHost>(['first-run', 'settings', 'registry'])(
  'stale probes and credential tests cannot authorize %s',
  async (host: RemoteSetupHost): Promise<void> => {
    const probe: ReturnType<typeof deferred<DesktopConnectionProbeResult>> = deferred<DesktopConnectionProbeResult>()

    const tested: ReturnType<typeof deferred<Awaited<ReturnType<Window['hermesDesktop']['testConnectionConfig']>>>> =
      deferred()

    const bridge = {
      probeConnectionConfig: vi.fn<Window['hermesDesktop']['probeConnectionConfig']>().mockReturnValue(probe.promise),
      testConnectionConfig: vi.fn<Window['hermesDesktop']['testConnectionConfig']>().mockReturnValue(tested.promise)
    } satisfies Pick<Window['hermesDesktop'], 'probeConnectionConfig' | 'testConnectionConfig'>

    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: bridge })

    const {
      result
    }: RenderHookResult<ReturnType<typeof useRemoteSetup>, void> = renderHook((): ReturnType<typeof useRemoteSetup> =>
      useRemoteSetup({ host })
    )

    act((): void => {
      result.current.setAuthMode('oauth')
      result.current.setUrl('https://a.example')
    })
    await act(async (): Promise<void> => {
      await vi.advanceTimersByTimeAsync(500)
    })
    act((): void => {
      result.current.setUrl('not-a-url')
    })
    await act(async (): Promise<void> => {
      probe.resolve(probeResult('token', 'old'))
    })
    expect(result.current.probeStatus).not.toBe('done')
    expect(result.current.canTest).toBe(false)
    expect(result.current.credentials.authMode).toBe('oauth')

    bridge.probeConnectionConfig.mockResolvedValue(probeResult('token', 'new'))
    act((): void => {
      result.current.setAuthMode('token')
      result.current.setUrl('https://b.example')
      result.current.setToken('token-a')
    })
    await act(async (): Promise<void> => {
      await vi.advanceTimersByTimeAsync(500)
    })
    expect(result.current.canTest).toBe(true)
    let pending!: Promise<void>
    await act(async (): Promise<void> => {
      pending = result.current.test()
    })
    expect(bridge.testConnectionConfig).toHaveBeenCalledExactlyOnceWith({
      mode: 'remote',
      remoteUrl: 'https://b.example',
      remoteAuthMode: 'token',
      remoteToken: 'token-a'
    })
    act((): void => {
      result.current.setToken('token-b')
    })
    await act(async (): Promise<void> => {
      tested.resolve({ ok: true, baseUrl: 'https://b.example', version: null })
      await pending
    })
    expect(result.current.success).toBeNull()
    expect(result.current.testing).toBe(false)
    expect(result.current.canCommit).toBe(host !== 'first-run')
  }
)

function unreachableProbe(error: string, errorKind?: 'tls'): DesktopConnectionProbeResult {
  return {
    authMode: 'unknown',
    baseUrl: 'https://a.example',
    error,
    errorKind,
    providers: [],
    reachable: false,
    version: null
  }
}

it('shows TLS certificate guidance when the remote probe fails verification', async () => {
  const probeConnectionConfig = vi
    .fn<Window['hermesDesktop']['probeConnectionConfig']>()
    .mockResolvedValue(unreachableProbe('self-signed certificate', 'tls'))

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { probeConnectionConfig } })
  const { result } = renderHook(() => useRemoteSetup({ host: 'settings' }))

  act(() => result.current.setUrl('https://a.example'))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(500)
  })

  expect(result.current.probeStatus).toBe('error')
  expect(result.current.probeErrorKind).toBe('tls')
  expect(result.current.credentials.authMode).toBe('token')
  expect(result.current.credentials.oauthConnected).toBe(false)

  await act(async () => {
    await result.current.test()
  })

  expect(result.current.error).toBe(en.settings.gateway.probeTlsError)
})

it('keeps non-TLS probe failures on the generic unreachable path', async () => {
  const probeConnectionConfig = vi
    .fn<Window['hermesDesktop']['probeConnectionConfig']>()
    .mockResolvedValue(unreachableProbe('connect ECONNREFUSED 10.0.0.1:443'))

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { probeConnectionConfig } })
  const { result } = renderHook(() => useRemoteSetup({ host: 'settings' }))

  act(() => result.current.setUrl('https://a.example'))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(500)
  })

  expect(result.current.probeStatus).toBe('error')
  expect(result.current.probeErrorKind).toBeUndefined()

  await act(async () => {
    await result.current.test()
  })

  expect(result.current.error).toBe('connect ECONNREFUSED 10.0.0.1:443')
  expect(result.current.credentials.authMode).toBe('token')
  expect(result.current.credentials.oauthConnected).toBe(false)
})

it.each<RemoteSetupHost>(['first-run', 'settings'])(
  'renders certificate guidance in the %s connection editor',
  async host => {
    const probeConnectionConfig = vi
      .fn<Window['hermesDesktop']['probeConnectionConfig']>()
      .mockResolvedValue(unreachableProbe('self-signed certificate', 'tls'))

    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { probeConnectionConfig } })

    function Editor() {
      const setup = useRemoteSetup({ host })

      return <RemoteSetupFields setup={setup} />
    }

    render(<Editor />)
    const copy = host === 'first-run' ? en.install : en.settings.gateway
    fireEvent.change(screen.getByRole('textbox', { name: copy.remoteUrlTitle }), {
      target: { value: 'https://a.example' }
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })

    expect(screen.getByText(copy.probeTlsError)).not.toBeNull()
    expect(screen.queryByText(copy.probeError)).toBeNull()
    expect(screen.queryByRole('button', { name: /sign in/i })).toBeNull()
  }
)
