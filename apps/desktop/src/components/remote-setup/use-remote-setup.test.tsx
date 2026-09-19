import { act, cleanup, renderHook, type RenderHookResult } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionProbeResult } from '@/global'
import { deferred } from '@/test/deferred'

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

it.each<RemoteSetupHost>(['first-run', 'settings', 'registry'])('stale probes and credential tests cannot authorize %s', async (host: RemoteSetupHost): Promise<void> => {
  const probe: ReturnType<typeof deferred<DesktopConnectionProbeResult>> = deferred<DesktopConnectionProbeResult>()
  const tested: ReturnType<typeof deferred<Awaited<ReturnType<Window['hermesDesktop']['testConnectionConfig']>>>> = deferred()

  const bridge = {
    probeConnectionConfig: vi.fn<Window['hermesDesktop']['probeConnectionConfig']>().mockReturnValue(probe.promise),
    testConnectionConfig: vi.fn<Window['hermesDesktop']['testConnectionConfig']>().mockReturnValue(tested.promise)
  } satisfies Pick<Window['hermesDesktop'], 'probeConnectionConfig' | 'testConnectionConfig'>

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: bridge })
  const { result }: RenderHookResult<ReturnType<typeof useRemoteSetup>, void> = renderHook((): ReturnType<typeof useRemoteSetup> => useRemoteSetup({ host }))
  act((): void => { result.current.setAuthMode('oauth'); result.current.setUrl('https://a.example') })
  await act(async (): Promise<void> => { await vi.advanceTimersByTimeAsync(500) })
  act((): void => { result.current.setUrl('not-a-url') })
  await act(async (): Promise<void> => { probe.resolve(probeResult('token', 'old')) })
  expect(result.current.probeStatus).not.toBe('done')
  expect(result.current.canTest).toBe(false)
  expect(result.current.credentials.authMode).toBe('oauth')

  bridge.probeConnectionConfig.mockResolvedValue(probeResult('token', 'new'))
  act((): void => { result.current.setAuthMode('token'); result.current.setUrl('https://b.example'); result.current.setToken('token-a') })
  await act(async (): Promise<void> => { await vi.advanceTimersByTimeAsync(500) })
  expect(result.current.canTest).toBe(true)
  let pending!: Promise<void>
  await act(async (): Promise<void> => { pending = result.current.test() })
  expect(bridge.testConnectionConfig).toHaveBeenCalledExactlyOnceWith({ mode: 'remote', remoteUrl: 'https://b.example', remoteAuthMode: 'token', remoteToken: 'token-a' })
  act((): void => { result.current.setToken('token-b') })
  await act(async (): Promise<void> => { tested.resolve({ ok: true, baseUrl: 'https://b.example', version: null }); await pending })
  expect(result.current.success).toBeNull()
  expect(result.current.testing).toBe(false)
  expect(result.current.canCommit).toBe(host !== 'first-run')
})
