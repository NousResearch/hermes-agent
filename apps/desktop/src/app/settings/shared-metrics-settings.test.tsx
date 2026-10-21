// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as GatewayStore from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import { $settingsOwner } from '@/store/settings-scope'
import type { SharedMetricsConsent } from '@/store/shared-metrics'

const mocks = vi.hoisted(() => ({
  gate: vi.fn(),
  notifyError: vi.fn(),
  request: vi.fn()
}))

// The real gateway store, with only the per-agent request routed to the fake backend below.
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  requestGatewayForAgent: (...args: unknown[]) => mocks.request(...args)
}))
vi.mock('@/store/desktop-metrics', () => ({ recordOnboarding: vi.fn(), setDesktopMetricsGate: mocks.gate }))
vi.mock('@/store/notifications', () => ({ notifyError: mocks.notifyError }))

const { en } = await import('@/i18n/en')
const { SharedMetricsSettings } = await import('./shared-metrics-settings')

const copy = en.sharedMetrics

function backend(initial: SharedMetricsConsent) {
  let stored = initial

  mocks.request.mockImplementation(
    async (_connection: unknown, _profile: unknown, method: string, params: Record<string, unknown> = {}) => {
      if (method === 'shared_metrics.set') {
        const enabled = params.enabled === true
        stored = { enabled, send: enabled && params.send === true, decided: true }
      }

      return stored
    }
  )
}

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(yes => {
    resolve = yes
  })

  return { promise, resolve }
}

const setCalls = () =>
  mocks.request.mock.calls
    .filter(call => call[2] === 'shared_metrics.set')
    .map(call => ({ enabled: call[3].enabled, send: call[3].send }))

beforeEach(() => {
  $activeGatewayProfile.set('default')
  $connection.set({
    baseUrl: 'https://original.example',
    connectionId: 'same-id',
    mode: 'remote',
    profile: 'default',
    token: 'original-token'
  } as never)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $connection.set(null)
})

describe('SharedMetricsSettings', () => {
  it('is one switch: on shares (both opt-ins), off clears both', async () => {
    backend({ enabled: false, send: false, decided: true })

    render(<SharedMetricsSettings />)

    const share = await screen.findByRole('switch', { name: copy.sendLabel })
    await waitFor(() => expect(share.hasAttribute('disabled')).toBe(false))
    expect(screen.getAllByRole('switch')).toHaveLength(1)
    expect(share.getAttribute('aria-checked')).toBe('false')

    fireEvent.click(share)
    await waitFor(() => expect(share.getAttribute('aria-checked')).toBe('true'))

    fireEvent.click(share)
    await waitFor(() => expect(share.getAttribute('aria-checked')).toBe('false'))

    expect(setCalls()).toEqual([
      { enabled: true, send: true },
      { enabled: false, send: false }
    ])
  })

  it('reads a retired collect-only answer as not sharing', async () => {
    backend({ enabled: true, send: false, decided: true })

    render(<SharedMetricsSettings />)

    const share = await screen.findByRole('switch', { name: copy.sendLabel })
    await waitFor(() => expect(share.hasAttribute('disabled')).toBe(false))
    expect(share.getAttribute('aria-checked')).toBe('false')
  })
})

describe('SharedMetricsSettings owner isolation', () => {
  it('pins consent writes to the captured owner and ignores their stale settlement', async () => {
    const save = deferred<SharedMetricsConsent>()
    mocks.request.mockImplementation(async (_connectionId, _profile, method) => {
      if (method === 'shared_metrics.set') {
        return save.promise
      }

      return { decided: true, enabled: false, send: false }
    })

    render(<SharedMetricsSettings />)
    const share = await screen.findByRole('switch', { name: copy.sendLabel })
    await waitFor(() => expect((share as HTMLButtonElement).disabled).toBe(false))
    const capturedOwner = $settingsOwner.get()!

    fireEvent.click(share)
    await waitFor(() =>
      expect(mocks.request).toHaveBeenCalledWith(
        'same-id',
        'default',
        'shared_metrics.set',
        expect.objectContaining({ enabled: true, profile: 'default', send: true }),
        undefined,
        undefined,
        expect.objectContaining({
          expectedConnectionOwner: capturedOwner.cacheOwner ?? capturedOwner.connectionOwner,
          expectedOwner: expect.objectContaining({ connectionOwner: capturedOwner.connectionOwner }),
          spawnPriority: 'foreground'
        })
      )
    )

    act(() => {
      $connection.set({
        baseUrl: 'https://replacement.example',
        connectionId: 'same-id',
        mode: 'remote',
        profile: 'default',
        token: 'replacement-token'
      } as never)
    })
    await act(async () => save.resolve({ decided: true, enabled: true, send: true }))

    const replacementOwner = $settingsOwner.get()!
    expect(replacementOwner).not.toBe(capturedOwner)
    await waitFor(() => expect((share as HTMLButtonElement).disabled).toBe(false))
    expect(
      mocks.request.mock.calls.filter(
        ([, , method, , , , options]) =>
          method === 'shared_metrics.set' && options?.expectedConnectionOwner === replacementOwner.cacheOwner
      )
    ).toHaveLength(0)
    expect(mocks.gate).not.toHaveBeenCalled()
    expect(mocks.notifyError).not.toHaveBeenCalled()
  })
})
