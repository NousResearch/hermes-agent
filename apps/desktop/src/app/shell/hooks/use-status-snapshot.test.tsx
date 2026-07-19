import { cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'

import { useStatusSnapshot } from './use-status-snapshot'

afterEach(() => {
  cleanup()
  $activeGatewayProfile.set('default')
  $connection.set(null)
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

describe('useStatusSnapshot', () => {
  it('polls readiness for the active named profile', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        api: vi.fn(async () => ({ gateway_running: true }))
      }
    })

    $activeGatewayProfile.set('coder')
    $connection.set({ mode: 'remote', source: 'settings' } as never)

    const calls: Array<[string, Record<string, unknown> | undefined]> = []

    const requestGateway = async <T = unknown,>(method: string, params?: Record<string, unknown>): Promise<T> => {
      calls.push([method, params])

      return (
        method === 'setup.status'
          ? { profile_name: 'coder', provider_configured: true }
          : { ok: true, profile_name: 'coder' }
      ) as T
    }

    renderHook(() => useStatusSnapshot('open', requestGateway))

    await waitFor(() => expect(calls).toHaveLength(2))
    expect(calls).toEqual([
      ['setup.status', { profile: 'coder' }],
      ['setup.runtime_check', { profile: 'coder' }]
    ])
  })

  it('does not accept unscoped readiness from an old app-global remote backend', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api: vi.fn(async () => ({ gateway_running: true })) }
    })
    $connection.set({ mode: 'remote', source: 'settings' } as never)

    const requestGateway = async <T = unknown,>(method: string): Promise<T> =>
      (method === 'setup.status' ? { provider_configured: true } : { ok: true }) as T

    const { result } = renderHook(() => useStatusSnapshot('open', requestGateway))

    await waitFor(() => expect(result.current.inferenceStatus).not.toBeNull())
    expect(result.current.inferenceStatus?.ready).toBe(false)
    expect(result.current.inferenceStatus?.reason).toMatch(/update/i)
  })
})
