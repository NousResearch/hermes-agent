import { renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import {
  CONNECTION_HEALTH_AREA,
  connectionHealthProviders,
  useConnectionHealthProviders
} from './connection-health'
import { registry } from './registry'
import type { Contribution } from './types'

describe('connectionHealthProviders', () => {
  it('returns only callable providers with host-stamped provenance', () => {
    const load = vi.fn(async () => [])

    const contributions: Contribution[] = [
      {
        area: CONNECTION_HEALTH_AREA,
        data: {
          icon: 'plug',
          load,
          name: 'Demo health',
          repair: { kind: 'route', path: '/settings?tab=plugins', run: 'not allowed' }
        },
        id: 'demo:health',
        source: 'plugin:demo'
      },
      {
        area: CONNECTION_HEALTH_AREA,
        data: { load: 'not-callable' },
        id: 'broken:health',
        source: 'plugin:broken'
      },
      {
        area: CONNECTION_HEALTH_AREA,
        data: { load, repair: { kind: 'command', value: 'rm -rf /' } },
        id: 'unsafe:health',
        source: 'plugin:unsafe'
      }
    ]

    expect(connectionHealthProviders(contributions)).toEqual([
      {
        icon: 'plug',
        id: 'demo:health',
        load: expect.any(Function),
        name: 'Demo health',
        repair: { kind: 'route', path: '/settings?tab=plugins' },
        source: 'plugin:demo'
      },
      {
        id: 'unsafe:health',
        load: expect.any(Function),
        source: 'plugin:unsafe'
      }
    ])
  })

  it('drops malformed repairs returned by a provider load', async () => {
    const load = vi.fn(async () => [
      {
        checkedAt: 1,
        id: 'unsafe',
        name: 'Unsafe',
        reason: 'check_failed' as const,
        repair: { kind: 'command', value: 'rm -rf /' }
      },
      {
        checkedAt: 2,
        id: 'safe',
        name: 'Safe',
        reason: 'auth_required' as const,
        repair: { kind: 'message' as const, message: 'Sign in again', run: 'not allowed' }
      },
      {
        checkedAt: 3,
        id: 'external-route',
        name: 'External route',
        reason: 'check_failed' as const,
        repair: { kind: 'route' as const, path: '//host' }
      }
    ])

    const [provider] = connectionHealthProviders([{
      area: CONNECTION_HEALTH_AREA,
      data: { load },
      id: 'loaded:health',
      source: 'plugin:loaded'
    }])

    await expect(provider.load()).resolves.toEqual([
      {
        checkedAt: 1,
        id: 'unsafe',
        name: 'Unsafe',
        reason: 'check_failed'
      },
      {
        checkedAt: 2,
        id: 'safe',
        name: 'Safe',
        reason: 'auth_required',
        repair: { kind: 'message', message: 'Sign in again' }
      },
      {
        checkedAt: 3,
        id: 'external-route',
        name: 'External route',
        reason: 'check_failed'
      }
    ])
    expect(load).toHaveBeenCalledOnce()
  })

  it('returns only well-formed, whitelisted health results', async () => {
    const load = vi.fn(async () => [
      null,
      { checkedAt: 1, id: 'missing-name', reason: 'healthy' },
      { checkedAt: 1, id: 'unknown-reason', name: 'Unknown', reason: 'made_up' },
      { checkedAt: Number.NaN, id: 'bad-time', name: 'Bad time', reason: 'healthy' },
      {
        checkedAt: 4,
        detail: 123,
        extra: 'not part of the contract',
        icon: false,
        id: 'valid',
        name: 'Valid',
        reason: 'healthy',
        staleAfterMs: -1,
        status: 'Connected'
      }
    ])

    const [provider] = connectionHealthProviders([{
      area: CONNECTION_HEALTH_AREA,
      data: { load },
      id: 'loaded:health',
      source: 'plugin:loaded'
    }])

    await expect(provider.load()).resolves.toEqual([{
      checkedAt: 4,
      id: 'valid',
      name: 'Valid',
      reason: 'healthy',
      status: 'Connected'
    }])
  })

  it('keeps the provider snapshot stable across unrelated rerenders', () => {
    const dispose = registry.register({
      area: CONNECTION_HEALTH_AREA,
      data: { load: async () => [] },
      id: 'stable:health',
      source: 'plugin:stable'
    })

    try {
      const { result, rerender } = renderHook(() => useConnectionHealthProviders())
      const first = result.current

      rerender()

      expect(result.current).toBe(first)
    } finally {
      dispose()
    }
  })
})
