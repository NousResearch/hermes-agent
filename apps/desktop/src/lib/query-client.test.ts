import type { ModelOptionsResult } from '@hermes/shared'
import { beforeEach, describe, expect, it } from 'vitest'

import { invalidateProfileScopedQueries, keepWarmNousRow, queryClient } from './query-client'

function invalidated(key: unknown[]): boolean {
  return queryClient.getQueryState(key)?.isInvalidated ?? false
}

describe('invalidateProfileScopedQueries', () => {
  beforeEach(() => {
    queryClient.clear()
  })

  it('invalidates profile-scoped caches and leaves account/global caches intact', () => {
    const profileScoped = [
      ['hermes-config-record'],
      ['hermes-config-schema'],
      ['skills-list'],
      ['toolsets-list'],
      ['model-options', 'global'],
      ['command-palette', 'sessions'],
      ['session-picker', 'sessions']
    ]

    const global = [
      ['billing', 'state'],
      ['billing', 'subscription'],
      ['marketplace-themes', 'all'],
      ['marketplace-themes-settings', 'x'],
      ['onboarding-model-options', 'y'],
      ['contrib-logs-tail']
    ]

    for (const key of [...profileScoped, ...global]) {
      queryClient.setQueryData(key, { seeded: true })
    }

    invalidateProfileScopedQueries()

    for (const key of profileScoped) {
      expect(invalidated(key), `${JSON.stringify(key)} should be invalidated`).toBe(true)
    }

    for (const key of global) {
      expect(invalidated(key), `${JSON.stringify(key)} should be left intact`).toBe(false)
    }
  })

  it('invalidates unknown/non-string-rooted keys by default (correctness-safe)', () => {
    queryClient.setQueryData(['some-future-profile-query'], 1)
    queryClient.setQueryData([{ scope: 'weird' }], 1)

    invalidateProfileScopedQueries()

    expect(invalidated(['some-future-profile-query'])).toBe(true)
    expect(invalidated([{ scope: 'weird' }])).toBe(true)
  })
})

const warm: ModelOptionsResult = {
  providers: [
    {
      models: ['example/model-a', 'example/model-free:free'],
      name: 'Nous Portal',
      slug: 'nous',
      unavailable_models: [],
      usage_levels: { 'example/model-a': 'top_tier', 'example/model-free:free': 'free' }
    }
  ]
}

const pending: ModelOptionsResult = {
  providers: [
    {
      free_tier_pending: true,
      models: ['example/model-a', 'example/model-free:free'],
      name: 'Nous Portal',
      slug: 'nous',
      unavailable_models: ['example/model-a', 'example/model-free:free']
    }
  ]
}

describe('keepWarmNousRow', () => {
  it('keeps the last warm Large / Free tags while a cache-only read is still pending', () => {
    const merged = keepWarmNousRow(warm, pending) as ModelOptionsResult
    const row = merged.providers?.[0]

    expect(row?.usage_levels).toEqual(warm.providers?.[0].usage_levels)
    expect(row?.unavailable_models).toEqual([])
    // Still pending, so the menu keeps re-reading until the warm row lands.
    expect(row?.free_tier_pending).toBe(true)
  })

  it('takes a warm reply as is, and leaves a first pending reply alone', () => {
    expect(keepWarmNousRow(pending, warm)).toEqual(warm)
    expect(keepWarmNousRow(undefined, pending)).toEqual(pending)
  })

  it('keeps the warm tags across several pending replies in a row', () => {
    const once = keepWarmNousRow(warm, pending)
    const twice = keepWarmNousRow(once, structuredClone(pending)) as ModelOptionsResult
    const row = twice.providers?.[0]

    expect(row?.usage_levels).toEqual(warm.providers?.[0].usage_levels)
    expect(row?.unavailable_models).toEqual([])
  })

  it("keeps the warm subscription-billed ids (no Free tag) on the pending row's fresh plan_usage", () => {
    const gauges = [{ label: 'day', left_fraction: 0.5, style: 'bar' }]
    const fresh = [{ label: 'day', left_fraction: 0.2, style: 'bar' }]

    const warmSubscribed: ModelOptionsResult = {
      providers: [
        {
          ...warm.providers![0],
          plan_usage: { gauges, kind: 'usage_window', subscription_model_ids: ['example/model-sub'] },
          usage_levels: { ...warm.providers![0].usage_levels, 'example/model-sub': 'free' }
        }
      ]
    }

    const pendingSubscribed: ModelOptionsResult = {
      providers: [{ ...pending.providers![0], plan_usage: { gauges: fresh, kind: 'usage_window' } }]
    }

    const row = (keepWarmNousRow(warmSubscribed, pendingSubscribed) as ModelOptionsResult).providers?.[0]

    expect(row?.usage_levels?.['example/model-sub']).toBe('free')
    expect(row?.plan_usage?.subscription_model_ids).toEqual(['example/model-sub'])
    // The gauges are the pending row's own (fresh), not the warm row's.
    expect(row?.plan_usage?.gauges).toEqual(fresh)
  })

  it('keeps warm per-model pricing for PAYG/credit Nous rows across repeated pending replies', () => {
    const warmPayg: ModelOptionsResult = {
      providers: [
        {
          free_tier: false,
          models: ['anthropic/claude-sonnet-5'],
          name: 'Nous Portal',
          pricing: {
            'anthropic/claude-sonnet-5': { cache: '$0.16', free: false, input: '$1.60', output: '$8.00' }
          },
          slug: 'nous',
          unavailable_models: []
        }
      ]
    }

    const pendingPayg: ModelOptionsResult = {
      providers: [
        {
          free_tier: false,
          models: ['anthropic/claude-sonnet-5'],
          name: 'Nous Portal',
          pricing_pending: true,
          slug: 'nous',
          unavailable_models: []
        }
      ]
    }

    const once = keepWarmNousRow(warmPayg, pendingPayg) as ModelOptionsResult
    const twice = keepWarmNousRow(once, structuredClone(pendingPayg)) as ModelOptionsResult

    expect(once.providers?.[0].pricing).toEqual(warmPayg.providers?.[0].pricing)
    expect(once.providers?.[0].pricing_pending).toBe(true)
    expect(twice.providers?.[0].pricing).toEqual(warmPayg.providers?.[0].pricing)
    expect(twice.providers?.[0].pricing_pending).toBe(true)
  })

  it('does not overlay a warm PAYG pricing row onto a pending usage-window plan row', () => {
    const warmPayg: ModelOptionsResult = {
      providers: [
        {
          free_tier: false,
          models: ['example/model-a'],
          name: 'Nous Portal',
          pricing: {
            'example/model-a': { cache: null, free: false, input: '$1.60', output: '$8.00' }
          },
          slug: 'nous',
          unavailable_models: []
        }
      ]
    }

    const pendingWindow: ModelOptionsResult = {
      providers: [
        {
          models: ['example/model-a'],
          name: 'Nous Portal',
          plan_usage: { gauges: [{ label: 'day', left_fraction: 0.9, style: 'green' }], kind: 'usage_window' },
          pricing_pending: true,
          slug: 'nous'
        }
      ]
    }

    const merged = keepWarmNousRow(warmPayg, pendingWindow) as ModelOptionsResult
    const row = merged.providers?.[0]

    expect(row?.pricing).toBeUndefined()
    expect(row?.plan_usage?.kind).toBe('usage_window')
  })

  it('ignores non-object previous data and free_tier_row guests safely', () => {
    expect(keepWarmNousRow('unexpected-primitive', pending)).toEqual(pending)

    const guestWarm: ModelOptionsResult = {
      providers: [
        {
          free_tier_row: true,
          models: ['nous/welcome'],
          name: 'Nous · free tier',
          pricing: { 'nous/welcome': { cache: null, free: true, input: 'free', output: 'free' } },
          slug: 'nous'
        }
      ]
    }

    const merged = keepWarmNousRow(guestWarm, pending) as ModelOptionsResult

    expect(merged.providers?.[0].pricing).toBeUndefined()
  })
})
