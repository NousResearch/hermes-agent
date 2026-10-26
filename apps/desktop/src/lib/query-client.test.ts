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
})
