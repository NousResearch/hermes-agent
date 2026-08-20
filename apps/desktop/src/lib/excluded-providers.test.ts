import { describe, expect, it } from 'vitest'

import {
  excludedProviderName,
  isProviderExcluded,
  readExcludedProviders,
  withExcludedProviders,
  withProviderExcluded
} from './excluded-providers'

describe('readExcludedProviders', () => {
  it('reads the model_catalog.excluded_providers list', () => {
    expect(readExcludedProviders({ model_catalog: { excluded_providers: ['copilot', 'openrouter'] } })).toEqual([
      'copilot',
      'openrouter'
    ])
  })

  it('treats a missing key, a missing block, or no config at all as nothing excluded', () => {
    expect(readExcludedProviders({ model_catalog: {} })).toEqual([])
    expect(readExcludedProviders({})).toEqual([])
    expect(readExcludedProviders(undefined)).toEqual([])
  })

  it('tolerates a hand-written scalar instead of a list', () => {
    // `excluded_providers: copilot` is a plausible hand edit of config.yaml.
    expect(readExcludedProviders({ model_catalog: { excluded_providers: 'copilot' } })).toEqual(['copilot'])
  })

  it('drops non-string entries rather than passing them to the picker', () => {
    expect(readExcludedProviders({ model_catalog: { excluded_providers: ['copilot', 7, null, ' '] } })).toEqual([
      'copilot'
    ])
  })
})

describe('excludedProviderName', () => {
  it("strips the picker's custom: prefix and separators so the row reads as before", () => {
    expect(excludedProviderName('custom:second-mock')).toBe('second mock')
    expect(excludedProviderName('my_llm')).toBe('my llm')
  })

  it('leaves a plain slug alone', () => {
    expect(excludedProviderName('copilot')).toBe('copilot')
  })
})

describe('isProviderExcluded', () => {
  it('matches case-insensitively, like the backend blocklist', () => {
    expect(isProviderExcluded(['copilot'], 'Copilot')).toBe(true)
    expect(isProviderExcluded(['Copilot'], 'copilot')).toBe(true)
    expect(isProviderExcluded(['copilot'], 'openrouter')).toBe(false)
  })
})

describe('withProviderExcluded', () => {
  it('appends a newly excluded slug', () => {
    expect(withProviderExcluded(['copilot'], 'openrouter', true)).toEqual(['copilot', 'openrouter'])
  })

  it('never lists the same provider twice', () => {
    expect(withProviderExcluded(['copilot'], 'Copilot', true)).toHaveLength(1)
  })

  it('removes a slug when the provider is re-enabled, case-insensitively', () => {
    expect(withProviderExcluded(['Copilot', 'openrouter'], 'copilot', false)).toEqual(['openrouter'])
  })
})

describe('withExcludedProviders', () => {
  it('writes the list into model_catalog without disturbing the rest of the config', () => {
    const config = { agent: { reasoning_effort: 'high' }, model_catalog: { refresh_hours: 24 } }

    expect(withExcludedProviders(config, ['copilot'])).toEqual({
      agent: { reasoning_effort: 'high' },
      model_catalog: { refresh_hours: 24, excluded_providers: ['copilot'] }
    })
  })

  it('creates the model_catalog block when the config has none', () => {
    expect(withExcludedProviders({}, ['copilot'])).toEqual({ model_catalog: { excluded_providers: ['copilot'] } })
  })

  // `PUT /api/config` deep-merges the body over the stored config, so a dropped
  // key reads as "no opinion" and the old list survives — switching the last
  // provider back on would silently leave it excluded.
  it('writes an explicit empty list when nothing is excluded, so the merge clears it', () => {
    const config = { model_catalog: { excluded_providers: ['copilot'] } }

    expect(withExcludedProviders(config, [])).toEqual({ model_catalog: { excluded_providers: [] } })
  })
})
