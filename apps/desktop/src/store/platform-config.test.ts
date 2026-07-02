import { describe, expect, it } from 'vitest'

import { configYamlEntries, shouldApplyClientConfig } from './platform-config'

describe('shouldApplyClientConfig', () => {
  it('applies only strictly newer positive-integer versions', () => {
    expect(shouldApplyClientConfig(2, 1)).toBe(true)
    expect(shouldApplyClientConfig(1, 0)).toBe(true)
    // First ever config with garbage/absent appliedVersion.
    expect(shouldApplyClientConfig(1, null)).toBe(true)
    expect(shouldApplyClientConfig(1, undefined)).toBe(true)
    // Same version and regressions never re-apply.
    expect(shouldApplyClientConfig(1, 1)).toBe(false)
    expect(shouldApplyClientConfig(2, 5)).toBe(false)
    // Garbage fetched versions never apply.
    expect(shouldApplyClientConfig(0, 0)).toBe(false)
    expect(shouldApplyClientConfig(-1, 0)).toBe(false)
    expect(shouldApplyClientConfig(1.5, 0)).toBe(false)
    expect(shouldApplyClientConfig('2', 1)).toBe(false)
    expect(shouldApplyClientConfig(null, 0)).toBe(false)
  })
})

describe('configYamlEntries', () => {
  it('extracts dotted-key scalar entries from payload.config_yaml', () => {
    const entries = configYamlEntries({
      config_yaml: {
        'display.show_reasoning': true,
        'agent.image_input_mode': 'auto',
        'agent.max_turns': 40,
        timezone: null
      }
    })

    expect(entries).toEqual([
      ['display.show_reasoning', true],
      ['agent.image_input_mode', 'auto'],
      ['agent.max_turns', 40],
      ['timezone', null]
    ])
  })

  it('drops non-scalar values and blank keys instead of failing (forward compat)', () => {
    const entries = configYamlEntries({
      config_yaml: {
        'display.show_reasoning': true,
        'future.nested': { not: 'a scalar' },
        'future.list': [1, 2],
        '   ': 'blank key',
        'future.fn': undefined
      }
    })

    expect(entries).toEqual([['display.show_reasoning', true]])
  })

  it('returns no entries for missing or malformed payloads', () => {
    expect(configYamlEntries(null)).toEqual([])
    expect(configYamlEntries(undefined)).toEqual([])
    expect(configYamlEntries({})).toEqual([])
    expect(configYamlEntries({ config_yaml: null })).toEqual([])
    expect(configYamlEntries({ config_yaml: 'display: {}' })).toEqual([])
    expect(configYamlEntries({ config_yaml: ['display.show_reasoning'] })).toEqual([])
    // Unknown top-level payload fields are ignored.
    expect(configYamlEntries({ some_future_field: { anything: true } })).toEqual([])
  })
})
