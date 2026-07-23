import { describe, expect, it } from 'vitest'

import type { MoaConfigResponse, MoaPresetConfig } from '@/types/hermes'

import {
  createMoaPreset,
  deleteMoaPreset,
  duplicateMoaPreset,
  moaConfigComplete,
  renameMoaPreset
} from './moa-preset-helpers'

const RESERVED_NAMES = ['__proto__', 'constructor', 'toString'] as const

const presetFixture = (): MoaPresetConfig => ({
  aggregator: { model: 'anthropic/claude-opus-4.8', provider: 'openrouter' },
  aggregator_temperature: null,
  enabled: true,
  fanout: 'user_turn',
  max_tokens: 4096,
  reference_max_tokens: 600,
  reference_models: [{ continuity_id: 'advisor', model: 'hermes-4', provider: 'nous' }],
  reference_temperature: null
})

const configFixture = (): MoaConfigResponse => ({
  active_preset: 'default',
  aggregator: { model: 'anthropic/claude-opus-4.8', provider: 'openrouter' },
  aggregator_temperature: null,
  default_preset: 'default',
  enabled: true,
  max_tokens: 4096,
  presets: { default: presetFixture() },
  reference_models: [{ continuity_id: 'advisor', model: 'hermes-4', provider: 'nous' }],
  reference_temperature: null
})

const hasOwn = (value: object, key: PropertyKey): boolean => Object.prototype.hasOwnProperty.call(value, key)

describe.each(RESERVED_NAMES)('MoA preset name %s', name => {
  it('renames to an own serializable data property that remains complete', () => {
    const original = configFixture()
    const source = original.presets.default
    const renamed = renameMoaPreset(original, 'default', name)

    expect(Object.keys(renamed.presets)).toEqual([name])
    expect(hasOwn(renamed.presets, name)).toBe(true)
    expect(Object.getOwnPropertyDescriptor(renamed.presets, name)).toMatchObject({
      enumerable: true,
      value: source,
      writable: true
    })
    expect(renamed.default_preset).toBe(name)
    expect(renamed.active_preset).toBe(name)
    expect(moaConfigComplete(renamed)).toBe(true)

    const serialized = JSON.parse(JSON.stringify(renamed)) as MoaConfigResponse
    expect(Object.keys(serialized.presets)).toEqual([name])
    expect(hasOwn(serialized.presets, name)).toBe(true)
    expect(serialized.presets[name].aggregator.model).toBe('anthropic/claude-opus-4.8')
    expect(moaConfigComplete(serialized)).toBe(true)
  })

  it('duplicates to an own serializable data property that remains complete', () => {
    const duplicated = duplicateMoaPreset(configFixture(), 'default', name)

    expect(Object.keys(duplicated.presets)).toEqual(['default', name])
    expect(hasOwn(duplicated.presets, name)).toBe(true)
    expect(duplicated.presets[name]).not.toBe(duplicated.presets.default)
    expect(moaConfigComplete(duplicated)).toBe(true)

    const serialized = JSON.parse(JSON.stringify(duplicated)) as MoaConfigResponse
    expect(Object.keys(serialized.presets)).toEqual(['default', name])
    expect(hasOwn(serialized.presets, name)).toBe(true)
    expect(serialized.presets[name].reference_models[0].continuity_id).toBe('advisor')
    expect(moaConfigComplete(serialized)).toBe(true)
  })

  it('creates an own serializable blank preset without prototype pollution', () => {
    const created = createMoaPreset(configFixture(), name)

    expect(Object.keys(created.presets)).toEqual(['default', name])
    expect(hasOwn(created.presets, name)).toBe(true)
    expect(created.presets[name]).not.toBe(created.presets.default)
    expect(created.presets[name].reference_models[0].continuity_id).toBeUndefined()
    expect(created.presets[name].aggregator).toEqual({ model: '', provider: '' })
    expect(moaConfigComplete(created)).toBe(false)

    const serialized = JSON.parse(JSON.stringify(created)) as MoaConfigResponse
    expect(Object.keys(serialized.presets)).toEqual(['default', name])
    expect(hasOwn(serialized.presets, name)).toBe(true)
    expect(moaConfigComplete(serialized)).toBe(false)
  })

  it('rejects inherited source names that are not own presets', () => {
    const config = configFixture()

    expect(() => renameMoaPreset(config, name, 'renamed')).toThrow('missing-preset')
    expect(() => duplicateMoaPreset(config, name, 'copy')).toThrow('missing-preset')
    expect(() => deleteMoaPreset(config, name)).toThrow('missing-preset')
  })
})

describe('MoA reserved-name pointer repair and validation', () => {
  it('deletes an own reserved preset and repairs default and active pointers to another own reserved preset', () => {
    const renamed = renameMoaPreset(configFixture(), 'default', '__proto__')
    const duplicated = duplicateMoaPreset(renamed, '__proto__', 'constructor')
    const deleted = deleteMoaPreset(duplicated, '__proto__')

    expect(Object.keys(deleted.presets)).toEqual(['constructor'])
    expect(hasOwn(deleted.presets, 'constructor')).toBe(true)
    expect(deleted.default_preset).toBe('constructor')
    expect(deleted.active_preset).toBe('constructor')
    expect(moaConfigComplete(deleted)).toBe(true)
  })

  it.each(RESERVED_NAMES)('rejects dangling default and active pointers inherited as %s', name => {
    const config = configFixture()

    expect(moaConfigComplete({ ...config, default_preset: name })).toBe(false)
    expect(moaConfigComplete({ ...config, active_preset: name })).toBe(false)
  })
})
