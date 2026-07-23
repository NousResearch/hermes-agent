import { describe, expect, it } from 'vitest'

import type { MoaConfigResponse } from '@/types/hermes'

import {
  addMoaReference,
  createMoaPreset,
  deleteMoaPreset,
  duplicateMoaPreset,
  moaConfigComplete,
  moveMoaReference,
  prepareMoaConfigForSave,
  renameMoaPreset,
  updateMoaSlot,
  validateMoaPresetName
} from './moa-preset-helpers'

const configFixture = (): MoaConfigResponse =>
  ({
    default_preset: 'default',
    active_preset: 'default',
    presets: {
      default: {
        reference_models: [
          {
            continuity_id: 'primary-auditor',
            provider: 'nous',
            model: 'hermes-4',
            reasoning_effort: 'high',
            hidden_lane_setting: { retain: true }
          },
          {
            continuity_id: 'secondary-auditor',
            provider: 'openrouter',
            model: 'deepseek/deepseek-v4-pro',
            hidden_lane_setting: { retain: 'also' }
          }
        ],
        aggregator: {
          provider: 'openrouter',
          model: 'anthropic/claude-opus-4.8',
          hidden_aggregator_setting: 'retain'
        },
        reference_temperature: 0,
        aggregator_temperature: null,
        max_tokens: 4096,
        reference_max_tokens: 600,
        fanout: 'per_iteration',
        enabled: true,
        hidden_preset_setting: 'retain'
      }
    },
    reference_models: [
      { continuity_id: 'primary-auditor', provider: 'nous', model: 'hermes-4' },
      { continuity_id: 'secondary-auditor', provider: 'openrouter', model: 'deepseek/deepseek-v4-pro' }
    ],
    aggregator: { provider: 'openrouter', model: 'anthropic/claude-opus-4.8' },
    reference_temperature: 0,
    aggregator_temperature: 0,
    max_tokens: 4096,
    enabled: true
  }) as MoaConfigResponse

describe('MoA preset lifecycle helpers', () => {
  it('rejects blank and duplicate names while allowing a preset to keep its own name', () => {
    const config = configFixture()

    expect(validateMoaPresetName(config, '   ')).toBe('blank')
    expect(validateMoaPresetName(config, ' default ')).toBe('duplicate')
    expect(validateMoaPresetName(config, ' default ', 'default')).toBeNull()
    expect(validateMoaPresetName(config, 'daily')).toBeNull()
  })

  it('renames a preset and updates both pointers without changing reference-seat identities or hidden fields', () => {
    const config = configFixture()
    const renamed = renameMoaPreset(config, 'default', 'daily-grok')

    expect(renamed.presets.default).toBeUndefined()
    expect(renamed.default_preset).toBe('daily-grok')
    expect(renamed.active_preset).toBe('daily-grok')
    expect(renamed.presets['daily-grok'].reference_models.map(slot => slot.continuity_id)).toEqual([
      'primary-auditor',
      'secondary-auditor'
    ])
    expect(renamed.presets['daily-grok'].reference_models[0]).toMatchObject({
      hidden_lane_setting: { retain: true }
    })
    expect(renamed.presets['daily-grok']).toMatchObject({ hidden_preset_setting: 'retain' })
  })

  it('duplicates a preset with deliberate continuity sharing but independent slot objects', () => {
    const config = configFixture()
    const duplicated = duplicateMoaPreset(config, 'default', 'daily-copy')
    const source = duplicated.presets.default
    const copy = duplicated.presets['daily-copy']

    expect(copy.reference_models.map(slot => slot.continuity_id)).toEqual(['primary-auditor', 'secondary-auditor'])
    expect(copy.reference_models[0]).not.toBe(source.reference_models[0])
    expect(copy.reference_models[0]).toMatchObject({ hidden_lane_setting: { retain: true } })
    expect(copy.aggregator).not.toBe(source.aggregator)
    expect(copy.aggregator).toMatchObject({ hidden_aggregator_setting: 'retain' })
  })

  it('creates an incomplete blank preset without copying the default settings', () => {
    const config = configFixture()
    const created = createMoaPreset(config, 'fresh')
    const fresh = created.presets.fresh

    expect(fresh).toEqual({
      aggregator: { model: '', provider: '' },
      aggregator_temperature: null,
      enabled: true,
      fanout: 'per_iteration',
      max_tokens: 4096,
      reference_max_tokens: null,
      reference_models: [{ model: '', provider: '' }],
      reference_temperature: null
    })
    expect(created.default_preset).toBe('default')
    expect(created.active_preset).toBe('default')
    expect(moaConfigComplete(created)).toBe(false)
  })

  it('never deletes the last preset and repairs default/active pointers when deleting another preset', () => {
    const duplicated = duplicateMoaPreset(configFixture(), 'default', 'other')

    expect(() => deleteMoaPreset(configFixture(), 'default')).toThrow('last-preset')

    const deleted = deleteMoaPreset(duplicated, 'default')
    expect(Object.keys(deleted.presets)).toEqual(['other'])
    expect(deleted.default_preset).toBe('other')
    expect(deleted.active_preset).toBe('other')
  })
})

describe('MoA seat and save validation helpers', () => {
  it('reorders complete slot objects without changing continuity identities or hidden fields', () => {
    const preset = configFixture().presets.default
    const moved = moveMoaReference(preset, 1, 0)

    expect(moved.reference_models.map(slot => slot.continuity_id)).toEqual(['secondary-auditor', 'primary-auditor'])
    expect(moved.reference_models[0]).toMatchObject({ hidden_lane_setting: { retain: 'also' } })
  })

  it('adds an independent reference seat with a new continuity id and preserves hidden seat defaults', () => {
    const preset = configFixture().presets.default
    const added = addMoaReference(preset)

    expect(added.reference_models).toHaveLength(3)
    expect(added.reference_models[2]).not.toBe(preset.reference_models[1])
    expect(added.reference_models[2]).toMatchObject({
      continuity_id: 'reference-3',
      hidden_lane_setting: { retain: 'also' },
      provider: 'openrouter',
      model: 'deepseek/deepseek-v4-pro'
    })
    expect(new Set(added.reference_models.map(slot => slot.continuity_id)).size).toBe(3)
  })

  it('preserves continuity and hidden fields when replacing a reference provider/model', () => {
    const slot = configFixture().presets.default.reference_models[0]
    const providerChanged = updateMoaSlot(slot, { provider: 'openrouter' })
    const modelChanged = updateMoaSlot(providerChanged, { model: 'anthropic/claude-opus-4.8' })

    expect(providerChanged).toMatchObject({
      continuity_id: 'primary-auditor',
      provider: 'openrouter',
      model: '',
      hidden_lane_setting: { retain: true }
    })
    expect(modelChanged).toMatchObject({
      continuity_id: 'primary-auditor',
      provider: 'openrouter',
      model: 'anthropic/claude-opus-4.8',
      hidden_lane_setting: { retain: true }
    })
  })

  it('strips continuity ids from aggregators while retaining every other hidden field', () => {
    const config = configFixture()

    const invalidAggregator = {
      ...config.presets.default.aggregator,
      continuity_id: 'must-not-survive'
    }

    const prepared = prepareMoaConfigForSave({
      ...config,
      aggregator: invalidAggregator,
      presets: {
        default: { ...config.presets.default, aggregator: invalidAggregator }
      }
    })

    expect(prepared.presets.default.aggregator.continuity_id).toBeUndefined()
    expect(prepared.aggregator.continuity_id).toBeUndefined()
    expect(prepared.presets.default.aggregator).toMatchObject({ hidden_aggregator_setting: 'retain' })
  })

  it('holds saves for incomplete, recursive, or backend-incompatible preset values', () => {
    const valid = configFixture()
    expect(moaConfigComplete(valid)).toBe(true)

    const incomplete = configFixture()
    incomplete.presets.default.reference_models[0] = updateMoaSlot(
      incomplete.presets.default.reference_models[0],
      { provider: 'openrouter' }
    )
    expect(moaConfigComplete(incomplete)).toBe(false)

    const recursive = configFixture()
    recursive.presets.default.aggregator = { provider: 'moa', model: 'default' }
    expect(moaConfigComplete(recursive)).toBe(false)

    const invalidLimit = configFixture()
    invalidLimit.presets.default.max_tokens = Number.NaN
    expect(moaConfigComplete(invalidLimit)).toBe(false)

    const invalidTemperature = configFixture()
    invalidTemperature.presets.default.reference_temperature = Number.POSITIVE_INFINITY
    expect(moaConfigComplete(invalidTemperature)).toBe(false)
  })
})
