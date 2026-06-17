import { beforeEach, describe, expect, it } from 'vitest'

import { $modelPresets, applyModelPreset, getModelPreset, modelPresetKey, setModelPreset } from './model-presets'

describe('model presets', () => {
  beforeEach(() => $modelPresets.set({}))

  it('round-trips a preset and merges patches without dropping prior fields', () => {
    setModelPreset('anthropic', 'claude-opus-4-8', { effort: 'high' })
    setModelPreset('anthropic', 'claude-opus-4-8', { fast: true })

    expect(getModelPreset('anthropic', 'claude-opus-4-8')).toEqual({ effort: 'high', fast: true })
  })

  it('returns an empty preset for unknown models', () => {
    expect(getModelPreset('x', 'y')).toEqual({})
  })

  it('keys by provider::model', () => {
    expect(modelPresetKey('openai', 'gpt-5.5')).toBe('openai::gpt-5.5')
  })

  it('pushes only the provided dimensions to the gateway', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const request = async <T>(method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return {} as T
    }

    await applyModelPreset({ effort: 'high' }, { failMessage: 'x', request, sessionId: 's1' })
    await applyModelPreset({}, { failMessage: 'x', request, sessionId: 's1' })

    expect(calls).toEqual([{ method: 'config.set', params: { key: 'reasoning', session_id: 's1', value: 'high' } }])
  })

  it('no-ops without a session so selecting a model cannot mutate global config', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const request = async <T>(method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return {} as T
    }

    await applyModelPreset({ effort: 'high', fast: true }, { failMessage: 'x', request, sessionId: null })

    expect(calls).toEqual([])
  })
})
