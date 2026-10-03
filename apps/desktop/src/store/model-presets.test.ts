import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $modelPresets, applyModelPreset, getModelPreset, setModelPreset } from './model-presets'
import {
  $currentFastMode,
  $currentReasoningEffort,
  $currentServiceTier,
  setCurrentFastMode,
  setCurrentReasoningEffort,
  setCurrentServiceTier
} from './session'

describe('model presets', () => {
  beforeEach(() => {
    $modelPresets.set({})
    setCurrentFastMode(false)
    setCurrentReasoningEffort('')
    setCurrentServiceTier('')
  })

  it('round-trips a preset and merges patches without dropping prior fields', () => {
    setModelPreset('anthropic', 'claude-opus-4-8', { effort: 'high' })
    setModelPreset('anthropic', 'claude-opus-4-8', { fast: true })

    expect(getModelPreset('anthropic', 'claude-opus-4-8')).toEqual({
      effort: 'high',
      fast: true,
      serviceTier: 'priority'
    })
  })

  it('returns an empty preset for unknown models', () => {
    expect(getModelPreset('x', 'y')).toEqual({})
  })

  it('keeps Ultrafast distinct and a later legacy Fast edit selects priority', async () => {
    const calls: unknown[] = []

    const request = async <T>(method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return {} as T
    }

    setModelPreset('openai-codex', 'gpt-6-astra', { serviceTier: 'ultrafast', effort: 'ultra' })
    await applyModelPreset(getModelPreset('openai-codex', 'gpt-6-astra'), {
      failMessage: 'failed',
      request,
      sessionId: 'session-a'
    })
    expect($currentServiceTier.get()).toBe('ultrafast')
    expect(calls).toContainEqual({
      method: 'config.set',
      params: { key: 'fast', session_id: 'session-a', value: 'ultrafast' }
    })
    setModelPreset('openai-codex', 'gpt-6-astra', { fast: true })
    expect(getModelPreset('openai-codex', 'gpt-6-astra')).toEqual({
      effort: 'ultra',
      fast: true,
      serviceTier: 'priority'
    })
    await applyModelPreset({ serviceTier: 'normal' }, { failMessage: 'failed', request, sessionId: null })
    expect($currentServiceTier.get()).toBe('normal')
    expect($currentFastMode.get()).toBe(false)
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

  it('applies a fresh-draft preset locally without mutating gateway config', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const request = async <T>(method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return {} as T
    }

    await applyModelPreset({ effort: 'high', fast: true }, { failMessage: 'x', request, sessionId: null })

    expect($currentReasoningEffort.get()).toBe('high')
    expect($currentFastMode.get()).toBe(true)
    expect(calls).toEqual([])
  })
  it('rolls back only the failed speed while still writing speed after an effort error', async () => {
    setCurrentServiceTier('normal')

    const request = vi.fn(async (_method: string, params?: Record<string, unknown>) => {
      if (params?.key === 'reasoning') {
        throw new Error('effort rejected')
      }

      return {} as never
    })

    await applyModelPreset(
      { effort: 'ultra', serviceTier: 'ultrafast' },
      {
        failMessage: 'failed',
        request,
        sessionId: 's1'
      }
    )
    expect(request).toHaveBeenCalledWith('config.set', { key: 'fast', session_id: 's1', value: 'ultrafast' })
    expect($currentReasoningEffort.get()).toBe('')
    expect($currentServiceTier.get()).toBe('ultrafast')

    await applyModelPreset(
      { serviceTier: 'priority' },
      {
        failMessage: 'failed',
        request: async () => {
          throw new Error('speed rejected')
        },
        sessionId: 's1'
      }
    )
    expect($currentServiceTier.get()).toBe('ultrafast')
    expect($currentFastMode.get()).toBe(true)
  })

  it('keeps a newer speed when an older write fails late', async () => {
    let rejectOld!: (error: Error) => void

    const old = applyModelPreset(
      { serviceTier: 'priority' },
      {
        failMessage: 'failed',
        sessionId: 's1',
        request: () =>
          new Promise((_resolve, reject) => {
            rejectOld = reject
          })
      }
    )

    const newer = applyModelPreset(
      { serviceTier: 'ultrafast' },
      {
        failMessage: 'failed',
        sessionId: 's1',
        request: async () => ({}) as never
      }
    )

    await vi.waitFor(() => expect(rejectOld).toBeTypeOf('function'))
    rejectOld(new Error('old write failed'))
    await old
    await newer
    expect($currentServiceTier.get()).toBe('ultrafast')
  })
  it('returns to confirmed Standard when both pending speed writes reject', async () => {
    setCurrentServiceTier('normal')
    const rejects: ((error: Error) => void)[] = []
    const request = () => new Promise<never>((_resolve, reject) => rejects.push(reject))
    const fast = applyModelPreset({ serviceTier: 'priority' }, { failMessage: 'failed', sessionId: 'both', request })

    const ultrafast = applyModelPreset(
      { serviceTier: 'ultrafast' },
      { failMessage: 'failed', sessionId: 'both', request }
    )

    await vi.waitFor(() => expect(rejects).toHaveLength(1))
    rejects[0](new Error('Fast rejected'))
    await fast
    await vi.waitFor(() => expect(rejects).toHaveLength(2))
    rejects[1](new Error('Ultrafast rejected'))
    await ultrafast
    expect($currentServiceTier.get()).toBe('normal')
    expect($currentFastMode.get()).toBe(false)
  })
  it('reserves speed independently while a preset effort write is pending', async () => {
    setCurrentServiceTier('normal')
    let finishEffort!: () => void
    const speedRejects: ((error: Error) => void)[] = []

    const request = (_method: string, params?: Record<string, unknown>) =>
      params?.key === 'reasoning'
        ? new Promise<never>(resolve => {
            finishEffort = () => resolve({} as never)
          })
        : new Promise<never>((_resolve, reject) => speedRejects.push(reject))

    const preset = applyModelPreset(
      { effort: 'ultra', serviceTier: 'priority' },
      { failMessage: 'failed', sessionId: 'mixed', request }
    )

    const newer = applyModelPreset({ serviceTier: 'ultrafast' }, { failMessage: 'failed', sessionId: 'mixed', request })
    await vi.waitFor(() => expect(speedRejects).toHaveLength(1))
    speedRejects[0](new Error('Fast rejected'))
    await vi.waitFor(() => expect(speedRejects).toHaveLength(2))
    speedRejects[1](new Error('Ultrafast rejected'))
    await newer
    expect($currentServiceTier.get()).toBe('normal')
    finishEffort()
    await preset
    expect($currentServiceTier.get()).toBe('normal')
  })
})
