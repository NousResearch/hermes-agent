import { beforeEach, describe, expect, it } from 'vitest'

import { $pinnedModelKeys, isModelPinned, pinModel, toggleModelPinned, unpinModel } from './model-pinned'

beforeEach(() => {
  $pinnedModelKeys.set([])
  localStorage.clear()
})

describe('model pinning', () => {
  it('starts with nothing pinned', () => {
    expect($pinnedModelKeys.get()).toEqual([])
    expect(isModelPinned('openai::gpt-6-astra')).toBe(false)
  })

  it('pins a model and preserves insertion order across multiple pins', () => {
    pinModel('openai::gpt-6-astra')
    pinModel('gemini::gemini-3.8-flash')

    expect($pinnedModelKeys.get()).toEqual(['openai::gpt-6-astra', 'gemini::gemini-3.8-flash'])
    expect(isModelPinned('openai::gpt-6-astra')).toBe(true)
    expect(isModelPinned('gemini::gemini-3.8-flash')).toBe(true)
  })

  it('pinning an already-pinned key is a no-op (no duplicate, no reorder)', () => {
    pinModel('openai::gpt-6-astra')
    pinModel('gemini::gemini-3.8-flash')
    pinModel('openai::gpt-6-astra')

    expect($pinnedModelKeys.get()).toEqual(['openai::gpt-6-astra', 'gemini::gemini-3.8-flash'])
  })

  it('unpins a model, leaving the rest in order', () => {
    pinModel('openai::gpt-6-astra')
    pinModel('gemini::gemini-3.8-flash')
    pinModel('anthropic::claude-opus-5')

    unpinModel('gemini::gemini-3.8-flash')

    expect($pinnedModelKeys.get()).toEqual(['openai::gpt-6-astra', 'anthropic::claude-opus-5'])
    expect(isModelPinned('gemini::gemini-3.8-flash')).toBe(false)
  })

  it('unpinning a key that was never pinned is a no-op', () => {
    pinModel('openai::gpt-6-astra')

    unpinModel('never-pinned::model')

    expect($pinnedModelKeys.get()).toEqual(['openai::gpt-6-astra'])
  })

  it('toggle flips pin state each call', () => {
    const key = 'openai::gpt-6-astra'

    toggleModelPinned(key)
    expect(isModelPinned(key)).toBe(true)

    toggleModelPinned(key)
    expect(isModelPinned(key)).toBe(false)
  })

  it('persists across a fresh read from storage', () => {
    pinModel('openai::gpt-6-astra')
    pinModel('gemini::gemini-3.8-flash')

    const raw = localStorage.getItem('hermes.desktop.pinned-models')

    expect(raw).not.toBeNull()
    expect(JSON.parse(raw!)).toEqual(['openai::gpt-6-astra', 'gemini::gemini-3.8-flash'])
  })

  it('clears storage entirely once the last pin is removed', () => {
    pinModel('openai::gpt-6-astra')
    unpinModel('openai::gpt-6-astra')

    expect(localStorage.getItem('hermes.desktop.pinned-models')).toBeNull()
  })
})
