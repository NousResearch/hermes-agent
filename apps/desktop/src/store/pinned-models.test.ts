import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $pinnedModels,
  isModelPinned,
  pinnedModelKey,
  setPinnedModels,
  togglePinnedModel
} from './pinned-models'

const STORAGE_KEY = 'hermes.desktop.pinned-models'

beforeEach(() => {
  window.localStorage.clear()
  $pinnedModels.set([])
})

afterEach(() => {
  window.localStorage.clear()
})

// Pin order IS the display order — the catalog renders pins in this array's
// order, so "most recently pinned wins the top slot" would be a behaviour
// change, not a detail. Appending keeps a user's first pin where they put it.
describe('pinned models keep their pin order', () => {
  it('appends new pins after existing ones', () => {
    togglePinnedModel('nous', 'opus-5')
    togglePinnedModel('anthropic', 'claude-sonnet-5')

    expect($pinnedModels.get()).toEqual([
      pinnedModelKey('nous', 'opus-5'),
      pinnedModelKey('anthropic', 'claude-sonnet-5')
    ])
  })

  it('unpinning leaves the remaining order intact', () => {
    togglePinnedModel('nous', 'opus-5')
    togglePinnedModel('anthropic', 'claude-sonnet-5')
    togglePinnedModel('google', 'gemini-3.1-pro')

    togglePinnedModel('anthropic', 'claude-sonnet-5')

    expect($pinnedModels.get()).toEqual([
      pinnedModelKey('nous', 'opus-5'),
      pinnedModelKey('google', 'gemini-3.1-pro')
    ])
  })

  it('re-pinning an unpinned model puts it at the end, not back in its old slot', () => {
    togglePinnedModel('nous', 'opus-5')
    togglePinnedModel('google', 'gemini-3.1-pro')

    togglePinnedModel('nous', 'opus-5')
    togglePinnedModel('nous', 'opus-5')

    expect($pinnedModels.get()).toEqual([
      pinnedModelKey('google', 'gemini-3.1-pro'),
      pinnedModelKey('nous', 'opus-5')
    ])
  })
})

// A pin is a durable preference: it has to survive the window closing, so the
// store must reach localStorage rather than only the atom.
describe('pins persist', () => {
  it('writes the pin list to storage', () => {
    togglePinnedModel('nous', 'opus-5')

    expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? '[]')).toEqual([
      pinnedModelKey('nous', 'opus-5')
    ])
  })

  it('clears the key once the last pin is removed', () => {
    togglePinnedModel('nous', 'opus-5')
    togglePinnedModel('nous', 'opus-5')

    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull()
  })
})

// Models from different providers can share an id (an aggregator and the lab
// both serve `claude-opus-5`); pinning one must not pin the other.
describe('pins are provider-scoped', () => {
  it('does not report a same-named model on another provider as pinned', () => {
    togglePinnedModel('nous', 'claude-opus-5')

    expect(isModelPinned('nous', 'claude-opus-5')).toBe(true)
    expect(isModelPinned('openrouter', 'claude-opus-5')).toBe(false)
  })
})

// The whole list can be staged in one write (tests, a future settings UI);
// it must not grow a duplicate entry for one model.
describe('the pin list dedupes on replace', () => {
  it('collapses duplicate keys', () => {
    const key = pinnedModelKey('google', 'gemini-3.1-pro')

    setPinnedModels([key, key])

    expect($pinnedModels.get()).toEqual([key])
  })
})
