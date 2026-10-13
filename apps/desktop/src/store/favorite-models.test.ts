import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $favoriteModels,
  favoriteModelKey,
  isModelFavorite,
  setFavoriteModels,
  toggleFavoriteModel
} from './favorite-models'

const STORAGE_KEY = 'hermes.desktop.favorite-models'

beforeEach(() => {
  window.localStorage.clear()
  $favoriteModels.set([])
})

afterEach(() => {
  window.localStorage.clear()
})

// Favorite order IS the display order — the catalog renders favorites in this
// array's order, so "most recently starred wins the top slot" would be a
// behaviour change, not a detail. Appending keeps a user's first star where
// they put it.
describe('favorite models keep their starred order', () => {
  it('appends new favorites after existing ones', () => {
    toggleFavoriteModel('nous', 'opus-5')
    toggleFavoriteModel('anthropic', 'claude-sonnet-5')

    expect($favoriteModels.get()).toEqual([
      favoriteModelKey('nous', 'opus-5'),
      favoriteModelKey('anthropic', 'claude-sonnet-5')
    ])
  })

  it('unfavoriting leaves the remaining order intact', () => {
    toggleFavoriteModel('nous', 'opus-5')
    toggleFavoriteModel('anthropic', 'claude-sonnet-5')
    toggleFavoriteModel('google', 'gemini-3.1-pro')

    toggleFavoriteModel('anthropic', 'claude-sonnet-5')

    expect($favoriteModels.get()).toEqual([
      favoriteModelKey('nous', 'opus-5'),
      favoriteModelKey('google', 'gemini-3.1-pro')
    ])
  })

  it('re-starring an unfavorited model puts it at the end, not back in its old slot', () => {
    toggleFavoriteModel('nous', 'opus-5')
    toggleFavoriteModel('google', 'gemini-3.1-pro')

    toggleFavoriteModel('nous', 'opus-5')
    toggleFavoriteModel('nous', 'opus-5')

    expect($favoriteModels.get()).toEqual([
      favoriteModelKey('google', 'gemini-3.1-pro'),
      favoriteModelKey('nous', 'opus-5')
    ])
  })
})

// A favorite is a durable preference: it has to survive the window closing,
// so the store must reach localStorage rather than only the atom.
describe('favorites persist', () => {
  it('writes the favorites list to storage', () => {
    toggleFavoriteModel('nous', 'opus-5')

    expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? '[]')).toEqual([favoriteModelKey('nous', 'opus-5')])
  })

  it('clears the key once the last favorite is removed', () => {
    toggleFavoriteModel('nous', 'opus-5')
    toggleFavoriteModel('nous', 'opus-5')

    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull()
  })
})

// Models from different providers can share an id (an aggregator and the lab
// both serve `claude-opus-5`); starring one must not star the other.
describe('favorites are provider-scoped', () => {
  it('does not report a same-named model on another provider as a favorite', () => {
    toggleFavoriteModel('nous', 'claude-opus-5')

    expect(isModelFavorite('nous', 'claude-opus-5')).toBe(true)
    expect(isModelFavorite('openrouter', 'claude-opus-5')).toBe(false)
  })
})

// The whole list can be staged in one write (tests, a future settings UI);
// it must not grow a duplicate entry for one model.
describe('the favorites list dedupes on replace', () => {
  it('collapses duplicate keys', () => {
    const key = favoriteModelKey('google', 'gemini-3.1-pro')

    setFavoriteModels([key, key])

    expect($favoriteModels.get()).toEqual([key])
  })
})
