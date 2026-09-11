import { beforeEach, describe, expect, it } from 'vitest'

import { readKey } from '@/lib/storage'
import { modelVisibilityKey } from '@/store/model-visibility'

import { $favoriteModels, setFavoriteModels, toggleFavoriteKey, toggleFavoriteModel } from './model-favorites'

const STORAGE_KEY = 'hermes.desktop.favorite-models'

beforeEach(() => {
  window.localStorage.clear()
  $favoriteModels.set([])
})

describe('model favorites', () => {
  it('stars a model and keeps the order the user starred them in', () => {
    const first = toggleFavoriteKey([], modelVisibilityKey('google', 'gemini-3.1-pro'))
    const second = toggleFavoriteKey(first, modelVisibilityKey('anthropic', 'claude-sonnet-4.6'))

    expect(second).toEqual([
      modelVisibilityKey('google', 'gemini-3.1-pro'),
      modelVisibilityKey('anthropic', 'claude-sonnet-4.6')
    ])
  })

  it('unstars without disturbing the remaining order', () => {
    const keys = [
      modelVisibilityKey('google', 'gemini-3.1-pro'),
      modelVisibilityKey('anthropic', 'claude-sonnet-4.6'),
      modelVisibilityKey('moonshot', 'kimi-k2')
    ]

    expect(toggleFavoriteKey(keys, modelVisibilityKey('anthropic', 'claude-sonnet-4.6'))).toEqual([
      modelVisibilityKey('google', 'gemini-3.1-pro'),
      modelVisibilityKey('moonshot', 'kimi-k2')
    ])
  })

  it('keys a star by provider AND model, so the same id on two providers is two rows', () => {
    const google = toggleFavoriteKey([], modelVisibilityKey('google', 'glm-5-med'))
    const zai = toggleFavoriteKey(google, modelVisibilityKey('zai', 'glm-5-med'))

    expect(zai).toHaveLength(2)
    expect(toggleFavoriteKey(zai, modelVisibilityKey('google', 'glm-5-med'))).toEqual([
      modelVisibilityKey('zai', 'glm-5-med')
    ])
  })

  it('persists the stars and clears the stored key when the last one goes', () => {
    toggleFavoriteModel(modelVisibilityKey('google', 'gemini-3.1-pro'))

    expect($favoriteModels.get()).toEqual([modelVisibilityKey('google', 'gemini-3.1-pro')])
    expect(readKey(STORAGE_KEY)).toBe(JSON.stringify([modelVisibilityKey('google', 'gemini-3.1-pro')]))

    toggleFavoriteModel(modelVisibilityKey('google', 'gemini-3.1-pro'))

    expect($favoriteModels.get()).toEqual([])
    expect(readKey(STORAGE_KEY)).toBeNull()
  })

  it('dedupes a list restored from storage', () => {
    setFavoriteModels([modelVisibilityKey('google', 'gemini-3.1-pro'), modelVisibilityKey('google', 'gemini-3.1-pro')])

    expect($favoriteModels.get()).toEqual([modelVisibilityKey('google', 'gemini-3.1-pro')])
  })
})
