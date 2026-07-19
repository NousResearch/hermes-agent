import { beforeEach, describe, expect, it } from 'vitest'

import type { HermesConnection } from '@/global'
import { storedStringArrayRecord } from '@/lib/storage'

import {
  $customModels,
  addCustomModel,
  getCustomModelsForProvider,
  isCustomModel,
  mergeCustomModels,
  removeCustomModel
} from './custom-models'
import { setConnection } from './session'

const LOCAL_KEY = 'hermes.desktop.custom-models'

function conn(partial: Partial<HermesConnection>): HermesConnection {
  return {
    baseUrl: '',
    isFullscreen: false,
    logs: [],
    nativeOverlayWidth: 0,
    token: '',
    windowButtonPosition: null,
    wsUrl: '',
    ...partial
  }
}

beforeEach(() => {
  window.localStorage.clear()
  setConnection(null)
  $customModels.set({})
})

describe('custom-models store', () => {
  it('persists an array-record and reloads it (regression: storedStringRecord dropped arrays)', () => {
    addCustomModel('openrouter', 'claude-opus-4.8')
    addCustomModel('openrouter', 'gpt-4-turbo')

    // The raw persisted shape is a Record<string, string[]> and survives the
    // array-aware codec round-trip (the old Record<string,string> codec filtered
    // arrays out, reloading as empty).
    expect(storedStringArrayRecord(LOCAL_KEY)).toEqual({
      openrouter: ['claude-opus-4.8', 'gpt-4-turbo']
    })
    expect($customModels.get()).toEqual({ openrouter: ['claude-opus-4.8', 'gpt-4-turbo'] })
  })

  it('ignores non-array and non-string entries when loading', () => {
    window.localStorage.setItem(
      LOCAL_KEY,
      JSON.stringify({ openrouter: ['ok', 123, ''], anthropic: 'not-an-array', google: [] })
    )
    expect(storedStringArrayRecord(LOCAL_KEY)).toEqual({ openrouter: ['ok'] })
  })

  it('dedupes and preserves insertion order', () => {
    addCustomModel('xai', 'grok-4')
    addCustomModel('xai', 'grok-4')
    addCustomModel('xai', 'grok-4-heavy')
    expect(getCustomModelsForProvider('xai')).toEqual(['grok-4', 'grok-4-heavy'])
  })

  it('trims and skips blank ids', () => {
    addCustomModel('  openrouter  ', '  claude-opus-4.8  ')
    addCustomModel('openrouter', '   ')
    expect(getCustomModelsForProvider('openrouter')).toEqual(['claude-opus-4.8'])
  })

  it('removes a model and prunes the provider key when empty', () => {
    addCustomModel('openrouter', 'a')
    addCustomModel('openrouter', 'b')
    removeCustomModel('openrouter', 'a')
    expect(getCustomModelsForProvider('openrouter')).toEqual(['b'])

    removeCustomModel('openrouter', 'b')
    expect($customModels.get()).toEqual({})
    expect(storedStringArrayRecord(LOCAL_KEY)).toEqual({})
  })

  it('mergeCustomModels appends custom entries after backend models, deduped', () => {
    addCustomModel('openrouter', 'custom-1')
    addCustomModel('openrouter', 'shared')
    const merged = mergeCustomModels('openrouter', ['backend-1', 'shared'])
    expect(merged).toEqual(['backend-1', 'shared', 'custom-1'])
  })

  it('isCustomModel reflects only user-added ids', () => {
    addCustomModel('openrouter', 'custom-1')
    expect(isCustomModel('openrouter', 'custom-1')).toBe(true)
    expect(isCustomModel('openrouter', 'backend-1')).toBe(false)
  })

  it('scopes storage per remote connection and re-homes the atom on switch', () => {
    setConnection(conn({ baseUrl: 'https://a.example', mode: 'remote', profile: 'work' }))
    addCustomModel('openrouter', 'model-A')

    // A different backend/profile must not see the first connection's model.
    setConnection(conn({ baseUrl: 'https://b.example', mode: 'remote', profile: 'work' }))
    expect($customModels.get()).toEqual({})
    addCustomModel('openrouter', 'model-B')

    // Switching back re-homes to the original scope's list.
    setConnection(conn({ baseUrl: 'https://a.example', mode: 'remote', profile: 'work' }))
    expect(getCustomModelsForProvider('openrouter')).toEqual(['model-A'])

    // The two scopes persisted under distinct keys.
    expect(storedStringArrayRecord(`${LOCAL_KEY}.remote.${encodeURIComponent('https://a.example')}.work`)).toEqual({
      openrouter: ['model-A']
    })
    expect(storedStringArrayRecord(`${LOCAL_KEY}.remote.${encodeURIComponent('https://b.example')}.work`)).toEqual({
      openrouter: ['model-B']
    })
  })

  it('local (non-remote) connections share the bare key', () => {
    setConnection(conn({ baseUrl: '', mode: 'local', profile: 'default' }))
    addCustomModel('anthropic', 'local-model')
    expect(storedStringArrayRecord(LOCAL_KEY)).toEqual({ anthropic: ['local-model'] })
  })
})
