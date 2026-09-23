import { expect, it } from 'vitest'

import { entityKey } from './identity'
import type { LunarEntity } from './model'
import { deriveWorkerPersonality } from './worker-personality'

it('keeps owner style independent of session, display title and operational state', () => {
  const identity = { kind: 'session', connectionId: 'remote', profile: 'worker', sessionId: 'one' } as const
  const entity: LunarEntity = { key: entityKey(identity), identity, observedAt: 1, authority: 'authoritative', destination: 'library', animation: 'idle' }
  const first = deriveWorkerPersonality(entity)
  const next = deriveWorkerPersonality({ ...entity, identity: { ...identity, sessionId: 'two' }, authority: 'stale', animation: 'error' })
  expect(next).toEqual(first)
  expect(first.provenance).toBe('authored-local')
  expect(entity.authority).toBe('authoritative')
})

it('only exposes exact configured roles from fresh metadata without inferring traits from labels', () => {
  const identity = { kind: 'profile', connectionId: 'remote', profile: 'worker' } as const
  const entity: LunarEntity = { key: entityKey(identity), identity, observedAt: 1, authority: 'authoritative', destination: 'library', animation: 'idle', presentation: { configuredTitle: 'Source title', groups: [{ id: 'g', name: 'Source group' }], metadata: { state: 'fresh', source: 'profiles' }, placement: { lodHint: 0, overflow: false } } }
  const fresh = deriveWorkerPersonality(entity)
  expect(fresh.configuredTitle).toBe('Source title')
  expect(fresh.groups).toEqual(entity.presentation!.groups)

  for (const state of ['stale', 'unavailable'] as const) {
    const result = deriveWorkerPersonality({ ...entity, presentation: { ...entity.presentation!, configuredTitle: 'Curious owl', metadata: { state, source: 'profiles' } } })
    expect(result.configuredTitle).toBeUndefined()
    expect(result.groups).toEqual([])
    expect(result.trait).toBe(fresh.trait)
    expect(result.metadataState).toBe(state)
  }
})
