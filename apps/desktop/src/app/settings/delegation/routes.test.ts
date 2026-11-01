import { describe, expect, it } from 'vitest'

import type { DelegationModelMap } from '@/hermes'

import {
  removeRoute,
  routeFromDraft,
  setDefaultRoute,
  toggleProfileRoute,
  upsertRoute,
  validateRouteDraft
} from './routes'

const route = (name: string, extra: Record<string, unknown> = {}) => ({
  name,
  description: `${name} route`,
  provider: 'openrouter',
  model: `m-${name}`,
  ...extra
})

describe('route editing preserves an optional default', () => {
  it('preserves inheritance when adding, editing, renaming or removing a route', () => {
    let map: DelegationModelMap = { options: [], default: null }

    map = upsertRoute(map, null, route('fast'))
    expect(map.default).toBeNull()
    map = upsertRoute(map, null, route('deep'))
    expect(map.default).toBeNull()
    map = upsertRoute(map, 'deep', route('deep', { description: 'Updated' }))
    expect(map.default).toBeNull()
    map = upsertRoute(map, 'deep', route('careful'))
    expect(map.default).toBeNull()
    expect(removeRoute(map, 'fast')).toEqual({ options: [route('careful')], default: null })
  })

  it('follows an explicitly chosen default through a rename, but never chooses a replacement', () => {
    const map = setDefaultRoute({ options: [route('fast'), route('deep')], default: null }, 'fast')
    expect(map.default).toBe('fast')

    const renamed = upsertRoute(map, 'fast', route('quick'))
    expect(renamed).toEqual({ options: [route('quick'), route('deep')], default: 'quick' })
    expect(removeRoute(renamed, 'deep').default).toBe('quick')
    expect(removeRoute(renamed, 'quick')).toEqual({ options: [route('deep')], default: null })
    expect(removeRoute(removeRoute(renamed, 'quick'), 'deep')).toEqual({ options: [], default: null })
    expect(setDefaultRoute(renamed, null)).toEqual({ ...renamed, default: null })
    expect(setDefaultRoute(renamed, 'missing').default).toBeNull()
  })

  it('keeps the profile-route default policy unchanged', () => {
    let profiles = toggleProfileRoute({ profiles: [], default: null }, 'coder', true)
    profiles = toggleProfileRoute(profiles, 'researcher', true)
    expect(profiles.default).toBe('coder')
    expect(toggleProfileRoute(profiles, 'coder', false)).toEqual({
      profiles: [{ name: 'researcher' }],
      default: 'researcher'
    })
  })

  it('refuses incomplete or clashing drafts; an edit keeps keys the dialog does not edit and names its source', () => {
    const draft = { name: 'deep', description: '', provider: '', model: '', reasoning_effort: '', service_tier: '' }

    expect(validateRouteDraft(draft, ['deep'])).toEqual({
      name: 'duplicate',
      description: 'required',
      provider: 'required',
      model: 'required'
    })
    expect(validateRouteDraft({ ...draft, name: 'a b' }, []).name).toBe('spaces')
    // A hand-written endpoint route has no provider and stays valid.
    expect(validateRouteDraft({ ...draft, description: 'd', model: 'm' }, [], true)).toEqual({})

    const stored = route('local', {
      base_url: 'http://127.0.0.1:8080/v1',
      api_key: '«redacted:sk-…1234»',
      service_tier: 'fast'
    })

    // A rename sends the preview key back with the old name, so the server keeps the stored key.
    const saved = routeFromDraft({ ...draft, name: 'lan', description: 'Private', model: 'qwen' }, stored)

    expect(saved).toEqual({
      name: 'lan',
      original_name: 'local',
      description: 'Private',
      model: 'qwen',
      base_url: 'http://127.0.0.1:8080/v1',
      api_key: '«redacted:sk-…1234»'
    })
  })
})
