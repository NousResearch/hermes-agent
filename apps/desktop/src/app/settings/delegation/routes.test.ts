import { describe, expect, it } from 'vitest'

import type { DelegationModelMap } from '@/hermes'

import {
  draftFromRoute,
  removeRoute,
  routeFromDraft,
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

describe('route editing keeps a default while routes exist', () => {
  it('names a default on the first route, follows a rename, and moves it off a deleted default', () => {
    let map: DelegationModelMap = { options: [], default: null }

    map = upsertRoute(map, null, route('fast'))
    map = upsertRoute(map, null, route('deep'))
    expect(map.default).toBe('fast')

    map = upsertRoute(map, 'fast', route('quick'))
    expect(map).toEqual({ options: [route('quick'), route('deep')], default: 'quick' })

    expect(removeRoute(map, 'quick')).toEqual({ options: [route('deep')], default: 'deep' })
    expect(removeRoute(removeRoute(map, 'quick'), 'deep')).toEqual({ options: [], default: null })

    let profiles = toggleProfileRoute({ profiles: [], default: null }, 'coder', true)
    profiles = toggleProfileRoute(profiles, 'researcher', true)
    expect(profiles.default).toBe('coder')
    expect(toggleProfileRoute(profiles, 'coder', false)).toEqual({
      profiles: [{ name: 'researcher' }],
      default: 'researcher'
    })
  })

  it('refuses incomplete or clashing drafts; an edit keeps keys the dialog does not edit and names its source', () => {
    const draft = {
      name: 'deep',
      description: '',
      provider: '',
      model: '',
      reasoning_effort: '',
      service_tier: '',
      max_iterations: '',
      child_timeout_seconds: ''
    }

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

describe('route max_iterations', () => {
  const base = { name: 'scout', description: 'd', provider: 'p', model: 'm' }

  it('loads a stored number into the draft as text', () => {
    expect(draftFromRoute({ ...base, max_iterations: 15 }).max_iterations).toBe('15')
    expect(draftFromRoute(base).max_iterations).toBe('')
  })

  it('refuses a value that is not a whole number of 1 or more', () => {
    const draft = { ...draftFromRoute(base), max_iterations: '0' }
    expect(validateRouteDraft(draft, []).max_iterations).toBe('turns')
    expect(validateRouteDraft({ ...draft, max_iterations: '2.5' }, []).max_iterations).toBe('turns')
    expect(validateRouteDraft({ ...draft, max_iterations: '15' }, []).max_iterations).toBeUndefined()
    expect(validateRouteDraft({ ...draft, max_iterations: '' }, []).max_iterations).toBeUndefined()
  })

  it('saves a number and drops a blank value', () => {
    expect(routeFromDraft({ ...draftFromRoute(base), max_iterations: '15' }).max_iterations).toBe(15)
    expect(
      routeFromDraft({ ...draftFromRoute(base), max_iterations: '' }, { ...base, max_iterations: 9 })
    ).not.toHaveProperty('max_iterations')
  })
})

describe('route limit edge cases', () => {
  const base = { name: 'scout', description: 'd', provider: 'p', model: 'm' }

  it('rejects turn digit strings beyond safe integers and seconds beyond a finite number', () => {
    const draft = draftFromRoute(base)
    expect(validateRouteDraft({ ...draft, max_iterations: '99999999999999999999' }, []).max_iterations).toBe('turns')
    expect(
      validateRouteDraft({ ...draft, child_timeout_seconds: '1' + '0'.repeat(400) }, []).child_timeout_seconds
    ).toBe('seconds')
  })

  it('accepts and saves a padded turn limit', () => {
    const draft = { ...draftFromRoute(base), max_iterations: ' 15 ' }
    expect(validateRouteDraft(draft, []).max_iterations).toBeUndefined()
    expect(routeFromDraft(draft).max_iterations).toBe(15)
  })
})

describe('route child_timeout_seconds', () => {
  const base = { name: 'scout', description: 'd', provider: 'p', model: 'm' }

  it('loads a stored number into the draft as text', () => {
    expect(draftFromRoute({ ...base, child_timeout_seconds: 300 }).child_timeout_seconds).toBe('300')
    expect(draftFromRoute({ ...base, child_timeout_seconds: 0 }).child_timeout_seconds).toBe('0')
    expect(draftFromRoute(base).child_timeout_seconds).toBe('')
  })

  it('refuses a value that is not a number of 0 or more', () => {
    const draft = { ...draftFromRoute(base), child_timeout_seconds: '-5' }
    expect(validateRouteDraft(draft, []).child_timeout_seconds).toBe('seconds')
    expect(validateRouteDraft({ ...draft, child_timeout_seconds: 'soon' }, []).child_timeout_seconds).toBe('seconds')
    expect(validateRouteDraft({ ...draft, child_timeout_seconds: '0' }, []).child_timeout_seconds).toBeUndefined()
    expect(validateRouteDraft({ ...draft, child_timeout_seconds: '300' }, []).child_timeout_seconds).toBeUndefined()
    expect(validateRouteDraft({ ...draft, child_timeout_seconds: '' }, []).child_timeout_seconds).toBeUndefined()

    for (const bad of ['2.', '.5', '1e3', '-0.5']) {
      expect(validateRouteDraft({ ...draft, child_timeout_seconds: bad }, []).child_timeout_seconds).toBe('seconds')
    }
  })

  it('accepts and saves a fractional timeout', () => {
    const draft = { ...draftFromRoute(base), child_timeout_seconds: '30.5' }
    expect(validateRouteDraft(draft, []).child_timeout_seconds).toBeUndefined()
    expect(routeFromDraft(draft).child_timeout_seconds).toBe(30.5)
  })

  it('does not block saving a route loaded with a fractional timeout', () => {
    const draft = draftFromRoute({ ...base, child_timeout_seconds: 30.5 })
    expect(draft.child_timeout_seconds).toBe('30.5')
    expect(validateRouteDraft(draft, [])).toEqual({})
  })

  it('saves a number and drops a blank value', () => {
    expect(routeFromDraft({ ...draftFromRoute(base), child_timeout_seconds: '300' }).child_timeout_seconds).toBe(300)
    expect(routeFromDraft({ ...draftFromRoute(base), child_timeout_seconds: '0' }).child_timeout_seconds).toBe(0)
    expect(
      routeFromDraft({ ...draftFromRoute(base), child_timeout_seconds: '' }, { ...base, child_timeout_seconds: 9 })
    ).not.toHaveProperty('child_timeout_seconds')
  })
})
