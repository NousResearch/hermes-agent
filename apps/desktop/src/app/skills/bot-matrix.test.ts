// @vitest-environment jsdom
import { describe, expect, it } from 'vitest'

import { buildSkillBotMatrix, scopeFromOptionValue, skillBotColumn } from './bot-matrix'

describe('scopeFromOptionValue', () => {
  it('passes legacy bare profile names through as string scopes', () => {
    expect(scopeFromOptionValue('researcher')).toBe('researcher')
    expect(scopeFromOptionValue('default')).toBe('default')
  })

  it('decodes roster picks into (connection, profile) pins', () => {
    expect(scopeFromOptionValue('homelab::inbox-bot')).toEqual({ connectionId: 'homelab', profile: 'inbox-bot' })
  })

  it('keeps a local-pool roster pick pinned to local (empty connection id survives)', () => {
    // `local::x` / `::x` decode to an explicit object whose empty connection id
    // means "the local pool" downstream — profileScopeKey then collapses it to
    // the bare profile, matching how capabilityScoped routes it.
    expect(scopeFromOptionValue('::solo')).toEqual({ connectionId: '', profile: 'solo' })
  })
})

describe('skillBotColumn', () => {
  it('derives the cache id from the decoded scope, not the raw option value', () => {
    // A remote roster row keeps its connection-prefixed key...
    expect(skillBotColumn('homelab::inbox-bot', 'inbox-bot — Homelab')).toEqual({
      id: 'homelab::inbox-bot',
      label: 'inbox-bot — Homelab',
      scope: { connectionId: 'homelab', profile: 'inbox-bot' }
    })

    // ...while a LOCAL pool row shares the bare-profile cache key the rest of
    // Capabilities uses — otherwise matrix reads would duplicate fetches.
    expect(skillBotColumn('local::default', 'Hermes (default)').id).toBe('default')
  })
})

describe('buildSkillBotMatrix', () => {
  const bots = [skillBotColumn('default', 'Hermes (default)'), skillBotColumn('researcher', 'researcher')]

  it('maps each skill to per-bot enabled flags', () => {
    const sets = new Map([
      ['default', new Set(['web-research', 'forge'])],
      ['researcher', new Set(['forge'])]
    ])

    const matrix = buildSkillBotMatrix(['web-research', 'forge'], bots, sets)

    expect(matrix.get('web-research')).toEqual(new Map([
      ['default', true],
      ['researcher', false]
    ]))
    expect(matrix.get('forge')).toEqual(new Map([
      ['default', true],
      ['researcher', true]
    ]))
  })

  it('marks columns whose read has not landed as null instead of guessing', () => {
    const sets = new Map([['default', new Set(['forge'])]])

    const matrix = buildSkillBotMatrix(['forge'], bots, sets)

    expect(matrix.get('forge')).toEqual(new Map([
      ['default', true],
      ['researcher', null]
    ]))
  })

  it('yields null rows for every column when no bot list has loaded', () => {
    const matrix = buildSkillBotMatrix(['forge'], bots, new Map())

    expect([...matrix.get('forge')!.values()]).toEqual([null, null])
  })

  it('is empty when there are no bot columns (single-profile users see nothing)', () => {
    expect(buildSkillBotMatrix(['forge'], [], new Map()).size).toBe(0)
  })
})
