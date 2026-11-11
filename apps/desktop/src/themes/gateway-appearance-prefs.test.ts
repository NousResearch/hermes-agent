import { registryBackendScopeKey } from '@hermes/shared'
import { beforeEach, describe, expect, it } from 'vitest'

import { dropAppearanceForProfile, migrateAppearanceForProfile, modePref, skinPref } from './appearance-prefs'

interface Pref {
  resolve: (profile: string, connectionId: string) => string
  assign: (profile: string, value: string, connectionId: string) => void
}

const cases = [
  {
    name: 'skin',
    pref: skinPref as unknown as Pref,
    record: 'hermes-desktop-profile-themes-v1',
    global: 'hermes-desktop-theme-v2',
    old: 'catppuccin',
    named: 'ember',
    next: 'nous',
    fallback: 'nous'
  },
  {
    name: 'mode',
    pref: modePref as unknown as Pref,
    record: 'hermes-desktop-profile-modes-v1',
    global: 'hermes-desktop-mode-v1',
    old: 'dark',
    named: 'system',
    next: 'light',
    fallback: 'system'
  }
] as const

beforeEach(() => window.localStorage.clear())

describe.each(cases)(
  'gateway-scoped $name upgrade fallback',
  ({ pref, record, global, old, named, next, fallback }) => {
    it('keeps legacy picks intact and never seeds an unvisited peer from another gateway pick', () => {
      window.localStorage.setItem(global, old)
      const original = JSON.stringify({ work: named })
      window.localStorage.setItem(record, original)
      expect(pref.resolve('default', 'local')).toBe(old)
      expect(pref.resolve('work', 'remote')).toBe(named)
      // The old format had no gateway owner: every gateway can still read it,
      // but no new selection is allowed to rewrite that ambiguous history.
      pref.assign('default', next, 'local')
      pref.assign('work', next, 'remote')
      expect(pref.resolve('default', 'unvisited-peer')).toBe(old)
      expect(pref.resolve('work', 'unvisited-peer')).toBe(named)
      expect(window.localStorage.getItem(global)).toBe(old)
      expect(window.localStorage.getItem(record)).toBe(original)
      expect(pref.resolve('default', 'local')).toBe(next)
      expect(pref.resolve('work', 'remote')).toBe(next)
    })

    it('inherits only from this gateway while explicit default and named picks stay pinned', () => {
      pref.assign('default', old, 'local')
      pref.assign('work', named, 'local')
      pref.assign('default', next, 'remote')
      expect(pref.resolve('default', 'local')).toBe(old)
      expect(pref.resolve('work', 'local')).toBe(named)
      expect(pref.resolve('new-bot', 'local')).toBe(named)
      expect(pref.resolve('new-bot', 'remote')).toBe(next)
      expect(pref.resolve('default', 'unvisited-peer')).toBe(fallback)
    })

    it('reads unanimous legacy records without rewriting global storage', () => {
      window.localStorage.setItem(record, JSON.stringify({ work: old, home: old }))
      expect(pref.resolve('new-bot', 'remote')).toBe(old)
      expect(window.localStorage.getItem(global)).toBeNull()
    })

    it('does not guess a global pick from disagreeing legacy profiles', () => {
      window.localStorage.setItem(record, JSON.stringify({ work: old, home: next }))
      expect(pref.resolve('new-bot', 'remote')).toBe(fallback)
      expect(pref.resolve('work', 'remote')).toBe(old)
      expect(pref.resolve('home', 'remote')).toBe(next)
    })

    it('ignores malformed records without blocking a scoped assignment', () => {
      window.localStorage.setItem(record, 'not json')
      window.localStorage.setItem(record.replace(/v1$/, 'v2'), '[]')
      expect(pref.resolve('work', 'remote')).toBe(fallback)
      pref.assign('work', old, 'remote')
      expect(pref.resolve('work', 'remote')).toBe(old)
    })
  }
)

it('preserves raw unknown skin names until a backend theme becomes resolvable', () => {
  window.localStorage.setItem('hermes-desktop-profile-themes-v1', JSON.stringify({ work: 'late-backend-skin' }))
  expect(skinPref.stored('work', 'remote')).toBe('late-backend-skin')
  expect(skinPref.resolve('work', 'remote')).toBe('nous')
  skinPref.assign('work', 'catppuccin', 'local')
  expect(skinPref.stored('work', 'remote')).toBe('late-backend-skin')
})

describe('scoped lifecycle writes', () => {
  const scoped = () => JSON.parse(window.localStorage.getItem('hermes-desktop-profile-themes-v2') ?? '{}')

  it('removes a deleted pick without leaving a mask when no legacy entry exists', () => {
    skinPref.assign('work', 'ember', 'remote')
    dropAppearanceForProfile('work', 'remote')

    expect(scoped()).not.toHaveProperty(registryBackendScopeKey('remote', 'work'))
    expect(skinPref.resolve('work', 'remote')).toBe('ember') // gateway last-pick still applies
  })

  it('masks a legacy pick only on the deleting gateway', () => {
    window.localStorage.setItem('hermes-desktop-profile-themes-v1', JSON.stringify({ work: 'ember' }))
    dropAppearanceForProfile('work', 'remote')

    expect(scoped()[registryBackendScopeKey('remote', 'work')]).toBe('')
    expect(skinPref.resolve('work', 'local')).toBe('ember')
  })

  it('carries only an explicit pick through rename, never an inherited one', () => {
    skinPref.assign('other', 'midnight', 'remote')
    migrateAppearanceForProfile('work', 'renamed', 'remote')

    expect(scoped()).not.toHaveProperty(registryBackendScopeKey('remote', 'renamed'))

    skinPref.assign('work', 'ember', 'remote')
    migrateAppearanceForProfile('work', 'renamed', 'remote')

    expect(scoped()[registryBackendScopeKey('remote', 'renamed')]).toBe('ember')
    expect(scoped()).not.toHaveProperty(registryBackendScopeKey('remote', 'work'))
    expect(skinPref.resolve('renamed', 'local')).toBe('nous')
  })
})
