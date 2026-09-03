import { describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'

import { defaultBindings, KEYBIND_ACTIONS, keybindAction } from './actions'

describe('session.archive keybind action', () => {
  it('is registered under the session category', () => {
    const action = keybindAction('session.archive')

    expect(action).toBeDefined()
    expect(action?.category).toBe('session')
  })

  it('ships unbound so it does not claim a chord for every user', () => {
    const action = keybindAction('session.archive')

    expect(action?.defaults).toEqual([])
    // A missing entry would silently drop from the panel; an accidental
    // default binding would change behaviour for everyone. Guard both.
    expect(defaultBindings()['session.archive']).toEqual([])
  })

  it('has an English label so it renders in the shortcuts panel', () => {
    expect(en.keybinds.actions['session.archive']).toBe('Archive current session')
  })

  it('appears exactly once in KEYBIND_ACTIONS', () => {
    const matches = KEYBIND_ACTIONS.filter(action => action.id === 'session.archive')

    expect(matches).toHaveLength(1)
  })
})
