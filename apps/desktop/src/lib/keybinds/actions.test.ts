import { describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'

import { defaultBindings, KEYBIND_ACTIONS, keybindAction } from './actions'

describe('session.pinned keybind actions', () => {
  it('registers both directions under the session category', () => {
    for (const id of ['session.pinned.next', 'session.pinned.previous']) {
      const action = keybindAction(id)

      expect(action).toBeDefined()
      expect(action?.category).toBe('session')
      expect(KEYBIND_ACTIONS.filter(entry => entry.id === id)).toHaveLength(1)
    }
  })

  it('ships unbound like session.togglePin so no chord is claimed by default', () => {
    expect(defaultBindings()['session.pinned.next']).toEqual([])
    expect(defaultBindings()['session.pinned.previous']).toEqual([])
  })

  it('has English labels so both rows render in the shortcuts panel', () => {
    expect(en.keybinds.actions['session.pinned.next']).toBe('Next pinned session')
    expect(en.keybinds.actions['session.pinned.previous']).toBe('Previous pinned session')
  })
})

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

describe('view.cycleSidebarGrouping keybind action', () => {
  it('is an unbound view action with a panel label', () => {
    const action = keybindAction('view.cycleSidebarGrouping')

    expect(action).toEqual({ id: 'view.cycleSidebarGrouping', category: 'view', defaults: [] })
    expect(defaultBindings()['view.cycleSidebarGrouping']).toEqual([])
    expect(en.keybinds.actions['view.cycleSidebarGrouping']).toBe('Cycle session grouping')
  })

  it('appears exactly once in KEYBIND_ACTIONS', () => {
    expect(KEYBIND_ACTIONS.filter(action => action.id === 'view.cycleSidebarGrouping')).toHaveLength(1)
  })
})
