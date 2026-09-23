import { describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'

import { defaultBindings, KEYBIND_ACTIONS, keybindAction } from './actions'

describe('session.archive keybind action', () => {
  it('is registered under the session category', () => {
    const action = keybindAction('session.archive')

    expect(action).toBeDefined()
    expect(action?.category).toBe('session')
  })

  it('ships bound to mod+l (⌘L / Ctrl+L)', () => {
    const action = keybindAction('session.archive')

    expect(action?.defaults).toEqual(['mod+l'])
    // A missing entry would silently drop from the panel; an accidental
    // empty default would leave archive unbound. Guard both.
    expect(defaultBindings()['session.archive']).toEqual(['mod+l'])
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
