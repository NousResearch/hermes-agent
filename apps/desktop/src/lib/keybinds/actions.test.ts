import { describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'

import { KEYBIND_ACTIONS } from './actions'

// Relationship checks between the action table and its consumers, not the
// specific chord or wording any one action ships with.
describe('KEYBIND_ACTIONS', () => {
  it('has unique ids (a duplicate would shadow a row in the shortcuts panel)', () => {
    const ids = KEYBIND_ACTIONS.map(action => action.id)

    expect(new Set(ids).size).toBe(ids.length)
  })

  it('gives every built-in action an English label so it renders in the shortcuts panel', () => {
    const labels = en.keybinds.actions as Record<string, string>
    const missing = KEYBIND_ACTIONS.filter(action => !labels[action.id]).map(action => action.id)

    expect(missing).toEqual([])
  })
})
