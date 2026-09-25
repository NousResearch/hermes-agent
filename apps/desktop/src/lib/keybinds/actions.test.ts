import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from '@/i18n/catalog'
import { en } from '@/i18n/en'

import { defaultBindings, KEYBIND_ACTIONS, KEYBIND_READONLY, keybindAction } from './actions'
import { canonicalizeCombo } from './combo'

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

  it('keeps session archive registered and unbound by default', () => {
    const action = keybindAction('session.archive')

    expect(action).toMatchObject({ category: 'session', defaults: [] })
    expect(defaultBindings()['session.archive']).toEqual([])
    expect(en.keybinds.actions['session.archive']).toBe('Archive current session')
    expect(KEYBIND_ACTIONS.filter(candidate => candidate.id === 'session.archive')).toHaveLength(1)
  })

  it('registers dictation with an English label and no default chord', () => {
    const action = keybindAction('composer.dictate')

    expect(action).toMatchObject({ category: 'composer', defaults: [] })
    expect(defaultBindings()['composer.dictate']).toEqual([])
    expect(en.keybinds.actions['composer.dictate']).toBe('Start / stop dictation')
    expect(KEYBIND_ACTIONS.filter(candidate => candidate.id === 'composer.dictate')).toHaveLength(1)
  })

  // jsdom never reports a Mac platform, so this is the Windows/Linux default.
  // Don't fake the host OS — assert the chord this runtime actually ships.
  it('ships a voice chord that does not claim the sidebar chord or any other shipped combo', () => {
    const voice = defaultBindings()['composer.voice'].map(canonicalizeCombo)

    expect(voice.length).toBeGreaterThan(0)

    const taken = new Set<string>()

    for (const action of KEYBIND_ACTIONS) {
      if (action.id === 'composer.voice') {
        continue
      }

      for (const combo of action.defaults) {
        taken.add(canonicalizeCombo(combo))
      }
    }

    for (const shortcut of KEYBIND_READONLY) {
      for (const combo of shortcut.keys) {
        taken.add(canonicalizeCombo(combo))
      }
    }

    expect(voice.filter(combo => taken.has(combo))).toEqual([])
    expect(voice).not.toContain(canonicalizeCombo('mod+b'))
  })

  it('points Voice settings hints at the voice conversation action, not dictation', () => {
    const englishVoice = en.keybinds.actions['composer.voice']

    for (const [locale, messages] of Object.entries(TRANSLATIONS)) {
      const voice = messages.keybinds.actions['composer.voice']
      const dictate = messages.keybinds.actions['composer.dictate']
      const hint = messages.settings.config.voiceShortcutHintDesc
      const namesVoice = hint.includes(voice) || hint.includes(englishVoice)

      expect(namesVoice, locale).toBe(true)

      if (dictate !== voice) {
        expect(hint.includes(dictate), locale).toBe(false)
      }
    }
  })
})
