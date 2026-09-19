import { normalizeComposerSendPrefs } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from '@/i18n/catalog'
import { en } from '@/i18n/en'

import {
  composerKeybindRows,
  defaultBindings,
  KEYBIND_ACTIONS,
  KEYBIND_READONLY,
  keybindAction,
  keybindActionAllowedInEditableTarget,
  readonlyShortcuts
} from './actions'
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

  // #71627: reasoning level up/down ship unbound (users pick their own chord)
  // and opt into firing from an editable target on MODIFIED combos only.
  it('registers reasoning level actions unbound, editable on modified combos only', () => {
    for (const id of ['composer.reasoningUp', 'composer.reasoningDown'] as const) {
      expect(keybindAction(id)).toMatchObject({ category: 'composer', defaults: [], editableTargetPolicy: 'modified' })
      expect(defaultBindings()[id]).toEqual([])
      expect(en.keybinds.actions[id]).toBeTruthy()
    }

    expect(keybindActionAllowedInEditableTarget('composer.reasoningUp', 'alt+.')).toBe(true)
    expect(keybindActionAllowedInEditableTarget('composer.reasoningDown', 'mod+alt+down')).toBe(true)
    // Bare / shift-only rebinds must never hijack typing.
    expect(keybindActionAllowedInEditableTarget('composer.reasoningUp', '.')).toBe(false)
    expect(keybindActionAllowedInEditableTarget('composer.reasoningUp', 'shift+.')).toBe(false)
    // Actions without the policy never qualify, modified combo or not.
    expect(keybindActionAllowedInEditableTarget('appearance.toggleMode', 'alt+x')).toBe(false)
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

describe('view.tabSlot.N layers over profile.switch.N on ⌘1…⌘9 (#92569)', () => {
  it('both actions ship on mod+N; the tab slot passes through, the profile switch does not', () => {
    // Over a tab strip the chord is "tab N"; with no eligible strip the tab
    // action declines and the same chord is "profile N". Two actions, one
    // chord: rebinding either changes only that one.
    for (let slot = 1; slot <= 9; slot += 1) {
      expect(defaultBindings()[`view.tabSlot.${slot}`]).toEqual([`mod+${slot}`])
      expect(defaultBindings()[`profile.switch.${slot}`]).toEqual([`mod+${slot}`])
      expect(keybindAction(`view.tabSlot.${slot}`)).toMatchObject({ category: 'view', passthrough: true })
      expect(keybindAction(`profile.switch.${slot}`)?.passthrough).toBeUndefined()
    }
  })

  it('tab-slot actions precede profile switchers so the chord reaches the tab first', () => {
    // The combo index is built in KEYBIND_ACTIONS order; the passthrough
    // action must sit ahead of the one it hands off to.
    const ids = KEYBIND_ACTIONS.map(action => action.id)
    const firstTabSlot = ids.indexOf('view.tabSlot.1')
    const firstProfileSwitch = ids.indexOf('profile.switch.1')

    expect(firstTabSlot).toBeGreaterThanOrEqual(0)
    expect(firstProfileSwitch).toBeGreaterThan(firstTabSlot)
  })
})

describe('composer keybind rows', () => {
  /** Real defaults, so a row is never tested against a shape the app cannot
   *  produce. `enterSends: false` is the interesting base: it is the only mode
   *  where any gesture can fire at all. */
  const prefs = (over: Record<string, unknown> = {}) => normalizeComposerSendPrefs({ enterSends: false, ...over })

  const keysFor = (over: Record<string, unknown>, id: string) =>
    composerKeybindRows(prefs(over)).find(row => row.id === id)?.keys

  it('prints the historical Enter binding when Enter sends', () => {
    const sending = normalizeComposerSendPrefs({})

    expect(composerKeybindRows(sending).find(row => row.id === 'composer.send')?.keys).toEqual(['enter'])
    expect(composerKeybindRows(sending).find(row => row.id === 'composer.newline')?.keys).toEqual(['shift+enter'])
  })

  it('prints the chord when Enter only breaks the line and nothing else is armed', () => {
    expect(keysFor({}, 'composer.send')).toEqual(['mod+enter'])
    expect(keysFor({}, 'composer.newline')).toEqual(['enter', 'shift+enter'])
  })

  it('gives each armed gesture its own row, because they are different instructions', () => {
    const rows = composerKeybindRows(prefs({ sendOnDoubleTap: true, sendOnHold: true }))
    const ids = rows.map(row => row.id)

    expect(ids).toContain('composer.send.double')
    expect(ids).toContain('composer.send.hold')
    // The generic row is replaced, not duplicated: "Enter" cannot describe both
    // a double tap and a long press.
    expect(keysFor({ sendOnDoubleTap: true, sendOnHold: true }, 'composer.send')).toBeUndefined()
  })

  it('keeps the queue chord printed in every configuration', () => {
    for (const over of [{}, { sendOnDoubleTap: true }, { sendOnHold: true, sendOnPause: true }]) {
      expect(keysFor(over, 'composer.queue')).toEqual(['mod+enter'])
    }
  })

  it('resolves every id the panel labels, so no row renders as a raw id', () => {
    const configurations = [
      normalizeComposerSendPrefs({}),
      prefs(),
      prefs({ sendOnDoubleTap: true, sendOnHold: true, sendOnPause: true })
    ]

    for (const config of configurations) {
      for (const row of readonlyShortcuts(config)) {
        const labelKey = row.labelKey ?? row.id

        expect(en.keybinds.actions[labelKey], labelKey).toBeDefined()
      }
    }
  })

  it('labels the send row per gesture, since the keys alone do not say what they do', () => {
    const labelFor = (over: Record<string, unknown>) =>
      composerKeybindRows(prefs(over)).find(row => row.id.startsWith('composer.send'))?.labelKey

    expect(labelFor({})).toBe('composer.send.mod')
    expect(labelFor({ sendOnDoubleTap: true })).toBe('composer.send.double')
    expect(labelFor({ sendOnHold: true })).toBe('composer.send.hold')
    expect(labelFor({ sendOnPause: true })).toBe('composer.send.pause')
  })

  it('prints a single Enter for the pause gesture, which only sometimes means send', () => {
    expect(keysFor({ sendOnPause: true }, 'composer.send.pause')).toEqual(['enter'])
    expect(keysFor({ sendOnPause: true }, 'composer.newline')).toEqual(['enter', 'shift+enter'])
  })
})
