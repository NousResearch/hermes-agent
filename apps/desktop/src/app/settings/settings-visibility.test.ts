import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from '@/i18n'

import { SECTIONS } from './constants'
import { settingSearchTargets } from './settings-manifest'
import { settingsSearchTargetShown } from './settings-search'
import { resolveSettingsView, settingShownInMode, settingsPageShown, settingsViewShown } from './settings-visibility'
import { resolveSettingsSubpage } from './subpages'
import type { SettingsView } from './types'

const RAIL_VIEWS: SettingsView[] = [
  ...SECTIONS.map(section => `config:${section.id}` as SettingsView),
  'vault',
  'notifications',
  'billing',
  'providers',
  'gateway',
  'keybinds',
  'keys',
  'sessions',
  'plugins',
  'about'
]

describe('settings visibility by interface mode', () => {
  it('hides the developer and security pages in Simple and every page stays in Advanced', () => {
    for (const hidden of ['config:safety', 'config:advanced', 'gateway', 'keys', 'plugins', 'vault'] as const) {
      expect(settingsViewShown(hidden, 'simple')).toBe(false)
    }

    // Accounts is how Simple signs in, Appearance hosts the mode switch, and
    // keyboard shortcuts must keep answering in both modes.
    for (const kept of ['providers', 'config:appearance', 'keybinds'] as const) {
      expect(settingsViewShown(kept, 'simple')).toBe(true)
    }

    expect(RAIL_VIEWS.filter(view => settingsViewShown(view, 'advanced'))).toEqual(RAIL_VIEWS)
  })

  it('lands a Simple deep link to a hidden page or sub-page on a visible one', () => {
    const view = resolveSettingsView('config:safety', 'simple')

    expect(settingsViewShown(view, 'simple')).toBe(true)
    expect(settingsPageShown(view, resolveSettingsSubpage(view, new URLSearchParams(), 'simple'), 'simple')).toBe(true)
    expect(resolveSettingsView('config:safety', 'advanced')).toBe('config:safety')
    expect(resolveSettingsSubpage('config:voice', new URLSearchParams({ page: 'speech' }), 'simple')).toBe(
      'conversation'
    )
    expect(resolveSettingsSubpage('config:voice', new URLSearchParams({ page: 'speech' }), 'advanced')).toBe('speech')
  })

  it('never searches into a page Simple hides', () => {
    const simple = settingSearchTargets(TRANSLATIONS.en, 'simple')
    const advanced = settingSearchTargets(TRANSLATIONS.en, 'advanced')

    expect(simple.length).toBeLessThan(advanced.length)
    expect(simple.some(target => target.view === 'gateway' || target.view === 'config:advanced')).toBe(false)
    expect(simple.some(target => target.view === 'config:appearance')).toBe(true)
    expect(settingsSearchTargetShown({ view: 'config:safety', field: 'approvals.mode' }, 'simple')).toBe(false)
    expect(settingsSearchTargetShown({ view: 'providers', providerView: 'keys' }, 'simple')).toBe(false)
    expect(settingsSearchTargetShown({ view: 'config:chat', field: 'timezone' }, 'simple')).toBe(true)
  })
})

describe('Simple-hidden technical settings', () => {
  const fields: [SettingsView, string][] = [
    ['config:chat', 'display.show_reasoning'],
    ['config:chat', 'agent.image_input_mode'],
    ['config:voice', 'voice.client_direct'],
    ['config:voice', 'voice.gpt_live.voice']
  ]

  it('are neither rendered nor searchable in Simple, but are in Advanced', () => {
    for (const [view, field] of fields) {
      expect(settingShownInMode(field, 'simple')).toBe(false)
      expect(settingShownInMode(field, 'advanced')).toBe(true)
      expect(settingsSearchTargetShown({ view, field }, 'simple')).toBe(false)
      expect(settingsSearchTargetShown({ view, field }, 'advanced')).toBe(true)
    }

    const pricing = (mode: 'advanced' | 'simple') =>
      settingSearchTargets(TRANSLATIONS.en, mode).some(target => target.id === 'appearance.model-pricing')

    expect(pricing('simple')).toBe(false)
    expect(pricing('advanced')).toBe(true)
    expect(settingShownInMode('voice.voice_chat_mode', 'simple')).toBe(true)
  })
})
