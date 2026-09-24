import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from '@/i18n'

import {
  APPEARANCE_SETTING_IDS,
  APPEARANCE_SETTINGS,
  APPEARANCE_SUBPAGES,
  appearanceSearchTargets,
  appearanceSubpageForSetting
} from './appearance-subpages'
import { CONFIG_SUBPAGES, configSubpageForField } from './config-subpages'
import { SECTIONS } from './constants'
import { OTHER_SUBPAGES } from './other-subpages'
import { settingsSearchTargetQuery } from './settings-search'
import { resolveSettingsSubpage, settingsSubpages } from './subpages'
import type { SettingsView } from './types'

const views: SettingsView[] = [
  ...SECTIONS.map(section => `config:${section.id}` as SettingsView),
  ...(Object.keys(OTHER_SUBPAGES) as SettingsView[])
]

describe('settings subpage routing', () => {
  it('opens the first ordered child for parents and keeps explicit child destinations', () => {
    for (const view of views) {
      const pages = settingsSubpages(view)
      expect(pages.length).toBeGreaterThan(0)
      expect(resolveSettingsSubpage(view, new URLSearchParams())).toBe(pages[0].id)
      expect(resolveSettingsSubpage(view, new URLSearchParams({ page: 'missing' }))).toBe(pages[0].id)

      if (pages.some(page => page.id === 'general')) {
        expect(pages[0].id).toBe('general')
      }

      for (const page of pages) {
        expect(resolveSettingsSubpage(view, new URLSearchParams({ page: page.id }))).toBe(page.id)

        for (const locale of Object.values(TRANSLATIONS)) {
          expect(locale.settings.subpages[page.labelKey]).toBeTruthy()
        }
      }
    }
  })

  it('routes every curated field and legacy target to its owning child before consuming the target', () => {
    for (const section of SECTIONS.filter(section => section.keys.length)) {
      for (const field of section.keys) {
        const owner = configSubpageForField(section.id, field)
        expect(CONFIG_SUBPAGES[section.id].some(page => page.id === owner)).toBe(true)
        const view = `config:${section.id}` as SettingsView
        const params = new URLSearchParams(settingsSearchTargetQuery({ view, field }))
        expect(params.get('page')).toBe(owner)
        expect(resolveSettingsSubpage(view, params)).toBe(owner)
      }
    }

    const cases: [SettingsView, string, string][] = [
      ['config:model', 'aux=vision', 'auxiliary'],
      ['keybinds', 'setting=hud-modifier', 'hud-gesture'],
      ['keybinds', 'page=shortcuts&setting=hud-modifier', 'hud-gesture'],
      ['sessions', 'session=archived-id', 'archived'],
      ['vault', 'kind=login', 'credentials'],
      ['vault', 'label=Example', 'credentials'],
      ['vault', 'origin=https%3A%2F%2Fexample.com', 'credentials'],
      ['config:appearance', 'setting=appearance.hide-thread-timeline', 'chat-display']
    ]

    for (const [view, search, expected] of cases) {
      expect(resolveSettingsSubpage(view, new URLSearchParams(search))).toBe(expected)
    }
  })

  it('makes every Appearance row a routed, translated palette hit', () => {
    const subpages = APPEARANCE_SUBPAGES.map(page => page.id)

    for (const [key, setting] of Object.entries(APPEARANCE_SETTINGS)) {
      const id = APPEARANCE_SETTING_IDS[key as keyof typeof APPEARANCE_SETTING_IDS]
      expect(id).toMatch(/^appearance\.[a-z-]+$/)
      expect(subpages).toContain(setting.subpage)
      expect(appearanceSubpageForSetting(id)).toBe(setting.subpage)
      expect(
        resolveSettingsSubpage(
          'config:appearance',
          new URLSearchParams(settingsSearchTargetQuery({ view: 'config:appearance', setting: id }))
        )
      ).toBe(setting.subpage)
      expect(setting.keywords.length).toBeGreaterThan(0)

      for (const locale of Object.values(TRANSLATIONS)) {
        expect(setting.copy(locale).label).toBeTruthy()
      }
    }

    // Platform-gated rows aside, the whole manifest reaches the palette.
    const targets = appearanceSearchTargets(TRANSLATIONS.en)
    const gated = Object.values(APPEARANCE_SETTINGS).filter(setting => 'available' in setting).length
    expect(targets.length).toBeGreaterThanOrEqual(Object.keys(APPEARANCE_SETTINGS).length - gated)
    expect(targets.map(target => target.label)).toEqual(expect.arrayContaining(['In-App Tips', 'Guided Tours']))
  })
})
