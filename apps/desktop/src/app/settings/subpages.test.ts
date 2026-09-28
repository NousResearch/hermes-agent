import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from '@/i18n'

import { CONFIG_SUBPAGES, configSubpageForField } from './config-subpages'
import { SECTIONS } from './constants'
import { OTHER_SUBPAGES } from './other-subpages'
import {
  manifestView,
  type ManifestViewKey,
  SETTING_IDS,
  settingDefinition,
  SETTINGS_MANIFEST,
  settingSearchTargets
} from './settings-manifest'
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

  it('gives every settings group its own nav label in every locale', () => {
    for (const locale of Object.values(TRANSLATIONS)) {
      const nav: Record<string, string> = locale.settings.nav

      for (const view of Object.keys(OTHER_SUBPAGES)) {
        expect(nav[view]).toBeTruthy()
      }

      // Sessions also holds the default project folder, so it can't wear the
      // label of the palette row that lands on its archive page.
      expect(nav.sessions).not.toBe(nav.archivedChats)
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
      ['keybinds', 'setting=keybinds.hud-modifier', 'hud-gesture'],
      ['keybinds', 'page=shortcuts&setting=keybinds.hud-modifier', 'hud-gesture'],
      ['notifications', 'setting=notifications.completion-sound', 'sounds'],
      ['config:advanced', 'setting=advanced.keep-awake', 'desktop'],
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

  it('makes every manifest row a routed, translated palette hit on a real page', () => {
    for (const viewKey of Object.keys(SETTINGS_MANIFEST) as ManifestViewKey[]) {
      const view = manifestView(viewKey)
      const subpages = settingsSubpages(view).map(page => page.id)
      expect(views).toContain(view)

      for (const [key, setting] of Object.entries(SETTINGS_MANIFEST[viewKey])) {
        const id = SETTING_IDS[viewKey][key as keyof (typeof SETTING_IDS)[typeof viewKey]]
        expect(id).toBe(`${viewKey}.${key.replace(/[A-Z]/g, c => `-${c.toLowerCase()}`)}`)
        expect(settingDefinition(view, id)).toBe(setting)
        expect(setting.keywords.length).toBeGreaterThan(0)

        if (setting.subpage) {
          expect(subpages).toContain(setting.subpage)
          expect(
            resolveSettingsSubpage(view, new URLSearchParams(settingsSearchTargetQuery({ view, setting: id })))
          ).toBe(setting.subpage)
        }

        for (const locale of Object.values(TRANSLATIONS)) {
          expect(setting.copy(locale).label).toBeTruthy()
        }
      }
    }

    // Platform-gated rows aside, the whole manifest reaches the palette.
    const definitions = Object.values(SETTINGS_MANIFEST).flatMap(rows => Object.values(rows))
    const gated = definitions.filter(setting => 'available' in setting).length
    const targets = settingSearchTargets(TRANSLATIONS.en)
    expect(targets.length).toBeGreaterThanOrEqual(definitions.length - gated)
    expect(new Set(targets.map(target => target.id)).size).toBe(targets.length)
    expect(targets.map(target => target.label)).toEqual(
      expect.arrayContaining([
        'In-App Tips',
        'Guided Tours',
        'Keep computer awake',
        'Auto-archive stale chats',
        TRANSLATIONS.en.settings.about.updates
      ])
    )
  })
})
