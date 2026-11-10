import { afterEach, describe, expect, it } from 'vitest'

import { registerLayoutPresets } from '@/app/contrib/layout-presets'
import { LAYOUTS_AREA } from '@/components/pane-shell/tree/presets'
import { registry } from '@/contrib/registry'
import { platformStatusLabel } from '@/lib/platform-status'

import { ar } from './ar'
import { en } from './en'
import { setRuntimeI18nLocale } from './runtime'

afterEach(() => setRuntimeI18nLocale('en'))

describe('Arabic shared chrome copy', () => {
  it('localizes and isolates the context occupancy percentage', () => {
    setRuntimeI18nLocale('ar')
    const percent = new Intl.NumberFormat('ar', { style: 'percent' }).format(0.47)

    expect(ar.shell.statusbar.contextUsagePanel.percentFull(47)).toBe(`ممتلئ \u2068${percent}\u2069`)
  })

  it('localizes settings navigation labels rather than falling back to English', () => {
    expect(ar.settings.sections.browser).toBe('المتصفح')
    expect(ar.settings.nav.billing).toBe('الفوترة')
  })

  it('updates every bundled preset consumer on a locale change, without changing user titles', () => {
    setRuntimeI18nLocale('en')
    const dispose = registerLayoutPresets()
    const disposeUser = registry.register({ area: LAYOUTS_AREA, id: 'user:test', source: 'user', title: 'My layout' })

    try {
      expect(registry.getArea(LAYOUTS_AREA).find(p => p.id === 'focus')?.title).toBe('Focus')
      setRuntimeI18nLocale('ar')

      for (const id of ['default', 'focus', 'basic', 'terminal-deck', 'quad', 'sidebar-left', 'sidebar-right']) {
        expect(registry.getArea(LAYOUTS_AREA).find(p => p.id === id)?.title).toBe(ar.zones.presetNames[id])
        expect(ar.zones.presetNames[id]).not.toBe(en.zones.presetNames[id])
      }

      expect(registry.getArea(LAYOUTS_AREA).find(p => p.id === 'user:test')?.title).toBe('My layout')
    } finally {
      disposeUser()
      dispose()
    }
  })

  it('uses the same localized platform states in Messaging and the gateway menu', () => {
    for (const state of Object.keys(ar.messaging.states)) {
      expect(platformStatusLabel(state, ar.messaging)).toBe(ar.messaging.states[state])
    }

    expect(platformStatusLabel('connected', ar.messaging)).toBe('متصل')
    expect(platformStatusLabel(null, ar.messaging)).toBe(ar.messaging.unknown)
    expect(platformStatusLabel('future_state', ar.messaging)).toBe('future state')
  })

  it('keeps technical glossary terms in English in legacy chrome and usage labels', () => {
    expect(ar.sidebar.terminal).toBe('Terminal')
    expect(ar.rightSidebar.terminal).toBe('Terminal')
    expect(ar.commandCenter.statTokens).toBe('Tokens')
    expect(ar.commandCenter.dailyTokens).toContain('Tokens')
    // Raw catalogs also cover interpolation functions and overridden legacy entries.
    const catalogs = import.meta.glob('./ar*.ts', { eager: true, query: '?raw', import: 'default' })

    const forbidden =
      /طرفية|طرفيت|هيرميز|هيرمز|توكن|توكين|(?:ويب|ويو)[ -]?هوك|ويبهوك|جي بي تي|إم سي بي|إس إس إتش|جيسون|نوس|ذاكرة الوصول العشوائي|وحدة معالجة الرسوم|مليون رمز|رموز البوابة/

    expect(
      Object.entries(catalogs)
        .filter(([path]) => !path.endsWith('.test.ts'))
        .flatMap(([path, text]) =>
          String(text)
            .split('\n')
            .filter(line => forbidden.test(line))
            .map(line => ({ path, line }))
        )
    ).toEqual([])
  })
})
