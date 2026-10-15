import { afterEach, describe, expect, it } from 'vitest'

import { TRANSLATIONS } from './catalog'
import {
  isLocale,
  isSupportedLocaleValue,
  languageOptions,
  localeConfigValue,
  localeMeta,
  normalizeLocale
} from './languages'
import {
  $appLocaleVersion,
  isRegisteredLocale,
  registerAppLocale,
  replaceAppLocaleSource,
  resetAppLocaleRegistry,
  resolveTranslations,
  unregisterAppLocaleSource
} from './registry'
import { translateNow } from './runtime'

afterEach(() => {
  resetAppLocaleRegistry()
})

describe('registerAppLocale', () => {
  it('layers a partial pack over English for a new language and falls back per key', () => {
    registerAppLocale('pl', {
      endonym: 'Polski',
      translations: { common: { save: 'Zapisz' } }
    })

    const pl = resolveTranslations('pl')

    expect(pl.common.save).toBe('Zapisz')
    expect(pl.common.cancel).toBe(TRANSLATIONS.en.common.cancel)
    expect(pl.language.label).toBe(TRANSLATIONS.en.language.label)
  })

  it('wraps a pack string over a function-valued English entry into a positional formatter', () => {
    registerAppLocale('pl', {
      translations: {
        'connectorsPage.searchPlaceholder': '{0} wyników',
        'connectorsPage.card.fact.toolsSomeOn': '{1} z {0} narzędzi'
      }
    })

    const pl = resolveTranslations('pl')

    expect(pl.connectorsPage.searchPlaceholder(3)).toBe('3 wyników')
    expect(pl.connectorsPage.card.fact.toolsSomeOn(10, 4)).toBe('4 z 10 narzędzi')
    expect(typeof pl.connectorsPage.card.fact.toolsSomeOn).toBe('function')
  })

  it('keeps dotted leaf keys (keybinds.actions) addressable from a flat pack', () => {
    registerAppLocale('pl', { translations: { 'keybinds.actions.session.new': 'Nowa sesja' } })

    const actions = resolveTranslations('pl').keybinds.actions

    expect(actions['session.new']).toBe('Nowa sesja')
    expect(actions['nav.settings']).toBe(TRANSLATIONS.en.keybinds.actions['nav.settings'])
  })

  it('layers a pack over the bundled catalog for a bundled id, not over English', () => {
    registerAppLocale('de', { translations: { common: { save: 'Sichern' } } }, 'backend')

    const de = resolveTranslations('de')

    expect(de.common.save).toBe('Sichern')
    expect(de.common.cancel).toBe(TRANSLATIONS.de.common.cancel)
    expect(de.common.cancel).not.toBe(TRANSLATIONS.en.common.cancel)
  })

  it('lets a later source win per key and drops exactly its own layer on dispose', () => {
    const disposeBackend = registerAppLocale(
      'pl',
      { translations: { common: { save: 'Zapisz', cancel: 'Anuluj' } } },
      'backend'
    )

    const disposePlugin = registerAppLocale(
      'pl',
      { translations: { common: { save: 'Zachowaj' } } },
      'plugin:hermes-lang-pl'
    )

    expect(resolveTranslations('pl').common.save).toBe('Zachowaj')
    expect(resolveTranslations('pl').common.cancel).toBe('Anuluj')

    disposePlugin()
    expect(resolveTranslations('pl').common.save).toBe('Zapisz')
    expect(isRegisteredLocale('pl')).toBe(true)

    disposeBackend()
    expect(isRegisteredLocale('pl')).toBe(false)
    expect(resolveTranslations('pl').common.save).toBe(TRANSLATIONS.en.common.save)
  })

  it('bumps the version on every change so translators re-resolve, and memoizes between', () => {
    const before = $appLocaleVersion.get()
    const dispose = registerAppLocale('pl', { translations: { common: { save: 'Zapisz' } } })

    expect($appLocaleVersion.get()).toBe(before + 1)
    expect(resolveTranslations('pl')).toBe(resolveTranslations('pl'))

    dispose()
    expect($appLocaleVersion.get()).toBe(before + 2)
  })

  it('normalizes ids like the backend and ignores an empty id', () => {
    registerAppLocale(' PT_br ', { endonym: 'Português (Brasil)' })
    registerAppLocale('', { endonym: 'nothing' })

    expect(isRegisteredLocale('pt-br')).toBe(true)
    expect(languageOptions().map(option => option.id)).not.toContain('')
  })

  it('replaces one source atomically and unregisters a whole source', () => {
    replaceAppLocaleSource('backend', [
      { id: 'pl', endonym: 'Polski' },
      { id: 'pt-br', endonym: 'Português (Brasil)' }
    ])
    registerAppLocale('pl', { translations: { common: { save: 'Zachowaj' } } }, 'plugin:x')

    const before = $appLocaleVersion.get()
    replaceAppLocaleSource('backend', [{ id: 'uk', endonym: 'Українська' }])

    expect($appLocaleVersion.get()).toBe(before + 1)
    expect(isRegisteredLocale('pt-br')).toBe(false)
    expect(isRegisteredLocale('uk')).toBe(true)
    // The plugin's layer for pl survives the backend swap.
    expect(isRegisteredLocale('pl')).toBe(true)

    unregisterAppLocaleSource('plugin:x')
    expect(isRegisteredLocale('pl')).toBe(false)
  })
})

describe('languages + registry', () => {
  it('accepts a registered id as a locale and its config value, still mapping aliases first', () => {
    expect(isLocale('pl')).toBe(false)
    expect(normalizeLocale('pl')).toBe('en')
    expect(localeConfigValue('pl')).toBe('en')

    registerAppLocale('pl', { endonym: 'Polski' })

    expect(isLocale('pl')).toBe(true)
    expect(isSupportedLocaleValue('PL')).toBe(true)
    expect(normalizeLocale('pl_PL')).toBe('en')
    expect(normalizeLocale('PL')).toBe('pl')
    expect(localeConfigValue('pl')).toBe('pl')
    expect(normalizeLocale('zh-TW')).toBe('zh-hant')
  })

  it('lists bundled then registered languages by endonym, with registry rtl and source', () => {
    registerAppLocale('pl', { endonym: 'Polski', englishName: 'Polish' }, 'backend')
    registerAppLocale('he', { endonym: 'עברית', rtl: true }, 'plugin:hermes-lang-he')

    const options = languageOptions()
    const ids = options.map(option => option.id)

    expect(ids.slice(0, 9)).toEqual(Object.keys(TRANSLATIONS))
    expect(ids.slice(9)).toEqual(['he', 'pl'])
    expect(options.find(option => option.id === 'pl')).toMatchObject({
      endonym: 'Polski',
      englishName: 'Polish',
      rtl: false,
      source: 'backend'
    })
    expect(options.find(option => option.id === 'he')).toMatchObject({ rtl: true, source: 'plugin:hermes-lang-he' })
    expect(options.find(option => option.id === 'ar')?.rtl).toBe(true)
    expect(localeMeta('xx').endonym).toBe('xx')
  })
})

describe('translateNow', () => {
  it('reads registered packs for the runtime locale', () => {
    registerAppLocale('en', { translations: { common: { save: 'Keep' } } }, 'backend')

    expect(translateNow('common.save')).toBe('Keep')
    expect(translateNow('connectorsPage.searchPlaceholder', 2)).toBe(
      TRANSLATIONS.en.connectorsPage.searchPlaceholder(2)
    )
  })
})
