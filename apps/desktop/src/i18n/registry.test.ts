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
    registerAppLocale('it', {
      endonym: 'Italiano',
      translations: { common: { save: 'Salva' } }
    })

    const it = resolveTranslations('it')

    expect(it.common.save).toBe('Salva')
    expect(it.common.cancel).toBe(TRANSLATIONS.en.common.cancel)
    expect(it.language.label).toBe(TRANSLATIONS.en.language.label)
  })

  it('wraps a pack string over a function-valued English entry into a positional formatter', () => {
    registerAppLocale('it', {
      translations: {
        'catalog.results': '{0} risultati',
        'connectorsPage.card.fact.toolsSomeOn': '{1} di {0} strumenti'
      }
    })

    const it = resolveTranslations('it')

    expect(it.catalog.results(3)).toBe('3 risultati')
    expect(it.connectorsPage.card.fact.toolsSomeOn(10, 4)).toBe('4 di 10 strumenti')
    expect(typeof it.catalog.installTitle).toBe('function')
  })

  it('keeps dotted leaf keys (keybinds.actions) addressable from a flat pack', () => {
    registerAppLocale('it', { translations: { 'keybinds.actions.session.new': 'Nuova sessione' } })

    const actions = resolveTranslations('it').keybinds.actions

    expect(actions['session.new']).toBe('Nuova sessione')
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
      'it',
      { translations: { common: { save: 'Salva', cancel: 'Annulla' } } },
      'backend'
    )

    const disposePlugin = registerAppLocale(
      'it',
      { translations: { common: { save: 'Conserva' } } },
      'plugin:hermes-lang-it'
    )

    expect(resolveTranslations('it').common.save).toBe('Conserva')
    expect(resolveTranslations('it').common.cancel).toBe('Annulla')

    disposePlugin()
    expect(resolveTranslations('it').common.save).toBe('Salva')
    expect(isRegisteredLocale('it')).toBe(true)

    disposeBackend()
    expect(isRegisteredLocale('it')).toBe(false)
    expect(resolveTranslations('it').common.save).toBe(TRANSLATIONS.en.common.save)
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
    expect(isLocale('it')).toBe(false)
    expect(normalizeLocale('it')).toBe('en')
    expect(localeConfigValue('it')).toBe('en')

    registerAppLocale('it', { endonym: 'Italiano' })

    expect(isLocale('it')).toBe(true)
    expect(isSupportedLocaleValue('IT')).toBe(true)
    expect(normalizeLocale('it_IT')).toBe('en')
    expect(normalizeLocale('IT')).toBe('it')
    expect(localeConfigValue('it')).toBe('it')
    expect(normalizeLocale('zh-TW')).toBe('zh-hant')
  })

  it('lists bundled then registered languages by endonym, with registry rtl and source', () => {
    registerAppLocale('it', { endonym: 'Italiano', englishName: 'Italian' }, 'backend')
    registerAppLocale('he', { endonym: 'עברית', rtl: true }, 'plugin:hermes-lang-he')

    const options = languageOptions()
    const ids = options.map(option => option.id)

    expect(ids.slice(0, Object.keys(TRANSLATIONS).length)).toEqual(Object.keys(TRANSLATIONS))
    expect(ids.slice(Object.keys(TRANSLATIONS).length)).toEqual(['he', 'it'])
    expect(options.find(option => option.id === 'it')).toMatchObject({
      endonym: 'Italiano',
      englishName: 'Italian',
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
    expect(translateNow('catalog.results', 2)).toBe(TRANSLATIONS.en.catalog.results(2))
  })
})
