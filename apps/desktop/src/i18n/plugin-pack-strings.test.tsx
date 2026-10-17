import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from './context'
import { registerPluginLocales, translatePlugin, usePluginI18n } from './plugin-i18n'
import { replaceAppLocaleSource, resetAppLocaleRegistry, resolveTranslations } from './registry'

afterEach(() => {
  cleanup()
  resetAppLocaleRegistry()
})

describe('language packs reach plugin-owned bundles', () => {
  it('routes `plugins.<id>.*` pack keys to that plugin, never into the app catalog', () => {
    const dispose = registerPluginLocales('board', {
      en: { title: 'Board', moveTo: (label: string) => `Move to ${label}`, other: 'Other' }
    })

    replaceAppLocaleSource('backend', [
      {
        id: 'pl',
        translations: {
          'common.save': 'Zapisz',
          'plugins.board.title': 'Tablica',
          'plugins.board.moveTo': 'Przenieś do {0}'
        }
      }
    ])

    expect(translatePlugin('board', 'pl', 'title', [])).toBe('Tablica')
    // A YAML string standing where the plugin has a function is a positional formatter.
    expect(translatePlugin('board', 'pl', 'moveTo', ['Gotowe'])).toBe('Przenieś do Gotowe')
    // Keys the pack does not carry fall back to the plugin's English.
    expect(translatePlugin('board', 'pl', 'other', [])).toBe('Other')
    // Other plugins are untouched, and the app catalog carries no plugin branch.
    expect(translatePlugin('elsewhere', 'pl', 'title', [])).toBe('title')
    expect(resolveTranslations('pl').common.save).toBe('Zapisz')
    expect(resolveTranslations('pl')).not.toHaveProperty('plugins')

    dispose()
  })

  it('layers over the plugin’s own bundle for a bundled language and drops with its source', () => {
    const dispose = registerPluginLocales('board', {
      en: { title: 'Board', other: 'Other' },
      ja: { title: 'ボード', other: 'その他' }
    })

    replaceAppLocaleSource('backend', [{ id: 'ja', translations: { 'plugins.board.title': 'かんばん' } }])

    expect(translatePlugin('board', 'ja', 'title', [])).toBe('かんばん')
    expect(translatePlugin('board', 'ja', 'other', [])).toBe('その他')

    replaceAppLocaleSource('backend', [])

    expect(translatePlugin('board', 'ja', 'title', [])).toBe('ボード')

    dispose()
  })

  it('re-renders a mounted plugin translator when the pack lands after first paint', () => {
    const dispose = registerPluginLocales('board', { en: { title: 'Board' } })

    function Title() {
      const t = usePluginI18n('board')

      return <span>{t('title')}</span>
    }

    replaceAppLocaleSource('backend', [{ id: 'pl', translations: { 'common.save': 'Zapisz' } }])
    render(
      <I18nProvider configClient={null} initialLocale="pl">
        <Title />
      </I18nProvider>
    )
    expect(screen.getByText('Board')).toBeTruthy()

    act(() => {
      replaceAppLocaleSource('backend', [
        { id: 'pl', translations: { 'common.save': 'Zapisz', 'plugins.board.title': 'Tablica' } }
      ])
    })

    expect(screen.getByText('Tablica')).toBeTruthy()

    dispose()
  })
})
