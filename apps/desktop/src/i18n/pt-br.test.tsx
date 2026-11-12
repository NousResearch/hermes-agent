import { flattenMessageKeys } from '@hermes/shared/i18n'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { LanguageSwitcher } from '@/components/language-switcher'
import type { HermesConfigRecord } from '@/hermes'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { type I18nConfigClient, I18nProvider, useI18n } from './context'
import { en } from './en'
import { languageOptions, localeConfigValue, normalizeLocale, resolveInitialLocale } from './languages'
import { ptBrOverrides } from './pt-br'
import { resolveTranslations } from './registry'

stubResizeObserver()
stubMenuDomApis()
afterEach(cleanup)

function LanguageSample() {
  const { t, saveError } = useI18n()

  return (
    <>
      <LanguageSwitcher />
      <p>{t.common.save}</p>
      <output>{saveError?.message}</output>
    </>
  )
}

describe('Brazilian Portuguese desktop language', () => {
  it('supplies the full desktop copy contract instead of an English-only overlay', () => {
    const visibleKeys = (tree: unknown) => flattenMessageKeys(tree).filter(key => !key.startsWith('intro.'))
    expect(visibleKeys(ptBrOverrides)).toEqual(visibleKeys(en))
  })

  it('offers Brazilian Portuguese as a bundled language and resolves regional aliases', () => {
    expect(languageOptions()).toContainEqual(
      expect.objectContaining({
        id: 'pt-br',
        endonym: 'Português (Brasil)',
        source: 'bundled'
      })
    )

    for (const alias of ['pt-BR', 'pt_br', ' PT-br ', 'pt']) {
      expect(normalizeLocale(alias)).toBe('pt-br')
    }

    expect(localeConfigValue('pt-br')).toBe('pt-br')
    expect(resolveInitialLocale(null, 'pt-BR')).toBe('pt-br')
    expect(resolveInitialLocale('en', 'pt-BR')).toBe('en')
    expect(resolveTranslations('pt-br').common.save).toBe('Salvar')
  })

  it('localizes the welcome rotation without changing personality identifiers', () => {
    const intro = resolveTranslations('pt-br').intro
    expect(intro.stock.helpful).toHaveLength(5)
    expect(intro.stock.helpful[0]).toContain('Peça')
    expect(intro.custom('MINHA_PERSONALIDADE')).toHaveLength(5)
    expect(intro.custom('MINHA_PERSONALIDADE').join(' ')).toContain('MINHA_PERSONALIDADE')
  })

  it('selects Portuguese in settings, preserves other config, and restores it on remount', async () => {
    let saved: HermesConfigRecord = { display: { language: 'en', skin: 'slate' }, terminal: { timeout: 60 } }

    const configClient: I18nConfigClient = {
      getConfig: async () => saved,
      saveConfig: async config => {
        saved = config

        return { ok: true }
      }
    }

    const view = render(
      <I18nProvider configClient={configClient}>
        <LanguageSample />
      </I18nProvider>
    )

    await screen.findByText('Save')
    fireEvent.click(screen.getByRole('button', { name: 'Switch language' }))
    fireEvent.change(screen.getByPlaceholderText(en.language.searchPlaceholder), { target: { value: 'pt-br' } })
    fireEvent.click(screen.getByRole('option', { name: /Português \(Brasil\)/ }))
    await waitFor(() =>
      expect(saved).toEqual({
        display: { language: 'pt-br', skin: 'slate' },
        terminal: { timeout: 60 }
      })
    )
    await screen.findByText('Salvar')
    expect(globalThis.document.documentElement.lang).toBe('pt-br')
    expect(globalThis.document.documentElement.dir).toBe('ltr')
    view.unmount()
    render(
      <I18nProvider configClient={configClient}>
        <LanguageSample />
      </I18nProvider>
    )
    await screen.findByText('Salvar')
    fireEvent.click(screen.getByRole('button'))
    fireEvent.click(screen.getByRole('option', { name: /English/ }))
    await screen.findByText('Save')
    await waitFor(() => expect((saved.display as { language: string }).language).toBe('en'))
  })
})
