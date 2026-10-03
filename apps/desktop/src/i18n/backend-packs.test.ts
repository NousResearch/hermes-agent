import { JsonRpcGatewayError } from '@hermes/shared'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type BackendLocaleRequest, syncBackendLocalePacks } from './backend-packs'
import { TRANSLATIONS } from './catalog'
import { languageOptions } from './languages'
import { isRegisteredLocale, registerAppLocale, resetAppLocaleRegistry, resolveTranslations } from './registry'

afterEach(() => {
  resetAppLocaleRegistry()
  vi.restoreAllMocks()
})

type Call = [string, Record<string, unknown>]

const backend =
  (languages: unknown, catalogs: Record<string, Record<string, string>>, calls: Call[] = []): BackendLocaleRequest =>
  async <T>(method: string, params: Record<string, unknown>) => {
    calls.push([method, params])

    if (method === 'i18n.languages') {
      return languages as T
    }

    if (method === 'i18n.catalog') {
      const lang = String(params.lang)

      return { lang, surface: params.surface, messages: catalogs[lang] ?? {} } as T
    }

    throw new JsonRpcGatewayError(`unknown method: ${method}`, { code: -32601 })
  }

describe('syncBackendLocalePacks', () => {
  it('registers the backend language list and the desktop pack for the requested language', async () => {
    const calls: Call[] = []

    const request = backend(
      {
        languages: [
          { id: 'en', endonym: 'English', rtl: false, source: 'bundled' },
          { id: 'pl', endonym: 'Polski', rtl: false, source: 'plugin:hermes-lang-pl' }
        ]
      },
      { pl: { 'common.save': 'Zapisz', 'catalog.results': '{0} wyników' } },
      calls
    )

    await syncBackendLocalePacks(request, 'pl')

    expect(calls).toContainEqual(['i18n.catalog', { lang: 'pl', surface: 'desktop' }])
    expect(isRegisteredLocale('pl')).toBe(true)
    expect(languageOptions().find(option => option.id === 'pl')).toMatchObject({ endonym: 'Polski', source: 'backend' })
    expect(resolveTranslations('pl').common.save).toBe('Zapisz')
    expect(resolveTranslations('pl').catalog.results(5)).toBe('5 wyników')
    expect(resolveTranslations('pl').common.cancel).toBe(TRANSLATIONS.en.common.cancel)
  })

  it('accepts a bare array from i18n.languages and skips the catalog call without a requested language', async () => {
    const calls: Call[] = []

    await syncBackendLocalePacks(backend([{ id: 'uk', endonym: 'Українська' }], {}, calls), null)

    expect(calls).toEqual([['i18n.languages', {}]])
    expect(isRegisteredLocale('uk')).toBe(true)
  })

  it('is silent on a backend that predates the i18n methods and leaves the registry alone', async () => {
    const debug = vi.spyOn(console, 'debug').mockImplementation(() => {})
    registerAppLocale('pl', { endonym: 'Polski' }, 'backend')

    const request: BackendLocaleRequest = async () => {
      throw new JsonRpcGatewayError('unknown method: i18n.languages', { code: -32601 })
    }

    await expect(syncBackendLocalePacks(request, 'pl')).resolves.toBeUndefined()

    expect(debug).not.toHaveBeenCalled()
    expect(isRegisteredLocale('pl')).toBe(true)
  })

  it('replaces the previous backend layer and never applies a stale reply', async () => {
    await syncBackendLocalePacks(backend([{ id: 'pl', endonym: 'Polski' }], {}), null)
    expect(isRegisteredLocale('pl')).toBe(true)

    await syncBackendLocalePacks(backend([{ id: 'uk', endonym: 'Українська' }], {}), null)
    expect(isRegisteredLocale('pl')).toBe(false)
    expect(isRegisteredLocale('uk')).toBe(true)

    await syncBackendLocalePacks(backend([{ id: 'pt-br', endonym: 'Português' }], {}), null, () => false)
    expect(isRegisteredLocale('uk')).toBe(true)
    expect(isRegisteredLocale('pt-br')).toBe(false)
  })
})
