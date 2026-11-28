import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type I18nConfigClient, I18nProvider, setRuntimeI18nLocale, TRANSLATIONS, useI18n } from '@/i18n'
import { $draftTitles, clearSessionDraft, stashSessionDraft } from '@/store/composer'

import { SessionDraftTitle } from './session-draft-title'

function LocaleSwitchButton() {
  const { locale, setLocale } = useI18n()

  return (
    <button onClick={() => void setLocale(locale === 'es' ? 'en' : 'es')} type="button">
      toggle-locale
    </button>
  )
}

describe('SessionDraftTitle i18n reactivity', () => {
  afterEach(() => {
    cleanup()
    clearSessionDraft('draft-tile-1')
    $draftTitles.set({})
    setRuntimeI18nLocale('en')
    vi.restoreAllMocks()
  })

  it('updates live when display.language loads asynchronously after first render', async () => {
    let resolveConfig: ((config: Record<string, unknown>) => void) | null = null

    const configClient: I18nConfigClient = {
      getConfig: () =>
        new Promise(resolve => {
          resolveConfig = resolve
        }),
      saveConfig: vi.fn().mockResolvedValue({ ok: true })
    }

    render(
      <I18nProvider configClient={configClient}>
        <span data-testid="draft-title">
          <SessionDraftTitle scope={null} />
        </span>
      </I18nProvider>
    )

    expect(screen.getByTestId('draft-title').textContent).toBe(TRANSLATIONS.en.sidebar.nav['new-session'])

    await act(async () => {
      resolveConfig?.({ display: { language: 'es' } })
    })

    await waitFor(() =>
      expect(screen.getByTestId('draft-title').textContent).toBe(TRANSLATIONS.es.sidebar.nav['new-session'])
    )
  })

  it('updates live when switching locales and yields to typed draft text', async () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <span data-testid="draft-title">
          <SessionDraftTitle scope="draft-tile-1" />
        </span>
        <LocaleSwitchButton />
      </I18nProvider>
    )

    expect(screen.getByTestId('draft-title').textContent).toBe(TRANSLATIONS.en.sidebar.nav['new-session'])

    fireEvent.click(screen.getByRole('button', { name: 'toggle-locale' }))
    await waitFor(() =>
      expect(screen.getByTestId('draft-title').textContent).toBe(TRANSLATIONS.es.sidebar.nav['new-session'])
    )

    act(() => {
      stashSessionDraft('draft-tile-1', 'Hola mundo', [])
    })
    expect(screen.getByTestId('draft-title').textContent).toBe('Hola mundo')

    act(() => {
      clearSessionDraft('draft-tile-1')
    })
    expect(screen.getByTestId('draft-title').textContent).toBe(TRANSLATIONS.es.sidebar.nav['new-session'])

    fireEvent.click(screen.getByRole('button', { name: 'toggle-locale' }))
    await waitFor(() =>
      expect(screen.getByTestId('draft-title').textContent).toBe(TRANSLATIONS.en.sidebar.nav['new-session'])
    )
  })
})
