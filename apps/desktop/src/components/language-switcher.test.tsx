import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesConfigRecord } from '@/hermes'
import { type I18nConfigClient, I18nProvider } from '@/i18n'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { LanguageSwitcher } from './language-switcher'

stubResizeObserver()
stubMenuDomApis()
describe('LanguageSwitcher', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it.each([
    ['ja', '日本語'],
    ['ko', '한국어']
  ])('persists %s through the language picker', async (locale, name) => {
    const saveConfig = vi.fn().mockResolvedValue({ ok: true })
    const latestConfig: HermesConfigRecord = { display: { language: 'en', skin: 'slate' } }

    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockResolvedValue(latestConfig),
      saveConfig
    }

    render(
      <I18nProvider configClient={configClient}>
        <LanguageSwitcher />
      </I18nProvider>
    )

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Switch language' }).hasAttribute('disabled')).toBe(false)
    })

    fireEvent.click(screen.getByRole('button', { name: 'Switch language' }))
    fireEvent.click(screen.getByRole('option', { name: new RegExp(name) }))

    await waitFor(() => expect(saveConfig).toHaveBeenCalledTimes(1))
    expect(saveConfig).toHaveBeenCalledWith({ display: { language: locale, skin: 'slate' } })
  })
})
