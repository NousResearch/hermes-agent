import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider, TRANSLATIONS, useI18n } from '@/i18n'
import type { I18nContextValue } from '@/i18n'

import { UninstallSection } from './uninstall-section'

let i18n: I18nContextValue

function Surface() {
  i18n = useI18n()

  return <UninstallSection />
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})
it.each(['gui', 'lite', 'full'] as const)(
  'localizes confirmation for %s without changing mode or running before confirmation',
  async mode => {
    const run = vi.fn().mockResolvedValue({ ok: false })
    vi.stubGlobal('hermesDesktop', {
      uninstall: { summary: async () => ({ agent_installed: true, running_app_path: '/fixture/Hermes.app' }), run }
    })
    render(
      <I18nProvider configClient={null} initialLocale="zh">
        <Surface />
      </I18nProvider>
    )
    const zh = TRANSLATIONS.zh.settings.uninstallSection
    await screen.findByText(zh.uninstallHermes)
    fireEvent.click(screen.getByRole('button', { name: new RegExp(zh.options[mode].title) }))
    expect(screen.getByText(zh.confirmBody(zh.options[mode].consequence))).toBeTruthy()
    expect(run).not.toHaveBeenCalled()
    await act(() => i18n.setLocale('ja'))
    const ja = TRANSLATIONS.ja.settings.uninstallSection
    expect(screen.getByText(ja.confirmBody(ja.options[mode].consequence))).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: ja.yesUninstall }))
    expect(run).toHaveBeenCalledWith(mode)
  }
)
