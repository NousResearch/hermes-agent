import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider, type Locale, useI18n } from '@/i18n'

import { Intro } from './intro'

function IntroLocaleProbe({ target }: { target: Locale }) {
  const { setLocale } = useI18n()

  return (
    <div>
      <Intro personality="none" seed={0} />
      <button onClick={() => void setLocale(target)} type="button">
        switch
      </button>
    </div>
  )
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('Intro i18n', () => {
  it('updates intro copy when I18nProvider changes locale', async () => {
    vi.spyOn(Math, 'random').mockReturnValue(0)

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <IntroLocaleProbe target="zh" />
      </I18nProvider>
    )

    expect(
      screen.getByText(
        'Ask a question, paste an error, or point me at a repo. I can read code, run tools, and help you ship.'
      )
    ).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'switch' }))

    await waitFor(() =>
      expect(screen.getByText('问问题、贴报错，或指向一个仓库。我可以读代码、运行工具，并帮你交付。')).toBeTruthy()
    )
  })

  it.each([
    ['zh-hant', '问问题、贴报错，或指向一个仓库。我可以读代码、运行工具，并帮你交付。'],
    ['ja', 'Ask a question, paste an error, or point me at a repo. I can read code, run tools, and help you ship.']
  ] as const)('uses the explicit intro fallback for %s', (locale, expectedBody) => {
    vi.spyOn(Math, 'random').mockReturnValue(0)

    render(
      <I18nProvider configClient={null} initialLocale={locale}>
        <Intro personality="none" seed={0} />
      </I18nProvider>
    )

    expect(screen.getByText(expectedBody)).toBeTruthy()
  })
})
