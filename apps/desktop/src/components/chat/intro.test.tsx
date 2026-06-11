import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { Intro } from './intro'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('Intro i18n', () => {
  it('uses Simplified Chinese intro copy for the zh locale', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0)

    render(
      <I18nProvider configClient={null} initialLocale="zh">
        <Intro personality="none" seed={0} />
      </I18nProvider>
    )

    expect(screen.getByText('问问题、贴报错，或指向一个仓库。我可以读代码、运行工具，并帮你交付。')).toBeTruthy()
    expect(screen.queryByText(/Ask a question/i)).toBeNull()
  })
})
