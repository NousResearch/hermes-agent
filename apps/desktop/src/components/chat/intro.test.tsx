import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { Intro } from './intro'

describe('Intro', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('preserves personality copy in English', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0)

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <Intro personality="concise" seed={0} />
      </I18nProvider>
    )

    expect(screen.getByText("Describe the task. I'll do it.")).toBeTruthy()
  })

  it('uses localized stable copy outside English', () => {
    render(
      <I18nProvider configClient={null} initialLocale="zh-hant">
        <Intro personality="none" seed={0} />
      </I18nProvider>
    )

    expect(screen.getByText('提出問題、貼上錯誤訊息，或告訴我儲存庫位置。我可以讀取程式碼、執行工具，協助你完成工作。')).toBeTruthy()
    expect(screen.queryByText(/Ask a question/)).toBeNull()
  })
})
