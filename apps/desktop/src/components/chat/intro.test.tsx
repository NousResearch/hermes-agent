import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { Intro } from './intro'

describe('Intro i18n', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('uses catalog fallback copy when the active locale has no intro catalog', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0)

    render(
      <I18nProvider configClient={null} initialLocale="zh">
        <Intro personality="matrix" seed={0} />
      </I18nProvider>
    )

    expect(
      screen.getByText(
        "Send the task, file, or rough idea. I'll use your configured voice and keep the work grounded in this repo."
      )
    ).toBeTruthy()
  })
})
