import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $threadScrolledUp, resetThreadScroll } from '@/store/thread-scroll'

import { ComposerStatusStack } from './index'

class TestResizeObserver {
  disconnect() {}
  observe() {}
  unobserve() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)

describe('ComposerStatusStack scroll treatment', () => {
  beforeEach(() => {
    $threadScrolledUp.set(true)
  })

  afterEach(() => {
    cleanup()
    resetThreadScroll()
  })

  it('dims only the status content while keeping the dock card opaque', () => {
    const view = render(
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <ComposerStatusStack queue={<div>Queued task</div>} sessionId={null} />
        </I18nProvider>
      </MemoryRouter>
    )

    const card = view.container.querySelector<HTMLElement>('[class*="bg-(--composer-fill)"]')
    const dimmedContent = screen.getByText('Queued task').closest<HTMLElement>('.opacity-30')

    expect(card).not.toBeNull()
    expect(card?.classList.contains('opacity-30')).toBe(false)
    expect(dimmedContent).not.toBeNull()
    expect(dimmedContent).not.toBe(card)
    expect(card?.contains(dimmedContent)).toBe(true)
  })
})
