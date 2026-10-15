import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import type { ComposerQuote } from '@/store/composer-quotes'

import { QuoteList } from './quote-list'

function makeQuote(id: string, text = 'quoted text'): ComposerQuote {
  return { id, role: 'assistant', text, timestamp: '12:45' }
}

async function renderWithI18n(ui: React.ReactNode) {
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
        {ui}
      </I18nProvider>
    )
  })

  return result!
}

describe('QuoteList', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders one card per quote with the quoted text', async () => {
    const { container } = await renderWithI18n(
      <QuoteList quotes={[makeQuote('a', '关闭 → 换包 → 自动重'), makeQuote('b', 'second')]} />
    )

    const cards = container.querySelectorAll('[data-slot="composer-quote-card"]')

    expect(cards).toHaveLength(2)
    expect(cards[0].textContent).toContain('关闭 → 换包 → 自动重')
    expect(cards[1].textContent).toContain('second')
  })

  it('collapses whitespace in the card line', async () => {
    const { container } = await renderWithI18n(<QuoteList quotes={[makeQuote('a', 'line one\nline two')]} />)

    expect(container.querySelector('[data-slot="composer-quote-card"]')?.textContent).toContain('line one line two')
  })

  it('calls onRemove with the quote id', async () => {
    const onRemove = vi.fn()

    const { container } = await renderWithI18n(
      <QuoteList onRemove={onRemove} quotes={[makeQuote('a'), makeQuote('b')]} />
    )

    const removeButtons = container.querySelectorAll('[data-slot="composer-quote-remove"]')

    expect(removeButtons).toHaveLength(2)
    fireEvent.click(removeButtons[1])

    expect(onRemove).toHaveBeenCalledWith('b')
  })

  it('does not crash when quotes contain undefined entries', async () => {
    const { container } = await renderWithI18n(
      <QuoteList quotes={[makeQuote('a'), undefined as unknown as ComposerQuote, makeQuote('b')]} />
    )

    expect(container.querySelectorAll('[data-slot="composer-quote-card"]')).toHaveLength(2)
  })

  it('exposes a remove label from i18n', async () => {
    await renderWithI18n(<QuoteList onRemove={() => undefined} quotes={[makeQuote('a')]} />)

    expect(screen.getByLabelText(/引用|Remove quote/)).toBeTruthy()
  })
})
