import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { KANBAN_LOCALES } from './i18n'
import { PrLink } from './pr-link'

const pr = 'https://github.com/acme/widgets/pull/123'

afterEach(cleanup)

function renderInCard(url: string, onCardClick: () => void) {
  const dispose = registerPluginLocales('kanban', KANBAN_LOCALES)

  // A button stands in for the card: one big click target with the chip inside.
  const result = render(
    <button onClick={onCardClick} type="button">
      <PrLink url={url} />
    </button>
  )

  return { ...result, dispose }
}

describe('PrLink', () => {
  it('renders the PR number and does not open the card it sits on', () => {
    const onCardClick = vi.fn()
    const { dispose } = renderInCard(pr, onCardClick)

    const link = screen.getByRole('link', { name: '#123' })
    expect(link.getAttribute('href')).toBe(pr)

    // A card is one big click target; the chip inside it opens the PR only.
    fireEvent.click(link)
    expect(onCardClick).not.toHaveBeenCalled()
    dispose()
  })

  it('renders nothing for a URL that is not a pull request', () => {
    const onCardClick = vi.fn()
    const { container, dispose } = renderInCard('https://github.com/acme/widgets', onCardClick)

    expect(container.querySelector('a')).toBeNull()
    dispose()
  })
})
