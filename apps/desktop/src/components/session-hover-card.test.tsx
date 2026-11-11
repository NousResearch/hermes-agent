import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { SessionHoverCard } from '@/components/session-hover-card'
import { I18nProvider } from '@/i18n'
import type { SessionInfo } from '@/types/hermes'

const SESSION: SessionInfo = {
  id: 's1',
  title: 'Refactor the sidebar',
  preview: 'first message preview',
  summary: 'The user asked for the sidebar to be refactored.',
  summary_updated_at: 1_700_000_000,
  summary_model: 'openai/gpt-4o-mini',
  message_count: 12,
} as SessionInfo

function mount(ui: React.ReactNode) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      {ui}
    </I18nProvider>
  )
}

describe('SessionHoverCard', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    cleanup()
  })

  it('renders no card until the open delay elapses', () => {
    mount(
      <SessionHoverCard session={SESSION}>
        <button type="button">row</button>
      </SessionHoverCard>
    )

    fireEvent.mouseEnter(screen.getByRole('button', { name: 'row' }))
    expect(screen.queryByRole('tooltip')).toBeNull()

    act(() => {
      vi.advanceTimersByTime(199)
    })
    expect(screen.queryByRole('tooltip')).toBeNull()

    act(() => {
      vi.advanceTimersByTime(1)
    })
    expect(screen.getByRole('tooltip')).not.toBeNull()
  })

  it('shows the cached summary rather than the preview when both exist', () => {
    mount(
      <SessionHoverCard session={SESSION}>
        <button type="button">row</button>
      </SessionHoverCard>
    )

    fireEvent.mouseEnter(screen.getByRole('button', { name: 'row' }))
    act(() => {
      vi.advanceTimersByTime(200)
    })

    const card = screen.getByRole('tooltip')
    expect(card.textContent).toContain('The user asked for the sidebar to be refactored.')
    expect(card.textContent).not.toContain('first message preview')
  })

  it('falls back to the preview when the session has no summary', () => {
    mount(
      <SessionHoverCard session={{ ...SESSION, summary: undefined }}>
        <button type="button">row</button>
      </SessionHoverCard>
    )

    fireEvent.mouseEnter(screen.getByRole('button', { name: 'row' }))
    act(() => {
      vi.advanceTimersByTime(200)
    })

    expect(screen.getByRole('tooltip').textContent).toContain('first message preview')
  })

  it('renders no card at all when there is neither a summary nor a preview', () => {
    mount(
      <SessionHoverCard session={{ ...SESSION, summary: undefined, preview: null }}>
        <button type="button">row</button>
      </SessionHoverCard>
    )

    fireEvent.mouseEnter(screen.getByRole('button', { name: 'row' }))
    act(() => {
      vi.advanceTimersByTime(1000)
    })

    expect(screen.queryByRole('tooltip')).toBeNull()
  })

  it('closes only after the close delay, so a swipe across the row does not flash the card', () => {
    mount(
      <SessionHoverCard session={SESSION}>
        <button type="button">row</button>
      </SessionHoverCard>
    )

    const trigger = screen.getByRole('button', { name: 'row' })
    fireEvent.mouseEnter(trigger)
    act(() => {
      vi.advanceTimersByTime(200)
    })
    expect(screen.getByRole('tooltip')).not.toBeNull()

    fireEvent.mouseLeave(trigger)
    act(() => {
      vi.advanceTimersByTime(99)
    })
    expect(screen.getByRole('tooltip')).not.toBeNull()

    act(() => {
      vi.advanceTimersByTime(1)
    })
    expect(screen.queryByRole('tooltip')).toBeNull()
  })

  it('opens on keyboard focus, not only on hover', () => {
    mount(
      <SessionHoverCard session={SESSION}>
        <button type="button">row</button>
      </SessionHoverCard>
    )

    fireEvent.focus(screen.getByRole('button', { name: 'row' }))
    act(() => {
      vi.advanceTimersByTime(200)
    })

    expect(screen.getByRole('tooltip')).not.toBeNull()
  })

  it('keeps the card open while the pointer moves onto the card itself', () => {
    mount(
      <SessionHoverCard session={SESSION}>
        <button type="button">row</button>
      </SessionHoverCard>
    )

    const trigger = screen.getByRole('button', { name: 'row' })
    fireEvent.mouseEnter(trigger)
    act(() => {
      vi.advanceTimersByTime(200)
    })

    const card = screen.getByRole('tooltip')
    fireEvent.mouseLeave(trigger)
    fireEvent.mouseEnter(card)
    act(() => {
      vi.advanceTimersByTime(500)
    })

    expect(screen.getByRole('tooltip')).not.toBeNull()
  })

  it('stays closed on a narrow viewport', () => {
    // The card is suppressed below 600px; the row keeps a shorter
    // highlight instead, so no popover may be mounted there.
    const original = window.matchMedia
    window.matchMedia = ((query: string) => ({
      matches: query.includes('max-width'),
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn()
    })) as unknown as typeof window.matchMedia

    try {
      mount(
        <SessionHoverCard session={SESSION}>
          <button type="button">row</button>
        </SessionHoverCard>
      )

      fireEvent.mouseEnter(screen.getByRole('button', { name: 'row' }))
      act(() => {
        vi.advanceTimersByTime(1000)
      })

      expect(screen.queryByRole('tooltip')).toBeNull()
    } finally {
      window.matchMedia = original
    }
  })

  it('uses the session title as the card heading', () => {
    mount(
      <SessionHoverCard session={SESSION}>
        <button type="button">row</button>
      </SessionHoverCard>
    )

    fireEvent.mouseEnter(screen.getByRole('button', { name: 'row' }))
    act(() => {
      vi.advanceTimersByTime(200)
    })

    expect(screen.getByRole('tooltip').textContent).toContain('Refactor the sidebar')
  })

  it('localizes the footer instead of hardcoding English', () => {
    mount(
      <SessionHoverCard session={SESSION}>
        <button type="button">row</button>
      </SessionHoverCard>
    )

    fireEvent.mouseEnter(screen.getByRole('button', { name: 'row' }))
    act(() => {
      vi.advanceTimersByTime(200)
    })

    const card = screen.getByRole('tooltip')
    // "Updated <age> · 12 messages · gpt-4o-mini" — the age and the
    // count come from the translation table.
    expect(card.textContent).toContain('12 messages')
    expect(card.textContent).toContain('gpt-4o-mini')
  })
})
