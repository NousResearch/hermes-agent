import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import type { VoiceActivityState } from './types'
import { VoiceActivity } from './voice-activity'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function pane(state: VoiceActivityState) {
  return (
    <I18nProvider configClient={null} initialLocale="en">
      <VoiceActivity state={state} />
    </I18nProvider>
  )
}

describe('VoiceActivity accessibility', () => {
  it('keeps the dictation timer visible without announcing every tick', () => {
    render(pane({ elapsedSeconds: 65, level: 0.5, status: 'recording' }))

    const status = screen.getByRole('status')
    const timer = screen.getByText('1:05')

    expect(status.getAttribute('aria-live')).toBe('polite')
    expect(status.textContent).toContain('Dictating')
    expect(status.textContent).toContain('1:05')
    expect(timer.getAttribute('aria-hidden')).toBe('true')
  })
})

describe('VoiceActivity live transcript (stt.streaming)', () => {
  const SHORT = 'so the plan for tomorrow'

  const LONG =
    'so the plan for tomorrow is to pick up the parcel before lunch, call the landlord about the heater, ' +
    'then finish the cover letter and send it before five'

  // jsdom has no layout: give every box more content than it shows, the way
  // the pane looks once speech outgrows it.
  const SCROLL_HEIGHT = 240

  const dictating = (partial: string): VoiceActivityState => ({
    elapsedSeconds: 12,
    level: 0.4,
    partial,
    status: 'recording'
  })

  /** Whether the element holding the text, or a box around it, is scrolled to its end. */
  function scrolledToEnd(text: HTMLElement) {
    for (let el: HTMLElement | null = text; el; el = el.parentElement) {
      if (el.scrollTop > 0 && el.scrollTop >= el.scrollHeight - el.clientHeight) {
        return true
      }
    }

    return false
  }

  function speak() {
    vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockReturnValue(SCROLL_HEIGHT)
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(60)
    const view = render(pane(dictating(SHORT)))
    view.rerender(pane(dictating(LONG)))

    return screen.getByText(LONG)
  }

  it('keeps the newest recognised words in view as speech outgrows the pane', () => {
    expect(scrolledToEnd(speak())).toBe(true)
  })

  it('announces the dictation state, not the whole transcript on every update', () => {
    const transcript = speak()

    expect(screen.getByRole('status').textContent).toContain('Dictating')
    expect(transcript.closest('[aria-live]')?.getAttribute('aria-live')).toBe('off')
  })
})
