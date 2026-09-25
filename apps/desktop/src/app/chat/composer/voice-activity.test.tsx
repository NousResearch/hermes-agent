import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'

import { VoiceActivity } from './voice-activity'

afterEach(cleanup)

describe('VoiceActivity accessibility', () => {
  it('keeps the dictation timer visible without announcing every tick', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <VoiceActivity state={{ elapsedSeconds: 65, level: 0.5, status: 'recording' }} />
      </I18nProvider>
    )

    const status = screen.getByRole('status')
    const timer = screen.getByText('1:05')

    expect(status.getAttribute('aria-live')).toBe('polite')
    expect(status.textContent).toContain('Dictating')
    expect(status.textContent).toContain('1:05')
    expect(timer.getAttribute('aria-hidden')).toBe('true')
  })
})
