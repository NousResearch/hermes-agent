import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import type { VoiceActivityState } from './types'
import { VoiceActivity } from './voice-activity'

// The cancel affordance lives in this panel — which only exists while a
// dictation is in flight — rather than in the composer control row, where
// inserting a button would shift the model pill and sibling icons sideways
// mid-recording.

function renderActivity(state: Partial<VoiceActivityState> = {}, onCancel?: () => void) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <VoiceActivity
        onCancel={onCancel}
        state={{ elapsedSeconds: 3, level: 0.4, status: 'recording', ...state }}
      />
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
})

describe('VoiceActivity cancel', () => {
  it('offers cancel while recording', () => {
    const onCancel = vi.fn()
    renderActivity({}, onCancel)

    fireEvent.click(screen.getByLabelText('Cancel dictation'))

    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it('hides cancel once transcription starts — the audio is already spent', () => {
    renderActivity({ status: 'transcribing' }, vi.fn())

    expect(screen.queryByLabelText('Cancel dictation')).toBeNull()
  })

  it('renders nothing at all when idle', () => {
    const { container } = renderActivity({ status: 'idle' }, vi.fn())

    expect(container.firstChild).toBeNull()
  })

  it('omits cancel when no handler is wired (back-compat)', () => {
    renderActivity()

    expect(screen.queryByLabelText('Cancel dictation')).toBeNull()
    expect(screen.getByRole('status')).toBeTruthy()
  })
})
