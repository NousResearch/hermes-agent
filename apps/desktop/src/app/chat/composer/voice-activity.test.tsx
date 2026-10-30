import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { setVoicePlaybackState } from '@/store/voice-playback'

import { VoiceActivity, VoicePlaybackActivity } from './voice-activity'

afterEach(() => {
  cleanup()
  setVoicePlaybackState({
    audioElement: null,
    messageId: null,
    sequence: 0,
    sessionId: null,
    source: null,
    status: 'idle'
  })
})

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

// One renderer, one speaker, several sessions: the pill belongs to the chat
// that is actually speaking. Rendering it unowned put "Reading aloud" in every
// open chat at once.
describe('VoicePlaybackActivity session ownership', () => {
  const speakIn = (sessionId: string) =>
    setVoicePlaybackState({
      audioElement: null,
      messageId: 'm1',
      sequence: 1,
      sessionId,
      source: 'read-aloud',
      status: 'speaking'
    })

  const renderPill = (sessionId: string) =>
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <VoicePlaybackActivity sessionId={sessionId} />
      </I18nProvider>
    )

  it('renders the pill in the session that started the clip', () => {
    speakIn('session-a')
    renderPill('session-a')

    expect(screen.getByRole('status').textContent).toContain('Reading aloud')
  })

  it('renders nothing in a sibling session while another one speaks', () => {
    speakIn('session-a')
    renderPill('session-b')

    expect(screen.queryByRole('status')).toBeNull()
  })

  it('renders nothing once the clip is idle', () => {
    renderPill('session-a')

    expect(screen.queryByRole('status')).toBeNull()
  })
})
