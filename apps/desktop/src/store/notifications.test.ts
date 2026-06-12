import { afterEach, describe, expect, it } from 'vitest'

import { $notifications, clearNotifications, notifyError } from './notifications'

describe('notifyError voice transcription summaries', () => {
  afterEach(() => {
    clearNotifications()
  })

  it('surfaces an actionable setup message when no STT provider is available', () => {
    notifyError(
      new Error(
        '400: {"detail":"No STT provider available. Install faster-whisper for free local transcription, configure HERMES_LOCAL_STT_COMMAND or set an API key."}'
      ),
      'Voice transcription failed'
    )

    expect($notifications.get()[0]).toMatchObject({
      kind: 'error',
      title: 'Voice transcription failed',
      message: 'No speech-to-text provider is configured. Install local STT or add a transcription API key in Settings.'
    })
  })

  it('explains local STT dependency failures without hiding the backend detail', () => {
    notifyError(new Error('400: {"detail":"Local transcription failed: faster-whisper not installed"}'), 'Voice transcription failed')

    expect($notifications.get()[0]).toMatchObject({
      message: 'Local speech-to-text is not ready. Install faster-whisper or choose another STT provider in Settings.',
      detail: 'Local transcription failed: faster-whisper not installed'
    })
  })

  it('explains configured local STT provider availability failures', () => {
    notifyError(new Error("400: {\"detail\":\"STT provider 'local' configured but unavailable\"}"), 'Voice transcription failed')

    expect($notifications.get()[0]).toMatchObject({
      message: 'Local speech-to-text is not ready. Install faster-whisper or choose another STT provider in Settings.',
      detail: "STT provider 'local' configured but unavailable"
    })
  })

  it('explains first-run transcription timeouts as a setup/loading condition', () => {
    notifyError(new Error('Timed out connecting to Hermes backend after 15000ms'), 'Voice transcription failed')

    expect($notifications.get()[0]).toMatchObject({
      message:
        'Transcription is still starting. First local STT setup can take a few minutes; try again once the model finishes loading.',
      detail: 'Timed out connecting to Hermes backend after 15000ms'
    })
  })
})
