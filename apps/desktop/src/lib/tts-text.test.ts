import { describe, expect, it } from 'vitest'

import { normalizeTextForSpeech } from './tts-text'

describe('normalizeTextForSpeech', () => {
  it('replaces filename, path, and identifier-dense tokens with speakable labels', () => {
    expect(
      normalizeTextForSpeech(
        'Saved peyton-sample-20260922.wav at /tmp/voice/peyton-sample-20260922.ogg for job_7f3a9c2e-19ab.'
      )
    ).toBe('Saved a file at a file for an identifier.')
  })

  it('preserves ordinary prose and punctuation', () => {
    expect(normalizeTextForSpeech('The samples played cleanly, so listen again.')).toBe(
      'The samples played cleanly, so listen again.'
    )
  })

  it('normalizes UUIDs and code-like calls without changing surrounding prose', () => {
    expect(normalizeTextForSpeech('Retry 550e8400-e29b-41d4-a716-446655440000 with client.playSpeech().')).toBe(
      'Retry an identifier with an identifier.'
    )
  })
})
