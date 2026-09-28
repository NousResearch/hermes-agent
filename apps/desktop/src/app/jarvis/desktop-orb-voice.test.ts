import { describe, expect, it } from 'vitest'

import type { ConversationStatus } from '@/app/chat/composer/hooks/use-voice-conversation'

import { desktopOrbVoice } from './desktop-orb-voice'

describe('desktopOrbVoice', () => {
  it('is idle with no conversation, whatever the last status was', () => {
    for (const status of ['idle', 'listening', 'transcribing', 'thinking', 'speaking'] as ConversationStatus[]) {
      expect(desktopOrbVoice(status, false)).toBe('idle')
    }
  })

  it('listens while the microphone is open, including while the request is being transcribed', () => {
    expect(desktopOrbVoice('listening', true)).toBe('listening')
    expect(desktopOrbVoice('transcribing', true)).toBe('listening')
    // Before the Live session reports its first status the mic is already ours.
    expect(desktopOrbVoice('idle', true)).toBe('listening')
  })

  it('speaks while the Live voice talks, so the orb follows Gemini rather than the gateway', () => {
    expect(desktopOrbVoice('speaking', true)).toBe('speaking')
  })

  it('hands a thinking turn back to the task phase instead of claiming the microphone', () => {
    expect(desktopOrbVoice('thinking', true)).toBe('idle')
  })
})
