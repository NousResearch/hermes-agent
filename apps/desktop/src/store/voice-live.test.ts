import { describe, expect, it } from 'vitest'

import { $voiceLiveGrokStatus, $voiceLiveStatus, selectedVoiceChatMode } from './voice-live'

describe('selectedVoiceChatMode', () => {
  it('falls back to chained when the backend has not answered', () => {
    expect(selectedVoiceChatMode(null)).toBe('chained')
  })

  it('falls back to chained when the backend predates the mode', () => {
    expect(
      selectedVoiceChatMode({ available: false, mode: 'unknown-future-mode' as never, model: '', reason: null, voice: '' })
    ).toBe('chained')
  })

  it('reports gpt-live when the status says so', () => {
    expect(selectedVoiceChatMode({ available: true, mode: 'gpt-live', model: 'gpt-live-1', reason: null, voice: 'marin' })).toBe(
      'gpt-live'
    )
  })

  it('reports grok-live when the status says so — the new third engine', () => {
    expect(
      selectedVoiceChatMode({ available: true, mode: 'grok-live', model: 'grok-voice-latest', reason: null, voice: 'eve' })
    ).toBe('grok-live')
  })

  it('reads the ambient $voiceLiveStatus atom when no argument is given', () => {
    $voiceLiveStatus.set({ available: true, mode: 'grok-live', model: 'grok-voice-latest', reason: null, voice: 'eve' })

    expect(selectedVoiceChatMode()).toBe('grok-live')

    $voiceLiveStatus.set(null)
  })
})

describe('$voiceLiveGrokStatus', () => {
  it('is a separate atom from $voiceLiveStatus (parallel status route, SPEC §11)', () => {
    $voiceLiveStatus.set({ available: true, mode: 'grok-live', model: 'x', reason: null, voice: 'x' })
    $voiceLiveGrokStatus.set(null)

    expect($voiceLiveGrokStatus.get()).toBeNull()
    expect($voiceLiveStatus.get()).not.toBeNull()

    $voiceLiveStatus.set(null)
  })
})
