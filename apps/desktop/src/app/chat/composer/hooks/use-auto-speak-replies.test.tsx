import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import * as voicePlayback from '@/lib/voice-playback'

const { ownsAmbientCue, playSpeechText } = vi.hoisted(() => ({
  ownsAmbientCue: vi.fn(),
  playSpeechText: vi.fn()
}))

vi.mock('@/store/ambient', () => ({ ownsAmbientCue }))
vi.mock('@/lib/voice-playback', async importOriginal => ({
  ...(await importOriginal<typeof voicePlayback>()),
  playSpeechText
}))

import { setVoicePlaybackState } from '@/store/voice-playback'
import { $autoSpeakReplies } from '@/store/voice-prefs'

import { type ComposerScope, ComposerScopeProvider } from '../scope'

import { useAutoSpeakReplies } from './use-auto-speak-replies'

const messages = atom([])

const scope = {
  $awaitingInput: atom(false),
  $messages: messages,
  attachments: {} as ComposerScope['attachments'],
  target: 'main'
} satisfies ComposerScope

function wrapper({ children }: { children: ReactNode }) {
  return <ComposerScopeProvider value={scope}>{children}</ComposerScopeProvider>
}

describe('useAutoSpeakReplies playback ownership', () => {
  beforeEach(() => {
    ownsAmbientCue.mockReset()
    playSpeechText.mockReset()
    playSpeechText.mockResolvedValue(true)
    messages.set([])
    $autoSpeakReplies.set(true)
    voicePlayback.stopVoicePlayback()
  })

  afterEach(() => {
    cleanup()
    $autoSpeakReplies.set(false)
    voicePlayback.stopVoicePlayback()
  })

  it('speaks the current reply after this window wins ambient ownership', async () => {
    let reply: { id: string; pending: boolean; text: string } | null = null
    let consumed = false

    ownsAmbientCue.mockResolvedValue(true)
    renderHook(
      () =>
        useAutoSpeakReplies({
          conversationActive: false,
          failureLabel: 'Playback failed',
          markSpoken: () => {
            consumed = true
          },
          pendingReply: () => (consumed ? null : reply),
          sessionId: 'session-1'
        }),
      { wrapper }
    )

    reply = { id: 'assistant-1', pending: false, text: 'The current automatic reply.' }
    consumed = false
    act(() => messages.set([{ id: 'assistant-1' }] as never[]))

    await waitFor(() =>
      expect(playSpeechText).toHaveBeenCalledWith('The current automatic reply.', {
        messageId: 'assistant-1',
        source: 'read-aloud'
      })
    )
    expect(consumed).toBe(true)
  })

  it('does not let delayed ambient authorization replace newer selected-text playback', async () => {
    let resolveOwnership!: (owns: boolean) => void
    let reply: { id: string; pending: boolean; text: string } | null = null
    let consumed = false

    ownsAmbientCue.mockImplementationOnce(
      () => new Promise<boolean>(resolve => (resolveOwnership = resolve))
    )

    renderHook(
      () =>
        useAutoSpeakReplies({
          conversationActive: false,
          failureLabel: 'Playback failed',
          markSpoken: () => {
            consumed = true
          },
          pendingReply: () => (consumed ? null : reply),
          sessionId: 'session-1'
        }),
      { wrapper }
    )

    reply = { id: 'assistant-1', pending: false, text: 'An older automatic reply.' }
    consumed = false
    act(() => messages.set([{ id: 'assistant-1' }] as never[]))
    expect(ownsAmbientCue).toHaveBeenCalledWith('speak:assistant-1')

    act(() => {
      voicePlayback.stopVoicePlayback()
      setVoicePlaybackState({
        audioElement: null,
        messageId: 'selection-read-aloud',
        sequence: voicePlayback.getVoicePlaybackSequence(),
        source: 'read-aloud',
        status: 'preparing'
      })
    })

    await act(async () => resolveOwnership(true))

    expect(playSpeechText).not.toHaveBeenCalled()
  })
})
