import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $voicePlayback } from '@/store/voice-playback'

import { PreviewReadAloudButton } from './preview-file'

const playback = vi.hoisted(() => ({
  playSpeechText: vi.fn(async () => true),
  stopVoicePlayback: vi.fn()
}))

vi.mock('@/lib/voice-playback', () => playback)

const IDLE = { audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' } as const

// The file-preview Read aloud control rides the message read-aloud path: same
// `playSpeechText` (the profile's configured voice), same `$voicePlayback` slot.
describe('PreviewReadAloudButton', () => {
  beforeEach(() => {
    $voicePlayback.set({ ...IDLE })
    playback.playSpeechText.mockClear()
    playback.stopVoicePlayback.mockClear()
  })

  afterEach(() => {
    cleanup()
  })

  it('reads the previewed document through the shared read-aloud path', () => {
    render(<PreviewReadAloudButton filePath="/home/me/notes.md" text={'# Title\n\nHello there.'} />)

    fireEvent.click(screen.getByRole('button', { name: 'Read aloud' }))

    expect(playback.playSpeechText).toHaveBeenCalledWith('# Title\n\nHello there.', {
      messageId: 'file:/home/me/notes.md',
      source: 'read-aloud'
    })
  })

  it('turns into Stop while this file is speaking', () => {
    render(<PreviewReadAloudButton filePath="/home/me/notes.md" text="Hello." />)

    act(() => {
      $voicePlayback.set({ ...IDLE, messageId: 'file:/home/me/notes.md', source: 'read-aloud', status: 'speaking' })
    })

    fireEvent.click(screen.getByRole('button', { name: 'Stop reading' }))

    expect(playback.stopVoicePlayback).toHaveBeenCalledTimes(1)
    expect(playback.playSpeechText).not.toHaveBeenCalled()
  })

  it('stays disabled while something else is being read', () => {
    render(<PreviewReadAloudButton filePath="/home/me/notes.md" text="Hello." />)

    act(() => {
      $voicePlayback.set({ ...IDLE, messageId: 'msg-1', source: 'read-aloud', status: 'speaking' })
    })

    expect(screen.getByRole('button', { name: 'Read aloud' })).toHaveProperty('disabled', true)
  })
})
