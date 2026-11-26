import { renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useDictation } from './dictation'

type Transcribe = (audio: Blob, owner?: unknown) => Promise<string>

const recorder = vi.hoisted(() => ({ options: null as null | { onTranscribeAudio?: Transcribe } }))
const direct = vi.hoisted(() => vi.fn<(audio: Blob, owner?: unknown) => Promise<null | string>>())
const relay = vi.hoisted(() => vi.fn())

vi.mock('@/app/chat/composer/hooks/use-voice-recorder', () => ({
  useVoiceRecorder: (options: { onTranscribeAudio?: Transcribe }) => {
    recorder.options = options

    return { dictate: vi.fn(), voiceActivityState: {}, voiceStatus: 'idle' }
  }
}))
vi.mock('@/lib/voice-client-direct', () => ({ transcribeAudioClientDirect: direct }))
vi.mock('@/hermes', () => ({ transcribeAudio: relay }))
vi.mock('@/app/session/hooks/use-prompt-actions/utils', () => ({
  blobToDataUrl: async () => 'data:audio/webm;base64,AAAA'
}))

const audio = new Blob(['x'], { type: 'audio/webm' })
const owner = { connectionId: 'c1', profile: 'p1' }

function transcribe() {
  renderHook(() => useDictation({ onTranscript: vi.fn() }))

  return recorder.options!.onTranscribeAudio!(audio, owner)
}

describe('useDictation transcription ladder', () => {
  beforeEach(() => {
    direct.mockReset()
    relay.mockReset()
  })

  it('uses the profile provider directly when it is client-callable', async () => {
    direct.mockResolvedValue('czarny granit')

    await expect(transcribe()).resolves.toBe('czarny granit')
    expect(direct).toHaveBeenCalledWith(audio, owner)
    expect(relay).not.toHaveBeenCalled()
  })

  it('falls back to the gateway relay with the same owner', async () => {
    direct.mockResolvedValue(null)
    relay.mockResolvedValue({ transcript: 'zlew podwieszany' })

    await expect(transcribe()).resolves.toBe('zlew podwieszany')
    expect(relay).toHaveBeenCalledWith('data:audio/webm;base64,AAAA', 'audio/webm', owner)
  })
})
