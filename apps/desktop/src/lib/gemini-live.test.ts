// @vitest-environment jsdom
import { describe, expect, it, vi, beforeEach } from 'vitest'

import {
  DEFAULT_GEMINI_LIVE_MODEL,
  DEFAULT_GEMINI_LIVE_VOICE,
  GEMINI_LIVE_MODEL_STORAGE,
  GEMINI_LIVE_MODELS,
  GEMINI_VOICES,
  GeminiLiveSession,
  getStoredGeminiLiveModel,
  resolveGeminiLiveApiKey,
  RETIRED_GEMINI_LIVE_MODELS,
  saveGeminiApiKey,
  thinkingConfigForModel
} from './gemini-live'
import { deleteEnvVar, revealEnvVar, setEnvVar } from '@/hermes'

vi.mock('@/hermes', () => ({
  revealEnvVar: vi.fn(),
  setEnvVar: vi.fn(),
  deleteEnvVar: vi.fn()
}))

describe('GeminiLiveSession', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  it('stopAudioStream sends audioStreamEnd without synthetic text commentary', () => {
    const sentMessages: string[] = []
    const mockWs = {
      close: vi.fn(),
      readyState: 1, // WebSocket.OPEN
      send: vi.fn((data: string) => sentMessages.push(data))
    }

    const session = new GeminiLiveSession(
      {
        onClosed: vi.fn(),
        onDelegation: vi.fn(),
        onError: vi.fn()
      },
      { apiKey: 'test-key' }
    )

    // Attach mock websocket and set connected
    ;(session as any).ws = mockWs
    ;(session as any).started = true

    const success = session.stopAudioStream()
    expect(success).toBe(true)
    expect(sentMessages.length).toBe(1)

    const parsed = JSON.parse(sentMessages[0])
    expect(parsed).toEqual({
      realtimeInput: {
        audioStreamEnd: true
      }
    })
    // Invariant: MUST NOT send clientContent with synthetic text that causes standby greetings
    expect(parsed.clientContent).toBeUndefined()
  })

  it('handles serverContent transcriptions in both camelCase and snake_case', () => {
    const transcripts: any[] = []
    const session = new GeminiLiveSession(
      {
        onClosed: vi.fn(),
        onDelegation: vi.fn(),
        onError: vi.fn(),
        onTranscript: frag => transcripts.push(frag)
      },
      { apiKey: 'test-key' }
    )

    // 1. camelCase input & output transcription
    ;(session as any).handleServerMessage({
      serverContent: {
        inputTranscription: { text: 'Hello Hermes' },
        outputTranscription: { text: 'Hello! How can I help?' }
      }
    })

    expect(transcripts).toContainEqual(
      expect.objectContaining({
        speaker: 'user',
        text: 'Hello Hermes'
      })
    )
    expect(transcripts).toContainEqual(
      expect.objectContaining({
        speaker: 'assistant',
        text: 'Hello! How can I help?'
      })
    )

    // 2. snake_case input & output transcription
    transcripts.length = 0
    ;(session as any).handleServerMessage({
      server_content: {
        input_transcription: { text: 'Check git status' },
        output_transcription: { text: 'Checking status now.' }
      }
    })

    expect(transcripts).toContainEqual(
      expect.objectContaining({
        speaker: 'user',
        text: 'Check git status'
      })
    )
    expect(transcripts).toContainEqual(
      expect.objectContaining({
        speaker: 'assistant',
        text: 'Checking status now.'
      })
    )
  })

  it('triggers instantaneous barge-in when serverContent.interrupted is received', () => {
    const onInterrupted = vi.fn()
    const session = new GeminiLiveSession(
      {
        onClosed: vi.fn(),
        onDelegation: vi.fn(),
        onError: vi.fn(),
        onInterrupted
      },
      { apiKey: 'test-key' }
    )

    const interruptSpy = vi.spyOn(session, 'interrupt')

    ;(session as any).handleServerMessage({
      serverContent: {
        interrupted: true
      }
    })

    expect(interruptSpy).toHaveBeenCalledTimes(1)
    expect(onInterrupted).toHaveBeenCalledTimes(1)
  })

  it('delegates ask_hermes tool calls to onDelegation handler', () => {
    const onDelegation = vi.fn()
    const session = new GeminiLiveSession(
      {
        onClosed: vi.fn(),
        onDelegation,
        onError: vi.fn()
      },
      { apiKey: 'test-key' }
    )

    ;(session as any).handleServerMessage({
      toolCall: {
        functionCalls: [
          {
            id: 'call_123',
            name: 'ask_hermes',
            args: { request: 'run test script' }
          }
        ]
      }
    })

    expect(onDelegation).toHaveBeenCalledWith('call_123', 'run test script')
  })

  it('sendToolResponse includes function name and id in functionResponses payload', () => {
    const sentMessages: string[] = []
    const mockWs = {
      close: vi.fn(),
      readyState: 1,
      send: vi.fn((data: string) => sentMessages.push(data))
    }

    const session = new GeminiLiveSession(
      {
        onClosed: vi.fn(),
        onDelegation: vi.fn(),
        onError: vi.fn()
      },
      { apiKey: 'test-key' }
    )

    ;(session as any).ws = mockWs
    ;(session as any).started = true

    const ok = session.sendToolResponse('call_xyz', 'task output', 'ask_hermes')
    expect(ok).toBe(true)
    expect(sentMessages.length).toBe(1)

    const parsed = JSON.parse(sentMessages[0])
    expect(parsed.toolResponse.functionResponses[0]).toEqual({
      id: 'call_xyz',
      name: 'ask_hermes',
      response: {
        output: {
          result: 'task output'
        }
      }
    })
  })

  it('tears down mic tracks if cancelled while getUserMedia is pending', async () => {
    const mockTrack = { stop: vi.fn() }
    const mockStream = { getTracks: () => [mockTrack] }

    let resolveGetUserMedia: (stream: any) => void
    const getUserMediaPromise = new Promise(resolve => {
      resolveGetUserMedia = resolve
    })

    const originalMediaDevices = navigator.mediaDevices
    Object.defineProperty(navigator, 'mediaDevices', {
      value: {
        getUserMedia: vi.fn(() => getUserMediaPromise)
      },
      configurable: true
    })

    const session = new GeminiLiveSession(
      {
        onClosed: vi.fn(),
        onDelegation: vi.fn(),
        onError: vi.fn()
      },
      { apiKey: 'test-key' }
    )

    // Start mic capture
    const capturePromise = (session as any).startMicrophoneCapture()

    // User closes session before getUserMedia resolves
    session.finish('cancelled')
    expect((session as any).finalized).toBe(true)

    // Now getUserMedia resolves
    resolveGetUserMedia!(mockStream)
    await capturePromise

    // Invariant: track.stop() must be called to avoid leaking mic
    expect(mockTrack.stop).toHaveBeenCalled()
    expect((session as any).micStream).toBeNull()
    expect((session as any).started).toBe(false)

    Object.defineProperty(navigator, 'mediaDevices', {
      value: originalMediaDevices,
      configurable: true
    })
  })

  it('resolveGeminiLiveApiKey resolves GEMINI_API_KEY from gateway environment', async () => {
    vi.mocked(revealEnvVar).mockImplementation(async (key: string) => {
      if (key === 'GEMINI_API_KEY') return { key, value: 'gemini-secret-123' }
      return { key, value: '' }
    })

    const key = await resolveGeminiLiveApiKey()
    expect(key).toBe('gemini-secret-123')
    expect(revealEnvVar).toHaveBeenCalledWith('GEMINI_API_KEY')
  })

  it('resolveGeminiLiveApiKey falls back to GOOGLE_API_KEY', async () => {
    vi.mocked(revealEnvVar).mockImplementation(async (key: string) => {
      if (key === 'GOOGLE_API_KEY') return { key, value: 'google-secret-456' }
      throw new Error('Not found')
    })

    const key = await resolveGeminiLiveApiKey()
    expect(key).toBe('google-secret-456')
  })

  it('resolveGeminiLiveApiKey purges legacy plaintext localStorage key', async () => {
    localStorage.setItem('hermes_gemini_live_api_key', 'insecure-plaintext-key')
    vi.mocked(revealEnvVar).mockResolvedValue({ key: 'GEMINI_API_KEY', value: 'real-key' })

    const key = await resolveGeminiLiveApiKey()
    expect(key).toBe('real-key')
    expect(localStorage.getItem('hermes_gemini_live_api_key')).toBeNull()
  })

  it('saveGeminiApiKey writes to GEMINI_API_KEY via setEnvVar', async () => {
    vi.mocked(setEnvVar).mockResolvedValue({ ok: true })

    const res = await saveGeminiApiKey('AIzaSyNewKey')
    expect(res).toEqual({ ok: true })
    expect(setEnvVar).toHaveBeenCalledWith('GEMINI_API_KEY', 'AIzaSyNewKey')
  })

  it('saveGeminiApiKey deletes GEMINI_API_KEY via deleteEnvVar when empty', async () => {
    vi.mocked(deleteEnvVar).mockResolvedValue({ ok: true })

    const res = await saveGeminiApiKey('   ')
    expect(res).toEqual({ ok: true })
    expect(deleteEnvVar).toHaveBeenCalledWith('GEMINI_API_KEY')
  })
})

describe('Gemini Live model and voice catalogs', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('offers only models that support bidiGenerateContent', () => {
    const ids = GEMINI_LIVE_MODELS.map(m => m.id)

    // `gemini-2.5-flash` does not support bidiGenerateContent and fails setup
    // with 1008; `gemini-2.5-flash-native-audio-latest` is the working 2.5 model.
    expect(ids).not.toContain('gemini-2.5-flash')
    expect(ids).toContain('gemini-2.5-flash-native-audio-latest')
    // Fastest measured time-to-first-audio; must stay selectable.
    expect(ids).toContain('gemini-3.1-flash-live-preview')
  })

  it('keeps the default model selectable', () => {
    expect(GEMINI_LIVE_MODELS.map(m => m.id)).toContain(DEFAULT_GEMINI_LIVE_MODEL)
  })

  it('exposes the full verified voice set without duplicates', () => {
    const ids = GEMINI_VOICES.map(v => v.id)

    expect(ids).toHaveLength(30)
    expect(new Set(ids).size).toBe(30)
    expect(ids).toContain(DEFAULT_GEMINI_LIVE_VOICE)
  })

  it('retires stored models that can no longer connect', () => {
    localStorage.setItem(GEMINI_LIVE_MODEL_STORAGE, 'gemini-2.5-flash')

    expect(RETIRED_GEMINI_LIVE_MODELS).toContain('gemini-2.5-flash')
    expect(getStoredGeminiLiveModel()).toBe(DEFAULT_GEMINI_LIVE_MODEL)
  })

  it('keeps a stored model that is still offered', () => {
    localStorage.setItem(GEMINI_LIVE_MODEL_STORAGE, 'gemini-3.1-flash-live-preview')

    expect(getStoredGeminiLiveModel()).toBe('gemini-3.1-flash-live-preview')
  })
})

describe('thinkingConfigForModel', () => {
  it('adds thinkingConfig for the model that requires it', () => {
    // Without it this model fails setup with
    // "Thinking level must be specified for this model" (1007).
    expect(thinkingConfigForModel('gemini-3.8-live-extended-thinking')).toEqual({
      thinkingLevel: 'LOW'
    })
  })

  it('omits thinkingConfig for models that reject it', () => {
    // Regression guard: sending thinkingConfig to gemini-3.8-live fails setup
    // with "Thinking level is not supported for this model" (1007).
    expect(thinkingConfigForModel('gemini-3.8-live')).toBeNull()
    expect(thinkingConfigForModel('gemini-3.1-flash-live-preview')).toBeNull()
    expect(thinkingConfigForModel('gemini-2.5-flash-native-audio-latest')).toBeNull()
  })

  it('accepts a models/ prefixed name', () => {
    expect(thinkingConfigForModel('models/gemini-3.8-live-extended-thinking')).toEqual({
      thinkingLevel: 'LOW'
    })
    expect(thinkingConfigForModel('models/gemini-3.8-live')).toBeNull()
  })
})
