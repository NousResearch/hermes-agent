import { describe, expect, it, vi } from 'vitest'

import { GeminiLiveSession } from './gemini-live'

describe('GeminiLiveSession', () => {
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
})
