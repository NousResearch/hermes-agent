// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { chunkForCommentary, toLiveHistory } from '@/lib/voice-live'
import type * as voiceLiveModule from '@/lib/voice-live'
import { notify, notifyError } from '@/store/notifications'

import { delegationPrompt, useVoiceLiveConversation } from './use-voice-live-conversation'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          couldNotStartSession: 'Could not start voice session',
          liveDelegationFailed: 'Could not hand the request to Hermes',
          liveEnded: 'Live voice session ended',
          liveEndedClosed: 'Session closed',
          liveEndedConnectionLost: 'Connection lost',
          liveError: 'Live voice',
          microphoneConstraintsUnsupported: 'Microphone constraints are not supported by this device.',
          microphoneInUse: 'Microphone is already in use by another app.',
          microphonePermissionDenied: 'Microphone permission was denied.',
          microphoneStartFailed: 'Could not start microphone recording.',
          noMicrophone: 'No microphone was found.'
        }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

interface MockLiveSession {
  onClosed: (reason: string, usageSeconds: number | null) => void
}

const createdSessions: MockLiveSession[] = []
let startImpl: () => Promise<void> = async () => undefined

vi.mock('@/lib/voice-live', async importOriginal => {
  const actual = await importOriginal<typeof voiceLiveModule>()

  return {
    ...actual,
    VoiceLiveSession: class {
      close = vi.fn()
      instruct = vi.fn()
      setMuted = vi.fn()
      speak = vi.fn()
      think = vi.fn()
      onClosed: MockLiveSession['onClosed']
      start = vi.fn(() => startImpl())

      constructor(options: { onClosed: MockLiveSession['onClosed'] }) {
        this.onClosed = options.onClosed
        createdSessions.push(this)
      }
    }
  }
})

function renderLiveConversation() {
  return renderHook(() =>
    useVoiceLiveConversation({
      busy: false,
      consumePendingResponse: vi.fn(),
      enabled: true,
      onFatalError: vi.fn(),
      onSubmit: vi.fn(),
      pendingResponse: () => null,
      seedHistory: () => []
    })
  )
}

describe('GPT-Live delegation → Hermes turn', () => {
  it('sends the latest user words as the turn and the exchange as model-only context', () => {
    // The delegation event carries no text: both are reconstructed from
    // transcript deltas, fragments of one speaker concatenated as received.
    const { context, prompt } = delegationPrompt([
      { endMs: 1000, speaker: 'assistant', startMs: 0, text: 'Hi, how ' },
      { endMs: 1500, speaker: 'assistant', startMs: 1000, text: 'can I help?' },
      { endMs: 2500, speaker: 'user', startMs: 1500, text: 'What is ' },
      { endMs: 3200, speaker: 'user', startMs: 2500, text: 'the weather in Paris?' }
    ])

    expect(prompt).toBe('What is the weather in Paris?')
    expect(context).toContain('Voice assistant: Hi, how can I help?')
    expect(context).toContain('User: What is the weather in Paris?')
  })

  it('splits a long reply into vendor-sized commentary appends on sentence boundaries', () => {
    const sentence = 'This is a sentence about the result. '
    const chunks = chunkForCommentary(sentence.repeat(80), 400)

    expect(chunks.length).toBeGreaterThan(1)
    expect(chunks.every(chunk => chunk.length <= 400)).toBe(true)
    expect(chunks.every(chunk => chunk.endsWith('.'))).toBe(true)
    expect(chunks.join(' ')).toBe(sentence.repeat(80).trim())
  })

  it('seeds the live session with the most recent text turns within budget', () => {
    const turns = Array.from({ length: 40 }, (_, index) => ({
      role: (index % 2 === 0 ? 'user' : 'assistant') as 'assistant' | 'user',
      text: `turn ${index}`
    }))

    const history = toLiveHistory(turns, 6)

    expect(history).toHaveLength(6)
    expect(history.at(-1)?.content[0]?.text).toBe('turn 39')
    expect(history[0]?.role).toBe('user')
    expect(history.find(m => m.role === 'assistant')?.content[0]?.type).toBe('output_text')
  })
})

describe('GPT-Live toast copy', () => {
  beforeEach(() => {
    createdSessions.length = 0
    startImpl = async () => undefined
    vi.clearAllMocks()
  })

  afterEach(() => {
    cleanup()
  })

  async function startSession() {
    const hook = renderLiveConversation()

    await act(async () => {
      await hook.result.current.start()
    })

    const session = createdSessions.at(-1)

    expect(session).toBeDefined()

    return { hook, session: session as MockLiveSession }
  }

  it('never shows the raw connection_lost session-end reason', async () => {
    const { session } = await startSession()

    act(() => {
      session.onClosed('connection_lost', 127)
    })

    const calls = vi.mocked(notify).mock.calls

    expect(calls).toHaveLength(1)
    expect(calls[0]?.[0]).toMatchObject({ kind: 'warning', title: 'Live voice session ended' })
    expect(calls[0]?.[0]?.message).toBe('Connection lost (127s)')
    expect(calls[0]?.[0]?.message).not.toContain('connection_lost')
  })

  it('never shows the raw closed session-end reason', async () => {
    const { session } = await startSession()

    act(() => {
      session.onClosed('closed', 42)
    })

    const calls = vi.mocked(notify).mock.calls

    expect(calls).toHaveLength(1)
    expect(calls[0]?.[0]?.message).toBe('Session closed (42s)')
  })

  it('passes unknown server-sent session-end reasons through verbatim', async () => {
    const { session } = await startSession()

    act(() => {
      session.onClosed('server_idle_timeout', 5)
    })

    const calls = vi.mocked(notify).mock.calls

    expect(calls).toHaveLength(1)
    expect(calls[0]?.[0]?.message).toBe('server_idle_timeout (5s)')
  })

  it.each([
    ['NotAllowedError', 'Microphone permission was denied.'],
    ['NotFoundError', 'No microphone was found.'],
    ['NotReadableError', 'Microphone is already in use by another app.']
  ])('maps mic DOMException %s to the recorder-path copy on live start', async (name, copy) => {
    startImpl = () => Promise.reject(new DOMException('raw mic failure', name))

    const hook = renderLiveConversation()

    await act(async () => {
      await hook.result.current.start()
    })

    const calls = vi.mocked(notifyError).mock.calls

    expect(calls).toHaveLength(1)
    expect(calls[0]?.[0]).toBeInstanceOf(Error)
    expect((calls[0]?.[0] as Error).message).toBe(copy)
    expect(calls[0]?.[1]).toBe('Could not start voice session')
  })

  it('keeps non-mic start failures verbatim', async () => {
    startImpl = () => Promise.reject(new Error('Missing local SDP offer'))

    const hook = renderLiveConversation()

    await act(async () => {
      await hook.result.current.start()
    })

    const calls = vi.mocked(notifyError).mock.calls

    expect(calls).toHaveLength(1)
    expect((calls[0]?.[0] as Error).message).toBe('Missing local SDP offer')
    expect(calls[0]?.[1]).toBe('Could not start voice session')
  })
})
