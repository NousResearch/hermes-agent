// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { chunkForCommentary, toLiveHistory, type VoiceLiveHandlers } from '@/lib/voice-live'
import { $notifications, clearNotifications } from '@/store/notifications'

import { delegationPrompt, useVoiceLiveConversation } from './use-voice-live-conversation'

// The live-voice toasts must not show machine strings — neither the wire close
// reasons (`connection_lost`, `closed`) nor the raw `DOMException` text a denied
// `getUserMedia` throws (#111987).
//
// The transport is the only seam in the live hook, so the fake session records
// the handlers it registers (tests drive the close/error paths directly) and
// lets `start` reject with exactly what the real `getUserMedia` would throw.
const transport = vi.hoisted(() => ({ failure: null as unknown, handlers: [] as VoiceLiveHandlers[] }))

vi.mock('@/lib/voice-live', async importOriginal => {
  const actual = (await importOriginal()) as Record<string, unknown>

  return {
    ...actual,
    VoiceLiveSession: class {
      close = vi.fn()
      instruct = vi.fn()
      setMuted = vi.fn()
      speak = vi.fn()
      think = vi.fn()

      constructor(handlers: VoiceLiveHandlers) {
        transport.handlers.push(handlers)
      }

      async start(): Promise<void> {
        if (transport.failure) {
          throw transport.failure
        }
      }
    }
  }
})

const voice = en.notifications.voice

function mountLive() {
  return renderHook(() =>
    useVoiceLiveConversation({
      busy: false,
      consumePendingResponse: vi.fn(),
      enabled: true,
      onSubmit: vi.fn(),
      pendingResponse: () => null,
      seedHistory: () => []
    })
  )
}

/** Mount, start, and hand back the handlers the started session registered. */
async function openSession(): Promise<VoiceLiveHandlers> {
  const hook = mountLive()

  await act(async () => {
    await hook.result.current.start()
  })

  return transport.handlers.at(-1) as VoiceLiveHandlers
}

/** Start a session whose transport fails the way the real one can. */
async function failedStart(failure: unknown): Promise<void> {
  transport.failure = failure
  const hook = mountLive()

  await act(async () => {
    await hook.result.current.start()
  })
}

function resetToasts() {
  clearNotifications()
  transport.failure = null
  transport.handlers.length = 0
}

describe('Voice-live toast copy', () => {
  beforeEach(resetToasts)
  afterEach(cleanup)

  it('names our own close reasons in copy and passes server-sent reasons through verbatim', async () => {
    let handlers = await openSession()

    act(() => {
      handlers.onClosed('connection_lost', 127)
    })

    expect($notifications.get()[0].title).toBe(voice.liveEnded)
    expect($notifications.get()[0].message).toBe(`${voice.liveEndedConnectionLost} (127s)`)

    cleanup()
    resetToasts()
    handlers = await openSession()

    act(() => {
      handlers.onClosed('quota_exhausted', 12)
    })

    // Server strings are unbounded: no mapping, no redaction claim.
    expect($notifications.get()[0].message).toBe('quota_exhausted (12s)')
  })

  it('maps a getUserMedia DOMException to the recorder mic copy and leaves non-mic failures alone', async () => {
    await failedStart(new DOMException('The request is not allowed by the user agent.', 'NotAllowedError'))

    expect($notifications.get()[0].title).toBe(voice.couldNotStartSession)
    expect($notifications.get()[0].message).toBe(voice.microphonePermissionDenied)

    cleanup()
    resetToasts()
    await failedStart(new Error('Missing local SDP offer'))

    expect($notifications.get()[0].message).toBe('Missing local SDP offer')
  })
})

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
