// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from 'vitest'

import { $apiRequestScope, resolveOwnerNow } from '@/api/client'
import type { CreateBackendSessionForSend } from '@/app/session/hooks/use-session-actions/create-overrides'
import { en } from '@/i18n/en'
import { createClientSessionState } from '@/lib/chat-runtime'
import { chunkForCommentary, toLiveHistory, type VoiceLiveHandlers } from '@/lib/voice-live'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $sessionStates } from '@/store/session-states'
import { $voiceLiveDrafts } from '@/store/voice-live-draft'

import { delegationPrompt, useVoiceLiveConversation } from './use-voice-live-conversation'

// The live-voice toasts must not show machine strings — neither the wire close
// reasons (`connection_lost`, `closed`) nor the raw `DOMException` text a denied
// `getUserMedia` throws (#111987).
//
// The transport is the only seam in the live hook, so the fake session records
// the handlers it registers (tests drive the close/error paths directly) and
// lets `start` reject with exactly what the real `getUserMedia` would throw.
const transport = vi.hoisted(() => ({
  captureStatus: 'recording' as 'recording' | 'unavailable',
  failure: null as unknown,
  handlers: [] as VoiceLiveHandlers[],
  start: vi.fn()
}))

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

      async start(history: unknown, storedSessionId: unknown, owner: unknown) {
        transport.start(history, storedSessionId, owner)

        if (transport.failure) {
          throw transport.failure
        }

        return transport.captureStatus
      }
    }
  }
})

const voice = en.notifications.voice

function mountLive(onSubmit = vi.fn(), onInterrupt = vi.fn(), busy = true) {
  return renderHook(({ storedSessionId }: { storedSessionId: null | string }) =>
    useVoiceLiveConversation({
      busy,
      consumePendingResponse: vi.fn(),
      enabled: true,
      onInterrupt,
      onSubmit,
      pendingResponse: () => null,
      seedHistory: () => [],
      storedSessionId
    }), { initialProps: { storedSessionId: 'stored-chat' as null | string } }
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
  transport.captureStatus = 'recording'
  transport.start.mockClear()
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

  it('warns without ending a call when canonical capture is unavailable', async () => {
    transport.captureStatus = 'unavailable'
    const hook = mountLive()
    await act(async () => hook.result.current.start())

    expect(transport.start).toHaveBeenCalledWith([], 'stored-chat', resolveOwnerNow())
    expect(hook.result.current.status).toBe('listening')
    expect($notifications.get()[0]?.kind).toBe('warning')
  })
})

describe('GPT-Live delegation → Hermes turn', () => {
  beforeEach(resetToasts)
  afterEach(cleanup)

  const request: Parameters<VoiceLiveHandlers['onDelegation']> = [
    'vendor-delegation', [{ endMs: 100, speaker: 'user', startMs: 0, text: 'Check the weather.' }], 'stable-input'
  ]

  it('preserves the submission ID and drops events after a chat switch', async () => {
    const submit = vi.fn()
    const interrupt = vi.fn()
    const hook = mountLive(submit, interrupt)

    try {
      await act(async () => hook.result.current.start())
      const handlers = transport.handlers.at(-1)!
      act(() => handlers.onDelegation(...request))
      expect(submit).toHaveBeenCalledWith('Check the weather.', 'User: Check the weather.', 'stable-input', 'stored-chat')
      hook.rerender({ storedSessionId: 'another-chat' })
      act(() => handlers.onDelegation(...request))
      expect(submit).toHaveBeenCalledTimes(1)
      expect(interrupt).toHaveBeenCalledTimes(1)
      expect(hook.result.current.status).toBe('idle')
    } finally {
      hook.unmount()
    }
  })

  it('submits recorded words and coverage in arrival order without copying transcript context', async () => {
    const submit = vi.fn()
    const interrupt = vi.fn()
    const hook = mountLive(submit, interrupt, false)

    await act(async () => hook.result.current.start())
    const handlers = transport.handlers.at(-1)!

    const args: Parameters<VoiceLiveHandlers['onDelegation']> = [
      'del', [
        { endMs: 100, event_id: 'reply', speaker: 'assistant', startMs: 0, text: 'What can I help with?' },
        { endMs: 300, event_id: 'first', speaker: 'user', startMs: 200, text: 'Check' },
        { endMs: 350, event_id: 'second', speaker: 'user', startMs: 175, text: ' the weather.' }
      ], 'voice-live:["call","del"]', 'call'
    ]

    act(() => {
      handlers.onDelegation(...args)
      handlers.onDelegation(...args)
    })
    expect(submit).toHaveBeenCalledExactlyOnceWith(
      'Check the weather.', '', args[2], 'stored-chat', undefined,
      { call_id: 'call', delegation_id: 'del', event_ids: ['first', 'second'] }
    )
    expect(interrupt).not.toHaveBeenCalled()
    hook.unmount()
  })

  it('will not submit a recorded utterance without full fragment IDs', async () => {
    const submit = vi.fn()
    const hook = mountLive(submit, vi.fn(), false)

    await act(async () => hook.result.current.start())
    act(() => transport.handlers.at(-1)!.onDelegation(request[0], request[1], request[2], 'recorded-call'))
    expect(submit).not.toHaveBeenCalled()
    expect($notifications.get()[0]?.title).toBe(voice.liveDelegationFailed)
    hook.unmount()
  })

  it('does not open audio or create a substitute on an unresolved existing chat', async () => {
    const submit = vi.fn()
    const interrupt = vi.fn()
    const hook = mountLive(submit, interrupt)

    try {
      hook.rerender({ storedSessionId: null })
      await act(async () => hook.result.current.start())
      expect(transport.start).not.toHaveBeenCalled()
      expect($notifications.get()[0]?.title).toBe(voice.couldNotStartSession)
      expect(interrupt).not.toHaveBeenCalled()
      expect(submit).not.toHaveBeenCalled()
    } finally {
      hook.unmount()
    }
  })

  it('opens a fresh aliased chat on its backend profile and delegates after its own route handoff', async () => {
    const submit = vi.fn()
    const owner = { connectionId: 'voice-host', profile: 'voice-label' }
    const previousScope = $apiRequestScope.get()
    onTestFinished(() => $apiRequestScope.set(previousScope))
    let finishCreate!: () => void

    const create = vi.fn<CreateBackendSessionForSend>(async (_preview, overrides) => {
      await new Promise<void>(resolve => { finishCreate = resolve })
      overrides!.voiceDraft!.onAssigned({
        backendProfile: 'voice-backend',
        owner,
        runtimeSessionId: 'created-runtime',
        storedSessionId: 'created-chat'
      })
      $apiRequestScope.set(owner)
      hook.rerender({ storedSessionId: 'created-chat' })

      return 'created-runtime'
    })

    const hook = renderHook(({ storedSessionId }: { storedSessionId: null | string }) =>
      useVoiceLiveConversation({
        busy: false,
        consumePendingResponse: vi.fn(),
        createVoiceDraftSession: !storedSessionId ? create : undefined,
        enabled: true,
        onSubmit: submit,
        pendingResponse: () => null,
        seedHistory: () => [{ type: 'message', role: 'user', content: [{ type: 'input_text', text: 'Old chat' }] }],
        storedSessionId,
        voiceDraftKey: !storedSessionId ? 'fresh-draft' : undefined
      }), { initialProps: { storedSessionId: null as null | string } }
    )

    let pending!: Promise<void>
    await act(async () => { pending = hook.result.current.start() })
    expect(create).toHaveBeenCalledWith(null, expect.objectContaining({ voiceDraft: expect.objectContaining({ key: 'fresh-draft' }) }))
    expect(transport.start).not.toHaveBeenCalled()
    await act(async () => { finishCreate(); await pending })
    expect(transport.start).toHaveBeenCalledExactlyOnceWith([], 'created-chat', { ...owner, profile: 'voice-backend' })
    act(() => transport.handlers.at(-1)!.onDelegation(...request))
    expect(submit).toHaveBeenCalledWith('Check the weather.', 'User: Check the weather.', 'stable-input', 'created-chat', 'created-runtime')
    hook.rerender({ storedSessionId: 'another-chat' })
    act(() => transport.handlers.at(-1)!.onDelegation(...request))
    expect(submit).toHaveBeenCalledTimes(1)
  })

  it('does not open audio if the voice start is canceled during draft creation', async () => {
    let finishCreate!: () => void

    const create = vi.fn<CreateBackendSessionForSend>(async (_preview, overrides) => {
      await new Promise<void>(resolve => { finishCreate = resolve })
      expect(overrides!.voiceDraft!.isCurrent()).toBe(false)

      return null
    })

    const submit = vi.fn()

    const hook = renderHook(() => useVoiceLiveConversation({
      busy: false,
      consumePendingResponse: vi.fn(),
      createVoiceDraftSession: create,
      enabled: true,
      onSubmit: submit,
      pendingResponse: () => null,
      seedHistory: () => [],
      voiceDraftKey: 'draft'
    }))

    let pending!: Promise<void>
    await act(async () => { pending = hook.result.current.start() })
    await act(async () => { await hook.result.current.end(); finishCreate(); await pending })
    expect(transport.start).not.toHaveBeenCalled()
    expect(submit).not.toHaveBeenCalled()
    expect($notifications.get()).toEqual([])
  })

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

  it('keeps a spoken request intact across an overlapping assistant backchannel', () => {
    const { context, prompt } = delegationPrompt([
      { endMs: 800, speaker: 'user', startMs: 0, text: 'Please move Thursday’s review' },
      { endMs: 1100, speaker: 'assistant', startMs: 650, text: 'Mm-hmm.' },
      { endMs: 1600, speaker: 'user', startMs: 850, text: ' to the afternoon.' }
    ])

    expect(prompt).toBe('Please move Thursday’s review to the afternoon.')
    expect(context).toBe('User: Please move Thursday’s review to the afternoon.\nVoice assistant: Mm-hmm.')
  })

  it('does not merge separate user turns just because an assistant delta arrived late', () => {
    const { context, prompt } = delegationPrompt([
      { endMs: 900, speaker: 'user', startMs: 0, text: 'Mornings are bad.' },
      { endMs: 3500, speaker: 'user', startMs: 2700, text: 'Draft the note.' },
      { endMs: 2200, speaker: 'assistant', startMs: 1000, text: 'Afternoons only?' }
    ])

    expect(prompt).toBe('Draft the note.')
    expect(context).toBe('User: Mornings are bad.\nVoice assistant: Afternoons only?\nUser: Draft the note.')
  })

  it('keeps separate requests on either side of a quick, non-overlapping reply', () => {
    const { context, prompt } = delegationPrompt([
      { endMs: 500, speaker: 'user', startMs: 0, text: 'Cancel that.' },
      { endMs: 800, speaker: 'assistant', startMs: 550, text: 'OK.' },
      { endMs: 1600, speaker: 'user', startMs: 850, text: 'Book the room.' }
    ])

    expect(prompt).toBe('Book the room.')
    expect(context).toBe('User: Cancel that.\nVoice assistant: OK.\nUser: Book the room.')
  })

  it('returns the provider IDs for exactly the user fragments used for delegation', () => {
    const { eventIds, prompt } = delegationPrompt([
      { endMs: 400, event_id: 'earlier', speaker: 'user', startMs: 0, text: 'Cancel that.' },
      { endMs: 800, event_id: 'reply', speaker: 'assistant', startMs: 450, text: 'Okay.' },
      { endMs: 1000, event_id: 'part-1', speaker: 'user', startMs: 850, text: 'Bring' },
      { endMs: 1400, event_id: 'part-2', speaker: 'user', startMs: 1000, text: ' tea.' }
    ])

    expect(prompt).toBe('Bring tea.')
    expect(eventIds).toEqual(['part-1', 'part-2'])
  })

  it('does not offer partial coverage when the delegated utterance contains unidentified fragments', () => {
    const { eventIds, prompt } = delegationPrompt([
      { endMs: 400, event_id: 'identified', speaker: 'user', startMs: 0, text: 'Bring' },
      { endMs: 600, speaker: 'user', startMs: 400, text: ' tea.' }
    ])

    expect(prompt).toBe('Bring tea.')
    expect(eventIds).toBeNull()
  })

  it('keeps the beginning of a request across a long pause with no other speaker', () => {
    const { prompt } = delegationPrompt([
      { endMs: 500, speaker: 'user', startMs: 0, text: 'Do not email' },
      { endMs: 2100, speaker: 'user', startMs: 1600, text: ' Alice.' }
    ])

    expect(prompt).toBe('Do not email Alice.')
  })

  it.each([false, true])('preserves arrival order with incomplete timing (mixed: %s)', mixed => {
    const { context, prompt } = delegationPrompt([
      { endMs: mixed ? 300 : 0, speaker: 'user', startMs: mixed ? 100 : 0, text: 'Cancel that.' },
      { endMs: mixed ? 500 : 0, speaker: 'assistant', startMs: mixed ? 300 : 0, text: 'OK.' },
      { endMs: mixed ? 800 : 0, speaker: 'user', startMs: 0, text: 'Book the room.', timingMissing: true }
    ])

    expect(prompt).toBe('Book the room.')
    expect(context).toBe('User: Cancel that.\nVoice assistant: OK.\nUser: Book the room.')
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

describe('Live words of a recorded call', () => {
  beforeEach(() => {
    resetToasts()
    $voiceLiveDrafts.set({})
    $sessionStates.set({})
  })
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    $voiceLiveDrafts.set({})
    $sessionStates.set({})
  })

  const fragment = (speaker: 'assistant' | 'user', text: string, startMs: number) =>
    ({ endMs: startMs + 200, event_id: `ev-${startMs}`, speaker, startMs, text })

  async function liveCall() {
    const hook = mountLive(vi.fn(), vi.fn(), false)
    await act(async () => hook.result.current.start())
    const handlers = transport.handlers.at(-1)!
    act(() => {
      handlers.onTranscript?.(fragment('user', 'Ask ', 0))
      handlers.onTranscript?.(fragment('user', 'Hermes', 400))
      handlers.onTranscript?.(fragment('assistant', 'Sure.', 800))
    })

    return hook
  }

  it('groups live fragments by speaker and keeps each run\'s timing', async () => {
    await liveCall()
    expect($voiceLiveDrafts.get()['stored-chat']?.map(run => [run.role, run.text, run.startMs, run.endMs, run.firstEventId])).toEqual([
      ['user', 'Ask Hermes', 0, 600, 'ev-0'],
      ['assistant', 'Sure.', 800, 1000, 'ev-800']
    ])
  })

  it('keeps an ended call\'s unsaved words while its chat is still busy, then retires them', async () => {
    const hook = await liveCall()
    $sessionStates.set({ runtime: { ...createClientSessionState(), busy: true, storedSessionId: 'stored-chat' } })
    vi.useFakeTimers()
    act(() => {
      void hook.result.current.end()
    })
    act(() => vi.advanceTimersByTime(5 * 60_000))
    expect($voiceLiveDrafts.get()['stored-chat']).toHaveLength(2)

    const idle = () => $sessionStates.set({ runtime: { ...createClientSessionState(), busy: false, storedSessionId: 'stored-chat' } })
    const busy = () => $sessionStates.set({ runtime: { ...createClientSessionState(), busy: true, storedSessionId: 'stored-chat' } })
    // Checks run every 15 s. A 9 s turn between two checks still restarts the idle wait:
    // idle at 300 s, busy 320–329 s, still up at 346 s, retired once idle 30 s.
    idle()
    act(() => vi.advanceTimersByTime(20_000))
    busy()
    act(() => vi.advanceTimersByTime(9_000))
    idle()
    act(() => vi.advanceTimersByTime(17_000))
    expect($voiceLiveDrafts.get()['stored-chat']).toHaveLength(2)

    act(() => vi.advanceTimersByTime(30_000))
    expect($voiceLiveDrafts.get()['stored-chat']).toBeUndefined()
  })

  it('shows nothing for a call the owner is not recording', async () => {
    transport.captureStatus = 'unavailable'
    const hook = mountLive(vi.fn(), vi.fn(), false)
    await act(async () => hook.result.current.start())
    act(() => transport.handlers.at(-1)!.onTranscript?.(fragment('user', 'Hello', 0)))
    expect($voiceLiveDrafts.get()).toEqual({})
  })
})
