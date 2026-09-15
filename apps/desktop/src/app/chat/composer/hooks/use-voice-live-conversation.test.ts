// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type * as VoiceLiveModule from '@/lib/voice-live'

const liveSession = vi.hoisted(() => ({
  handlers: null as null | {
    onDelegation: (
      id: string,
      context: Array<{ endMs: number; speaker: 'assistant' | 'user'; startMs: number; text: string }>
    ) => void
  }
}))

vi.mock('@/lib/voice-live', async importOriginal => {
  const actual = await importOriginal<typeof VoiceLiveModule>()

  return {
    ...actual,
    VoiceLiveSession: class {
      constructor(handlers: NonNullable<typeof liveSession.handlers>) {
        liveSession.handlers = handlers
      }

      close() {}
      async start() {}
    }
  }
})

import { chunkForCommentary, LiveTranscriptBuffer, toLiveHistory } from '@/lib/voice-live'

import { delegationPrompt, useVoiceLiveConversation } from './use-voice-live-conversation'

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
    expect(context).not.toContain('User: What is the weather in Paris?')
  })

  it('keeps every fragment of the latest utterance when the supplemental window caps at 80', () => {
    const buffer = new LiveTranscriptBuffer()
    const beginning = 'BEGIN-FRAGMENT-WINDOW'
    const middle = 'MIDDLE-FRAGMENT-WINDOW'
    const end = 'END-FRAGMENT-WINDOW'

    for (let index = 0; index < 140; index += 1) {
      const marker = index === 0 ? beginning : index === 70 ? middle : index === 139 ? end : `part-${index}`
      buffer.record({ endMs: index + 1, speaker: 'user', startMs: index, text: `${marker} ` })
    }

    const snapshot = buffer.takeDelegationSnapshot()

    expect(snapshot.context).toHaveLength(80)
    expect(snapshot.diagnostics.droppedByCount).toBe(60)
    expect(snapshot.diagnostics.latestUserFragments).toBe(140)
    expect(snapshot.latestUserUtterance).toContain(beginning)
    expect(snapshot.latestUserUtterance).toContain(middle)
    expect(snapshot.latestUserUtterance).toContain(end)

    buffer.record({ endMs: 141, speaker: 'user', startMs: 140, text: 'AFTER-DELEGATION-TAIL' })
    expect(buffer.takeDelegationSnapshot().latestUserUtterance).toContain(beginning)
    expect(buffer.takeDelegationSnapshot().latestUserUtterance).toContain('AFTER-DELEGATION-TAIL')
  })

  it('does not merge user utterances when transcript retention drops the separating history', () => {
    const buffer = new LiveTranscriptBuffer()
    buffer.record({ endMs: 1, speaker: 'user', startMs: 0, text: 'OLD-USER-UTTERANCE' })
    buffer.record({ endMs: 2, speaker: 'assistant', startMs: 1, text: 'separator' })

    for (let index = 0; index < 2_100; index += 1) {
      buffer.record({ endMs: index + 3, speaker: 'assistant', startMs: index + 2, text: 'filler' })
    }

    buffer.record({ endMs: 2_104, speaker: 'user', startMs: 2_103, text: 'NEW-USER-UTTERANCE' })
    const snapshot = buffer.takeDelegationSnapshot()

    expect(snapshot.latestUserUtterance).toBe('NEW-USER-UTTERANCE')
    expect(snapshot.diagnostics.droppedByRetention).toBeGreaterThan(0)
  })

  it('preserves a complete long latest utterance outside the bounded transcript window', () => {
    const beginning = 'BEGIN-LONG-REQUEST'
    const middle = 'MIDDLE-LONG-REQUEST'
    const end = 'END-LONG-REQUEST'
    const authoritativeUtterance = `${beginning} ${'alpha '.repeat(900)}${middle} ${'omega '.repeat(900)}${end}`

    const clippedWindow = [
      { endMs: 500, speaker: 'assistant' as const, startMs: 0, text: 'Earlier answer' },
      { endMs: 1_000, speaker: 'user' as const, startMs: 500, text: `${middle} ${'omega '.repeat(20)}${end}` }
    ]

    const result = delegationPrompt(clippedWindow, authoritativeUtterance)

    expect(result.prompt).toContain(beginning)
    expect(result.prompt).toContain(middle)
    expect(result.prompt).toContain(end)
    expect(result.prompt.length).toBeGreaterThan(10_000)
    expect(result.context).toBe('Voice assistant: Earlier answer')
    expect(result.diagnostics.promptSource).toBe('authoritative-utterance')
    expect(result.diagnostics.contextCapApplied).toBe(false)
  })

  it('prefers a demonstrably more complete latest user window over a partial accumulator', () => {
    const result = delegationPrompt(
      [{ endMs: 1_000, speaker: 'user', startMs: 0, text: 'BEGIN recovered middle recovered END' }],
      'middle recovered END'
    )

    expect(result.prompt).toBe('BEGIN recovered middle recovered END')
    expect(result.diagnostics.promptSource).toBe('window-last-user-recovery')
  })

  it('uses the complete latest user window turn as an explicit recovery and never assistant transcript tail', () => {
    const beginning = 'BEGIN-FALLBACK'
    const middle = 'MIDDLE-FALLBACK'
    const end = 'END-FALLBACK'
    const latest = `${beginning} ${'detail '.repeat(100)}${middle} ${'finish '.repeat(100)}${end}`

    const recovered = delegationPrompt([
      { endMs: 500, speaker: 'assistant', startMs: 0, text: 'Earlier answer' },
      { endMs: 1_000, speaker: 'user', startMs: 500, text: latest }
    ])

    const missing = delegationPrompt([
      { endMs: 500, speaker: 'assistant', startMs: 0, text: 'Do not submit these assistant words' }
    ])

    expect(recovered.prompt).toContain(beginning)
    expect(recovered.prompt).toContain(middle)
    expect(recovered.prompt).toContain(end)
    expect(recovered.diagnostics.promptSource).toBe('window-last-user-recovery')
    expect(missing.prompt).toBe('')
    expect(missing.diagnostics.promptSource).toBe('missing-user-utterance')
  })

  it('caps only earlier voice context from the oldest side', () => {
    const newestMarker = 'NEWEST-CONTEXT-MARKER'

    const result = delegationPrompt(
      [
        { endMs: 500, speaker: 'assistant', startMs: 0, text: `OLDEST-CONTEXT-MARKER ${'old '.repeat(2_000)}` },
        { endMs: 1_000, speaker: 'user', startMs: 500, text: 'prior question' },
        { endMs: 1_500, speaker: 'assistant', startMs: 1_000, text: `${'new '.repeat(1_000)}${newestMarker}` },
        { endMs: 2_000, speaker: 'user', startMs: 1_500, text: 'latest request' }
      ],
      'latest request'
    )

    expect(result.prompt).toBe('latest request')
    expect(result.context.length).toBeLessThanOrEqual(6_000)
    expect(result.context).toContain(newestMarker)
    expect(result.context).not.toContain('OLDEST-CONTEXT-MARKER')
    expect(result.diagnostics.contextCapApplied).toBe(true)
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

  it('queues a second delegation while Hermes is busy without interrupting or disarming the first', async () => {
    const consumePendingResponse = vi.fn()
    const onInterrupt = vi.fn()
    const onSubmit = vi.fn()

    const { rerender, result } = renderHook(
      ({ busy }) =>
        useVoiceLiveConversation({
          busy,
          consumePendingResponse,
          enabled: true,
          onInterrupt,
          onSubmit,
          pendingResponse: () => null,
          queueBusyDelegations: true,
          seedHistory: () => []
        }),
      { initialProps: { busy: false } }
    )

    await act(async () => result.current.start())
    act(() => {
      liveSession.handlers?.onDelegation('delegation-1', [
        { endMs: 500, speaker: 'user', startMs: 0, text: 'Do this first' }
      ])
    })
    rerender({ busy: true })
    act(() => {
      liveSession.handlers?.onDelegation('delegation-2', [
        { endMs: 1000, speaker: 'user', startMs: 500, text: 'Do this next' }
      ])
    })

    expect(onInterrupt).not.toHaveBeenCalled()
    expect(onSubmit).toHaveBeenNthCalledWith(1, 'Do this first', '', false, expect.any(Function))
    expect(onSubmit).toHaveBeenNthCalledWith(2, 'Do this next', '', true, expect.any(Function))
    expect(consumePendingResponse).toHaveBeenCalledTimes(1)

    const onQueuedDrain = onSubmit.mock.calls[1]?.[3]
    act(() => onQueuedDrain())
    expect(consumePendingResponse).toHaveBeenCalledTimes(2)
  })

  it('submits a busy delegation immediately under the canonical interrupt policy', async () => {
    const onSubmit = vi.fn()

    const { result } = renderHook(() =>
      useVoiceLiveConversation({
        busy: true,
        consumePendingResponse: vi.fn(),
        enabled: true,
        onSubmit,
        pendingResponse: () => null,
        queueBusyDelegations: false,
        seedHistory: () => []
      })
    )

    await act(async () => result.current.start())
    act(() => {
      liveSession.handlers?.onDelegation('delegation-1', [
        { endMs: 500, speaker: 'user', startMs: 0, text: 'Replace the current turn' }
      ])
    })

    expect(onSubmit).toHaveBeenCalledWith('Replace the current turn', '', false, expect.any(Function))
  })
})
