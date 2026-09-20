import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useSubmitPrompt } from '@/app/session/hooks/use-prompt-actions/submit'
import { chatMessageText, textPart } from '@/lib/chat-messages'

import { renderMessageStream } from './test-harness'
import { STREAM_DELTA_FLUSH_MS } from './utils'

const input = (id: string, ref?: string, session = 's'): GatewayEvent => ({
  type: 'message.input',
  session_id: session,
  turn: { id: 'run', source: { kind: 'unknown' } },
  payload: {
    kind: 'redirect',
    input: { role: 'user', text: 'Same words', display_kind: 'steer' },
    inputs: [{ id, ref }]
  }
})

describe('shared correction observation', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('flushes prior output, inserts each occurrence once, and keeps later output below it', async () => {
    vi.useFakeTimers()
    const h = renderMessageStream('s')
    act(() => {
      h.handleEvent({ type: 'message.start', session_id: 's', turn: { id: 'run', source: { kind: 'unknown' } } })
      h.handleEvent({ type: 'message.delta', session_id: 's', payload: { text: 'Before' } })
      h.handleEvent(input('one'))
      h.handleEvent(input('one')) // replay overlap
      h.handleEvent(input('two')) // identical words, different submission
      h.handleEvent({ type: 'message.delta', session_id: 's', payload: { text: 'After' } })
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })
    expect(h.state().messages.map(m => [m.role, chatMessageText(m)])).toEqual([
      ['assistant', 'Before'],
      ['user', 'Same words'],
      ['user', 'Same words'],
      ['assistant', 'After']
    ])
    expect(h.state().busy).toBe(true)
  })

  it('recognizes its own optimistic row by reference and rejects foreign executions or unscoped input', () => {
    const h = renderMessageStream('s')
    act(() => {
      h.handleEvent({ type: 'message.start', session_id: 's', turn: { id: 'run', source: { kind: 'unknown' } } })
    })
    h.states.set('s', { ...h.state(), messages: [{ id: 'local-ref', role: 'user', parts: [textPart('Same words')] }] })
    act(() => {
      h.handleEvent(input('own', 'local-ref'))
      h.handleEvent(input('own', 'local-ref'))
      h.handleEvent({ ...input('stale'), turn: { id: 'old-run', source: { kind: 'unknown' } } })
      h.handleEvent({ ...input('unscoped'), session_id: undefined })
      h.handleEvent({ ...input('hidden'), payload: { kind: 'redirect', input: null, inputs: [{ id: 'hidden' }] } })
    })
    expect(h.state().messages.map(m => m.id)).toEqual(['local-ref'])
    act(() => {
      h.handleEvent(input('second-occurrence', 'local-ref'))
    })
    expect(h.state().messages.map(chatMessageText)).toEqual(['Same words', 'Same words'])
    // A background session's correction belongs to its own state and never the focused chat.
    act(() => {
      h.handleEvent({ type: 'message.start', session_id: 'background', turn: { id: 'run', source: { kind: 'unknown' } } })
      h.handleEvent(input('background', undefined, 'background'))
    })
    expect(h.state().messages).toHaveLength(2)
    expect(h.state('background').messages.map(chatMessageText)).toEqual(['Same words'])
  })
})

const start = (execution: string, id: string, ref?: string, session = 's'): GatewayEvent => ({
  type: 'message.start',
  session_id: session,
  turn: { id: execution, source: { kind: 'unknown' } },
  payload: {
    input: { role: 'user', text: 'Same words' },
    inputs: [{ id, ref }]
  }
})

describe('shared starting input observation', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('paints peer starts once before output while rejecting stale, interrupted and unscoped observations', async () => {
    vi.useFakeTimers()
    const h = renderMessageStream('s')
    act(() => {
      h.handleEvent(start('run', 'first'))
    })
    expect(h.state().messages.map(m => [m.role, chatMessageText(m)])).toEqual([['user', 'Same words']])
    expect(h.state().messages[0]?.inputIds).toEqual(['first'])
    act(() => {
      h.handleEvent({ type: 'message.delta', session_id: 's', payload: { text: 'Before' } })
      h.handleEvent(input('correction'))
      h.handleEvent({ type: 'message.delta', session_id: 's', payload: { text: 'After' } })
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })
    const streaming = h.state()
    act(() => {
      h.handleEvent(start('run', 'first')) // replay must not reset the live stream or correction ids
      h.handleEvent({ ...input('foreign'), turn: { id: 'foreign-run', source: { kind: 'unknown' } } })
      h.handleEvent({ ...start('unscoped-run', 'unscoped'), session_id: undefined })
      h.handleEvent(input('correction'))
    })
    expect(h.state()).toBe(streaming)
    expect(h.state().messages.map(m => [m.role, chatMessageText(m)])).toEqual([
      ['user', 'Same words'],
      ['assistant', 'Before'],
      ['user', 'Same words'],
      ['assistant', 'After']
    ])
    act(() => {
      h.handleEvent({ type: 'message.complete', session_id: 's', turn: { id: 'run', source: { kind: 'unknown' } }, payload: { text: 'After' } })
      h.handleEvent({ type: 'session.info', session_id: 's', payload: { running: true } })
      h.handleEvent(start('second-run', 'second')) // same words, fresh execution and occurrence
      h.handleEvent(start('run', 'first')) // replay of a retired execution
      h.handleEvent({ ...input('late-correction'), turn: { id: 'run', source: { kind: 'unknown' } } })
    })
    expect(h.state().observedExecutionId).toBe('second-run')
    expect(
      h
        .state()
        .messages.filter(m => m.role === 'user')
        .map(m => m.inputIds)
    ).toEqual([['first'], ['correction'], ['second']])
    h.states.set('s', { ...h.state(), interrupted: true })
    const stopped = h.state()
    act(() => {
      h.handleEvent(start('interrupted-run', 'interrupted'))
      h.handleEvent({ ...input('interrupted'), turn: { id: 'second-run', source: { kind: 'unknown' } } })
      h.handleEvent(start('background-run', 'background', undefined, 'background'))
    })
    expect(h.state()).toBe(stopped)
    expect(h.state('background').messages.map(chatMessageText)).toEqual(['Same words'])

    for (const projection of [null, { role: 'user', text: 'internal', display_kind: 'hidden' }]) {
      const sid = projection ? 'hidden' : 'null'
      act(() => {
        h.handleEvent({
          ...start(sid, sid, undefined, sid),
          payload: { input: projection, inputs: [{ id: sid }] }
        })
      })
      expect(h.state(sid).messages).toEqual([])
      expect(h.state(sid).observedInputIds).toEqual([sid])
    }

    act(() => {
      h.handleEvent({ type: 'message.start', session_id: 'legacy' })
    })
    expect(h.state('legacy')).toMatchObject({ busy: true, awaitingResponse: true, messages: [] })
    // A fresh execution must still be visible when this observer missed the prior terminal.
    act(() => {
      h.handleEvent(start('missed-terminal', 'old-input', undefined, 'missed'))
      h.handleEvent({ type: 'session.info', session_id: 'missed', payload: { running: true } })
      h.handleEvent(start('fresh-execution', 'new-input', undefined, 'missed'))
      h.handleEvent(start('missed-terminal', 'old-input', undefined, 'missed'))
    })
    expect(h.state('missed').observedExecutionId).toBe('fresh-execution')
    expect(h.state('missed').messages.map(m => m.inputIds)).toEqual([['old-input'], ['new-input']])
  })

  it('renders the complete merged projection without hiding peer input or duplicating optimistic constituents', () => {
    for (const ownCount of [0, 1, 2]) {
      const h = renderMessageStream('s')
      h.states.set('s', {
        ...h.state(),
        messages: Array.from({ length: ownCount }, (_, index) => ({
          id: `own-${index}`,
          role: 'user' as const,
          parts: [textPart(`Part ${index}`)]
        }))
      })

      const event: GatewayEvent = {
        ...start('merged-run', 'first'),
        payload: {
          input: { role: 'user', text: 'Part 0\nPart 1\nPeer addition' },
          inputs: [{ id: 'first', ref: 'own-0' }, { id: 'second', ref: 'own-1' }, { id: 'peer' }],
          inputs_complete: true
        }
      }

      act(() => {
        h.handleEvent(event)
        h.handleEvent(event)
      })
      expect(h.state().messages.map(chatMessageText)).toEqual(['Part 0\nPart 1\nPeer addition'])
      expect(h.state().messages[0]?.inputIds).toEqual(['first', 'second', 'peer'])

      if (ownCount) {
        expect(h.state().messages[0]?.id).toBe('own-0')
      }

      cleanup()
    }

    const h = renderMessageStream('s')
    act(() => {
      h.handleEvent({
        ...start('evicted-run', 'unused'),
        payload: {
          input: { role: 'user', text: 'Visible even after correlation eviction' },
          inputs: [],
          inputs_complete: false
        }
      })
    })
    expect(h.state().messages.map(chatMessageText)).toEqual(['Visible even after correlation eviction'])
  })

  it('binds the real optimistic submit by reference in either ACK order and never reuses it for a later occurrence', async () => {
    for (const ackFirst of [false, true]) {
      const h = renderMessageStream('s')
      let acknowledge!: () => void
      let submittedRef = ''

      const ack = new Promise<void>(resolve => {
        acknowledge = resolve
      })

      const requestGateway = async <T,>(method: string, params?: Record<string, unknown>): Promise<T> => {
        expect(method).toBe('prompt.submit')
        submittedRef = params?.submission_ref as string
        await ack

        return { status: 'accepted' } as T
      }

      const { result } = renderHook(() =>
        useSubmitPrompt({
          activeSessionIdRef: { current: 's' },
          busyRef: { current: false },
          copy: {} as Parameters<typeof useSubmitPrompt>[0]['copy'],
          createBackendSessionForSend: async () => 's',
          getRoutedStoredSessionId: () => null,
          getRuntimeIdForStoredSession: () => null,
          getRouteToken: () => 'stable',
          requestGateway,
          runtimeIdByStoredSessionIdRef: { current: new Map() },
          resumeStoredSession: async () => undefined,
          selectedStoredSessionIdRef: { current: null },
          syncAttachmentsForSubmit: async sessionId => ({ sessionId, attachments: [] }),
          updateSessionState: (sid, update) => {
            const next = update(h.state(sid))
            h.states.set(sid, next)

            return next
          },
          scope: {
            removeAttachments: () => undefined,
            readAttachments: () => [],
            setAwaitingResponse: () => undefined,
            setBusy: () => undefined,
            setMessages: () => undefined
          }
        })
      )

      let pending!: Promise<boolean>
      await act(async () => {
        pending = result.current('Same words')
      })
      expect(submittedRef).not.toBe('')
      expect(h.state().messages.map(m => m.id)).toEqual([submittedRef])

      if (ackFirst) {
        await act(async () => {
          acknowledge()
          expect(await pending).toBe(true)
        })
      }

      act(() => {
        h.handleEvent(start('own-run', 'own-input', submittedRef))
        h.handleEvent(start('own-run', 'own-input', submittedRef))
      })

      if (!ackFirst) {
        await act(async () => {
          acknowledge()
          expect(await pending).toBe(true)
        })
      }

      expect(h.state().messages.map(m => [m.id, m.inputIds, chatMessageText(m)])).toEqual([
        [submittedRef, ['own-input'], 'Same words']
      ])
      act(() => {
        h.handleEvent({ type: 'message.complete', session_id: 's', turn: { id: 'own-run', source: { kind: 'unknown' } }, payload: { text: 'Done' } })
        h.handleEvent(start('retry-run', 'retry-input', submittedRef))
      })
      expect(
        h
          .state()
          .messages.filter(m => m.role === 'user')
          .map(m => m.inputIds)
      ).toEqual([['own-input'], ['retry-input']])
      cleanup()
    }
  })
})
