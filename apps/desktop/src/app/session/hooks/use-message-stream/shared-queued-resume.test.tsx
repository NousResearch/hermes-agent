import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { appendLiveSessionProjection, preserveEquivalentTranscript } from '@/app/session/hooks/use-session-actions/utils'
import { chatMessageText } from '@/lib/chat-messages'

import { renderMessageStream } from './test-harness'

afterEach(cleanup)

it('binds resumed queued occurrences across merging and replay without collapsing repeated inputs', () => {
  for (const complete of [true, false]) {
    const h = renderMessageStream('s')
    const occurrence = { id: 'queue-input', ref: 'peer-ref' }
    const unresolved = appendLiveSessionProjection([], {
      session_id: 's',
      queued: { user: 'Peer queued', inputs: [], inputs_complete: false }
    })
    const messages = preserveEquivalentTranscript(unresolved, appendLiveSessionProjection([], {
      session_id: 's',
      queued: { user: 'Peer queued', inputs: [occurrence], inputs_complete: complete }
    }))

    h.states.set('s', { ...h.state(), messages })
    const start = {
      type: 'message.start' as const,
      session_id: 's',
      turn: { id: 'run', source: { kind: 'unknown' as const } },
      payload: {
        input: { role: 'user' as const, text: 'Peer queued\n\nMerged later' },
        inputs: [occurrence, { id: 'merged-input' }],
        inputs_complete: complete
      }
    }

    act(() => {
      h.handleEvent(start)
      h.handleEvent(start)
    })
    expect(h.state().messages.map(chatMessageText)).toEqual(['Peer queued\n\nMerged later'])
    expect(h.state().messages[0].inputIds).toEqual(['queue-input', 'merged-input'])
    act(() => {
      h.handleEvent({
        ...start,
        turn: { id: 'next-run', source: { kind: 'unknown' } },
        payload: { ...start.payload, inputs: [{ id: 'different-occurrence', ref: 'peer-ref' }] }
      })
    })
    expect(h.state().messages.map(chatMessageText)).toEqual([
      'Peer queued\n\nMerged later',
      'Peer queued\n\nMerged later'
    ])
    cleanup()
  }
})

it('keeps unresolved queued text when the starting occurrence has no matching retained identity', () => {
  const h = renderMessageStream('s')
  const messages = appendLiveSessionProjection([], {
    session_id: 's',
    queued: { user: 'Peer queued', inputs: [], inputs_complete: false }
  })

  h.states.set('s', { ...h.state(), messages })
  act(() => {
    h.handleEvent({
      type: 'message.start',
      session_id: 's',
      turn: { id: 'run', source: { kind: 'unknown' } },
      payload: { input: { role: 'user', text: 'Peer queued' }, inputs: [{ id: 'new-input' }] }
    })
  })
  expect(h.state().messages.map(chatMessageText)).toEqual(['Peer queued', 'Peer queued'])
})
