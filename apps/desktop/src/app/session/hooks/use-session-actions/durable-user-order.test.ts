import { describe, expect, it } from 'vitest'

import { chatMessageText, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage, SessionResumeResult } from '@/types/hermes'

import { reconcilePersistedLiveTurn } from './persisted-live-turn'
import { appendLiveSessionProjection } from './utils'

const prompt = 'Build the release'
const corrections = ['hurry up', 'include the migration steps']
const steer = 'Continue with the approved plan'
const answer = 'The release and its migration guide are ready.'

const toolRound = (id: number, timestamp: number): SessionMessage => ({
  id,
  role: 'assistant',
  content: `Working on step ${id}.`,
  timestamp,
  tool_calls: [{ id: `call-${id}`, type: 'function', function: { name: 'terminal', arguments: '{}' } }]
})

describe('durable user order during live projection', () => {
  it.each(
    [false, true].flatMap(streaming => [
      { streaming, corrections: [...corrections, steer], snapshot: 'plain and steer corrections' },
      { streaming, corrections: [steer], snapshot: 'steer-only corrections' }
    ])
  )('keeps persisted mid-turn prompts above their final answer ($snapshot, streaming=$streaming)', snapshot => {
    // #141351: durable tool rounds separate the prompt, corrections and steer.
    // None of these user rows needs an optimistic renderer twin to retain its place.
    const rows: SessionMessage[] = [
      { id: 90, role: 'user', content: prompt, timestamp: 100 },
      toolRound(99, 150),
      { id: 100, role: 'user', content: corrections[0], timestamp: 157 },
      toolRound(144, 600),
      { id: 145, role: 'user', content: corrections[1], timestamp: 624 },
      toolRound(150, 630),
      { id: 151, role: 'user', content: steer, timestamp: 635, display_kind: 'steer' },
      { id: 274, role: 'assistant', content: answer, timestamp: 1900 }
    ]

    const stored = toChatMessages(rows)

    const projection: SessionResumeResult = {
      session_id: 'runtime',
      resumed: 'runtime',
      message_count: rows.length,
      messages: rows,
      turn_started_at: 100,
      inflight: { user: prompt, corrections: snapshot.corrections, assistant: answer, streaming: snapshot.streaming }
    }

    // The stored final reply sends session activation through the projection
    // fallback rather than the anchored reconciliation for partial turns.
    expect(reconcilePersistedLiveTurn(stored, [], rows, projection)).toBeNull()

    const projected = appendLiveSessionProjection(stored, projection)

    expect(projected.filter(message => message.role === 'user').map(message => message.rowId)).toEqual([
      90, 100, 145, 151
    ])
    expect(projected.at(-1)?.rowId).toBe(274)
    expect(projected.map(chatMessageText)).toEqual(stored.map(chatMessageText))
  })

  it.each([false, true])('keeps a repeated prompt distinct from a completed prior turn (tool-bearing=%s)', toolBearing => {
    const stored = toChatMessages([
      { id: 80, role: 'user', content: prompt, timestamp: 1 },
      {
        ...(toolBearing ? toolRound(81, 2) : { id: 81, role: 'assistant' as const, timestamp: 2 }),
        content: 'The previous release is complete.'
      },
      { id: 90, role: 'user', content: 'A different task', timestamp: 100 },
      toolRound(99, 150),
      { id: 100, role: 'user', content: 'Check it carefully', timestamp: 157 }
    ])

    const projected = appendLiveSessionProjection(stored, {
      session_id: 'runtime',
      turn_started_at: 100,
      inflight: { user: prompt, assistant: '', streaming: true }
    })

    expect(projected.filter(message => message.role === 'user').map(chatMessageText)).toEqual([
      prompt, 'A different task', 'Check it carefully', prompt
    ])
    expect(projected.find(message => message.id === 'user-inflight-runtime')?.rowId).toBeUndefined()
    expect(projected.at(-1)).toMatchObject({ role: 'assistant', pending: true })
  })
})
