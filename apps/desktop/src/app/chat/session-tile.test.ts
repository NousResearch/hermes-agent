import { afterEach, describe, expect, it } from 'vitest'

import { setSessions } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { sessionTileResumeFailure, tileTitle } from './session-tile'

function listed(over: Partial<SessionInfo>): SessionInfo {
  return {
    ended_at: null,
    id: 'bot-chat',
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 2,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: null,
    tool_call_count: 0,
    ...over
  }
}

describe('sessionTileResumeFailure', () => {
  it('keeps a confirmed durable session retryable instead of repeating a stale 404', () => {
    expect(sessionTileResumeFailure('session not found', true, true)).toBe(
      'Session is still available — retry resuming it.'
    )
  })

  it('fails safe on an inconclusive durable lookup', () => {
    expect(sessionTileResumeFailure('404', false, true)).toBe('Session unavailable — you can retry resuming it.')
  })

  it('does not overwrite a tile that rebound while the lookup was pending', () => {
    expect(sessionTileResumeFailure('session not found', true, false)).toBeUndefined()
  })
})

describe('tileTitle canonical Bot Chat identity', () => {
  afterEach(() => {
    setSessions([])
    $sessionTiles.set([])
  })

  it('keeps Bot Chat on re-bind instead of a preview-derived listing caption', () => {
    setSessions([listed({ id: 'bot-chat', preview: 'what is the weather in oslo', root_title: 'Bot Chat', title: '' })])
    $sessionTiles.set([{ storedSessionId: 'bot-chat', workspaceMode: 'bots' }])

    expect(tileTitle('bot-chat')).toBe('Bot Chat')
  })

  it('keeps Bot Chat when the tile already carries the canonical tab title', () => {
    setSessions([listed({ id: 'bot-chat', preview: 'what is the weather in oslo', title: '' })])
    $sessionTiles.set([{ storedSessionId: 'bot-chat', workspaceMode: 'bots', workspaceTabTitle: 'Bot Chat' }])

    expect(tileTitle('bot-chat')).toBe('Bot Chat')
  })

  it('keeps preview captions for non-canonical sessions', () => {
    setSessions([listed({ id: 'notes', preview: 'what is the weather in oslo', title: '' })])
    $sessionTiles.set([{ storedSessionId: 'notes' }])

    expect(tileTitle('notes')).toBe('what is the weather in oslo')
  })
})
