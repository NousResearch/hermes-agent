import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'

let stream: MessageStreamHarness

function emit(type: GatewayEvent['type'], payload: GatewayEvent['payload'] = {}, sessionId = SID) {
  act(() => stream.handleEvent({ payload, session_id: sessionId, type }))
}

describe('btw.progress event', () => {
  beforeEach(() => {
    stream = renderMessageStream(SID)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  // A /btw --explore thread's steps fold into ONE live row per thread, so the side work reads as
  // its own panel instead of a row per tool call.
  it('folds every step of a thread into one live system row', () => {
    emit('btw.progress', { task_id: 'btw_ab12cd', text: 'exploring with read_file, search_files' })
    emit('btw.progress', { task_id: 'btw_ab12cd', text: 'search_files · retry' })
    emit('btw.progress', { task_id: 'btw_ff0011', text: 'read_file · net.py' })

    const rows = stream.state(SID).messages.filter(message => message.id.startsWith('btw-progress-'))

    expect(rows.map(row => row.id)).toEqual(['btw-progress-btw_ab12cd', 'btw-progress-btw_ff0011'])
    expect(rows[0].parts.map(part => ('text' in part ? part.text : '')).join('')).toBe(
      '[btw btw_ab12cd]\n┊ exploring with read_file, search_files\n┊ search_files · retry'
    )
  })

  it('ignores an empty step', () => {
    emit('btw.progress', { task_id: 'btw_ab12cd', text: '  ' })

    expect(stream.state(SID).messages).toEqual([])
  })
})
