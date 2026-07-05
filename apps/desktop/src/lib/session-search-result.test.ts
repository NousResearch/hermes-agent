import { describe, expect, it } from 'vitest'

import { searchResultToSession } from './session-search-result'

describe('searchResultToSession', () => {
  it('preserves safe shared-channel origin labels for search-only rows', () => {
    const session = searchResultToSession({
      channel_origin: {
        chat_name: 'Build Room',
        chat_topic: 'Release coordination',
        chat_type: 'channel',
        display_name: 'Build Room',
        has_thread: true,
        platform: 'webhook'
      },
      lineage_root: 'root-session',
      model: 'claude',
      role: null,
      session_id: 'tip-session',
      session_started: 1_234,
      snippet: 'Shared channel: Build Room',
      source: 'webhook'
    })

    expect(session.id).toBe('tip-session')
    expect(session._lineage_root_id).toBe('root-session')
    expect(session.channel_origin?.display_name).toBe('Build Room')
    expect(session.channel_origin?.chat_topic).toBe('Release coordination')
  })
})
