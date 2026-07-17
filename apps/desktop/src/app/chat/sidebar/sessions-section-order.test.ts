import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { sessionEntriesForSection } from './sessions-section-order'

const session = (id: string, lastActive: number): SessionInfo =>
  ({
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: false,
    last_active: lastActive,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: lastActive,
    title: id,
    tool_call_count: 0
  }) as SessionInfo

describe('sessionEntriesForSection', () => {
  it('preserves the persisted user order for pinned sessions', () => {
    const older = session('older', 1)
    const newer = session('newer', 2)

    expect(sessionEntriesForSection([older, newer], true).map(entry => entry.session.id)).toEqual(['older', 'newer'])
  })

  it('keeps the activity ordering used by regular session lists', () => {
    const older = session('older', 1)
    const newer = session('newer', 2)

    expect(sessionEntriesForSection([older, newer], false).map(entry => entry.session.id)).toEqual(['newer', 'older'])
  })
})
