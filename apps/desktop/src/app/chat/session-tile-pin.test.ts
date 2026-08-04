import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $pinnedSessionIds, pinSession, unpinSession } from '@/store/layout'
import { $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { toggleTilePin } from './session-tile'

const STORED = 'tile-session-1'

function row(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    cwd: null,
    ended_at: null,
    id: STORED,
    input_tokens: 0,
    is_active: true,
    last_active: 1,
    message_count: 1,
    model: null,
    output_tokens: 0,
    parent_session_id: null,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: null,
    tool_call_count: 0,
    ...overrides
  }
}

describe('toggleTilePin', () => {
  beforeEach(() => {
    $sessions.set([])
    $pinnedSessionIds.set([])
  })

  afterEach(() => {
    $sessions.set([])
    $pinnedSessionIds.set([])
  })

  it('pins a session that is not yet pinned', () => {
    $sessions.set([row()])

    toggleTilePin(STORED)

    expect($pinnedSessionIds.get()).toContain(STORED)
  })

  it('unpins a session that is already pinned', () => {
    $sessions.set([row()])
    pinSession(STORED)

    toggleTilePin(STORED)

    expect($pinnedSessionIds.get()).not.toContain(STORED)
  })

  it('keys on the durable lineage-root id when available', () => {
    // After auto-compression, the live id rotates but the lineage root stays.
    // The original stored id is the lineage root; the compressed row has a
    // new live id but the same lineage root.
    const rotated = row({ id: 'tile-session-1-compressed', _lineage_root_id: STORED })
    $sessions.set([rotated])

    toggleTilePin(STORED)

    // Pin is keyed on the lineage root, not the rotated live id.
    expect($pinnedSessionIds.get()).toContain(STORED)
    expect($pinnedSessionIds.get()).not.toContain('tile-session-1-compressed')
  })

  it('falls back to storedSessionId when the row is not loaded', () => {
    // A tab-strip "+" tab that hasn't persisted a turn yet has no $sessions row.
    $sessions.set([])

    toggleTilePin(STORED)

    expect($pinnedSessionIds.get()).toContain(STORED)
  })

  it('is idempotent-safe: toggling twice returns to unpinned', () => {
    $sessions.set([row()])

    toggleTilePin(STORED)
    toggleTilePin(STORED)

    expect($pinnedSessionIds.get()).not.toContain(STORED)
  })

  it('agrees with the tab-menu pin path (pinSession/unpinSession)', () => {
    $sessions.set([row()])

    // The tab menu calls pinSession(pinId) directly; toggleTilePin should
    // produce the same result.
    toggleTilePin(STORED)
    const viaToggle = [...$pinnedSessionIds.get()]

    $pinnedSessionIds.set([])
    pinSession(STORED)
    const viaTabMenu = [...$pinnedSessionIds.get()]

    expect(viaToggle).toEqual(viaTabMenu)
  })
})