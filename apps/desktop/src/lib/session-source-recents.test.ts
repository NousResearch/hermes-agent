import { describe, expect, it } from 'vitest'

import {
  isMessagingSource,
  LOCAL_SESSION_SOURCE_IDS,
  MESSAGING_SESSION_SOURCE_IDS,
  recentsExcludedSources,
  sessionOriginBadge
} from './session-source'

describe('recentsExcludedSources', () => {
  it('keeps every messaging platform out of recents by default', () => {
    const excluded = new Set(recentsExcludedSources(false))

    for (const id of MESSAGING_SESSION_SOURCE_IDS) {
      expect(excluded.has(id)).toBe(true)
    }
  })

  it('lets messaging platforms into recents when opted in, but never background sources', () => {
    const optedIn = new Set(recentsExcludedSources(true))
    const byDefault = recentsExcludedSources(false)

    for (const id of MESSAGING_SESSION_SOURCE_IDS) {
      expect(optedIn.has(id)).toBe(false)
    }

    // The opt-in only removes messaging ids: whatever else was excluded stays excluded.
    for (const id of byDefault.filter(id => !isMessagingSource(id))) {
      expect(optedIn.has(id)).toBe(true)
    }
  })

  it('never excludes local interactive sources either way', () => {
    for (const on of [false, true]) {
      const excluded = new Set(recentsExcludedSources(on))

      // Interactive local chats; kanban/oneshot are local but background-only.
      for (const id of LOCAL_SESSION_SOURCE_IDS.filter(id => ['cli', 'desktop', 'tui'].includes(id))) {
        expect(excluded.has(id)).toBe(false)
      }
    }
  })
})

describe('sessionOriginBadge', () => {
  it('badges a live messaging session with its own platform', () => {
    expect(sessionOriginBadge({ source: 'api_server' })).toEqual({ kind: 'live', source: 'api_server' })
    expect(sessionOriginBadge({ source: 'WhatsApp' })).toEqual({ kind: 'live', source: 'whatsapp' })
  })

  it('badges a local session only when it was handed off from a platform', () => {
    expect(sessionOriginBadge({ source: 'desktop' })).toBeNull()
    expect(sessionOriginBadge({ handoff_platform: 'telegram', handoff_state: 'completed', source: 'desktop' })).toEqual(
      { kind: 'handoff', source: 'telegram' }
    )
  })
})
