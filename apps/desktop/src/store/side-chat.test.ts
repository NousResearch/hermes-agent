import { beforeEach, describe, expect, it } from 'vitest'

import { readJson } from '@/lib/storage'

import {
  $sideChatOrigins,
  dismissSideChatOriginBanner,
  markSideChatOrigin,
  mostRecentSideChatId,
  SIDE_CHAT_ORIGINS_KEY,
  sideChatOriginFor
} from './side-chat'

const origin = { fromStoredSessionId: 'main-1', fromMessageId: 'msg-7', fromTitle: 'Settings loader' }

beforeEach(() => {
  $sideChatOrigins.set({})
  window.localStorage.clear()
})

describe('side chat origins', () => {
  it('resolves an origin for the side chat it was recorded for', () => {
    markSideChatOrigin('side-1', origin)

    expect(sideChatOriginFor('side-1')).toEqual(origin)
  })

  it('reports no origin for an unrelated session, so the banner stays off ordinary chats', () => {
    expect(sideChatOriginFor('main-1')).toBeUndefined()
    expect(sideChatOriginFor(null)).toBeUndefined()
  })

  it('keeps the persisted payload in step with the atom, so a restored pane can read it', () => {
    markSideChatOrigin('side-1', origin)

    expect(readJson<Record<string, unknown>>(SIDE_CHAT_ORIGINS_KEY)).toEqual({ 'side-1': origin })
  })

  it('dismisses the notice without forgetting the origin, which is what staging needs', () => {
    markSideChatOrigin('side-1', origin)

    dismissSideChatOriginBanner('side-1')

    expect(sideChatOriginFor('side-1')).toEqual({ ...origin, bannerDismissed: true })
    // Still persisted, still readable: dismissing is a display choice.
    expect(readJson<Record<string, unknown>>(SIDE_CHAT_ORIGINS_KEY)).toHaveProperty('side-1')
  })

  it('records independent origins for concurrent side chats', () => {
    markSideChatOrigin('side-1', origin)
    markSideChatOrigin('side-2', { ...origin, fromMessageId: 'msg-9', fromStoredSessionId: 'main-2' })

    expect(sideChatOriginFor('side-1')?.fromStoredSessionId).toBe('main-1')
    expect(sideChatOriginFor('side-2')?.fromStoredSessionId).toBe('main-2')
  })

  it('ignores an empty session id rather than writing an unreachable entry', () => {
    markSideChatOrigin('', origin)

    expect($sideChatOrigins.get()).toEqual({})
  })
})

describe('mostRecentSideChatId', () => {
  it('picks the newest side chat among the ones actually open', () => {
    markSideChatOrigin('side-1', origin)
    markSideChatOrigin('side-2', origin)

    expect(mostRecentSideChatId(['side-1', 'side-2'])).toBe('side-2')
  })

  it('takes recency from when the side chat was opened, not from pane order', () => {
    markSideChatOrigin('side-1', origin)
    markSideChatOrigin('side-2', origin)

    // A dragged/re-docked tile can sit before an older one in the strip; the
    // most recently OPENED side chat is still side-2.
    expect(mostRecentSideChatId(['side-2', 'side-1'])).toBe('side-2')
  })

  it('never targets a closed pane, so the toggle cannot act on nothing', () => {
    markSideChatOrigin('side-1', origin)
    markSideChatOrigin('side-2', origin)

    // side-2's pane was closed: it keeps its origin (the strip returns if the
    // session is reopened) but is not a toggle target.
    expect(mostRecentSideChatId(['side-1'])).toBe('side-1')
  })

  it('returns nothing when no open tile is a side chat', () => {
    markSideChatOrigin('side-1', origin)

    expect(mostRecentSideChatId(['ordinary-session'])).toBeNull()
    expect(mostRecentSideChatId([])).toBeNull()
  })
})
