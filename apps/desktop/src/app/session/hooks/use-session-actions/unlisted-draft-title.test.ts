import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { NEW_SESSION_TITLE } from '@/lib/chat-runtime'
import {
  $unlistedSessionOwnerRows,
  ownerLookupSessionRows,
  setUnlistedSessionOwnerRows
} from '@/store/session'
import type { SessionCreateResponse } from '@/types/hermes'

import { upsertUnlistedSessionOwner } from './unlisted-owners'

// The shape openNewSessionTile answers for a `listed: false` draft
// (tab-strip "+", ⌘T, ⌘N). Zero messages, no runtime info to seed from.
const created: SessionCreateResponse = {
  message_count: 0,
  session_id: 'rt-1',
  stored_session_id: 'stored-1'
}

describe('upsertUnlistedSessionOwner (unlisted draft stubs)', () => {
  beforeEach(() => {
    setUnlistedSessionOwnerRows([])
  })

  afterEach(() => {
    // The atom is module state; leave it clean for the next test file.
    setUnlistedSessionOwnerRows([])
  })

  it('stamps the draft stub with the "New session" placeholder', () => {
    // An untouched draft reads "New session" everywhere, never
    // sessionTitle()'s "Untitled session" fallback: the draft is new, not
    // untitled. Backend row stays untitled, so auto-titling is unaffected.
    upsertUnlistedSessionOwner(created, 'stored-1')

    expect($unlistedSessionOwnerRows.get()).toHaveLength(1)
    expect($unlistedSessionOwnerRows.get()[0].title).toBe(NEW_SESSION_TITLE)
  })

  it('surfaces the placeholder through the owner-lookup path the tab strip reads', () => {
    // tileTitle() resolves a tab's row via ownerLookupSessionRows(); the stub
    // rides that path (never the sidebar render path), so the lookup must
    // yield a titled row.
    upsertUnlistedSessionOwner(created, 'stored-1')

    const row = ownerLookupSessionRows().find(session => session.id === 'stored-1')

    expect(row?.title).toBe(NEW_SESSION_TITLE)
  })

  it('replaces an earlier stub for the same id instead of stacking a twin', () => {
    upsertUnlistedSessionOwner(created, 'stored-1')
    upsertUnlistedSessionOwner({ ...created, session_id: 'rt-2' }, 'stored-1')

    const rows = $unlistedSessionOwnerRows.get()

    expect(rows).toHaveLength(1)
    expect(rows[0].title).toBe(NEW_SESSION_TITLE)
  })
})
