import { NEW_SESSION_TITLE } from '@/lib/chat-runtime'
import { setSessionOwnerHint, setUnlistedSessionOwnerRows } from '@/store/session'
import type { SessionProfileRoute } from '@/store/session-request-router'
import type { SessionCreateResponse } from '@/types/hermes'

import { buildOptimisticSession } from './utils'

/**
 * Unlisted-draft owner stubs — the rows a tab-strip "+"/⌘T/⌘N draft rides
 * until its first turn persists and lists it. Lives in its own topical
 * sibling of utils.ts (moved there: that file sits on the FILE_LINES
 * ratchet and may only go down; moved code keeps its cap).
 */

/**
 * Record the owner of an UNLISTED draft tile without touching the visible
 * sidebar list (see `openNewSessionTile` with `listed: false`, #102792). The
 * stub carries the same stamps an optimistic row would — ambient profile for
 * an unrouted create, exact connection tag for a routed one — and rides only
 * the owner-lookup path, never the sidebar render path. A later real row for
 * the same id shadows it; the first send replaces it outright.
 *
 * The stub is TITLED "New session": every surface that resolves a row through
 * `ownerLookupSessionRows` (the tab strip's `tileTitle`, the document header)
 * must read the same placeholder a pure view-reset draft does, not
 * `sessionTitle()`'s "Untitled session" fallback — an untouched draft is
 * "new", not "untitled". Client-side only: the backend row stays untitled, so
 * derived/llm auto-titling still lands on the first exchange.
 */
export function upsertUnlistedSessionOwner(
  created: SessionCreateResponse,
  id: string,
  owner?: null | SessionProfileRoute
) {
  const stub = buildOptimisticSession(created, id, NEW_SESSION_TITLE, null, null, undefined, owner)
  setSessionOwnerHintForStub(id, owner)
  setUnlistedSessionOwnerRows(prev => [stub, ...prev.filter(s => s.id !== id)])
}

function setSessionOwnerHintForStub(id: string, owner?: null | SessionProfileRoute): void {
  if (owner) {
    setSessionOwnerHint(id, owner)
  }
}
