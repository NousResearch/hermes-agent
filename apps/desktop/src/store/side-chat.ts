/**
 * Where a side chat was opened from.
 *
 * A side chat is an ordinary session — that is exactly what makes it
 * independent and what lets it reuse the tile chassis for free. The one thing
 * an ordinary session cannot hold is provenance: which conversation, and which
 * message, it was opened from. That is presentation state the renderer owns (no
 * other Hermes surface changes it), so it lives here rather than on the session
 * row.
 *
 * Persisted, because a restored pane must still show the banner and still offer
 * "stage in main composer" after a restart.
 */

import { atom } from 'nanostores'

import { readJson, writeJson } from '@/lib/storage'

export const SIDE_CHAT_ORIGINS_KEY = 'hermes.desktop.sideChatOrigins.v1'

export interface SideChatOrigin {
  /** True once the user dismissed the provenance strip. The record stays: the
   *  origin is what enables "stage in main composer", so hiding the notice must
   *  not cost the capability. */
  bannerDismissed?: boolean
  /** The message the selection came from, when it resolved to one. */
  fromMessageId?: string
  /** The conversation the selection was taken from. */
  fromStoredSessionId: string
  /** Its title at capture time — the banner names where the context came from. */
  fromTitle: string
}

function load(): Record<string, SideChatOrigin> {
  const parsed = readJson<Record<string, SideChatOrigin>>(SIDE_CHAT_ORIGINS_KEY) ?? {}
  const origins: Record<string, SideChatOrigin> = {}

  for (const [storedSessionId, origin] of Object.entries(parsed)) {
    // A hand-edited or half-written payload must not render a broken banner.
    if (origin && typeof origin.fromStoredSessionId === 'string' && typeof origin.fromTitle === 'string') {
      origins[storedSessionId] = origin
    }
  }

  return origins
}

export const $sideChatOrigins = atom<Record<string, SideChatOrigin>>(load())

function publish(next: Record<string, SideChatOrigin>): void {
  $sideChatOrigins.set(next)
  writeJson(SIDE_CHAT_ORIGINS_KEY, next)
}

export function markSideChatOrigin(storedSessionId: string, origin: SideChatOrigin): void {
  if (!storedSessionId) {
    return
  }

  publish({ ...$sideChatOrigins.get(), [storedSessionId]: origin })
}

export function sideChatOriginFor(storedSessionId: string | null | undefined): SideChatOrigin | undefined {
  return storedSessionId ? $sideChatOrigins.get()[storedSessionId] : undefined
}

/** Hide the provenance strip without forgetting where the side chat came from.
 *  The origin is what keeps "stage in main composer" available, so dismissing
 *  the notice must not take the capability with it. */
export function dismissSideChatOriginBanner(storedSessionId: string): void {
  const current = $sideChatOrigins.get()[storedSessionId]

  if (!current || current.bannerDismissed) {
    return
  }

  publish({ ...$sideChatOrigins.get(), [storedSessionId]: { ...current, bannerDismissed: true } })
}

export interface SideChatRequest {
  /** The conversation the selection was taken from. Omitted by entrances that
   *  only exist in the primary chat; the wiring then falls back to the primary
   *  session. A selection made in another pane carries its own. */
  fromStoredSessionId?: string
  /** The message the selection came from, when it resolved to one. */
  messageId?: string
  /** The selected text to carry in as the side chat's opening context. */
  text: string
}

/**
 * Ask the controller to open a side chat for a selection.
 *
 * The entry points (the transcript context menu, the selection toolbar) are
 * chrome — they hold no session hooks, and duplicating the create path there is
 * how two entrances drift apart. They post the request here and the wiring
 * consumes it, the same shape the new-project and fresh-session entrances use.
 */
export const $sideChatRequest = atom<null | SideChatRequest>(null)

export function requestSideChat(request: SideChatRequest): void {
  $sideChatRequest.set(request)
}

/**
 * The newest side chat that is still OPEN, for the panel toggle to act on when
 * the user has not focused a particular one. Recency comes from the origins map
 * (insertion order — the order side chats were opened), filtered by which of
 * those are open now: a side chat whose pane was closed keeps its origin on
 * purpose (the strip and staging come back if the session is reopened), so it
 * must not be a toggle target while it has no pane.
 */
export function mostRecentSideChatId(openStoredSessionIds: readonly string[]): null | string {
  const origins = $sideChatOrigins.get()
  const open = new Set(openStoredSessionIds)
  const originIds = Object.keys(origins)

  for (let index = originIds.length - 1; index >= 0; index -= 1) {
    if (open.has(originIds[index])) {
      return originIds[index]
    }
  }

  return null
}
