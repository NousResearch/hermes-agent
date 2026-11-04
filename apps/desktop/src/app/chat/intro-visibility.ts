import { useStore } from '@nanostores/react'

import { isBlankSessionRow } from '@/lib/chat-runtime'
import { useStoresSelector } from '@/lib/use-session-slice'
import {
  $cronSessions,
  $messagingSessions,
  $sessions,
  $unlistedSessionOwnerRows,
  ownerLookupSessionRows,
  sessionMatchesStoredId
} from '@/store/session'

import { useSessionView } from './session-view'

/**
 * Whether the empty-chat intro splash renders.
 *
 * The splash is the full-height empty state of a NEW session: a fresh draft in
 * the primary chat, or a tab that is one (`tileDraft`, see `useBlankTileDraft`)
 * — the "+" / ⌘T tab and "New session" into a focused tab both mint a tile, and
 * without the splash it opened as an empty pane instead of a new session.
 * Auxiliary windows are scratch surfaces, a routed or active session already
 * owns the primary view, and any transcript at all means the conversation
 * started.
 *
 * `enabled` is the user's Appearance toggle and outranks every other clause:
 * turning the splash off never depends on which window asks.
 */
export function shouldShowIntro(input: {
  activeSessionId: null | string
  auxiliaryWindow: boolean
  enabled: boolean
  freshDraftReady: boolean
  messagesEmpty: boolean
  primary: boolean
  routedSessionView: boolean
  selectedSessionId: null | string
  tileDraft?: boolean
}): boolean {
  if (!input.enabled || input.auxiliaryWindow || !input.messagesEmpty) {
    return false
  }

  if (!input.primary) {
    return Boolean(input.tileDraft)
  }

  return input.freshDraftReady && !input.routedSessionView && !input.selectedSessionId && !input.activeSessionId
}

type IntroInput = Parameters<typeof shouldShowIntro>[0]

/** `shouldShowIntro` for the surface rendering it: the primary chat, or a tab
 *  (whose new-session state comes from its row, see `useBlankTileDraft`). */
export function useShowIntro(input: Omit<IntroInput, 'primary' | 'tileDraft'>): boolean {
  const view = useSessionView()
  const primary = view.kind === 'primary'
  const storedSessionId = useStore(view.$storedId)
  const tileDraft = useBlankTileDraft(primary, storedSessionId)

  return shouldShowIntro({ ...input, primary, tileDraft })
}

/** A tab whose session is still a new one: its row is blank (no title, no
 *  preview, no messages) — the owner stub a "+" draft gets at creation, or that
 *  draft probed back after a relaunch. An unresolved row is NOT a draft: a
 *  restored conversation outside the recents page has no row until its probe
 *  lands, and must not flash the splash over its loading transcript. */
export function useBlankTileDraft(primary: boolean, storedSessionId: null | string): boolean {
  return useStoresSelector(
    [$sessions, $cronSessions, $messagingSessions, $unlistedSessionOwnerRows],
    () =>
      !primary &&
      Boolean(storedSessionId) &&
      isBlankSessionRow(ownerLookupSessionRows().find(row => sessionMatchesStoredId(row, storedSessionId ?? '')))
  )
}
