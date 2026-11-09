import { useStore } from '@nanostores/react'

import { $introSplash } from '@/store/intro-splash'
import { $freshDraftReady, $sessions, $unlistedSessionOwnerRows, sessionMatchesStoredId } from '@/store/session'
import { $botChatSessionIds } from '@/store/session-states'
import { isAuxiliaryWindow } from '@/store/windows'
import type { SessionInfo } from '@/types/hermes'

import { shouldShowIntro } from './intro-visibility'

/**
 * Whether a session row positively reads as a never-sent DRAFT — the shape
 * that also paints the empty-chat splash (see `shouldShowIntro`).
 *
 * Positive evidence only: the row must EXIST and carry an explicit zero
 * message count. An unknown session (still resuming, row not loaded yet) is
 * never a draft, so reopening an older chat cannot flash the splash next to
 * its loading spinner. Hidden rows are bot plumbing (the canonical Bot Chat
 * and group rooms) — their empty state belongs to the owning plugin's
 * CHAT_EMPTY_AREA contribution, not the Hermes splash — and a bot's chat
 * keeps its face for the same reason even when its row is visible.
 */
export function isDraftSessionRow(row: SessionInfo | undefined, botScoped = false): boolean {
  if (!row || botScoped || row.hidden) {
    return false
  }

  return row.message_count === 0
}

/**
 * Whether this view's session is a DRAFT SESSION — the shape that also paints
 * the empty-chat splash (see `shouldShowIntro`): a REAL session that reports
 * zero messages everywhere. That covers the unlisted drafts minted by
 * openNewSessionTab (tab-strip "+", ⌘T, ⌘N — their off-list owner stubs carry
 * message_count 0) and a restored blank draft once its row resolves.
 *
 * A row KNOWN to have history (message_count > 0) disqualifies the view even
 * while its transcript is still loading, so a resumed tab never flashes the
 * splash mid-resume. Live in its own module because ChatViewContent sits on
 * the code-health ratchet (CC / FUNC_LINES may only go down).
 */
export function useDraftSessionMarker(input: { messagesEmpty: boolean; storedId: null | string }): boolean {
  const sessions = useStore($sessions)
  const unlistedOwnerRows = useStore($unlistedSessionOwnerRows)
  const botChatSessionIds = useStore($botChatSessionIds)
  const { messagesEmpty, storedId } = input

  if (!storedId || !messagesEmpty) {
    return false
  }

  const row = [...sessions, ...unlistedOwnerRows].find(candidate => sessionMatchesStoredId(candidate, storedId))

  return isDraftSessionRow(row, botChatSessionIds.has(storedId))
}

/**
 * The view-local inputs `shouldShowIntro` needs beyond global state. The
 * toggle (`$introSplash`) and the pure-draft flag (`$freshDraftReady`) are
 * read here so ChatViewContent (on the FUNC_LINES ratchet) stays put.
 */
export function useShowIntro(input: {
  activeSessionId: null | string
  isPrimary: boolean
  messagesEmpty: boolean
  routedSessionView: boolean
  selectedSessionId: null | string
  storedId: null | string
}): boolean {
  const introSplash = useStore($introSplash)
  const freshDraftReady = useStore($freshDraftReady)

  const draftSession = useDraftSessionMarker({
    messagesEmpty: input.messagesEmpty,
    storedId: input.storedId
  })

  return shouldShowIntro({
    activeSessionId: input.activeSessionId,
    auxiliaryWindow: isAuxiliaryWindow(),
    draftSession,
    enabled: introSplash,
    freshDraftReady,
    messagesEmpty: input.messagesEmpty,
    primary: input.isPrimary,
    routedSessionView: input.routedSessionView,
    selectedSessionId: input.selectedSessionId
  })
}

