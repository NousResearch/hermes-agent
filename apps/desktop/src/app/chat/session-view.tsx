import { computed, type ReadableAtom } from 'nanostores'
import { createContext, useContext } from 'react'

import type { ClientSessionState } from '@/app/types'
import type { ChatMessage } from '@/lib/chat-messages'
import {
  $activeSessionId,
  $awaitingResponse,
  $busy,
  $currentCwd,
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $messages,
  $selectedStoredSessionId,
  $sessions,
  $turnStartedAt,
  idsShareLineage
} from '@/store/session'
import { $sessionStates, $workingSessionIds } from '@/store/session-states'

import { lastVisibleMessageIsUser } from './thread-loading'

/**
 * SESSION VIEW — the store surface a ChatView renders from. Every session,
 * including the one in the workspace pane, renders from ITS OWN slice of
 * `$sessionStates`. The workspace pane is just the first tab: a session
 * surface with no privileged state of its own.
 *
 * That symmetry is load-bearing. The pane used to render off the global
 * `$messages`/`$busy` atoms — a mirror of whichever session was active — so
 * with two turns in flight (⌘T tabs made that routine), navigating away from
 * a still-streaming session left it painting into the surface now showing a
 * different conversation. Reading the per-session slice makes that
 * structurally impossible rather than merely guarded.
 *
 * The global atoms stay the DRAFT surface: a new chat has no runtime id, and
 * therefore no slice, until its first turn creates one.
 *
 * Everything is atoms (not values) so subscription granularity survives:
 * ChatView subscribes only to the coarse edges; `$messages` stays boundary-
 * only exactly like the primary view's perf contract.
 */
export interface SessionView {
  kind: 'primary' | 'tile'
  $runtimeId: ReadableAtom<string | null>
  $storedId: ReadableAtom<string | null>
  $messages: ReadableAtom<ChatMessage[]>
  $busy: ReadableAtom<boolean>
  $awaitingResponse: ReadableAtom<boolean>
  $messagesEmpty: ReadableAtom<boolean>
  $lastVisibleIsUser: ReadableAtom<boolean>
  /** Epoch ms this surface's current turn began, null when idle. Per-surface
   *  for the same reason $busy is: a tile's activity timer must count its own
   *  turn, not whichever session the global mirror last reflected. */
  $turnStartedAt: ReadableAtom<number | null>
  $cwd: ReadableAtom<string>
  $model: ReadableAtom<string>
  $provider: ReadableAtom<string>
  $fast: ReadableAtom<boolean>
  $reasoningEffort: ReadableAtom<string>
}

/** The active session's own slice, or `undefined` while it's a draft.
 *
 *  The runtime id keying this only rebinds once `resumeSession()` lands — and
 *  is nulled outright on the cold path — while the STORED id (selection/route)
 *  flips synchronously on navigate. So mid-switch the pane can hold session A's
 *  stored id next to session B's runtime slice. A slice that does not own the
 *  current selection describes a different conversation and must not answer for
 *  this one: it is how the outgoing session's cwd/model kept painting under the
 *  incoming session (and, for `busy`, how the composer's queue drained into a
 *  turn that was still running). An unpersisted conversation has no stored id
 *  yet, so its slice is the only account of itself and still counts. */
const $primaryState = computed(
  [$activeSessionId, $sessionStates, $selectedStoredSessionId, $sessions],
  (runtimeId, states, selected, sessions) => {
    const state = runtimeId ? states[runtimeId] : undefined

    if (!state || !selected) {
      return state
    }

    const sliceStoredId = state.storedSessionId

    return !sliceStoredId || idsShareLineage(sliceStoredId, selected, sessions) ? state : undefined
  }
)

/**
 * Read one field from the active session's slice, falling back to the global
 * draft atom while no runtime exists yet. Once a session HAS a slice, that
 * slice is authoritative — a background session publishing its own state can
 * never reach this view.
 */
function primaryField<T>(select: (state: ClientSessionState) => T, $draft: ReadableAtom<T>): ReadableAtom<T> {
  const $field: ReadableAtom<T> = computed([$primaryState, $draft], (state, draft: T) =>
    state ? select(state) : draft
  )

  return $field
}

const $primaryMessages = primaryField<ChatMessage[]>(state => state.messages, $messages)

/**
 * Turn-busy for the workspace pane.
 *
 * `$primaryState` already drops a slice that does not own the selection, so
 * mid-switch this is left with no slice at all — and answering "idle" there is
 * what fired the composer's level-triggered queue auto-drain into a session
 * still running its turn. An unknown defers to the authoritative working set
 * rather than guessing the permissive answer; that set already publishes every
 * lineage alias, so a compression tip matches its root. It is the same oracle
 * `use-background-queue-drain` consults offscreen.
 *
 * The global `$busy` atom stays reserved for a true new chat (no stored id) so
 * the first-send optimistic lock still paints. Inheriting it for a selected
 * session is how focusing B while A ran marked B busy.
 */
const $primaryBusy = computed(
  [$primaryState, $busy, $selectedStoredSessionId, $workingSessionIds],
  (state, draftBusy, selected, working) => {
    if (!selected) {
      return state ? state.busy : draftBusy
    }

    if (state) {
      return state.busy
    }

    return working.includes(selected)
  }
)

export const PRIMARY_SESSION_VIEW: SessionView = {
  kind: 'primary',
  $awaitingResponse: primaryField<boolean>(state => state.awaitingResponse, $awaitingResponse),
  $busy: $primaryBusy,
  $cwd: primaryField<string>(state => state.cwd, $currentCwd),
  $fast: primaryField<boolean>(state => state.fast, $currentFastMode),
  $lastVisibleIsUser: computed($primaryMessages, lastVisibleMessageIsUser),
  $messages: $primaryMessages,
  $messagesEmpty: computed($primaryMessages, messages => messages.length === 0),
  $model: primaryField<string>(state => state.model, $currentModel),
  $provider: primaryField<string>(state => state.provider, $currentProvider),
  $reasoningEffort: primaryField<string>(state => state.reasoningEffort, $currentReasoningEffort),
  $runtimeId: $activeSessionId,
  $storedId: $selectedStoredSessionId,
  $turnStartedAt: primaryField<number | null>(state => state.turnStartedAt, $turnStartedAt)
}

const SessionViewContext = createContext<SessionView>(PRIMARY_SESSION_VIEW)

export const SessionViewProvider = SessionViewContext.Provider

export const useSessionView = (): SessionView => useContext(SessionViewContext)
