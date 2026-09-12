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

/** The active session's own slice, or `undefined` while it's a draft. */
const $primaryState = computed([$activeSessionId, $sessionStates], (runtimeId, states) =>
  runtimeId ? states[runtimeId] : undefined
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
 * The slice is authoritative only for the session the pane actually SHOWS.
 * `$primaryState` keys on `$activeSessionId` (the bound runtime) while the
 * composer and its queue key on the selected/routed STORED id, and a switch
 * moves those at different times: selection lands at once, the runtime rebinds
 * a beat later behind an awaited `session.activate`. In that window the slice
 * still describes the session the user just LEFT.
 *
 * That gap is the queued-prompt bug: queue a turn in A while A runs, switch to
 * idle B, switch back. Selection is A, the runtime is still B, so the pane
 * inherits B's `busy: false` — and `shouldAutoDrain` reads exactly this atom,
 * firing A's queued prompts into A's live turn. Treating "no slice yet" (cold
 * resume) as idle had the same effect.
 *
 * So trust the slice only when it belongs to the selected conversation, else
 * fall back to `$workingSessionIds`: the per-session projection the sidebar arc
 * paints from, published under every lineage alias, so it survives both the
 * slice gap and compression tip rotation. `$busy` stays DRAFT-only (no stored
 * id) for the first-send optimistic lock — it mirrors whichever session last
 * published and must never be inherited by a stored session.
 */
const $primaryBusy = computed(
  [$primaryState, $busy, $selectedStoredSessionId, $workingSessionIds, $sessions],
  (state, draftBusy, selected, working, sessions) => {
    if (!selected) {
      return state ? state.busy : draftBusy
    }

    return state?.storedSessionId && idsShareLineage(state.storedSessionId, selected, sessions)
      ? state.busy
      : working.includes(selected)
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
