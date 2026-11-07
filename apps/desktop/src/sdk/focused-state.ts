// Focused-session state atoms projected onto `host.state`: turn flags, model /
// provider / reasoning effort, connection-qualified owner, profile and usage.

import { computed, type ReadableAtom } from 'nanostores'

import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import type { ClientSessionState } from '@/app/types'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import {
  $connection,
  $selectedStoredSessionId,
  $sessions,
  getSessionOwnerHints,
  rememberedSessionProfile,
  sessionMatchesStoredId
} from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $focusedSessionState } from '@/store/session-states'

/**
 * One field of the FOCUSED chat, read off the slice the UI already renders
 * from. While the focused surface is the primary workspace (or a draft with no
 * runtime slice yet) this reads the primary view, which itself falls back to
 * the global draft atoms. Once a session TILE holds focus, the tile's own state
 * slice is authoritative — a background session can never leak in. `absent`
 * is what a focused tile without a slice yet reports.
 */
const focusedField = <T>(
  select: (state: ClientSessionState) => T,
  $primary: ReadableAtom<T>,
  absent: T
): ReadableAtom<T> =>
  computed(
    [$focusedStoredSessionId, $selectedStoredSessionId, $focusedSessionState, $primary],
    (focused, selected, state, primary) => (!focused || focused === selected ? primary : state ? select(state) : absent)
  )

/** Turn flag for the FOCUSED chat — same semantics as the statusbar's busy pulse. */
const focusedTurnFlag = (
  select: (state: ClientSessionState) => boolean,
  $primary: ReadableAtom<boolean>
): ReadableAtom<boolean> => focusedField(state => Boolean(select(state)), $primary, false)

export const $focusedBusy = focusedTurnFlag(state => state.busy, PRIMARY_SESSION_VIEW.$busy)

export const $focusedAwaitingResponse = focusedTurnFlag(
  state => state.awaitingResponse,
  PRIMARY_SESSION_VIEW.$awaitingResponse
)

// What the focused chat is running. `state.model` is the composer DRAFT atom:
// global rather than per session, never written by a tile pick, and stale for
// a resumed session whose slice carries its own model. There was no provider
// at all, so a plugin had to rebuild the pair from localStorage keys plus
// `session.info` and got it wrong on every seam (#85776). These read the same
// resolved value the model pill paints.
export const $focusedModel = focusedField<string>(state => state.model, PRIMARY_SESSION_VIEW.$model, '')

export const $focusedProvider = focusedField<string>(state => state.provider, PRIMARY_SESSION_VIEW.$provider, '')

export const $focusedReasoningEffort = focusedField<string>(
  state => state.reasoningEffort,
  PRIMARY_SESSION_VIEW.$reasoningEffort,
  ''
)

export interface PluginFocusedSessionOwner {
  connectionId: string
  profile: string
}

/**
 * Connection-qualified owner of the FOCUSED chat. The gateway-routing atom
 * (`$activeGatewayProfile`) answers "which backend is the live socket homed
 * on" — but tab/tile focus moves without swapping the socket, and a cold
 * start can restore a route into a session the booting gateway doesn't own.
 * Any per-bot readout must follow the chat the user is LOOKING AT, so this
 * resolves the focused stored session to a unique immutable owner hint or a
 * unique connection-qualified aggregated row. Ambiguous or unresolved focused
 * ids fail closed with null; only a draft/no focused id uses the active gateway
 * owner. `focusedSessionProfile` remains the profile-only compatibility ladder.
 */
export const $focusedSessionOwner = computed(
  [$focusedStoredSessionId, $sessions, $activeGatewayProfile, $connection],
  (focused, sessions, activeProfile, connection): PluginFocusedSessionOwner | null => {
    const activeConnectionId = String(connection?.connectionId || (connection?.mode === 'local' ? 'local' : '')).trim()

    const fallback = {
      connectionId: activeConnectionId,
      profile: normalizeProfileKey(activeProfile)
    }

    if (!focused) {
      return fallback
    }

    const hints = getSessionOwnerHints(focused)

    if (hints.length === 1) {
      return {
        connectionId: hints[0].connectionId,
        profile: normalizeProfileKey(hints[0].profile)
      }
    }

    if (hints.length > 1) {
      return null
    }

    const owners = new Map<string, PluginFocusedSessionOwner>()

    for (const row of sessions.filter(session => sessionMatchesStoredId(session, focused))) {
      const connectionId = String(row.connection_id || '').trim()
      const profile = normalizeProfileKey(row.profile)

      if (connectionId) {
        owners.set(`${connectionId}::${profile}`, { connectionId, profile })
      }
    }

    return owners.size === 1 ? [...owners.values()][0] : null
  }
)

export const $focusedSessionProfile = computed(
  [$focusedSessionOwner, $focusedStoredSessionId, $sessions, $activeGatewayProfile],
  (owner, focused, sessions, activeProfile) =>
    owner?.profile || rememberedSessionProfile(sessions, focused, activeProfile)
)

/** Live usage of the FOCUSED session, projected out of the streamed session
 *  state — the same readout the core statusbar's context chip paints. */
export const $focusedUsage = computed($focusedSessionState, state => state?.usage ?? null)
