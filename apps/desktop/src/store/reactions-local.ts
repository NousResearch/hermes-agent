import { registryBackendScopeKey } from '@hermes/shared'
import { atom } from 'nanostores'

import { $activeGatewayProfile } from '@/store/profile'
import { applyReaction } from '@/store/reactions'
import type { MessageReaction } from '@/types/hermes'

/**
 * Reactions the user has set in THIS window, keyed by renderer message id.
 *
 * The UI owns this outright. A tapback is a direct manipulation — it flips the
 * instant you click it, with no round-trip, no gateway, and no dependency on a
 * message having been persisted yet. Durable state (and the agent's own
 * reactions) still arrive through `metadata.custom.reactions`; this layer sits
 * on top of it so the interaction never waits on the backend to feel alive.
 */
export const $localReactions = atom<Record<string, MessageReaction[]>>({})

/**
 * Agent reactions announced live (`message.reaction` events), keyed by the
 * DURABLE row id — never the renderer message id, which the end-of-turn
 * resume regenerates (an overlay keyed on the old id would orphan the instant
 * the transcript rebuilds; "identity is not incidental", AGENTS.md). The
 * resume also rebuilds from the gateway's in-memory history, which doesn't
 * carry a reaction written to the DB mid-turn — this overlay outlives that
 * clobber and a real reload hydrates the same reaction from disk.
 */
export interface AgentReactionOverlay {
  /** Composite source identity (connection + profile) the recording event
   *  came from — see the overlay-scope note below. */
  scope: string
  reactions: MessageReaction[]
}

/**
 * Row ids are per-DATABASE, so a bare row id only names a message within ONE
 * source's state.db. Each entry therefore carries the full source identity
 * (`registryBackendScopeKey(connectionId, profile)`) of the event that
 * recorded it, and a reader only sees an entry whose scope matches the
 * session it is displaying: an overlay recorded while source A was on screen
 * can never repaint source B's row 42, because B's read keys B's own scope
 * and falls back to B's persisted reactions. Clearing on profile/connection
 * swaps remains as memory hygiene; correctness comes from the scope key.
 */
export const $agentReactions = atom<Record<number, AgentReactionOverlay>>({})

/** The live overlay for `rowId` when it was recorded by `scope`'s source. */
export function agentLiveReactions(
  overlays: Record<number, AgentReactionOverlay>,
  rowId: number,
  scope: string
): MessageReaction[] | undefined {
  const overlay = overlays[rowId]

  return overlay && overlay.scope === scope ? overlay.reactions : undefined
}

/** Record an agent reaction painted from a live gateway event. `scope` is the
 *  event's composite source identity (connection + profile). */
export function recordAgentReaction(rowId: number, reactions: MessageReaction[], scope: string): void {
  $agentReactions.set({
    ...$agentReactions.get(),
    [rowId]: { reactions: reactions.filter(reaction => reaction.author === 'agent'), scope }
  })
}

/** The composite source identity a gateway event carries. Pool secondaries
 *  and the legacy primary have no registry connection, so their scope is the
 *  bare profile name — exactly how `session-states.ts` records event scopes. */
export function reactionOverlayScope(event: { connectionId?: string; profile?: string }): string {
  return registryBackendScopeKey(event.connectionId ?? null, event.profile ?? null)
}

/**
 * Drop both overlays, called when the row-id space they describe changes.
 *
 * $agentReactions is keyed by bare DB row id and row ids are per-database, so
 * after a profile swap or a gateway switch every entry could stamp onto a
 * DIFFERENT message that happens to share the row id in the new backend's
 * state.db. $localReactions is keyed by renderer message id, which is
 * regenerated when the transcript reloads; both maps describe messages that
 * no longer exist, so they go together.
 *
 * Correctness no longer depends on this wipe — entries are scope-keyed and a
 * different source's read never claims them — but the wipe keeps the maps from
 * describing a backend this window may never show again.
 *
 * Two boundaries: the $activeGatewayProfile subscribe below covers the
 * profile swap (same backend, different home/DB); wipeSessionListsForGatewaySwitch
 * calls this for the connection switch, where the profile name can stay the
 * same while the backend (and its row ids) changes. A same-backend reconnect
 * keeps the entries: the DB is unchanged and the overlay is still true.
 */
export function clearLiveReactionOverlays(): void {
  $agentReactions.set({})
  $localReactions.set({})
}

// Guard on a real change: a same-value set (reconnect re-asserts the profile)
// notifies listeners too, and wiping live overlays mid-turn would drop the
// reactions it exists to survive.
let wipedProfileScope = $activeGatewayProfile.get()

$activeGatewayProfile.subscribe(value => {
  if (value !== wipedProfileScope) {
    wipedProfileScope = value
    clearLiveReactionOverlays()
  }
})

/**
 * Merge the durable reaction list with anything this window knows live.
 *
 * The user's slot: local wins (they just clicked it — newer by definition).
 * The agent's slot: the live-event overlay wins over persisted (a mid-turn
 * reaction reaches the DB before the in-memory history the next resume
 * projects from), falling back to what the transcript carried.
 */
export function mergeReactions(
  persisted: MessageReaction[] | undefined,
  local: MessageReaction[] | undefined,
  agentLive?: MessageReaction[]
): MessageReaction[] {
  const persistedList = persisted ?? []

  const userSide = local
    ? local.filter(reaction => reaction.author === 'user')
    : persistedList.filter(reaction => reaction.author === 'user')

  const agentSide = agentLive ?? persistedList.filter(reaction => reaction.author === 'agent')

  return [...userSide, ...agentSide]
}

/** Toggle the user's reaction on a message — instant, local, no round-trip. */
export function setLocalReaction(messageId: string, emoji: null | string): MessageReaction[] {
  const next = applyReaction($localReactions.get()[messageId], emoji, 'user')

  $localReactions.set({ ...$localReactions.get(), [messageId]: next })

  return next
}
