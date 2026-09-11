import { atom } from 'nanostores'

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
export const $agentReactions = atom<Record<number, MessageReaction[]>>({})

/** Record an agent reaction painted from a live gateway event. */
export function recordAgentReaction(rowId: number, reactions: MessageReaction[]): void {
  $agentReactions.set({
    ...$agentReactions.get(),
    [rowId]: reactions.filter(reaction => reaction.author === 'agent')
  })
}

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
