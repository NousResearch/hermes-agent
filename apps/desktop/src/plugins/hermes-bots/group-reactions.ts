/**
 * Emoji reactions on room messages.
 *
 * A group chat between people carries reactions — a tapback says "seen it",
 * "agreed", "well done" without spending a message on it. Rooms had none, so
 * the only way to acknowledge a member's post was to talk over it.
 *
 * One reaction per person per message (the tapback rule the 1:1 thread already
 * follows): reacting again with the same emoji takes it back, a different emoji
 * replaces it. Reactions ride the message itself, so they travel with the room
 * log to every member exactly like the text does.
 */

import type { GroupReaction } from './types'

/** The six the strip offers first — the same set the 1:1 thread's picker
 *  leads with, so a reaction means the same gesture on both surfaces. */
export const GROUP_QUICK_REACTIONS = ['❤️', '👍', '👎', '😂', '‼️', '❓'] as const

/** Reactions a single message may carry. The room log is synced through
 *  profile ui_meta with a byte budget, so this is a hard cap, not a nicety. */
export const GROUP_REACTION_LIMIT = 12

/** The human's key in a message's reaction list. Members use their name. */
export const GROUP_REACTION_USER = 'user'

/** One chip in the strip under a message. */
export interface GroupReactionChip {
  /** How many people picked this emoji. */
  count: number
  emoji: string
  /** The human is among them, so the chip can be clicked to take it back. */
  mine: boolean
}

/** Add, replace or retract `by`'s reaction. */
export function toggleGroupReaction(
  reactions: readonly GroupReaction[] | undefined,
  by: string,
  emoji: string,
  at = Math.floor(Date.now() / 1000)
): GroupReaction[] {
  const list = [...(reactions ?? [])]
  const mine = list.find(reaction => reaction.by === by)

  // Same emoji again = take it back (the gesture that added it removes it).
  if (mine?.emoji === emoji) {
    return list.filter(reaction => reaction.by !== by)
  }

  const next = mine
    ? list.map(reaction => (reaction.by === by ? { at, by, emoji } : reaction))
    : [...list, { at, by, emoji }]

  // Trim oldest first: the newest reactions are the ones the room is looking at.
  return next.length > GROUP_REACTION_LIMIT ? next.slice(next.length - GROUP_REACTION_LIMIT) : next
}

/** Collapse a message's reactions into one chip per emoji, first-seen order. */
export function groupReactionChips(
  reactions: readonly GroupReaction[] | undefined,
  viewer = GROUP_REACTION_USER
): GroupReactionChip[] {
  const chips: GroupReactionChip[] = []

  for (const reaction of reactions ?? []) {
    const chip = chips.find(candidate => candidate.emoji === reaction.emoji)

    if (chip) {
      chip.count += 1
      chip.mine ||= reaction.by === viewer
    } else {
      chips.push({ count: 1, emoji: reaction.emoji, mine: reaction.by === viewer })
    }
  }

  return chips
}
