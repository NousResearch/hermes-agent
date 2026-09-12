import { clearBotAttention, noteBotAttention } from './data'
import { recordGroupActivity } from './group-activity'
import {
  $groupChats,
  appendGroupChatEntry,
  GROUP_CHAT_HISTORY_LIMIT,
  GROUP_CHAT_MAX_CONTINUATIONS,
  GROUP_CHAT_MAX_MESSAGES,
  groupThreadOf,
  shouldCommitMemberTurn,
  updateGroupChat
} from './group-chat'
import type { GroupChatRoom } from './group-chat'
import { groupMemberKey } from './group-membership'
import { buildGroupChatTurnPrompt, formatGroupChatLine } from './group-round-prompt'
import { isGroupPassText, runGroupChatMemberTurn } from './group-turns'
import type { Attachment, GroupMember, GroupMessage } from './types'

export interface GroupRoundMemberContext {
  group: string
  members: GroupMember[]
  thread: string
  startEpoch: number
  binding: { isLive(): boolean }
  isCurrent(): boolean
  // Per-drive count of transient (thrown) turn failures per member key. A
  // thrown turn (e.g. a ws_orphan_reap 4001 the turn-level retry couldn't
  // recover) must NOT silently consume the user's delta and go stale: leave
  // the watermark unadvanced so the next round re-drives the member, up to
  // MAX_FAILED_TURN_RETRIES, after which the room settles against a
  // genuinely-down member.
  failedRetries?: Map<string, number>
}

/** How many extra times a member whose turn THREW is re-driven before the
 *  round loop gives up and lets the room settle. Timeouts (reply===null
 *  without a throw) keep their existing stranded-harvest path untouched. */
export const MAX_FAILED_TURN_RETRIES = 1

/** #93129: a held member's skip must consume its delta exactly once —
 *  advance the watermark past the current log so the same entries never
 *  re-trigger the skip. Null = nothing to consume (no write, no spin). */
export function heldMemberWatermarkAdvance(seen: number | undefined, logLength: number): null | number {
  return logLength > (seen || 0) ? logLength : null
}

function prepareGroupRoundMember(context: GroupRoundMemberContext, member: GroupMember) {
  const { members, thread } = context

  const room = $groupChats.get()[context.group] || {
    log: [],
    watermarks: {}
  }

  const memberKey = groupMemberKey(member)
  const markKey = `${thread}::${memberKey}`
  const seen = room.watermarks[markKey] || 0
  // Delta: NEW room entries, narrowed to this thread — the member's
  // turn sees only the conversation it's part of.
  const delta = room.log.slice(seen).filter((e: GroupMessage) => groupThreadOf(e) === thread)

  if (!delta.length) {
    return null
  }

  // #93129: a member the user told to stop is HELD — no turn until an
  // explicit release (resume / @all resume / a direct non-stop
  // mention). Consume the delta exactly once (watermark past the
  // current log) so the same entries never re-trigger this skip, and
  // surface WHY the bot is silent in the activity feed the first time.
  const heldEntry = (room.holds || {})[memberKey]

  if (heldEntry) {
    const advance = heldMemberWatermarkAdvance(seen, room.log.length)
    updateGroupChat(context.group, (r: GroupChatRoom) => {
      if (advance !== null) {
        r.watermarks[markKey] = advance
      }

      if (r.holds?.[memberKey] && !r.holds[memberKey].noted) {
        r.holds = {
          ...r.holds,
          [memberKey]: {
            ...r.holds[memberKey],
            noted: true
          }
        }
      }

      return r
    })

    if (!heldEntry.noted) {
      recordGroupActivity(context.group, {
        kind: 'held',
        member: member.name,
        thread
      })
    }

    return null
  }

  const prompt = buildGroupChatTurnPrompt({
    groupName: context.group,
    members,
    viewer: member,
    deltaLines: delta.slice(-GROUP_CHAT_HISTORY_LIMIT).map((e: GroupMessage) => formatGroupChatLine(e, member))
  })

  // Images riding this delta (user attachments — member entries don't
  // carry images today, but flatMap keeps this future-proof) get staged
  // into the member's session so the model sees the pixels, not just
  // the transcript's [attached image: …] marker.
  const deltaImages = delta.flatMap((e: GroupMessage) => (Array.isArray(e.images) ? e.images : []))

  return { room, memberKey, markKey, prompt, deltaImages }
}

/** Each invocation owns its descriptor, so an old completion cannot clear a newer turn. */
async function runVisibleMemberTurn(
  context: GroupRoundMemberContext,
  member: GroupMember,
  prompt: string,
  images?: Attachment[]
) {
  const turn = { ...member }
  updateGroupChat(context.group, (room: GroupChatRoom) => ({ ...room, turn }), { sync: false })

  try {
    return await runGroupChatMemberTurn(context.group, member, prompt, context.thread, images)
  } finally {
    if (context.binding.isLive() && $groupChats.get()[context.group]?.turn === turn) {
      updateGroupChat(context.group, (room: GroupChatRoom) => ({ ...room, turn: null }), { sync: false })
    }
  }
}

export async function runGroupRoundMember(
  context: GroupRoundMemberContext,
  member: GroupMember
): Promise<boolean | null | 'retry'> {
  const { thread, startEpoch, binding } = context
  const prepared = prepareGroupRoundMember(context, member)

  if (!prepared) {
    return false
  }

  const { room, markKey, prompt, deltaImages } = prepared
  let reply: null | string = null
  let turnFailed = false

  try {
    reply = await runVisibleMemberTurn(context, member, prompt, deltaImages)

    // Needs-attention hook (#93091 item 3): a turn that produced a real
    // reply (or an explicit pass) is a good turn — clear the badge.
    // A timed-out turn also returns null but never threw; leaving any
    // prior badge in place there is the conservative choice.
    if (reply !== null) {
      clearBotAttention(groupMemberKey(member))
    }
  } catch (error: any) {
    if (!binding.isLive()) {
      return null
    }

    const reason = String(error?.data?.reason || '').trim()
    recordGroupActivity(context.group, {
      kind: 'failed',
      member: member.name,
      thread,
      ...(reason
        ? {
            reason
          }
        : {})
    })
    noteBotAttention(groupMemberKey(member), reason || error?.message || error)
    reply = null // a failed turn is a pass, never a room error
    turnFailed = true
  }

  // #93127: the turn may have finished AFTER a newer user send bumped
  // the room epoch. That newer send's loop re-drives this member with
  // the full delta, so committing this stale result (watermark advance
  // + append) would double-deliver the same reply. Drop it here —
  // BEFORE the watermark advance and BEFORE the append. Only a newer
  // USER entry in THIS thread makes the re-drive premise true: a
  // cross-thread send bumps the epoch too, but its loop filters this
  // thread out and would never regenerate the finished reply. The
  // during-turn tail is anchored by entry id, not index — the history
  // trim drops entries from the FRONT, so an index slice could
  // overshoot after a mid-turn trim and silently commit a stale turn.
  if (!binding.isLive()) {
    return null
  }

  const roomNow = $groupChats.get()[context.group] || {
    log: []
  }

  const epochNow = roomNow.epoch || 0
  const anchorId = room.log.length ? room.log[room.log.length - 1].id : null
  const anchorIdx = anchorId === null ? -1 : roomNow.log.findIndex((e: GroupMessage) => e.id === anchorId)
  // Anchor trimmed away ⇒ every pre-turn entry was dropped, so every
  // surviving entry is newer — scanning the whole log stays exact.
  const turnTail = anchorIdx >= 0 ? roomNow.log.slice(anchorIdx + 1) : roomNow.log

  const newerUserEntryInThread = turnTail.some(
    (e: GroupMessage) => e.from?.kind === 'user' && groupThreadOf(e) === thread
  )

  if (!shouldCommitMemberTurn(startEpoch, epochNow, newerUserEntryInThread)) {
    recordGroupActivity(context.group, {
      kind: 'cancelled',
      member: member.name,
      thread
    })

    return null
  }

  // A THROWN turn (transient reap) that still has retries left must NOT
  // advance the watermark: leaving the user's delta unseen lets the NEXT
  // round re-drive this member instead of silently consuming the mention.
  // Only real throws are retried (timeouts keep the stranded-harvest path),
  // and only up to MAX_FAILED_TURN_RETRIES so the room still settles against
  // a genuinely-down member. Return true so spokeThisRound stays > 0 and the
  // round loop runs another round (the room would otherwise settle after a
  // silent round before the re-drive) — no entry is appended, so this only
  // keeps the loop alive; the actual reply lands on the retry round.
  if (turnFailed && context.failedRetries) {
    const memberKey = groupMemberKey(member)
    const priorRetries = context.failedRetries.get(memberKey) || 0

    if (priorRetries < MAX_FAILED_TURN_RETRIES) {
      context.failedRetries.set(memberKey, priorRetries + 1)

      return 'retry'
    }
  }

  // The member has now seen everything up to the pre-reply log length.
  updateGroupChat(context.group, (r: GroupChatRoom) => {
    r.watermarks[markKey] = r.log.length

    return r
  })

  if (reply !== null && !isGroupPassText(reply)) {
    appendGroupChatEntry(
      context.group,
      {
        kind: 'member',
        name: member.name,
        ...(member.remoteSource
          ? {
              source: member.connectionLabel || member.connectionId
            }
          : {})
      },
      reply,
      thread
    )
    // Its own message counts as seen too.
    updateGroupChat(context.group, (r: GroupChatRoom) => {
      r.watermarks[markKey] = r.log.length

      return r
    })

    return true
  }

  return false
}

async function runGroupContinuationMember(
  context: GroupRoundMemberContext,
  member: GroupMember
): Promise<boolean | null> {
  const { members, thread, binding, isCurrent } = context

  const room = $groupChats.get()[context.group] || {
    log: [],
    watermarks: {}
  }

  const memberKey = groupMemberKey(member)
  const markKey = `${thread}::${memberKey}`
  const seen = room.watermarks[markKey] || 0
  const delta = room.log.slice(seen).filter((e: GroupMessage) => groupThreadOf(e) === thread)

  // A cited member always has delta here (the citing reply IS in
  // its tail); skip defensively anyway so an empty prompt never
  // fires.
  if (!delta.length) {
    return false
  }

  const heldEntry = (room.holds || {})[memberKey]

  if (heldEntry) {
    return false // holds still apply to continuation turns (#93129)
  }

  const prompt = buildGroupChatTurnPrompt({
    groupName: context.group,
    members,
    viewer: member,
    // The continuation prompt centers on what the member missed:
    // everything since its watermark, which includes the reply
    // that cites it.
    deltaLines: delta.slice(-GROUP_CHAT_HISTORY_LIMIT).map((e: GroupMessage) => formatGroupChatLine(e, member))
  })

  let continuationReply: null | string = null

  try {
    continuationReply = await runVisibleMemberTurn(context, member, prompt)

    if (continuationReply !== null) {
      clearBotAttention(memberKey)
    }
  } catch (error: any) {
    if (!binding.isLive()) {
      return null
    }

    recordGroupActivity(context.group, {
      kind: 'failed',
      member: member.name,
      thread
    })
    noteBotAttention(memberKey, error?.message || error)
    continuationReply = null
  }

  if (!isCurrent()) {
    return null
  }

  updateGroupChat(context.group, (r: GroupChatRoom) => {
    r.watermarks[markKey] = r.log.length

    return r
  })

  if (continuationReply !== null && !isGroupPassText(continuationReply)) {
    appendGroupChatEntry(
      context.group,
      {
        kind: 'member',
        name: member.name,
        ...(member.remoteSource
          ? {
              source: member.connectionLabel || member.connectionId
            }
          : {})
      },
      continuationReply,
      thread
    )
    updateGroupChat(context.group, (r: GroupChatRoom) => {
      r.watermarks[markKey] = r.log.length

      return r
    })

    return true
  }

  return false
}

export async function runGroupContinuationMembers(
  context: GroupRoundMemberContext,
  pendingKeys: string[],
  continuations: number,
  posted: number
): Promise<number | null> {
  const { members, isCurrent } = context
  let spokeThisRound = 0

  if (pendingKeys.length && continuations <= GROUP_CHAT_MAX_CONTINUATIONS) {
    const citedMembers = members.filter((member: GroupMember) => pendingKeys.includes(groupMemberKey(member)))

    if (citedMembers.length && posted < GROUP_CHAT_MAX_MESSAGES) {
      const strandedNow = ($groupChats.get()[context.group] || {}).stranded || {}

      const continuationResponders = citedMembers.filter(
        (member: GroupMember) => !Object.prototype.hasOwnProperty.call(strandedNow, groupMemberKey(member))
      )

      for (const member of continuationResponders) {
        if (!isCurrent() || posted >= GROUP_CHAT_MAX_MESSAGES || continuations > GROUP_CHAT_MAX_CONTINUATIONS) {
          break
        }

        const result = await runGroupContinuationMember(context, member)

        if (result === null) {
          return null
        }

        if (result) {
          posted += 1
          spokeThisRound += 1
        }
      }
    }
  }

  return spokeThisRound
}
