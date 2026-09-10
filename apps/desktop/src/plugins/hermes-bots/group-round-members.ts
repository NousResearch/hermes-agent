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
import type { GroupMember, GroupMessage } from './types'

export interface GroupRoundMemberContext {
  group: string
  members: GroupMember[]
  thread: string
  startEpoch: number
  binding: { isLive(): boolean }
  isCurrent(): boolean
}

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
    deltaLines: delta.slice(-GROUP_CHAT_HISTORY_LIMIT).map((e: GroupMessage) => formatGroupChatLine(e, member.name))
  })

  // Images riding this delta (user attachments — member entries don't
  // carry images today, but flatMap keeps this future-proof) get staged
  // into the member's session so the model sees the pixels, not just
  // the transcript's [attached image: …] marker.
  const deltaImages = delta.flatMap((e: GroupMessage) => (Array.isArray(e.images) ? e.images : []))

  return { room, memberKey, markKey, prompt, deltaImages }
}

export async function runGroupRoundMember(
  context: GroupRoundMemberContext,
  member: GroupMember
): Promise<boolean | null> {
  const { thread, startEpoch, binding } = context
  const prepared = prepareGroupRoundMember(context, member)

  if (!prepared) {
    return false
  }

  const { room, markKey, prompt, deltaImages } = prepared
  // Surface WHO is on turn (runtime-only, like running/epoch) so the
  // room shows "Radar is thinking…" instead of a generic working line —
  // long model turns otherwise read as the room being stuck.
  updateGroupChat(context.group, (r: GroupChatRoom) => {
    r.turn = member.name

    return r
  })
  let reply: null | string = null

  try {
    reply = await runGroupChatMemberTurn(context.group, member, prompt, thread, deltaImages)

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
    deltaLines: delta.slice(-GROUP_CHAT_HISTORY_LIMIT).map((e: GroupMessage) => formatGroupChatLine(e, member.name))
  })

  updateGroupChat(context.group, (r: GroupChatRoom) => {
    r.turn = member.name

    return r
  })
  let continuationReply: null | string = null

  try {
    continuationReply = await runGroupChatMemberTurn(context.group, member, prompt, thread)

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
