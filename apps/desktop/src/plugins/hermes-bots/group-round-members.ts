import { clearBotAttention, noteBotAttention } from './data'
import { groupFailureReason, recordGroupActivity } from './group-activity'
import {
  $groupChats,
  appendGroupChatEntry,
  GROUP_CHAT_HISTORY_LIMIT,
  GROUP_CHAT_MAX_CONTINUATIONS,
  GROUP_CHAT_MAX_MESSAGES,
  groupChatRoomKey,
  groupThreadOf,
  shouldCommitMemberTurn,
  updateGroupChat
} from './group-chat'
import type { GroupChatRoom } from './group-chat'
import { followGroupChat, groupMemberAuthor, groupMemberKey } from './group-membership'
import {
  buildGroupChatTurnPrompt,
  formatGroupDeltaLines,
  GROUP_ADDRESSED_NUDGE_SUFFIX,
  isGroupChatSelf
} from './group-round-prompt'
import { isGroupPassText, runGroupChatMemberTurn } from './group-turns'
import type { Attachment, GroupMember, GroupMessage } from './types'

export interface GroupRoundMemberContext {
  group: string
  members: GroupMember[]
  thread: string
  startEpoch: number
  binding: { isLive(): boolean }
  isCurrent(): boolean
  failedMembers?: Set<string>
  /** #129443: member keys the driving user send explicitly addressed
   *  (@everyone / @mention). An addressed member's "(pass)" gets one bounded
   *  nudge; a second "(pass)" is recorded as explicit noncompliance — never
   *  ordinary silence. Absent/empty = collaborative turn, where a plain
   *  "(pass)" stays legitimate. */
  addressedKeys?: null | Set<string>
  /** A priority-mention lane turn: runs beside the foreground drive, so it
   *  must not claim the room's single `turn` slot — that slot belongs to the
   *  busy member the lane exists to route AROUND, and clobbering it would
   *  break Stop's interrupt targeting and the presence roster. */
  background?: boolean
}

/** #93129: a held member's skip must consume its delta exactly once —
 *  advance the watermark past the current log so the same entries never
 *  re-trigger the skip. Null = nothing to consume (no write, no spin). */
export function heldMemberWatermarkAdvance(seen: number | undefined, logLength: number): null | number {
  return logLength > (seen || 0) ? logLength : null
}

/** Remember held entries after their watermarks advance. Keep the first entry
 *  (the hold-triggering instruction) plus the newest bounded tail, and dedupe
 *  by id so repeated held passes remain idempotent. */
export function heldMemberMessageIds(existing: string[] | undefined, delta: GroupMessage[]): string[] {
  const ids = [...(existing || []), ...delta.map(entry => entry.id).filter((id): id is string => Boolean(id))]
  const unique = [...new Set(ids)]

  if (unique.length <= GROUP_CHAT_HISTORY_LIMIT) {
    return unique
  }

  return [unique[0], ...unique.slice(-(GROUP_CHAT_HISTORY_LIMIT - 1))]
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

      r.heldMessages = {
        ...(r.heldMessages || {}),
        [memberKey]: heldMemberMessageIds(r.heldMessages?.[memberKey], delta)
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
        member: groupMemberKey(member),
        thread
      })
    }

    return null
  }

  const heldIds = room.heldMessages?.[memberKey] || []
  const heldSet = new Set(heldIds)
  const delivered = room.log.filter((entry: GroupMessage) => Boolean(entry.id && heldSet.has(entry.id)))
  const deliveredIds = new Set(delivered.map((entry: GroupMessage) => entry.id).filter(Boolean))

  const visibleDelta = [
    ...delivered,
    ...delta.filter((entry: GroupMessage) => !entry.id || !deliveredIds.has(entry.id))
  ].sort((left, right) => Number(left.at || 0) - Number(right.at || 0))

  const prompt = buildGroupChatTurnPrompt({
    groupName: context.group,
    members,
    viewer: member,
    deltaLines: formatGroupDeltaLines(visibleDelta, member, context.group)
  })

  // Images riding this delta (user attachments — member entries don't
  // carry images today, but flatMap keeps this future-proof) get staged
  // into the member's session so the model sees the pixels, not just
  // the transcript's [attached image: …] marker.
  const deltaImages = visibleDelta.flatMap((e: GroupMessage) => (Array.isArray(e.images) ? e.images : []))

  return { room, memberKey, markKey, prompt, deltaImages, heldIds }
}

/** Each invocation owns its descriptor, so an old completion cannot clear a newer turn. */
async function runVisibleMemberTurn(
  context: GroupRoundMemberContext,
  member: GroupMember,
  prompt: string,
  images?: Attachment[]
) {
  const turn = { ...member }

  // A background lane turn never claims the room's single `turn` slot: the
  // foreground drive owns it, and Stop keys its interrupt on `room.turn`.
  if (!context.background) {
    updateGroupChat(context.group, (room: GroupChatRoom) => ({ ...room, turn }), { sync: false })
  }

  try {
    return await runGroupChatMemberTurn(context.group, member, prompt, context.thread, images)
  } finally {
    if (!context.background && context.binding.isLive() && $groupChats.get()[context.group]?.turn === turn) {
      updateGroupChat(context.group, (room: GroupChatRoom) => ({ ...room, turn: null }), { sync: false })
    }
  }
}

export async function runGroupRoundMember(
  context: GroupRoundMemberContext,
  member: GroupMember
): Promise<boolean | null> {
  const { thread, startEpoch, binding } = context

  if (
    context.failedMembers?.has(groupMemberKey(member)) ||
    (!context.background && groupPriorityLaneClaimed(context.group, thread, member))
  ) {
    // Recheck at dispatch, not only when the round snapshots its responders:
    // a lane can acquire this member while an earlier member is awaited.
    return false
  }

  const prepared = prepareGroupRoundMember(context, member)

  if (!prepared) {
    return false
  }

  const { room, memberKey, markKey, prompt, deltaImages, heldIds } = prepared
  const anchorId = room.log.at(-1)?.id ?? null
  let reply: null | string = null
  let accepted = false

  try {
    reply = await runVisibleMemberTurn(context, member, prompt, deltaImages)

    // #129443: the participation rules ("reply only with something new,
    // else (pass)") let a directly addressed member pass just because a
    // teammate already answered — the room then settles as ordinary
    // consensus and the user's explicit address silently disappears. One
    // bounded nudge re-asks with the directive spelled out; the recovered
    // reply replaces the pass and flows through the exact same commit path
    // below. A nudge that throws falls into the catch like any failed turn.
    // `isCurrent()` (not the bare binding): a superseded drive's pass is
    // discarded by the epoch check right below anyway, so nudging it would
    // burn a model call whose result can never be committed.
    if (context.isCurrent() && reply !== null && isGroupPassText(reply) && context.addressedKeys?.has(memberKey)) {
      reply = await runVisibleMemberTurn(context, member, `${prompt}${GROUP_ADDRESSED_NUDGE_SUFFIX}`, deltaImages)
    }

    accepted = true

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

    const reason = groupFailureReason(error)
    recordGroupActivity(context.group, {
      kind: 'failed',
      member: groupMemberKey(member),
      thread,
      ...(reason
        ? {
            reason
          }
        : {})
    })
    noteBotAttention(groupMemberKey(member), reason || error?.message || error)
    context.failedMembers?.add(groupMemberKey(member))
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
  const anchorIdx = anchorId === null ? -1 : roomNow.log.findIndex((e: GroupMessage) => e.id === anchorId)
  // Anchor trimmed away ⇒ every pre-turn entry was dropped, so every
  // surviving entry is newer — scanning the whole log stays exact.
  const turnTail = anchorIdx >= 0 ? roomNow.log.slice(anchorIdx + 1) : roomNow.log

  const newerUserEntryInThread = turnTail.some(
    (e: GroupMessage) => e.from?.kind === 'user' && groupThreadOf(e) === thread
  )

  if (
    (roomNow.stoppedEpoch || 0) > startEpoch ||
    !shouldCommitMemberTurn(startEpoch, epochNow, newerUserEntryInThread)
  ) {
    recordGroupActivity(context.group, {
      kind: 'cancelled',
      member: groupMemberKey(member),
      thread
    })

    return null
  }

  // Resolve the frozen submit boundary against the retained log. If it was
  // trimmed away, every surviving entry is still unseen. Throws do not
  // acknowledge input, and a timed-out turn keeps its submitted boundary.
  if (accepted) {
    updateGroupChat(context.group, (r: GroupChatRoom) => {
      r.watermarks[markKey] = anchorIdx + 1

      if (heldIds.length) {
        const deliveredIds = new Set(heldIds)
        const remaining = (r.heldMessages?.[memberKey] || []).filter(id => !deliveredIds.has(id))
        r.heldMessages = {
          ...(r.heldMessages || {})
        }

        if (remaining.length) {
          r.heldMessages[memberKey] = remaining
        } else {
          delete r.heldMessages[memberKey]
        }
      }

      return r
    })
  }

  const spoke = reply !== null && !isGroupPassText(reply)

  if (reply !== null && spoke) {
    appendGroupChatEntry(context.group, groupMemberAuthor(member), reply, thread)
  } else if (reply !== null && context.isCurrent() && context.addressedKeys?.has(memberKey)) {
    // #129443: the nudge was passed on too. Record the noncompliance where
    // the user looks (activity row + roster badge) instead of letting the
    // user's explicit address settle as ordinary quiet consensus. Only a
    // CURRENT drive's pass is disclosed: a turn the epoch check just
    // tolerated committing (a cross-thread send superseded its drive) is
    // stale residue, and flagging it would point the user at an address
    // their newer send already moved past.
    const reason = 'explicitly addressed member passed twice'

    recordGroupActivity(context.group, {
      kind: 'failed',
      member: groupMemberKey(member),
      reason,
      thread
    })
    noteBotAttention(groupMemberKey(member), reason)
  }

  // A member's own entries — its reply, and the rows group-external-writes.ts
  // mirrored out of its own session — are never news to their author, so the
  // watermark steps over them. A user entry that arrived during inference
  // stops the walk: a reply cannot acknowledge what it never saw. Entries of
  // OTHER threads are invisible to this thread's delta (the delta filter is
  // thread-scoped), so the walk steps over them too — a concurrent lane's
  // writes in a sibling thread must not wedge this member's watermark
  // between the commit boundary and its own reply, or the next round
  // re-delivers that reply to its own author as "news" (#head-of-line).
  updateGroupChat(context.group, (r: GroupChatRoom) => {
    let mark = r.watermarks[markKey] || 0

    while (mark < r.log.length) {
      const entry = r.log[mark]

      if (groupThreadOf(entry) !== thread) {
        mark += 1

        continue
      }

      if (!authoredByMember(entry, member)) {
        break
      }

      mark += 1
    }

    if (mark !== (r.watermarks[markKey] || 0)) {
      r.watermarks[markKey] = mark
    }

    return r
  })

  return spoke
}

function authoredByMember(entry: GroupMessage, member: GroupMember): boolean {
  return entry?.from?.kind === 'member' && isGroupChatSelf(entry.from, member)
}

// ── priority mention lanes (#head-of-line) ───────────────────────────────────
//
// The room drive is serial on purpose: one member turn at a time, user sends
// queue behind it. But head-of-line blocking made a DIRECT user mention of an
// IDLE member wait out another member's arbitrarily long turn — minutes of
// "Chief is quiet" while the Chief was never even prompted.
//
// A lane is a single-shot background drive for exactly ONE member+thread,
// dispatched at send time when the room is busy and the mention is explicit
// (non-@everyone). It reuses the whole turn path — delta/watermarks, hold
// semantics, the #93127 commit check, stranded markers — so a lane turn and
// the later foreground pass over the same thread cannot double-deliver:
// whichever finishes first advances the watermark; the other finds no delta.
//
// One lane per member+thread at a time. The thread stays in the foreground
// queue regardless, as the safety net for mentions of members that are busy
// right now (their work must not be interrupted — the lane never re-submits
// into a live session) and for the bot-to-bot handoffs the round loop owns.

/** In-flight lane claims, keyed `${roomKey}::${thread}::${memberKey}`. The
 *  value carries the live descriptors so Stop can resolve and interrupt each
 *  claim's session without re-parsing the key. */
interface GroupPriorityLaneClaim {
  group: string
  member: GroupMember
  thread: string
}

const groupPriorityLanes = new Map<string, GroupPriorityLaneClaim>()

function groupPriorityLaneKey(group: string, thread: string, member: GroupMember) {
  return `${groupChatRoomKey(group, $groupChats.get()[group])}::${thread}::${groupMemberKey(member)}`
}

/** Claim a member+thread lane for one background dispatch. False when a lane
 *  is already running there — the foreground queue still owns the thread, so
 *  a second claim would be a duplicate dispatch. */
export function claimGroupPriorityLane(group: string, thread: string, member: GroupMember): boolean {
  const key = groupPriorityLaneKey(group, thread, member)

  if (groupPriorityLanes.has(key)) {
    return false
  }

  groupPriorityLanes.set(key, { group, member, thread })

  return true
}

/** Whether a live lane claim exists for this member+thread — the foreground
 *  round loop skips such members so it can never re-prepare the delta the
 *  lane is already driving (a duplicate submit into a live session). */
export function groupPriorityLaneClaimed(group: string, thread: string, member: GroupMember): boolean {
  return groupPriorityLanes.has(groupPriorityLaneKey(group, thread, member))
}

/** Live lane claims in a room, for the Stop path: every claim owns an
 *  in-flight member turn the stop must cancel AND interrupt. */
export function activeGroupPriorityLaneClaims(group: string): GroupPriorityLaneClaim[] {
  return [...groupPriorityLanes.values()].filter(claim => claim.group === group)
}

/** Release a lane claim when its background turn settles (any outcome). A
 *  rename mid-lane re-keys the room for legacy (roomId-less) rooms, so both
 *  the claim key and the current-name key are cleared — a leaked claim would
 *  silence that member+thread's lanes forever. */
function releaseGroupPriorityLane(claimKey: string, group: string, thread: string, member: GroupMember) {
  groupPriorityLanes.delete(claimKey)
  groupPriorityLanes.delete(groupPriorityLaneKey(group, thread, member))
}

/** Is this member permitted to run a lane turn RIGHT NOW? False when it is
 *  the member the foreground drive is awaiting (`room.turn` — the very work
 *  the lane exists to route around; a lane must never re-submit or interrupt
 *  it), or when a stranded marker says its previous turn is still running on
 *  the backend. Presence in `room.stranded` is the same still-running signal
 *  the round loop's responder filter uses. */
export function memberIdleForPriorityLane(room: GroupChatRoom, member: GroupMember): boolean {
  if (room.turn && groupMemberKey(room.turn) === groupMemberKey(member)) {
    return false
  }

  return !Object.prototype.hasOwnProperty.call(room.stranded || {}, groupMemberKey(member))
}

/** Drive ONE member's turn in a background lane: claim, run, release. Returns
 *  the turn outcome so a caller can surface failures. Reuses
 *  runGroupRoundMember wholesale — delta scoping, holds, the #93127 commit
 *  check and stranded markers are all its behavior — with a `background`
 *  context so the lane never clobbers the room's `turn` slot (Stop's
 *  interrupt key). `failedMembers` is the LIVE drive's set, so a member the
 *  drive already gave up on (ambiguous admission/submit failure) is not
 *  retried by the lane, and a lane failure poisons the drive's set the same
 *  way a foreground one does. */
export async function runGroupPriorityLaneTurn(
  group: string,
  members: GroupMember[],
  thread: string,
  member: GroupMember,
  failedMembers?: Set<string>
): Promise<boolean | null> {
  let claimKey = groupPriorityLaneKey(group, thread, member)

  if (!claimGroupPriorityLane(group, thread, member)) {
    return null
  }

  const claim = groupPriorityLanes.get(claimKey)!

  // Follow the room and the claim together, including name-keyed legacy rooms.
  let laneGroup = group

  const binding = followGroupChat(group, name => {
    groupPriorityLanes.delete(claimKey)
    laneGroup = name
    claim.group = name
    claimKey = groupPriorityLaneKey(name, thread, member)
    groupPriorityLanes.set(claimKey, claim)
  })

  const startEpoch = ($groupChats.get()[laneGroup] || {}).epoch || 0
  const isCurrent = () => binding.isLive() && (($groupChats.get()[laneGroup] || {}).epoch || 0) === startEpoch

  try {
    return await runGroupRoundMember(
      {
        get group() {
          return laneGroup
        },
        members,
        thread,
        startEpoch,
        binding,
        isCurrent,
        failedMembers,
        background: true
      },
      member
    )
  } finally {
    releaseGroupPriorityLane(claimKey, laneGroup, thread, member)
    binding.dispose()
  }
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
        (member: GroupMember) =>
          !Object.prototype.hasOwnProperty.call(strandedNow, groupMemberKey(member)) &&
          // Same live-lane exclusion as the round loop's responder filter: a
          // lane turn in flight for this member+thread owns that delta.
          !groupPriorityLaneClaimed(context.group, context.thread, member)
      )

      for (const member of continuationResponders) {
        if (!isCurrent() || posted >= GROUP_CHAT_MAX_MESSAGES || continuations > GROUP_CHAT_MAX_CONTINUATIONS) {
          break
        }

        const result = await runGroupRoundMember(context, member)

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
