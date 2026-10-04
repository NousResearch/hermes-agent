/**
 * Manual compression of a room member's hidden plumbing sessions (#102291).
 *
 * Member sessions (`Group: <roomId> · <thread>`, hidden, room_plumbing) grow
 * with every room drive and are reachable from no other surface: they are
 * excluded from session lists, the focused-chat `/compress` never targets
 * them, and `-q` chat forwards slash text to the model. This is the room
 * settings' "Compress history" action: resume each session the room knows
 * for the member, then run the gateway's `session.compress` against the live
 * runtime id — the same RPC pair the reporter ran by hand over JSON-RPC.
 */

import { $groupChats } from './group-chat'
import { groupMemberKey, groupSessionMemberKey } from './group-membership'
import { requestForBot } from './routing'
import type { GroupMember } from './types'

/** Matches the focused-chat `/compress` budget: the gateway's compute-host
 *  wait is legitimately this long, and the socket's 30 s default reports a
 *  false timeout while the summary is still committing (#97948). */
const GROUP_SESSION_COMPRESS_TIMEOUT_MS = 660_000

interface GroupSessionCompressResult {
  after_messages?: number
  before_messages?: number
  compressed?: boolean
  message?: string
  status?: string
  summary?: { aborted?: boolean; headline?: string; note?: string }
}

export interface GroupMemberCompressOutcome {
  /** Sessions actually compressed. */
  compressed: number
  /** Human-readable headline per compressed session, e.g. "compressed 1588 → 134 messages". */
  lines: string[]
  /** Sessions the gateway still compresses in the background (`status: pending`). */
  pending: number
  /** Sessions the room knows but the gateway no longer has (4007) or that refused. */
  skipped: number
}

/** Every stored session id the room holds for this member: the thread-scoped
 *  pointers plus the pre-thread pointer of a legacy room. The pre-thread title
 *  is the last resort for a rehydrated room whose pointer was lost. */
function memberSessionTargets(group: string, member: GroupMember): string[] {
  const room = $groupChats.get()[group] || {}
  const memberKey = groupMemberKey(member)

  const stored = Object.entries(room.sessions || {})
    .filter(([key, value]) => typeof value === 'string' && groupSessionMemberKey(key) === memberKey)
    .map(([, value]) => value as string)

  return stored.length > 0 ? Array.from(new Set(stored)) : [`Group: ${room.roomId || group}`]
}

export async function compressGroupMemberHistory(
  group: string,
  member: GroupMember
): Promise<GroupMemberCompressOutcome> {
  const outcome: GroupMemberCompressOutcome = { compressed: 0, lines: [], pending: 0, skipped: 0 }

  for (const target of memberSessionTargets(group, member)) {
    let runtime: null | string = null

    try {
      const resumed = (await requestForBot(member, 'session.resume', {
        session_id: target,
        profile: member.name,
        omit_messages: true
      })) as { session_id?: string }

      runtime = resumed?.session_id || null
    } catch (error: any) {
      // 4007 = the gateway genuinely has no such session; anything else is a
      // real failure the caller should see.
      if (error?.code !== 4007) {
        throw error
      }
    }

    if (!runtime) {
      outcome.skipped += 1

      continue
    }

    const result = await requestForBot<GroupSessionCompressResult>(
      member,
      'session.compress',
      { session_id: runtime },
      { timeoutMs: GROUP_SESSION_COMPRESS_TIMEOUT_MS }
    )

    if (result?.status === 'pending') {
      outcome.pending += 1
    } else if (result?.status === 'aborted' || result?.summary?.aborted || result?.compressed === false) {
      outcome.skipped += 1
    } else {
      outcome.compressed += 1

      const headline =
        result?.summary?.headline ||
        (typeof result?.before_messages === 'number' && typeof result?.after_messages === 'number'
          ? `${result.before_messages} → ${result.after_messages} messages`
          : '')

      if (headline) {
        outcome.lines.push(headline)
      }
    }
  }

  return outcome
}
