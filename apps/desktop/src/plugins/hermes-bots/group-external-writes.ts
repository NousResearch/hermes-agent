/**
 * Mirror of what a member said and heard OUTSIDE the room engine. (#93813)
 *
 * A member's hidden per-group session is an ordinary Hermes session, so other
 * writers legitimately append to it: the user resuming it from the CLI
 * (`hermes -p <bot> chat --resume "Group: <room> · <thread>"`), a cron job, the
 * agent's own tools. Those rows reach the transcript but never the room log,
 * so the room silently diverges from the member's real conversation. The
 * sweep below reads each member session's unseen tail and appends the rows
 * the room did not itself produce, authored by that member.
 *
 * Cursor identity is the SESSION key (`thread:<t>::<memberKey>`): one member
 * owns one session per thread, and a cursor shared across threads would skip
 * one thread's rows after another thread's sweep advanced it.
 *
 * The cursor is an absolute row index into the transcript `session.resume`
 * reports. Two edges follow from that:
 *  - First sight of a session (no cursor yet — a room hydrated from the
 *    gateway mirror, which does not carry cursors, or one that predates this
 *    sweep) seeds the cursor at the transcript's current length. Trade-off:
 *    "late, never lost" holds from that moment on, but history already in the
 *    session is not replayed — a second Desktop taking the room over would
 *    otherwise post every historical external row again.
 *  - Compaction shrinks the transcript. A cursor past the current end is
 *    reset to the end: whatever compaction folded away is gone from the
 *    view anyway, and the alternative (waiting for the session to regrow past
 *    the old length) starts the next slice mid-exchange.
 */

import { $groupChats, appendGroupChatEntry, updateGroupChat } from './group-chat'
import type { GroupChatRoom } from './group-chat'
import { groupMemberKey, groupSessionKey, groupSessionMemberKey, groupSessionThread } from './group-membership'
import { GROUP_PROMPT_HEADER_PREFIX } from './group-round-prompt'
import { requestForBot } from './routing'
import type { GroupMember } from './types'

/** A transcript row as `session.resume` reports it; `content` is a plain string
 *  on most providers and a part array on the rest. */
export interface GroupTranscriptRow {
  content?: string | Array<string | { text?: string }>
  /** The gateway's display type for scaffolding rows it persists typed
   *  (`persist_user_display_kind`); absent on real user words. */
  display_kind?: string
  role?: string
  text?: string
}

export function groupTranscriptRowText(row: GroupTranscriptRow): string {
  const text =
    typeof row.content === 'string'
      ? row.content
      : Array.isArray(row.content)
        ? row.content.map(part => (typeof part === 'string' ? part : part?.text || '')).join('')
        : row.text || ''

  return String(text).trim()
}

/** Openers of the user-role rows the agent loop, compressor, cron and
 *  delegation plumbing inject into a transcript. Mirrors
 *  `agent/context_compressor.py::_SYNTHETIC_USER_ROW_PREFIXES` — keep the two
 *  lists in step. `session.resume` already drops `[System:` and
 *  `display_kind: hidden` rows and projects compaction carriers, so most of
 *  these only matter for rows persisted before the gateway typed them. */
export const SYNTHETIC_USER_ROW_PREFIXES = [
  '[System:',
  '[CONTEXT',
  '[PRIOR CONTEXT',
  '[IMPORTANT: Background',
  '[Your active task list',
  '[Planning state preserved',
  '[ASYNC DELEGATION',
  '[OUT-OF-BAND',
  'Cronjob Response:'
]

/** The Hermes-authored assistant row that closes a turn which failed before
 *  the model answered (a provider 401, retry exhaustion, a refusal), typed
 *  `display_kind: failed_turn` by `agent/turn_failure_copy.py`. A transcript
 *  boundary, never the member's reply: read as one, the room posts it as the
 *  bot speaking and loses the failure (#92760). */
export function failedTurnBoundaryRow(row: GroupTranscriptRow): boolean {
  return row.role === 'assistant' && row.display_kind === 'failed_turn'
}

/** A user row that carries no user words: typed scaffolding (auto-continue
 *  notes, steer markers, model-switch notices — anything but a skill
 *  invocation, which is the user's own `/command`) or an untyped row opening
 *  with one of the canonical prefixes. */
export function syntheticGroupUserRow(row: GroupTranscriptRow, text = groupTranscriptRowText(row)): boolean {
  if (row.display_kind && row.display_kind !== 'skill_invocation') {
    return true
  }

  return SYNTHETIC_USER_ROW_PREFIXES.some(prefix => text.startsWith(prefix))
}

/** The rows in `rows` the room engine did not write itself. A user row that
 *  does not open with the room prompt header is external; an assistant row
 *  answers whichever user row preceded it, so it inherits that row's origin.
 *  Synthetic user rows are plumbing: neither they nor the assistant row
 *  reacting to them (a compaction handoff, a cron report, a finished
 *  delegation) is a member speaking to anyone, so they close the exchange
 *  instead of continuing it. Tool rows and empty rows are never mirrored. */
export function externalGroupTranscriptRows(rows: GroupTranscriptRow[]): string[] {
  const external: string[] = []
  let answeringExternal = false

  for (const row of rows) {
    const text = groupTranscriptRowText(row)

    if (!text || (row.role !== 'user' && row.role !== 'assistant') || failedTurnBoundaryRow(row)) {
      continue
    }

    if (row.role === 'user') {
      answeringExternal = !syntheticGroupUserRow(row, text) && !text.startsWith(GROUP_PROMPT_HEADER_PREFIX)

      if (!answeringExternal) {
        continue
      }
    }

    if (answeringExternal) {
      external.push(text)
    }
  }

  return external
}

/** Append the rows written to `member`'s `thread` session since the last sweep
 *  to the room log, then move that session's cursor to the end of `messages`.
 *  Idempotent per row: the cursor persists with the room, so a restarted
 *  window never mirrors a row twice. `undefined` messages (no snapshot) leave
 *  the cursor alone — seeding from a failed resume would replay history on
 *  the next successful one. */
export function mirrorExternalGroupWrites(
  group: string,
  member: GroupMember,
  thread: string,
  messages: GroupTranscriptRow[] | undefined
) {
  if (!Array.isArray(messages)) {
    return
  }

  const rows = messages
  const key = groupSessionKey(thread, member)
  const room = ($groupChats.get()[group] || {}) as GroupChatRoom
  const cursor = room.externalCursors?.[key]
  // First sight or compaction shrink: park the cursor at the end (see header).
  const seen = typeof cursor === 'number' && cursor >= 0 && cursor <= rows.length ? cursor : rows.length

  if (seen === rows.length && cursor === seen) {
    return
  }

  const markKey = `${thread}::${groupMemberKey(member)}`
  const logLengthBefore = (room.log || []).length

  const mirrored = externalGroupTranscriptRows(rows.slice(seen)).map(text =>
    appendGroupChatEntry(
      group,
      {
        kind: 'member',
        name: member.name,
        ...(member.remoteSource ? { source: member.connectionLabel || member.connectionId } : {})
      },
      text,
      thread
    )
  )

  updateGroupChat(group, (r: GroupChatRoom) => {
    r.externalCursors = { ...(r.externalCursors || {}), [key]: rows.length }

    // The gateway mirror merge orders same-millisecond entries by id, and a
    // burst appended in one tick would come back shuffled: give the mirrored
    // rows strictly increasing stamps so they keep their transcript order.
    for (let i = logLengthBefore + 1; i < r.log.length; i += 1) {
      if (mirrored.includes(r.log[i])) {
        r.log[i].at = Math.max(r.log[i].at, (r.log[i - 1].at || 0) + 1)
      }
    }

    // The member already lived these rows in its own session; a watermark
    // sitting at the pre-mirror tail steps over them instead of feeding the
    // member its own conversation back as room news.
    if (r.watermarks[markKey] === logLengthBefore) {
      r.watermarks[markKey] = r.log.length
    }

    return r
  })
}

/** Idle trigger: read every member session the room still shows a thread for
 *  and mirror what reached it, without the room driving anyone. Runs when the
 *  room is opened, so a Bot posting reports into its own room session between
 *  rounds surfaces the next time the user looks — not only once the user
 *  types and that member happens to be a responder. One `session.resume` per
 *  stored session, bounded by the room's own (trimmed) log; no polling. A
 *  room mid-round is left to the round, which sweeps its responders itself. */
export async function sweepExternalGroupWrites(group: string, members: GroupMember[]) {
  const room = ($groupChats.get()[group] || {}) as GroupChatRoom

  if (room.running || room.tombstone) {
    return
  }

  const shown = new Set<string>(['legacy', ...(room.log || []).map(entry => entry.thread || 'legacy')])
  const sessions = room.sessions || {}

  for (const member of members) {
    const memberKey = groupMemberKey(member)

    for (const [key, stored] of Object.entries(sessions)) {
      const thread = groupSessionThread(key)

      if (typeof stored !== 'string' || groupSessionMemberKey(key) !== memberKey || !shown.has(thread)) {
        continue
      }

      let state: { messages?: GroupTranscriptRow[]; running?: boolean } | null = null

      try {
        state = await requestForBot(member, 'session.resume', { session_id: stored, profile: member.name })
      } catch {
        continue // unreachable or gone: nothing to mirror from here
      }

      if (!state?.running && ($groupChats.get()[group] || {}).sessions?.[key] === stored) {
        mirrorExternalGroupWrites(group, member, thread, state?.messages)
      }
    }
  }
}
