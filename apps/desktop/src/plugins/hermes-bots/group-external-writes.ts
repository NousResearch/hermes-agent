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
import { kickGroupChatDrive, postToGroupChat, resolveGroupMentionTargets } from './group-rounds'
import { requestForBot } from './routing'
import type { GroupMember } from './types'

/** A transcript row as `session.resume` reports it; `content` is a plain string
 *  on most providers and a part array on the rest. */
export interface GroupTranscriptRow {
  /** Parsed tool-call arguments — the ONLY part of a tool row the gateway's
   *  `session.resume` projection ships besides its name (`session_history.py`
   *  projects `{role, name, context, args?}` and drops result content). */
  args?: unknown
  content?: string | Array<string | { text?: string }>
  /** The gateway's display type for scaffolding rows it persists typed
   *  (`persist_user_display_kind`); absent on real user words. */
  display_kind?: string
  /** Tool name, on a projected tool row. */
  name?: string
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

    if (!text || (row.role !== 'user' && row.role !== 'assistant')) {
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

/** The tool a member posts with — `tools/bot_room_post.py`. The name is the
 *  provenance the room trusts, so it is spelled once, here. */
export const ROOM_POST_TOOL_NAME = 'room_post'

/** A deliberate post a member made into the room, out of a `room_post` tool row
 *  row: the member said something TO the room rather than answering in it.
 *
 *  `kind` is the marker the tool writes into its acknowledgement, and `post`
 *  carries the words — the row's content is the tool RESULT, so the reader never
 *  has to reach for tool-call arguments that an older transcript does not
 *  project. A row whose content is not this shape is not a post. */
export interface GroupRoomPost {
  mentions: string[]
  room: string
  text: string
}

/** A post the sweep appended that named someone — the idle trigger kicks their
 *  turns once the sweep is done, never mid-read. */
export interface MirroredGroupPost {
  member: GroupMember
  post: GroupRoomPost
  thread: string
}

export function roomPostFromToolRow(row: GroupTranscriptRow): GroupRoomPost | null {
  // Provenance, not shape: the post is the row's NAME plus its ARGS, which is
  // what the resume projection actually preserves. Reading the tool RESULT body
  // would look for words that never arrive, and accepting any tool output shaped
  // like a post would let an unrelated row impersonate one.
  if (row.role !== 'tool' || String(row.name || '').trim() !== ROOM_POST_TOOL_NAME) {
    return null
  }

  const args = row.args && typeof row.args === 'object' ? (row.args as Record<string, unknown>) : null
  const room = String(args?.room ?? '').trim()
  const text = String(args?.text ?? '').trim()

  if (!room || !text) {
    return null
  }

  const mentions = Array.isArray(args?.mentions)
    ? (args?.mentions as unknown[]).map(entry => String(entry).trim()).filter(Boolean)
    : []

  return { mentions, room, text }
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
  messages: GroupTranscriptRow[] | undefined,
  members?: GroupMember[]
): MirroredGroupPost[] {
  if (!Array.isArray(messages)) {
    return []
  }

  const rows = messages
  const key = groupSessionKey(thread, member)
  const room = ($groupChats.get()[group] || {}) as GroupChatRoom
  const cursor = room.externalCursors?.[key]
  // First sight or compaction shrink: park the cursor at the end (see header).
  const seen = typeof cursor === 'number' && cursor >= 0 && cursor <= rows.length ? cursor : rows.length

  if (seen === rows.length && cursor === seen) {
    return []
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

  // A member that POSTED deliberately (`room_post`) is the other half of the same
  // sweep: the tool row is not a turn's prose, so it is not mirrored as text —
  // it is delivered as the member's message, and a post that names someone starts
  // that member's turn exactly as a human send would. A post addressed to another
  // room is not this room's to deliver.
  // Kick the drive with the members the caller drives with — the live roster —
  // and fall back to the room's durable descriptors only when a caller has none.
  // The durable rows are source-qualified: they route on the machine that holds
  // that connection, which is exactly the wrong list to hand a fresh drive on
  // the machine reading this transcript.
  const roster = Array.isArray(members) && members.length ? members : (room.members as GroupMember[]) || []
  const addressed: MirroredGroupPost[] = []

  // Where the member's own words end and a post begins. The watermark below
  // steps the member over its own rows so it is not fed its conversation back —
  // but a POST is addressed TO the room, and a member whose watermark already
  // covers it is never driven to answer it. So the step stops here.
  const mirroredEnd = (($groupChats.get()[group] || {}).log || []).length

  for (const post of rows.slice(seen).map(roomPostFromToolRow)) {
    if (!post || post.room.trim().toLowerCase() !== group.trim().toLowerCase()) {
      continue
    }

    // `kick: false` — this function also runs inside a live turn (the turn
    // mirrors its own session), and a kick from there supersedes that turn.
    postToGroupChat(group, roster, member, post.text, thread, { kick: false, mentions: post.mentions })

    if (post.mentions.length) {
      addressed.push({ member, post, thread })
    }
  }

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
      r.watermarks[markKey] = Math.min(mirroredEnd, r.log.length)
    }

    return r
  })

  return addressed
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
  const addressed: MirroredGroupPost[] = []

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
        addressed.push(...mirrorExternalGroupWrites(group, member, thread, state?.messages, members))
      }
    }
  }

  // A post that named someone starts their turn — here, once, after the sweep.
  // Kicking inside the loop would bump the room's epoch while the room is still
  // reading transcripts, and an epoch bump is how a live round is superseded: a
  // post would cancel the very turn that was about to read it.
  for (const item of addressed) {
    // Targets come from the validated list, not from the post's text: a post
    // whose text names nobody must drive exactly the members it named in its
    // arguments, not the whole room.
    kickGroupChatDrive(group, members, item.thread, resolveGroupMentionTargets(item.post.mentions, members))
  }
}
