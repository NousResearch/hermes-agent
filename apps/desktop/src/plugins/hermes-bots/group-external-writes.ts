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
 */

import { $groupChats, appendGroupChatEntry, updateGroupChat } from './group-chat'
import type { GroupChatRoom } from './group-chat'
import { groupMemberKey, groupSessionKey } from './group-membership'
import { GROUP_PROMPT_HEADER_PREFIX } from './group-round-prompt'
import type { GroupMember } from './types'

/** A transcript row as `session.resume` reports it; `content` is a plain string
 *  on most providers and a part array on the rest. */
export interface GroupTranscriptRow {
  content?: string | Array<string | { text?: string }>
  role?: string
  text?: string
}

function transcriptRowText(row: GroupTranscriptRow): string {
  const text =
    typeof row.content === 'string'
      ? row.content
      : Array.isArray(row.content)
        ? row.content.map(part => (typeof part === 'string' ? part : part?.text || '')).join('')
        : row.text || ''

  return String(text).trim()
}

/** Opens the user-role rows the agent loop injects into its own transcript
 *  (continuation nudges, truncation recovery — `agent/conversation_loop.py`,
 *  `agent/context_compressor.py`). They continue whichever exchange is open. */
const AGENT_INJECTED_ROW_PREFIX = '[System:'

/** The rows in `rows` the room engine did not write itself. A user row that
 *  does not open with the room prompt header is external; an assistant row
 *  answers whichever user row preceded it, so it inherits that row's origin;
 *  agent-injected user rows, tool rows and empty rows are plumbing and never
 *  mirrored. */
export function externalGroupTranscriptRows(rows: GroupTranscriptRow[]): string[] {
  const external: string[] = []
  let answeringExternal = false

  for (const row of rows) {
    const text = transcriptRowText(row)

    if (!text || (row.role !== 'user' && row.role !== 'assistant')) {
      continue
    }

    if (row.role === 'user') {
      if (text.startsWith(AGENT_INJECTED_ROW_PREFIX)) {
        continue
      }

      answeringExternal = !text.startsWith(GROUP_PROMPT_HEADER_PREFIX)
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
 *  window never mirrors a row twice. */
export function mirrorExternalGroupWrites(
  group: string,
  member: GroupMember,
  thread: string,
  messages: GroupTranscriptRow[] | undefined
) {
  const rows = Array.isArray(messages) ? messages : []
  const key = groupSessionKey(thread, member)
  const room = ($groupChats.get()[group] || {}) as GroupChatRoom
  const seen = Math.max(0, Number(room.externalCursors?.[key] || 0))

  if (rows.length <= seen) {
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
