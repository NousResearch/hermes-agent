import { botMentionTag } from './data'
import {
  compactGroupChatSyncText,
  GROUP_CHAT_HISTORY_CHARS,
  GROUP_CHAT_HISTORY_LIMIT,
  GROUP_CHAT_HISTORY_LINE_CHARS,
  groupSpeakerLabel
} from './group-chat'
import { groupMemberKey } from './group-membership'
import type { GroupMember, GroupMessage, GroupMessageAuthor } from './types'

// Openers of Hermes' own control frames (the mid-turn steer marker, the compaction
// handoff, runtime/system notes). A member reply is republished to every peer inside
// a role=user prompt, so a reply reproducing one of these reads as harness input to
// the peers; the opener is relabelled visibly (the words stay, the exact trusted
// shape does not). Genuine user lines are never touched. Keep in sync with
// agent/prompt_builder.py::CONTROL_FRAME_OPENERS (the source of
// gateway/hosted_room_discussion.py::_MEMBER_CONTROL_FRAME_RE).
const MEMBER_CONTROL_FRAME_RE =
  /\[(?=\/?OUT-OF-BAND USER MESSAGE|CONTEXT COMPACTION|CONTEXT SUMMARY\]|PRIOR CONTEXT|Runtime note:|System note:|System:|SYSTEM\]|IMPORTANT:|Planning state preserved|ASYNC DELEGATION)/gi

function relabelMemberControlFrames(text: string) {
  return text.replace(MEMBER_CONTROL_FRAME_RE, '[member-quoted ')
}

/** Viewer identity for a room-log line. A bare string is the local, unsourced
 *  profile name (legacy call sites and single-connection jobs). */
export type GroupChatLineViewer =
  | string
  | (Pick<GroupMember, 'name'> & Partial<Pick<GroupMember, 'connectionId' | 'connectionLabel' | 'installId' | 'remoteSource'>>)

/** Room-log line as a member sees it: `Name (user): …` / `Name: …` /
 *  `Name (you): …`. */
export function formatGroupChatLine(entry: GroupMessage, viewer: GroupChatLineViewer, group?: null | string) {
  // Attachments are staged into each member's session as real payloads; the
  // transcript line names them so the delta text and the bytes line up.
  const attached =
    Array.isArray(entry.images) && entry.images.length
      ? ` ${entry.images
          .map(img => {
            const label = img.kind === 'pdf' ? 'attached PDF' : img.kind === 'file' ? 'attached file' : 'attached image'

            return `[${label}: ${img.name || 'image'}]`
          })
          .join(' ')}`
      : ''

  if (entry.from.kind === 'user') {
    return `${entry.from.name || 'User'} (user): ${entry.text}${attached}`
  }

  const suffix = isGroupChatSelf(entry.from, viewer) ? ' (you)' : ''
  // Cross-connection speakers carry their device so same-named agents on
  // two machines stay tellable apart in every member's transcript.
  const source = entry.from.source ? ` [${entry.from.source}]` : ''

  return `${groupSpeakerLabel(entry.from.name, group)}${suffix}${source}: ${relabelMemberControlFrames(entry.text)}${attached}`
}

/** #114341: a member's turn renders the newest delta lines that fit the
 *  window — GROUP_CHAT_HISTORY_LIMIT entries AND GROUP_CHAT_HISTORY_CHARS
 *  characters, each body first cut to GROUP_CHAT_HISTORY_LINE_CHARS — while
 *  the watermark commit advances past the whole tail, so the head of an
 *  over-long delta is never delivered on any later turn either. Mark the
 *  cut with the exact count — without it a member has no way to know its
 *  view of the room is partial (typically missing the very user instruction
 *  that started the exchange). The newest entry is always kept. */
export function formatGroupDeltaLines(delta: GroupMessage[], viewer: GroupChatLineViewer, group?: null | string) {
  const lines: string[] = []
  let chars = 0

  for (let i = delta.length - 1; i >= 0 && lines.length < GROUP_CHAT_HISTORY_LIMIT; i--) {
    const entry = delta[i]
    const line = formatGroupChatLine(
      { ...entry, text: compactGroupChatSyncText(entry.text, GROUP_CHAT_HISTORY_LINE_CHARS).text },
      viewer,
      group
    )

    if (lines.length && chars + line.length > GROUP_CHAT_HISTORY_CHARS) {
      break
    }

    lines.push(line)
    chars += line.length + 1
  }

  lines.reverse()
  const omitted = delta.length - lines.length

  if (omitted > 0) {
    lines.unshift(`… ${omitted} earlier room message${omitted === 1 ? '' : 's'} omitted since your last turn`)
  }

  return lines
}

function viewerNameOf(viewer: GroupChatLineViewer): string {
  return typeof viewer === 'string' ? viewer : viewer?.name || ''
}

/** Members stamp `from.source` as `connectionLabel || connectionId` (local
 *  ones too, once they know their connection). A string viewer or a member
 *  without a connection exposes no tokens. */
function viewerConnectionSources(viewer: GroupChatLineViewer): string[] {
  if (typeof viewer === 'string') {
    return []
  }

  return [viewer.connectionLabel, viewer.connectionId].filter((token): token is string => Boolean(token))
}

function isGroupChatSelf(from: GroupMessageAuthor, viewer: GroupChatLineViewer): boolean {
  if (!from.name || from.name !== viewerNameOf(viewer)) {
    return false
  }

  // Gateway identity first: the install_id is the same token on every
  // Desktop, while `source` is whatever THIS Desktop labelled the connection
  // (two Desktops calling one gateway "Central" / "Studio" agree here and
  // disagree below). Only decisive when both sides carry it.
  if (from.gateway && typeof viewer !== 'string' && viewer?.installId) {
    return from.gateway === viewer.installId
  }

  const speakerSource = from.source || ''

  // An unsourced same-name line is local by the room's resolution rule
  // (routing.ts: no source ⇒ `!remoteSource`), so only a local viewer owns it.
  if (!speakerSource) {
    return typeof viewer === 'string' || !viewer?.remoteSource
  }

  return viewerConnectionSources(viewer).includes(speakerSource)
}

interface GroupChatTurnPromptInput {
  deltaLines: string[]
  groupName: string
  members: GroupMember[]
  viewer: GroupMember
}

/** The full per-turn payload for one member: participation rules + the room
 *  delta. Rules travel in the turn payload (not SOUL) so every existing bot
 *  can join a group chat without a profile migration. */
export function buildGroupChatTurnPrompt({ groupName, members, viewer, deltaLines }: GroupChatTurnPromptInput) {
  const viewerKey = groupMemberKey(viewer)
  const peers = members.filter(m => groupMemberKey(m) !== viewerKey)

  const peerNames = peers
    .map(m => {
      const handle = m.title ? `${m.title} (@${botMentionTag(m)})` : `@${botMentionTag(m)}`

      return m.remoteSource ? `${handle} [on ${m.connectionLabel || m.connectionId}]` : handle
    })
    .join(', ')

  return [
    `[Group chat: "${groupName}"] You are @${botMentionTag(viewer)}, one participant in a group chat with ${peerNames || 'no one else yet'} and the user.`,
    '',
    'New messages in the room since your last turn (oldest first):',
    ...deltaLines.map(line => `  ${line}`),
    '',
    'Rules for this room:',
    '- Reply with ONE conversational message ONLY if you have something new worth adding: build on what was just said, claim or hand off work, answer a question aimed at you, or report a real result. Keep chatter short (1-3 sentences) — but when you are delivering a result, an answer the user asked for, or substantive work, give it at full quality and length; never thin out real content to fit the room.',
    '- If you have nothing new to add, reply with exactly "(pass)". Passing is good — it lets the conversation settle.',
    '- Mention a teammate as @name to pull them in; mention @user only for a judgment call or a result the user needs. Do not repeat points already made.',
    '- Never reveal content from your private 1:1 chats. Your reply text goes to the room verbatim — no preamble, no meta-commentary.'
  ].join('\n')
}
