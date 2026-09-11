import { botHandle } from './data'
import { groupSpeakerLabel } from './group-chat'
import { groupMemberKey } from './group-membership'
import type { GroupMember, GroupMessage, GroupMessageAuthor } from './types'

/** Viewer identity for a room-log line. A bare string is the local, unsourced
 *  profile name (legacy call sites and single-connection jobs). */
export type GroupChatLineViewer =
  string | (Pick<GroupMember, 'name'> & Partial<Pick<GroupMember, 'connectionId' | 'connectionLabel' | 'remoteSource'>>)

/** Room-log line as a member sees it: `Name (user): …` / `Name: …` /
 *  `Name (you): …`. */
export function formatGroupChatLine(entry: GroupMessage, viewer: GroupChatLineViewer) {
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

  return `${groupSpeakerLabel(entry.from.name)}${suffix}${source}: ${entry.text}${attached}`
}

function viewerNameOf(viewer: GroupChatLineViewer): string {
  return typeof viewer === 'string' ? viewer : viewer?.name || ''
}

/** Remote members stamp `from.source` as `connectionLabel || connectionId`.
 *  Only a remoteSource viewer exposes those tokens; a string or local member
 *  is unsourced so same-name remote lines fail open (no `(you)`). */
function viewerConnectionSources(viewer: GroupChatLineViewer): string[] {
  if (typeof viewer === 'string' || !viewer?.remoteSource) {
    return []
  }

  return [viewer.connectionLabel, viewer.connectionId].filter((token): token is string => Boolean(token))
}

function isGroupChatSelf(from: GroupMessageAuthor, viewer: GroupChatLineViewer): boolean {
  if (!from.name || from.name !== viewerNameOf(viewer)) {
    return false
  }

  const speakerSource = from.source || ''
  const viewerSources = viewerConnectionSources(viewer)

  if (!speakerSource && viewerSources.length === 0) {
    return true
  }

  return Boolean(speakerSource) && viewerSources.includes(speakerSource)
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
      const handle = m.title ? `${m.title} (@${botHandle(m.name, m)})` : `@${botHandle(m.name, m)}`

      return m.remoteSource ? `${handle} [on ${m.connectionLabel || m.connectionId}]` : handle
    })
    .join(', ')

  return [
    `[Group chat: "${groupName}"] You are @${botHandle(viewer.name, viewer)}, one participant in a group chat with ${peerNames || 'no one else yet'} and the user.`,
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
