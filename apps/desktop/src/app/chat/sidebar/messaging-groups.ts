import { normalizeSessionSource, sessionSourceLabel } from '@/lib/session-source'
import type { ConversationBindingInfo, ProjectInfo, SessionInfo } from '@/types/hermes'

export interface SessionConversationIdentity {
  profile: string
  platform: string
  chatId: string
  chatType: null | string
  threadId: null | string
  displayName: null | string
  chatTopic: null | string
  targetKey: string
}

export interface MessagingTopicGroup {
  id: string
  identity: SessionConversationIdentity
  label: string
  binding: ConversationBindingInfo | null
  canManageBinding: boolean
  sessions: SessionInfo[]
}

export interface MessagingConversationGroup {
  id: string
  label: string
  topics: MessagingTopicGroup[]
}

export interface MessagingPlatformGroup {
  sourceId: string
  label: string
  conversations: MessagingConversationGroup[]
  flatSessions: SessionInfo[]
  sessions: SessionInfo[]
  total: number
  hasMore: boolean
}

const normalizeText = (value: null | string | undefined): string => (value ?? '').trim()
const normalizeThread = (value: null | string | undefined): null | string => normalizeText(value) || null
const profileKey = (session: SessionInfo): string => normalizeText(session.profile) || 'default'

function parseOrigin(raw: null | string | undefined): Record<string, unknown> {
  if (!raw?.trim()) {
    return {}
  }

  try {
    const parsed = JSON.parse(raw)

    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as Record<string, unknown>) : {}
  } catch {
    return {}
  }
}

const originString = (origin: Record<string, unknown>, key: string): null | string => {
  const value = origin[key]

  return typeof value === 'string' || typeof value === 'number' ? String(value) : null
}

export function conversationTargetKey(platform: string, chatId: string, threadId: null | string): string {
  return JSON.stringify({ chat_id: chatId, platform, thread_id: threadId })
}

export function sessionConversationIdentity(session: SessionInfo): null | SessionConversationIdentity {
  const origin = parseOrigin(session.origin_json)
  const platform = normalizeSessionSource(session.source) || normalizeSessionSource(originString(origin, 'platform'))
  const chatId = normalizeText(session.chat_id) || normalizeText(originString(origin, 'chat_id'))

  if (!platform || !chatId) {
    return null
  }

  const threadId = normalizeThread(session.thread_id) ?? normalizeThread(originString(origin, 'thread_id'))

  const displayName =
    normalizeText(session.display_name) ||
    normalizeText(originString(origin, 'display_name')) ||
    normalizeText(originString(origin, 'chat_title')) ||
    null

  const chatTopic =
    normalizeText(originString(origin, 'chat_topic')) ||
    normalizeText(originString(origin, 'thread_name')) ||
    normalizeText(originString(origin, 'topic')) ||
    null

  return {
    profile: profileKey(session),
    platform,
    chatId,
    chatType: normalizeText(session.chat_type) || normalizeText(originString(origin, 'chat_type')) || null,
    threadId,
    displayName,
    chatTopic,
    targetKey: conversationTargetKey(platform, chatId, threadId)
  }
}

function bindingMap(projects: readonly ProjectInfo[]): Map<string, ConversationBindingInfo> {
  const out = new Map<string, ConversationBindingInfo>()

  for (const project of projects) {
    for (const binding of project.conversation_bindings ?? []) {
      out.set(conversationTargetKey(binding.platform, binding.chat_id, binding.thread_id), binding)
    }
  }

  return out
}

const sessionTime = (session: SessionInfo): number => session.last_active || session.started_at || 0
const topicTime = (topic: MessagingTopicGroup | undefined): number => (topic?.sessions[0] ? sessionTime(topic.sessions[0]) : 0)

export function buildMessagingGroups(params: {
  sessions: readonly SessionInfo[]
  projects: readonly ProjectInfo[]
  projectProfile: null | string
  platformTotals: Record<string, number>
  truncated: boolean
}): MessagingPlatformGroup[] {
  const bindings = bindingMap(params.projects)
  const byPlatform = new Map<string, { identified: SessionInfo[]; flat: SessionInfo[] }>()

  for (const session of params.sessions) {
    const sourceId = normalizeSessionSource(session.source)

    if (!sourceId) {
      continue
    }

    const bucket = byPlatform.get(sourceId) ?? { identified: [], flat: [] }
    const identity = sessionConversationIdentity(session)
    bucket[identity ? 'identified' : 'flat'].push(session)
    byPlatform.set(sourceId, bucket)
  }

  return [...byPlatform.entries()]
    .map(([sourceId, bucket]) => {
      const conversations = new Map<string, MessagingConversationGroup>()

      for (const session of bucket.identified) {
        const identity = sessionConversationIdentity(session)

        if (!identity) {
          continue
        }

        const conversationId = `${identity.profile}:${identity.platform}:${identity.chatId}`
        const topicId = `${conversationId}:${identity.threadId ?? ''}`

        const conversation = conversations.get(conversationId) ?? {
          id: conversationId,
          label: identity.displayName || identity.chatId,
          topics: []
        }

        let topic = conversation.topics.find(item => item.id === topicId)

        if (!topic) {
          // Messaging rows are intentionally fetched across every profile, while
          // Projects and their bindings come from the active profile's projects.db.
          // Never decorate or mutate another profile's identical platform target
          // with the active profile's binding.
          const canManageBinding = params.projectProfile !== null && identity.profile === params.projectProfile
          const binding = canManageBinding ? (bindings.get(identity.targetKey) ?? null) : null
          topic = {
            id: topicId,
            identity,
            binding,
            canManageBinding,
            label: binding?.alias || identity.chatTopic || (identity.threadId ? `${conversation.label} · ${identity.threadId}` : conversation.label),
            sessions: []
          }
          conversation.topics.push(topic)
        }

        topic.sessions.push(session)
        conversations.set(conversationId, conversation)
      }

      for (const conversation of conversations.values()) {
        for (const topic of conversation.topics) {
          topic.sessions.sort((a, b) => sessionTime(b) - sessionTime(a) || a.id.localeCompare(b.id))
        }

        conversation.topics.sort(
          (a, b) =>
            sessionTime(b.sessions[0]) - sessionTime(a.sessions[0]) ||
            a.label.localeCompare(b.label, undefined, { sensitivity: 'base' })
        )
      }

      const nested = [...conversations.values()].sort(
        (a, b) =>
          topicTime(b.topics[0]) - topicTime(a.topics[0]) ||
          a.label.localeCompare(b.label, undefined, { sensitivity: 'base' })
      )

      const sessions = [...bucket.identified, ...bucket.flat].sort((a, b) => sessionTime(b) - sessionTime(a))
      const known = params.platformTotals[sourceId]
      const total = Math.max(sessions.length, known ?? 0)

      return {
        sourceId,
        label: sessionSourceLabel(sourceId) ?? sourceId,
        conversations: nested,
        flatSessions: bucket.flat.sort((a, b) => sessionTime(b) - sessionTime(a)),
        sessions,
        total,
        hasMore: known != null ? known > sessions.length : params.truncated
      }
    })
    .sort((a, b) => sessionTime(b.sessions[0]) - sessionTime(a.sessions[0]))
}
