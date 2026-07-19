import { describe, expect, it } from 'vitest'

import type { ConversationBindingInfo, ProjectInfo, SessionInfo } from '@/types/hermes'

import { buildMessagingGroups, sessionConversationIdentity } from './messaging-groups'

const session = (over: Partial<SessionInfo> & Pick<SessionInfo, 'id'>): SessionInfo =>
  ({
    ended_at: null,
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'telegram',
    started_at: 0,
    title: null,
    tool_call_count: 0,
    ...over
  }) as SessionInfo

const project = (binding: ConversationBindingInfo): ProjectInfo =>
  ({
    archived: false,
    board_slug: null,
    color: null,
    conversation_bindings: [binding],
    created_at: 0,
    description: null,
    folders: [],
    icon: null,
    id: binding.project_id,
    name: 'Project',
    primary_path: null,
    slug: 'project'
  }) as ProjectInfo

describe('sessionConversationIdentity', () => {
  it('uses direct compact-row fields before origin_json fallback', () => {
    const identity = sessionConversationIdentity(
      session({
        id: 's1',
        chat_id: 'direct-chat',
        thread_id: 'direct-thread',
        display_name: 'Direct',
        origin_json: JSON.stringify({ chat_id: 'origin-chat', thread_id: 'origin-thread', chat_topic: 'Topic' })
      })
    )

    expect(identity?.chatId).toBe('direct-chat')
    expect(identity?.threadId).toBe('direct-thread')
    expect(identity?.displayName).toBe('Direct')
    expect(identity?.chatTopic).toBe('Topic')
  })

  it('falls back gracefully for malformed origin_json and missing identity', () => {
    expect(sessionConversationIdentity(session({ id: 'bad', origin_json: '{nope' }))).toBeNull()
    expect(
      sessionConversationIdentity(session({ id: 'ok', origin_json: JSON.stringify({ chat_id: 'c', thread_id: '' }) }))
        ?.threadId
    ).toBeNull()
  })
})

describe('buildMessagingGroups', () => {
  it('groups by platform, profile, chat, and topic with deterministic label priority', () => {
    const bound = {
      alias: 'Bound alias',
      chat_id: 'chat',
      created_at: 0,
      platform: 'telegram',
      project_id: 'p_1',
      target_key: '',
      thread_id: 'topic',
      updated_at: 0
    }

    const groups = buildMessagingGroups({
      projects: [project(bound)],
      projectProfile: 'default',
      platformTotals: {},
      truncated: false,
      sessions: [
        session({ id: 'old', chat_id: 'chat', thread_id: 'topic', origin_json: JSON.stringify({ chat_topic: 'Topic' }), last_active: 1 }),
        session({ id: 'new', chat_id: 'chat', thread_id: 'topic', last_active: 2 })
      ]
    })

    expect(groups).toHaveLength(1)
    expect(groups[0].conversations).toHaveLength(1)
    expect(groups[0].conversations[0].topics[0].label).toBe('Bound alias')
    expect(groups[0].conversations[0].topics[0].canManageBinding).toBe(true)
    expect(groups[0].conversations[0].topics[0].sessions.map(s => s.id)).toEqual(['new', 'old'])
  })

  it('keeps legacy rows as flat fallback beside discovered topics', () => {
    const groups = buildMessagingGroups({
      projects: [],
      projectProfile: 'default',
      platformTotals: { telegram: 4 },
      truncated: false,
      sessions: [session({ id: 'legacy' }), session({ id: 'topic', chat_id: 'chat', thread_id: 't' })]
    })

    expect(groups[0].flatSessions.map(s => s.id)).toEqual(['legacy'])
    expect(groups[0].conversations[0].topics[0].sessions.map(s => s.id)).toEqual(['topic'])
    expect(groups[0].total).toBeGreaterThan(groups[0].sessions.length)
    expect(groups[0].hasMore).toBe(true)
  })

  it('does not merge the same thread id across different profiles or chats', () => {
    const groups = buildMessagingGroups({
      projects: [],
      projectProfile: 'default',
      platformTotals: {},
      truncated: false,
      sessions: [
        session({ id: 'a', profile: 'default', chat_id: 'chat-a', thread_id: 'same' }),
        session({ id: 'b', profile: 'work', chat_id: 'chat-a', thread_id: 'same' }),
        session({ id: 'c', profile: 'default', chat_id: 'chat-b', thread_id: 'same' })
      ]
    })

    expect(groups[0].conversations).toHaveLength(3)
  })

  it('does not apply or expose one profile binding on another profile target', () => {
    const bound = {
      alias: 'Default alias',
      chat_id: 'same-chat',
      created_at: 0,
      platform: 'telegram',
      project_id: 'p_1',
      target_key: '',
      thread_id: 'same-topic',
      updated_at: 0
    }

    const groups = buildMessagingGroups({
      projects: [project(bound)],
      projectProfile: 'default',
      platformTotals: {},
      truncated: false,
      sessions: [
        session({ id: 'default', profile: 'default', chat_id: 'same-chat', thread_id: 'same-topic' }),
        session({ id: 'work', profile: 'work', chat_id: 'same-chat', thread_id: 'same-topic' })
      ]
    })

    const topics = groups[0].conversations.flatMap(conversation => conversation.topics)
    const defaultTopic = topics.find(topic => topic.identity.profile === 'default')
    const workTopic = topics.find(topic => topic.identity.profile === 'work')

    expect(defaultTopic?.binding?.alias).toBe('Default alias')
    expect(defaultTopic?.canManageBinding).toBe(true)
    expect(workTopic?.binding).toBeNull()
    expect(workTopic?.canManageBinding).toBe(false)
  })

  it('disables project binding management in the all-profiles scope', () => {
    const groups = buildMessagingGroups({
      projects: [],
      projectProfile: null,
      platformTotals: {},
      truncated: false,
      sessions: [session({ id: 'default', profile: 'default', chat_id: 'chat', thread_id: 'topic' })]
    })

    expect(groups[0].conversations[0].topics[0].canManageBinding).toBe(false)
  })
})
