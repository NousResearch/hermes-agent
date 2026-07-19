import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import type { MessagingConversationGroup } from './messaging-groups'
import { SidebarSessionsSection } from './sessions-section'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: {
        close: 'Close'
      },
      sidebar: {
        projects: {
          topicAliasPlaceholder: 'Local topic alias',
          topicLink: 'Save binding',
          topicProjectAction: 'Project binding',
          topicProjectDesc: 'Bind this topic to a project.',
          topicProjectPlaceholder: 'Select project',
          topicProjectTitle: 'Topic project',
          topicUnlink: 'Unlink'
        }
      }
    }
  })
}))

vi.mock('./session-row', () => ({
  SidebarSessionRow: ({ session }: { session: SessionInfo }) => <div data-testid={`session-${session.id}`}>{session.id}</div>
}))

const session = (id: string): SessionInfo =>
  ({
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'telegram',
    started_at: 0,
    title: id,
    tool_call_count: 0
  }) as SessionInfo

const conversations = (topicSessions: SessionInfo[], canManageBinding = false): MessagingConversationGroup[] => [
  {
    id: 'default:telegram:chat',
    label: 'Team chat',
    topics: [
      {
        binding: null,
        canManageBinding,
        id: 'default:telegram:chat:topic',
        identity: {
          chatId: 'chat',
          chatTopic: 'Release topic',
          chatType: 'group',
          displayName: 'Team chat',
          platform: 'telegram',
          profile: 'default',
          targetKey: '{"chat_id":"chat","platform":"telegram","thread_id":"topic"}',
          threadId: 'topic'
        },
        label: 'Release topic',
        sessions: topicSessions
      }
    ]
  }
]

function renderSection(options: {
  flatSessions?: SessionInfo[]
  messagingConversations: MessagingConversationGroup[]
  projects?: Parameters<typeof SidebarSessionsSection>[0]['projects']
}) {
  return render(
    <SidebarSessionsSection
      activeSessionId={null}
      emptyState={<div data-testid="empty">Empty</div>}
      label="Telegram"
      messagingConversations={options.messagingConversations}
      onArchiveSession={vi.fn()}
      onDeleteSession={vi.fn()}
      onResumeSession={vi.fn()}
      onToggle={vi.fn()}
      onTogglePin={vi.fn()}
      open
      pinned={false}
      projects={options.projects}
      sessions={options.flatSessions ?? []}
      workingSessionIdSet={new Set()}
    />
  )
}

describe('SidebarSessionsSection messaging conversations', () => {
  it('renders identified topics when there are no flat legacy rows', () => {
    renderSection({ messagingConversations: conversations([session('topic')]) })

    expect(screen.queryByTestId('empty')).toBeNull()
    expect(screen.getByText('Team chat')).not.toBeNull()
    expect(screen.getByText('Release topic')).not.toBeNull()
    expect(screen.getByTestId('session-topic')).not.toBeNull()
  })

  it('renders flat legacy rows alongside identified topics', () => {
    renderSection({
      flatSessions: [session('legacy')],
      messagingConversations: conversations([session('topic')])
    })

    expect(screen.getByTestId('session-topic')).not.toBeNull()
    expect(screen.getByTestId('session-legacy')).not.toBeNull()
  })

  it('exposes conversation and topic disclosure state', () => {
    const { container } = renderSection({ messagingConversations: conversations([session('topic')]) })
    const ownerDocument = container.ownerDocument

    const conversationToggle = screen.getByRole('button', { name: 'Team chat' })
    const topicToggle = screen.getByRole('button', { name: 'Release topic' })
    const conversationContentId = conversationToggle.getAttribute('aria-controls')
    const topicContentId = topicToggle.getAttribute('aria-controls')

    expect(conversationToggle.getAttribute('aria-expanded')).toBe('true')
    expect(topicToggle.getAttribute('aria-expanded')).toBe('true')
    expect(ownerDocument.getElementById(conversationContentId ?? '')).not.toBeNull()
    expect(ownerDocument.getElementById(topicContentId ?? '')).not.toBeNull()

    fireEvent.click(topicToggle)
    expect(topicToggle.getAttribute('aria-expanded')).toBe('false')
    expect(ownerDocument.getElementById(topicContentId ?? '')).toBeNull()

    fireEvent.click(conversationToggle)
    expect(conversationToggle.getAttribute('aria-expanded')).toBe('false')
    expect(ownerDocument.getElementById(conversationContentId ?? '')).toBeNull()
  })

  it('gives the project and alias controls accessible names', () => {
    renderSection({
      messagingConversations: conversations([session('topic')], true),
      projects: [
        {
          archived: false,
          board_slug: null,
          color: null,
          conversation_bindings: [],
          created_at: 0,
          description: null,
          folders: [],
          icon: null,
          id: 'p_1',
          name: 'Project',
          primary_path: null,
          slug: 'project'
        }
      ]
    })

    fireEvent.click(screen.getByLabelText('Project binding'))

    expect(screen.getByLabelText('Select project')).not.toBeNull()
    expect(screen.getByLabelText('Local topic alias')).not.toBeNull()
  })

  it('resets unsaved dialog state whenever the binding dialog reopens', () => {
    renderSection({
      messagingConversations: conversations([session('topic')], true),
      projects: [
        {
          archived: false,
          board_slug: null,
          color: null,
          conversation_bindings: [],
          created_at: 0,
          description: null,
          folders: [],
          icon: null,
          id: 'p_1',
          name: 'Project',
          primary_path: null,
          slug: 'project'
        }
      ]
    })

    fireEvent.click(screen.getByLabelText('Project binding'))
    fireEvent.change(screen.getByLabelText('Local topic alias'), { target: { value: 'Unsaved alias' } })
    fireEvent.click(screen.getByLabelText('Close'))
    fireEvent.click(screen.getByLabelText('Project binding'))

    expect((screen.getByLabelText('Local topic alias') as HTMLInputElement).value).toBe('Release topic')
  })
})
