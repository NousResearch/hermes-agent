// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { StableNavigateProvider } from '@/app/stable-navigate'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { en } from '@/i18n/en'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $selectedStoredSessionId, $sessions } from '@/store/session'
import { makeSessionInfo } from '@/test/session-info'

import { SimplifiedSessionList } from '../simplified-session-list'

import { AGENT_CHAT_CAP, capAgentChats } from './agent-chat-cap'

const noop = () => {}

const chats: SessionInfo[] = Array.from({ length: 6 }, (_, index) =>
  makeSessionInfo({ id: `chat-${index}`, connection_id: 'local', profile: 'default', title: `Chat ${index}` })
)

function AgentChats({
  activeSessionId = null,
  searching = false,
  sessions = chats
}: {
  activeSessionId?: null | string
  searching?: boolean
  sessions?: SessionInfo[]
}) {
  const [showAll, setShowAll] = useState(false)

  return (
    <MemoryRouter><StableNavigateProvider>
      <RootTooltipProvider>
        <SimplifiedSessionList
          activeSessionId={activeSessionId}
          ambiguousIds={new Set()}
          cap={{ limit: AGENT_CHAT_CAP, onToggle: () => setShowAll(!showAll), showAll }}
          onArchiveSession={noop}
          onBranchSession={noop}
          onDeleteSession={noop}
          onResumeSession={noop}
          onUpdate={noop}
          pinnedIds={new Set()}
          searching={searching}
          sessions={sessions}
        />
      </RootTooltipProvider>
    </StableNavigateProvider></MemoryRouter>
  )
}

const titles = () => screen.queryAllByText(/^Chat \d$/).map(node => node.textContent)

beforeEach(() => {
  $apiRequestScope.set({ connectionId: 'local', profile: 'default' })
  $activeGatewayProfile.set('default')
  $selectedStoredSessionId.set(null)
  $activeSessionId.set(null)
  $sessions.set([])
})

afterEach(cleanup)

describe('capAgentChats', () => {
  it('keeps the most recent rows and any later row that must stay in reach, in order', () => {
    expect(capAgentChats([1, 2, 3, 4, 5], 2, row => row === 4)).toEqual([1, 2, 4])
    expect(capAgentChats([1, 2], 3, () => false)).toEqual([1, 2])
  })
})

describe("an agent card's chats", () => {
  it('shows the three most recent, then Show all with the full count, then Show less', () => {
    render(<AgentChats />)
    expect(titles()).toEqual(['Chat 0', 'Chat 1', 'Chat 2'])

    fireEvent.click(screen.getByRole('button', { name: en.simplifiedAgents.showAllChats(6) }))
    expect(titles()).toHaveLength(6)

    fireEvent.click(screen.getByRole('button', { name: en.simplifiedAgents.showFewerChats }))
    expect(titles()).toHaveLength(3)
  })

  it('keeps the open chat in the card even when it is older than the cap', () => {
    render(<AgentChats activeSessionId="chat-5" />)
    expect(titles()).toEqual(['Chat 0', 'Chat 1', 'Chat 2', 'Chat 5'])
  })

  it('keeps unread chats in the card', () => {
    render(<AgentChats sessions={chats.map((row, index) => (index === 4 ? { ...row, unread: true } : row))} />)
    expect(titles()).toContain('Chat 4')
  })

  it('lists every match while searching, with no Show all row', () => {
    render(<AgentChats searching />)
    expect(titles()).toHaveLength(6)
    expect(screen.queryByRole('button', { name: en.simplifiedAgents.showAllChats(6) })).toBeNull()
  })

  it('offers no Show all row when everything fits', () => {
    render(<AgentChats sessions={chats.slice(0, 3)} />)
    expect(screen.queryByRole('button', { name: /Show all/ })).toBeNull()
  })
})
