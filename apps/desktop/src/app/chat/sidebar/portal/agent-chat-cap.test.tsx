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

import { AGENT_CHAT_CAP, agentChatCount, capAgentChats, computeAgentChatLimits } from './agent-chat-cap'

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

  it('bounds soft-kept unread rows so unread bursts never swallow sibling agents', () => {
    expect(
      capAgentChats([1, 2, 3, 4, 5, 6, 7, 8], 3, () => false, {
        maxSoftKeep: 2,
        softKeep: row => row >= 4
      })
    ).toEqual([1, 2, 3, 4, 5])
  })
})

describe('computeAgentChatLimits', () => {
  it('falls back to AGENT_CHAT_CAP when height is unmeasured', () => {
    expect(
      computeAgentChatLimits({
        availableHeight: 0,
        collapsedAgentCount: 0,
        groupRows: 0,
        machineCount: 1,
        openAgents: [{ key: 'a', count: 12 }],
        pinnedRows: 0
      })
    ).toEqual({ a: AGENT_CHAT_CAP })
  })

  it('water-fills a single agent across the entire tall sidebar with no arbitrary ceiling', () => {
    const limits = computeAgentChatLimits({
      availableHeight: 1100,
      collapsedAgentCount: 0,
      groupRows: 0,
      machineCount: 1,
      openAgents: [{ key: 'a', count: 30, active: true }],
      pinnedRows: 0
    })
    expect(limits.a).toBeGreaterThan(20)
  })

  it('keeps sibling agents above the fold by accounting for real card chrome and Bot Screen rows', () => {
    const withBotScreens = computeAgentChatLimits({
      availableHeight: 820,
      botScreenRows: 3,
      collapsedAgentCount: 0,
      groupRows: 0,
      machineCount: 1,
      openAgents: [
        { key: 'default', count: 25, active: true },
        { key: 'ops', count: 12 },
        { key: 'scout', count: 8 }
      ],
      pinnedRows: 0
    })
    expect(withBotScreens).toEqual({ default: AGENT_CHAT_CAP, ops: AGENT_CHAT_CAP, scout: AGENT_CHAT_CAP })

    const roomyNoScreens = computeAgentChatLimits({
      availableHeight: 900,
      collapsedAgentCount: 0,
      groupRows: 0,
      machineCount: 1,
      openAgents: [
        { key: 'small', count: 1 },
        { key: 'busy', count: 25, active: true },
        { key: 'other', count: 18 }
      ],
      pinnedRows: 0
    })
    // Water-fills spare rows round-robin across busy + other without overflowing the 900px budget
    expect(roomyNoScreens.busy).toBeGreaterThanOrEqual(roomyNoScreens.other)
    expect(roomyNoScreens.busy + roomyNoScreens.other).toBeLessThanOrEqual(16)

    const tight = computeAgentChatLimits({
      availableHeight: 280,
      collapsedAgentCount: 0,
      groupRows: 0,
      machineCount: 1,
      openAgents: [
        { key: 'a', count: 10 },
        { key: 'b', count: 10 }
      ],
      pinnedRows: 2
    })
    expect(tight).toEqual({ a: AGENT_CHAT_CAP, b: AGENT_CHAT_CAP })
  })

  it('never hides a single lone overflow chat behind a Show all button', () => {
    const limits = computeAgentChatLimits({
      availableHeight: 280,
      collapsedAgentCount: 0,
      groupRows: 0,
      machineCount: 1,
      openAgents: [{ key: 'a', count: AGENT_CHAT_CAP + 1 }],
      pinnedRows: 2
    })
    expect(limits.a).toBe(AGENT_CHAT_CAP + 1)
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

  it('offers no Show all row when the chats it keeps in reach are already every chat', () => {
    const rows = chats.slice(0, 4).map((row, index) => (index === 3 ? { ...row, unread: true } : row))

    render(<AgentChats sessions={rows} />)
    expect(titles()).toHaveLength(4)
    expect(screen.queryByRole('button', { name: /Show all/ })).toBeNull()
  })

  it('offers no Show all row when everything fits', () => {
    render(<AgentChats sessions={chats.slice(0, 3)} />)
    expect(screen.queryByRole('button', { name: /Show all/ })).toBeNull()
  })

  it('steps +10 chats at a time with paired Show less and snaps to all when <= 12 remain', () => {
    const manyChats: SessionInfo[] = Array.from({ length: 24 }, (_, index) =>
      makeSessionInfo({ id: `chat-${index}`, connection_id: 'local', profile: 'default', title: `Chat ${index}` })
    )

    function SteppedAgentChats() {
      const [extra, setExtra] = useState(0)

      return (
        <MemoryRouter>
          <StableNavigateProvider>
            <RootTooltipProvider>
              <SimplifiedSessionList
                activeSessionId={null}
                ambiguousIds={new Set()}
                cap={{
                  extra,
                  limit: 3,
                  showAll: false,
                  onShowLess: () => setExtra(0),
                  onShowMore: () =>
                    setExtra(current => {
                      const remaining = Math.max(0, manyChats.length - (3 + current))

                      return current + (remaining <= 12 ? remaining : 10)
                    }),
                  onToggle: () => setExtra(current => (current > 0 ? 0 : manyChats.length))
                }}
                onArchiveSession={noop}
                onBranchSession={noop}
                onDeleteSession={noop}
                onResumeSession={noop}
                onUpdate={noop}
                pinnedIds={new Set()}
                searching={false}
                sessions={manyChats}
              />
            </RootTooltipProvider>
          </StableNavigateProvider>
        </MemoryRouter>
      )
    }

    render(<SteppedAgentChats />)
    expect(screen.queryAllByText(/^Chat \d+$/)).toHaveLength(3)
    expect(screen.queryByRole('button', { name: en.simplifiedAgents.showFewerChats })).toBeNull()

    // Step 1: 3 -> 13 (21 hidden -> > 12 -> stepped "+10")
    fireEvent.click(screen.getByRole('button', { name: en.simplifiedAgents.showMoreChats(10, 21) }))
    expect(screen.queryAllByText(/^Chat \d+$/)).toHaveLength(13)
    // Paired Show less is immediately available without dumping all 24 chats
    expect(screen.getByRole('button', { name: en.simplifiedAgents.showFewerChats })).toBeTruthy()

    // Step 2: 11 remain (<= 12) -> snaps straight to Show all (24)
    fireEvent.click(screen.getByRole('button', { name: en.simplifiedAgents.showAllChats(24) }))
    expect(screen.queryAllByText(/^Chat \d+$/)).toHaveLength(24)

    // Show less collapses cleanly back to 3
    fireEvent.click(screen.getByRole('button', { name: en.simplifiedAgents.showFewerChats }))
    expect(screen.queryAllByText(/^Chat \d+$/)).toHaveLength(3)
  })
})

describe('agentChatCount', () => {
  it('states the loaded chats when the list holds all of them', () => {
    expect(agentChatCount(134, false)).toEqual({ value: 134, more: false })
  })

  it('states the profile\'s own count when the backend sends one, the chats past the page included', () => {
    expect(agentChatCount(134, true, 1873)).toEqual({ value: 1873, more: false })
    // A chat listed before the count catches up is still counted.
    expect(agentChatCount(4, false, 3)).toEqual({ value: 4, more: false })
  })

  it('states a floor that a new chat does not move when more chats exist than are loaded', () => {
    expect(agentChatCount(134, true)).toEqual({ value: 100, more: true })
    expect(agentChatCount(135, true)).toEqual(agentChatCount(134, true))
  })
})
