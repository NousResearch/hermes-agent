// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { StableNavigateProvider } from '@/app/stable-navigate'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import {
  $portalAgentChatOrder,
  agentSidebarKey,
  dropAgentSidebarPreferences,
  migrateAgentSidebarPreferences,
  setAgentChatOrder
} from '@/store/agent-sidebar-preferences'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $selectedStoredSessionId, $sessions } from '@/store/session'
import { makeSessionInfo } from '@/test/session-info'

import { sessionRowKey } from '../simplified-session-identity'
import { SimplifiedSessionList } from '../simplified-session-list'

import { arrangeAgentChats, orderAgentChats } from './agent-chat-order'

const noop = () => {}

// Most recent first, as the sidebar hands an agent's chats over: chat-0 was active at 400s.
const chat = (index: number, lastActive: number) =>
  makeSessionInfo({
    id: `chat-${index}`,
    connection_id: 'local',
    profile: 'default',
    title: `Chat ${index}`,
    last_active: lastActive
  })

const chats: SessionInfo[] = [chat(0, 400), chat(1, 300), chat(2, 200), chat(3, 100)]
const ids = (rows: SessionInfo[]) => rows.map(row => row.id)

function List({
  onReorder,
  pinnedIds = new Set(),
  searching = false
}: {
  onReorder: (rows: SessionInfo[]) => void
  pinnedIds?: ReadonlySet<string>
  searching?: boolean
}) {
  return (
    <MemoryRouter>
      <StableNavigateProvider>
        <RootTooltipProvider>
          <SimplifiedSessionList
            activeSessionId={null}
            ambiguousIds={new Set()}
            cap={{ limit: 3, onToggle: noop, showAll: false }}
            onArchiveSession={noop}
            onBranchSession={noop}
            onDeleteSession={noop}
            onReorder={onReorder}
            onResumeSession={noop}
            onUpdate={noop}
            pinnedIds={pinnedIds}
            searching={searching}
            sessions={chats}
          />
        </RootTooltipProvider>
      </StableNavigateProvider>
    </MemoryRouter>
  )
}

beforeEach(() => {
  $apiRequestScope.set({ connectionId: 'local', profile: 'default' })
  $activeGatewayProfile.set('default')
  $selectedStoredSessionId.set(null)
  $activeSessionId.set(null)
  $sessions.set([])
  $portalAgentChatOrder.set({})
})

afterEach(cleanup)

describe('orderAgentChats', () => {
  it('leaves recency alone until the user arranges the chats', () => {
    expect(orderAgentChats(chats, undefined)).toBe(chats)
  })

  it('holds the arrangement for the chats it names', () => {
    const order = { at: 400_000, ids: ['chat-2', 'chat-0', 'chat-3', 'chat-1'] }

    expect(ids(orderAgentChats(chats, order))).toEqual(order.ids)
  })

  it('lifts a chat with news since the arrangement above it, and sinks chats it never named', () => {
    const order = { at: 350_000, ids: ['chat-2', 'chat-0', 'chat-1'] }

    // chat-0 (400s) was active after the arrangement; chat-3 loaded later and was never arranged.
    expect(ids(orderAgentChats(chats, order))).toEqual(['chat-0', 'chat-2', 'chat-1', 'chat-3'])
  })

  it('stamps an arrangement no earlier than its latest chat, so a fast gateway clock cannot undo it', () => {
    const arranged = [chats[2], chats[0], chats[1], chats[3]]
    const order = arrangeAgentChats(arranged, 1_000)

    expect(ids(orderAgentChats(chats, order))).toEqual(ids(arranged))
  })
})

describe('reordering a chat list', () => {
  it('moves a row a place with Alt+ArrowDown and leaves rows past the cap where they were', () => {
    const onReorder = vi.fn()

    render(<List onReorder={onReorder} />)
    fireEvent.keyDown(screen.getByRole('button', { name: 'Chat 0' }), { altKey: true, key: 'ArrowDown' })

    expect(ids(onReorder.mock.calls[0][0])).toEqual(['chat-1', 'chat-0', 'chat-2', 'chat-3'])
  })

  it('does nothing at the end of the list or without Alt', () => {
    const onReorder = vi.fn()

    render(<List onReorder={onReorder} />)
    fireEvent.keyDown(screen.getByRole('button', { name: 'Chat 0' }), { altKey: true, key: 'ArrowUp' })
    fireEvent.keyDown(screen.getByRole('button', { name: 'Chat 0' }), { key: 'ArrowDown' })

    expect(onReorder).not.toHaveBeenCalled()
  })

  it('keeps a move inside its pinned or unpinned run, so the pinned-first order never undoes it', () => {
    const onReorder = vi.fn()

    // chat-2 is pinned: it leads the list, then chat-0 and chat-1 (chat-3 is past the cap).
    render(<List onReorder={onReorder} pinnedIds={new Set([sessionRowKey(chats[2])])} />)
    fireEvent.keyDown(screen.getByRole('button', { name: 'Chat 0' }), { altKey: true, key: 'ArrowUp' })
    fireEvent.keyDown(screen.getByRole('button', { name: 'Chat 2' }), { altKey: true, key: 'ArrowDown' })
    expect(onReorder).not.toHaveBeenCalled()

    fireEvent.keyDown(screen.getByRole('button', { name: 'Chat 1' }), { altKey: true, key: 'ArrowUp' })
    expect(ids(onReorder.mock.calls[0][0])).toEqual(['chat-2', 'chat-1', 'chat-0', 'chat-3'])
  })

  it('never rearranges a search result', () => {
    const onReorder = vi.fn()

    render(<List onReorder={onReorder} searching />)
    fireEvent.keyDown(screen.getByRole('button', { name: 'Chat 0' }), { altKey: true, key: 'ArrowDown' })

    expect(onReorder).not.toHaveBeenCalled()
  })
})

describe('an agent’s chat order as a preference', () => {
  const owner = { connectionId: null, profile: 'research' }
  const order = { at: 1, ids: ['a', 'b'] }

  it('follows the agent through a rename and goes when the agent does', () => {
    setAgentChatOrder(owner, order)
    setAgentChatOrder({ connectionId: 'remote', profile: 'research' }, order)
    migrateAgentSidebarPreferences(owner, 'renamed')

    const renamed = { connectionId: null, profile: 'renamed' }

    expect($portalAgentChatOrder.get()).toEqual({
      [agentSidebarKey(renamed)]: order,
      [agentSidebarKey({ connectionId: 'remote', profile: 'research' })]: order
    })

    dropAgentSidebarPreferences(renamed)
    expect(Object.keys($portalAgentChatOrder.get())).toEqual([
      agentSidebarKey({ connectionId: 'remote', profile: 'research' })
    ])
  })
})
