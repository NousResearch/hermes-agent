// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { StableNavigateProvider } from '@/app/stable-navigate'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { $portalPinnedSectionOpen } from '@/store/agent-sidebar-preferences'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $selectedStoredSessionId, $sessions } from '@/store/session'
import { makeSessionInfo } from '@/test/session-info'

import { SimplifiedPinnedSessions } from './simplified-pinned-sessions'
import { sessionRowKey } from './simplified-session-identity'
import { SimplifiedSessionList } from './simplified-session-list'

const noop = () => {}

const chat = (index: number) =>
  makeSessionInfo({ id: `chat-${index}`, connection_id: 'local', profile: 'default', title: `Chat ${index}` })

const chats: SessionInfo[] = [chat(0), chat(1), chat(2)]

// jsdom lays nothing out: Pinned takes the top 100px, the agent's rows 30px each from 200 down.
const PINNED = { left: 0, top: 0, right: 300, bottom: 100 }
const ROW_TOP = 200
const ROW_HEIGHT = 30

const rect = (box: { left: number; top: number; right: number; bottom: number }) =>
  ({ ...box, width: box.right - box.left, height: box.bottom - box.top, x: box.left, y: box.top }) as DOMRect

function layOut() {
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
    if (this.matches('[data-slot="pinned-section"]')) {
      return rect(PINNED)
    }

    const key = this.closest('.portal-agent-sessions')?.contains(this) && this.getAttribute('data-session-key')
    const index = key ? chats.findIndex(row => sessionRowKey(row) === key) : -1

    if (index < 0) {
      return rect({ left: 0, top: 0, right: 0, bottom: 0 })
    }

    const top = ROW_TOP + index * ROW_HEIGHT

    return rect({ left: 0, top, right: 300, bottom: top + ROW_HEIGHT })
  })
}

interface SidebarProps {
  ambiguousIds?: ReadonlySet<string>
  onReorder?: (rows: SessionInfo[]) => void
  onUpdate: (row: SessionInfo, field: 'pinned' | 'unread', value: boolean) => void
  pinned?: SessionInfo[]
  pinPending?: (row: SessionInfo) => boolean
}

/** The Pinned section over one agent's chats, as the redesigned sidebar stacks them. */
function Sidebar({ ambiguousIds = new Set(), onReorder = noop, onUpdate, pinned = [], pinPending }: SidebarProps) {
  const shared = {
    activeSessionId: null,
    ambiguousIds,
    onArchiveSession: noop,
    onBranchSession: noop,
    onDeleteSession: noop,
    onResumeSession: noop,
    onUpdate,
    pinnedIds: new Set(pinned.map(row => sessionRowKey(row))),
    pinPending
  }

  return (
    <MemoryRouter>
      <StableNavigateProvider>
        <RootTooltipProvider>
          <SimplifiedPinnedSessions {...shared} sessions={pinned} />
          <div className="portal-agent-sessions">
            <SimplifiedSessionList {...shared} onReorder={onReorder} sessions={chats} />
          </div>
        </RootTooltipProvider>
      </StableNavigateProvider>
    </MemoryRouter>
  )
}

const agentRow = (title: string) =>
  screen
    .getAllByRole('button', { name: title })
    .map(button => button.closest<HTMLElement>('.portal-session-row')!)
    .find(row => row.closest('.portal-agent-sessions'))!

const doc = window.document
const pointer = (clientX: number, clientY: number) => ({ clientX, clientY, isPrimary: true, pointerId: 1 })
const frame = () => new Promise<void>(resolve => requestAnimationFrame(() => resolve()))

/** Press a row and carry it to (x, y). */
async function carry(row: HTMLElement, x: number, y: number) {
  const top = row.parentElement!.getBoundingClientRect().top + 10

  // The first move past the threshold lifts the row; the list measures itself before the next.
  await act(async () => {
    fireEvent.pointerDown(row, { ...pointer(20, top), button: 0, pointerType: 'mouse' })
    fireEvent.pointerMove(doc, pointer(20, top + 10))
    await frame()
  })
  await act(async () => {
    fireEvent.pointerMove(doc, pointer(x, y))
    await frame()
  })
}

async function letGo(x: number, y: number) {
  await act(async () => {
    fireEvent.pointerUp(doc, pointer(x, y))
    await Promise.resolve()
  })
}

/** Press a row, carry it to (x, y) and let go there. */
async function drag(row: HTMLElement, x: number, y: number) {
  await carry(row, x, y)
  await letGo(x, y)
}

beforeEach(() => {
  $apiRequestScope.set({ connectionId: 'local', profile: 'default' })
  $activeGatewayProfile.set('default')
  $selectedStoredSessionId.set(null)
  $activeSessionId.set(null)
  $sessions.set([])
  $portalPinnedSectionOpen.set(true)
  layOut()
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

it('pins a chat dropped on Pinned, the way its menu does, and leaves the agent’s order alone', async () => {
  const onUpdate = vi.fn()
  const onReorder = vi.fn()

  render(<Sidebar onReorder={onReorder} onUpdate={onUpdate} />)
  await drag(agentRow('Chat 2'), 150, 50)

  expect(onUpdate).toHaveBeenCalledTimes(1)
  expect(onUpdate).toHaveBeenCalledWith(chats[2], 'pinned', true)
  expect(onReorder).not.toHaveBeenCalled()
})

it('marks Pinned as the target only while the chat is held over it', async () => {
  render(<Sidebar onUpdate={noop} />)
  const section = doc.querySelector<HTMLElement>('[data-slot="pinned-section"]')!

  await carry(agentRow('Chat 2'), 150, 50)
  expect(section.getAttribute('data-session-drop-hover')).toBe('true')

  await letGo(150, 50)
  expect(section.hasAttribute('data-session-drop-hover')).toBe(false)
})

it('still reorders a chat dropped inside its own list', async () => {
  const onUpdate = vi.fn()
  const onReorder = vi.fn()

  render(<Sidebar onReorder={onReorder} onUpdate={onUpdate} />)
  await drag(agentRow('Chat 2'), 20, ROW_TOP + 5)

  expect(onUpdate).not.toHaveBeenCalled()
  expect(onReorder.mock.calls[0][0].map((row: SessionInfo) => row.id)).toEqual(['chat-2', 'chat-0', 'chat-1'])
})

it('pins nothing when the chat is let go anywhere else', async () => {
  const onUpdate = vi.fn()

  render(<Sidebar onUpdate={onUpdate} />)
  await drag(agentRow('Chat 1'), 600, 400)

  expect(onUpdate).not.toHaveBeenCalled()
})

it('calls the whole drag off with Esc: the release then neither pins nor rearranges', async () => {
  const onUpdate = vi.fn()
  const onReorder = vi.fn()

  render(<Sidebar onReorder={onReorder} onUpdate={onUpdate} />)
  await carry(agentRow('Chat 2'), 150, 50)
  fireEvent.keyDown(doc.body, { code: 'Escape', key: 'Escape' })
  await letGo(150, 50)

  expect(onUpdate).not.toHaveBeenCalled()
  expect(onReorder).not.toHaveBeenCalled()

  // The next drag is a fresh one.
  await drag(agentRow('Chat 2'), 20, ROW_TOP + 5)
  expect(onReorder).toHaveBeenCalledTimes(1)
})

it('offers no pin drop to a chat that is pinned already, mid-write, or has no menu to pin it from', async () => {
  const onUpdate = vi.fn()

  render(
    <Sidebar
      ambiguousIds={new Set(['chat-2'])}
      onUpdate={onUpdate}
      pinned={[chats[0]]}
      pinPending={row => row.id === 'chat-1'}
    />
  )

  for (const title of ['Chat 0', 'Chat 1', 'Chat 2']) {
    await drag(agentRow(title), 150, 50)
  }

  expect(onUpdate).not.toHaveBeenCalled()
})
