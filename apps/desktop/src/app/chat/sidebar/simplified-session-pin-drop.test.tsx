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

import { startSessionDrag } from '../session-drag'

import { SimplifiedPinnedSessions } from './simplified-pinned-sessions'
import { sessionRowKey } from './simplified-session-identity'
import { SimplifiedSessionList } from './simplified-session-list'

// The shared session drag is spied, never run: a press must arm it (HADESK-167), while dnd-kit's
// own drag stays real for the drop tests. The real module is kept for the rest — the drop selectors.
// The stand-in does the one thing the session does to Esc: takes the key with a window CAPTURE
// listener that stops it before dnd-kit's own document listener hears it — and tears that listener
// down there and then, the synchronous teardown the real Esc-as-top-layer abort performs. The row's
// own forwarder re-dispatches the key a microtask later, once this listener is gone.
vi.mock('../session-drag', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  startSessionDrag: vi.fn(() => {
    const stop = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation()
        window.removeEventListener('keydown', stop, true)
      }
    }

    window.addEventListener('keydown', stop, true)
    window.addEventListener('pointerup', () => window.removeEventListener('keydown', stop, true), {
      capture: true,
      once: true
    })
  })
}))

const noop = () => {}
const startDrag = vi.mocked(startSessionDrag)

const chat = (index: number) =>
  makeSessionInfo({ id: `chat-${index}`, connection_id: 'local', profile: 'default', title: `Chat ${index}` })

const chats: SessionInfo[] = [chat(0), chat(1), chat(2)]

// jsdom lays nothing out: Pinned takes the top 100px, the Agents section — the unpin target every
// non-pinned list lives in — the band under it, and the agent's rows 30px each from 200 down.
const PINNED = { left: 0, top: 0, right: 300, bottom: 100 }
const AGENTS = { left: 0, top: 200, right: 300, bottom: 400 }
const ROW_TOP = 200
const ROW_HEIGHT = 30

const rect = (box: { left: number; top: number; right: number; bottom: number }) =>
  ({ ...box, width: box.right - box.left, height: box.bottom - box.top, x: box.left, y: box.top }) as DOMRect

// jsdom has no hit testing either: a point inside Pinned's box hits Pinned, one inside the Agents
// section hits it.
const inside = (box: typeof PINNED, x: number, y: number) =>
  x >= box.left && x <= box.right && y >= box.top && y <= box.bottom

function layOut() {
  window.document.elementsFromPoint = (x, y) =>
    inside(AGENTS, x, y)
      ? [...window.document.querySelectorAll('[data-session-unpin-drop]')]
      : inside(PINNED, x, y)
        ? [...window.document.querySelectorAll('[data-slot="pinned-section"]')]
        : []

  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
    if (this.matches('[data-session-unpin-drop]')) {
      return rect(AGENTS)
    }

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
          {/* Both lists arrange, as the redesigned sidebar's do: Pinned's own drag is what carries a
              chat out of it and lands it on the Agents section below. */}
          <SimplifiedPinnedSessions {...shared} onReorder={noop} sessions={pinned} />
          <section data-session-unpin-drop="" data-slot="agent-section">
            <div className="portal-agent-sessions">
              <SimplifiedSessionList {...shared} onReorder={onReorder} sessions={chats} />
            </div>
          </section>
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

const pinnedRow = (title: string) =>
  screen
    .getAllByRole('button', { name: title })
    .map(button => button.closest<HTMLElement>('.portal-session-row')!)
    .find(row => row.closest('[data-slot="pinned-section"]'))!

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
  startDrag.mockClear()
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

// Carried sideways out of the sidebar — toward a workspace pane to split the chat there — the row
// must no longer read as held over Pinned: the list hit-tests the pointer's real x, so the drop
// clears and the release neither pins nor unpins (a stale hover would commit pin AND split at once).
it('clears Pinned and pins nothing when a chat is carried sideways out of the sidebar', async () => {
  const onUpdate = vi.fn()

  render(<Sidebar onUpdate={onUpdate} />)
  const section = doc.querySelector<HTMLElement>('[data-slot="pinned-section"]')!

  await carry(agentRow('Chat 2'), 900, 50)
  expect(section.hasAttribute('data-session-drop-hover')).toBe(false)

  await letGo(900, 50)
  expect(onUpdate).not.toHaveBeenCalled()
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

it.each([
  ['Esc', () => fireEvent.keyDown(doc.body, { code: 'Escape', key: 'Escape' })],
  ['the pointer being taken away', () => fireEvent.pointerCancel(doc, pointer(150, 50))]
])('calls the drag off on %s: the chat is neither pinned nor moved', async (_, callOff) => {
  const onUpdate = vi.fn()
  const onReorder = vi.fn()

  render(<Sidebar onReorder={onReorder} onUpdate={onUpdate} />)
  await carry(agentRow('Chat 2'), 150, 50)
  await act(async () => {
    callOff()
    await frame()
  })
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

// Out of Pinned the same way: carried down onto the lists below, the chat is unpinned by its menu's
// own write — and by that one alone (the session drag leaves pinning to the list, `pins: false`).
it('unpins a pinned chat dropped on the lists below, the way its menu does', async () => {
  const onUpdate = vi.fn()

  render(<Sidebar onUpdate={onUpdate} pinned={[chats[0]]} />)
  await drag(pinnedRow('Chat 0'), 150, 250)

  expect(onUpdate).toHaveBeenCalledTimes(1)
  expect(onUpdate).toHaveBeenCalledWith(chats[0], 'pinned', false)
})

// The row is the handle of both drags at once (HADESK-167): the list's own reorder, and the shared
// session drag that carries a chat into a pane to split it there.
it('arms the session drag on a press, leaving the pin targets to the list’s own drag', async () => {
  render(<Sidebar onUpdate={noop} />)

  await act(async () => {
    fireEvent.pointerDown(agentRow('Chat 2'), { ...pointer(20, ROW_TOP + 10), button: 0, pointerType: 'mouse' })
  })
  await letGo(20, ROW_TOP + 10)

  expect(startDrag).toHaveBeenCalledTimes(1)
  // The connection-tagged row hands the session drag its exact owner, so the split's tile routes to
  // the backend that owns the chat rather than whatever profile is ambient at drop time (HIGH #3).
  expect(startDrag.mock.calls[0][0]).toEqual({
    id: 'chat-2',
    ownerRoute: { connectionId: 'local', profile: 'default' },
    profile: 'default',
    title: 'Chat 2'
  })
  // Pin · unpin stay with the list: a second resolver on this press would resolve one drop twice.
  expect(startDrag.mock.calls[0][2]).toEqual({ pins: false })
})

// A right click is not a grab: it arms neither the shared session drag nor the row's Esc forwarder.
it('leaves the session drag unarmed on a non-primary press', async () => {
  render(<Sidebar onUpdate={noop} />)

  await act(async () => {
    fireEvent.pointerDown(agentRow('Chat 2'), { ...pointer(20, ROW_TOP + 10), button: 2, pointerType: 'mouse' })
  })
  await letGo(20, ROW_TOP + 10)

  expect(startDrag).not.toHaveBeenCalled()
})

// A chat whose id is live in two profiles has no owner of its own to open (its menu stands down).
it('leaves a chat with no owner of its own to open its press to the list alone', async () => {
  render(<Sidebar ambiguousIds={new Set(['chat-2'])} onUpdate={noop} />)

  await act(async () => {
    fireEvent.pointerDown(agentRow('Chat 2'), { ...pointer(20, ROW_TOP + 10), button: 0, pointerType: 'mouse' })
  })
  await letGo(20, ROW_TOP + 10)

  expect(startDrag).not.toHaveBeenCalled()
})

// The session takes Esc before dnd-kit hears it, so the row hands the key on: the list's own drag is
// called off there and then — the chat is set back down and the target unmarked at once, rather than
// staying lifted until the pointer is let go (HADESK-58).
it('hands Esc to the list’s own drag, so a carried chat is set back down at once', async () => {
  const onUpdate = vi.fn()

  render(<Sidebar onUpdate={onUpdate} />)
  const row = agentRow('Chat 2')
  const section = doc.querySelector<HTMLElement>('[data-slot="pinned-section"]')!

  await carry(row, 150, 50)
  expect(section.getAttribute('data-session-drop-hover')).toBe('true')

  await act(async () => {
    fireEvent.keyDown(doc.body, { code: 'Escape', key: 'Escape' })
    await frame()
  })

  expect(row.parentElement!.hasAttribute('data-dragging')).toBe(false)
  expect(section.hasAttribute('data-session-drop-hover')).toBe(false)

  await letGo(150, 50)
  expect(onUpdate).not.toHaveBeenCalled()
})
