import { KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core'
import { sortableKeyboardCoordinates } from '@dnd-kit/sortable'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { startSessionDrag } from '@/app/chat/session-drag'
import type * as SessionDrag from '@/app/chat/session-drag'
import type { SessionInfo } from '@/hermes'
import type * as ChatRuntime from '@/lib/chat-runtime'
import type * as GatewayStore from '@/store/gateway'
import type * as ProjectsStore from '@/store/projects'
import type * as SessionStore from '@/store/session'
import type * as SessionColorStore from '@/store/session-color'
import type * as SessionStatesStore from '@/store/session-states'
import type * as WindowsStore from '@/store/windows'

import { ReorderableList, useSortableBindings } from './reorderable-list'
import { SidebarSessionRow } from './session-row'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

// The MOUSE-side sibling of the #83617 regression covered in
// session-row.test.tsx. The row renders its ⋯ menu (and through it the rename
// dialog) INSIDE its own React subtree, and DialogContent portals into <body>.
// React re-dispatches an event fired in a portal along the REACT tree, so a
// pointerdown inside the dialog's input still reaches the row shell's own
// onPointerDown with a `target` that is NOT a DOM descendant of the row —
// selecting the title with the mouse then lifted the row onto the shared drag
// session (ghost = the session title, every zone lit as a drop target, and a
// release over the composer inserts an @session chip) while also arming the
// dnd-kit reorder.
//
// The menu and the dialog are the REAL components here (no stub): the press
// really does travel through a portal. Only the drag sink and the sinks the
// row/menu reach for on open are mocked.
vi.mock('@/app/chat/session-drag', async importOriginal => {
  const actual = await importOriginal<typeof SessionDrag>()

  // Spy-WRAPPER, not a stub: the assertions count the calls, while the real
  // session still runs so the engage chrome a user sees (the grabbing cursor,
  // the row's lifted look) is exercised rather than assumed.
  return { ...actual, startSessionDrag: vi.fn(actual.startSessionDrag) }
})
vi.mock('@/components/pane-shell/tree/store', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return {
    ...actual,
    closeAllTreeTabs: vi.fn(),
    closeOtherTreeTabs: vi.fn(),
    closeTreeTabsToRight: vi.fn(),
    treeTabCloseTargets: vi.fn(() => null)
  }
})
vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return { ...actual, renameSession: vi.fn() }
})
vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      assistant: {
        thread: {
          today: (time: string) => `Today at ${time}`,
          yesterday: (time: string) => `Yesterday at ${time}`
        }
      },
      common: {
        cancel: 'Cancel',
        close: 'Close',
        confirm: 'Confirm',
        delete: 'Delete',
        done: 'Done',
        loading: 'Loading…',
        save: 'Save'
      },
      errors: { genericFailure: 'Something went wrong' },
      sidebar: {
        messageCount: (count: number) => `${count} messages`,
        projects: {
          home: 'Home',
          menuAppearance: 'Appearance',
          moveFailed: 'Could not move session',
          moveNoProjects: 'No other projects',
          movedTo: (name: string) => `Moved to ${name}`,
          moveToProject: 'Move to project',
          noColor: 'No color'
        },
        row: {
          ageMin: 'm',
          ageNow: 'now',
          archive: 'Archive',
          backgroundRunning: 'Running in background',
          branchFrom: 'Branch from here',
          copyId: 'Copy ID',
          copyIdFailed: 'Failed to copy ID',
          deleteDesc: (title: string) => `Delete ${title}?`,
          deleteTitle: 'Delete session?',
          deleted: 'Session deleted',
          deleting: 'Deleting…',
          export: 'Export',
          finishedUnread: 'Finished',
          handoffOrigin: (platform: string) => `Started on ${platform}`,
          hideTabBar: 'Hide tab bar',
          markRead: 'Mark as read',
          messageCount: (count: number) => `${count} messages`,
          needsInput: 'Needs input',
          pin: 'Pin',
          rename: 'Rename',
          renamed: 'Renamed',
          renameFailed: 'Rename failed',
          renameTitle: 'Rename session',
          sessionActions: 'Session actions',
          sessionRunning: 'Running',
          todoProgress: 'Tasks completed',
          unpin: 'Unpin',
          untitledPlaceholder: 'Untitled',
          waitingForAnswer: 'Waiting for answer'
        },
        toolCallCount: (count: number) => `${count} tool calls`
      },
      zones: { closeAll: 'Close all', closeOthers: 'Close others', closeToRight: 'Close to the right' }
    }
  })
}))
vi.mock('@/lib/haptics', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return { ...actual, triggerHaptic: vi.fn() }
})
vi.mock('@/lib/profile-color', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return { ...actual, PROFILE_SWATCHES: [] }
})
vi.mock('@/lib/session-export', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return { ...actual, exportSession: vi.fn() }
})
vi.mock('@/lib/session-source', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return {
    ...actual,
    handoffOriginSource: (state?: string, platform?: string) => (state && platform ? platform : null),
    sessionSourceLabel: (source: string) => source
  }
})
vi.mock('@/lib/time', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return { ...actual, coarseElapsed: () => ({ unit: 'minute' as const, value: 5 }) }
})

// importOriginal + named overrides (never a wholesale replacement): both the
// row and the menu read more of these stores than this file names, and an
// unlisted export silently becomes `undefined` and crashes a nanostores
// `computed()` downstream.
vi.mock('@/lib/chat-runtime', async importOriginal => {
  const actual = await importOriginal<typeof ChatRuntime>()

  return { ...actual, sessionTitle: (s: SessionInfo) => (s as unknown as { title: string }).title }
})
vi.mock('@/store/gateway', async importOriginal => {
  const actual = await importOriginal<typeof GatewayStore>()

  return { ...actual, activeGateway: vi.fn(() => null) }
})
vi.mock('@/store/notifications', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return { ...actual, notify: vi.fn(), notifyError: vi.fn() }
})
vi.mock('@/store/projects', async importOriginal => {
  const actual = await importOriginal<typeof ProjectsStore>()

  return { ...actual, $projectTree: atom<unknown[]>([]) }
})
vi.mock('@/store/session', async importOriginal => {
  const actual = await importOriginal<typeof SessionStore>()

  return { ...actual, $unreadFinishedSessionIds: atom<string[]>([]) }
})
vi.mock('@/store/session-color', async importOriginal => {
  const actual = await importOriginal<typeof SessionColorStore>()

  return { ...actual, $sessionColorOverrides: atom<Record<string, string>>({}) }
})
vi.mock('@/store/session-states', async importOriginal => {
  const actual = await importOriginal<typeof SessionStatesStore>()

  return {
    ...actual,
    $attentionSessionIds: atom<string[]>([]),
    $sessionTiles: atom<unknown[]>([]),
    $stalledSessionIds: atom<string[]>([]),
    openSessionTile: vi.fn()
  }
})
vi.mock('@/store/windows', async importOriginal => {
  const actual = await importOriginal<typeof WindowsStore>()

  return {
    ...actual,
    canOpenSessionInTerminal: () => false,
    canOpenSessionWindow: () => false,
    openSessionInNewWindow: vi.fn(),
    openSessionInTerminal: vi.fn()
  }
})

function makeSession(overrides: Partial<SessionInfo> & { title: string }): SessionInfo {
  return {
    handoff_platform: null,
    handoff_state: null,
    id: 's1',
    last_active: 0,
    profile: 'default',
    started_at: 0,
    ...overrides
  } as unknown as SessionInfo
}

const noop = vi.fn()

function SortableRow({ session }: { session: SessionInfo }) {
  const { dragHandleProps, dragging, ref, reorderable, style } = useSortableBindings(session.id)

  return (
    <SidebarSessionRow
      dragging={dragging}
      dragHandleProps={dragHandleProps}
      isPinned={false}
      isSelected={false}
      onArchive={noop}
      onDelete={noop}
      onPin={noop}
      onResume={noop}
      onToggleUnread={noop}
      ref={ref}
      reorderable={reorderable}
      session={session}
      style={style}
      unread={false}
    />
  )
}

function Host({ session }: { session: SessionInfo }) {
  // The sidebar's own sensor set (index.tsx dndSensors).
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  )

  return (
    <ReorderableList ids={[session.id]} onReorder={noop} sensors={sensors}>
      <SortableRow session={session} />
    </ReorderableList>
  )
}

/** Open the row's ⋯ menu and click Rename — the real dialog, opened the real way. */
async function openRenameDialog() {
  const trigger = screen.getByRole('button', { name: 'Session actions' })

  // Radix's dropdown trigger opens on pointerdown, not on a bare click.
  fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.click(trigger)

  fireEvent.click(await screen.findByRole('menuitem', { name: /rename/i }))

  const dialog = await screen.findByRole('dialog')

  return within(dialog).getByRole('textbox')
}

// drag session coalesces its moves into that frame (drag-session.ts: onMove ->
// rAF -> processMove -> engage). Stub it with a 16ms timer so the engage the
// user actually sees (the grabbing cursor, the lifted row) happens here too.
beforeAll(() => {
  vi.stubGlobal(
    'requestAnimationFrame',
    (callback: FrameRequestCallback) => setTimeout(() => callback(Date.now()), 16) as unknown as number
  )
  vi.stubGlobal('cancelAnimationFrame', (handle: number) => clearTimeout(handle))
})

afterAll(() => vi.unstubAllGlobals())

/** Both rAF (now a timer) and the drag session's own bookkeeping have run. */
const nextFrame = () => new Promise(resolve => setTimeout(resolve, 48))

const rowShell = (container: HTMLElement) => container.querySelector<HTMLElement>('.row-hover')!

/** What the drag session paints on engage — the row's lifted look. */
const LIFTED_OPACITY = '0.45'

describe('SidebarSessionRow drag and the rename dialog (portal press)', () => {
  it("ignores the dialog's own presses, so selecting the title cannot lift the row", async () => {
    const { container } = render(<Host session={makeSession({ title: 'Renamable' })} />)
    const input = await openRenameDialog()
    const row = rowShell(container)
    const doc = container.ownerDocument

    // Opening the menu and the dialog is itself clean: the ⋯ press lands on the
    // row's own data-row-actions cluster, and the menu-item click starts nothing.
    expect(startSessionDrag).not.toHaveBeenCalled()

    // A mouse text-selection drag: press inside the dialog's input (a portal in
    // <body>) and move past both thresholds — the drag session's own 4px and
    // dnd-kit's 6px activation distance.
    fireEvent.pointerDown(input, {
      button: 0,
      clientX: 20,
      clientY: 20,
      isPrimary: true,
      pointerId: 1,
      pointerType: 'mouse'
    })
    await act(async () => {
      fireEvent.pointerMove(doc, { clientX: 80, clientY: 20, isPrimary: true, pointerId: 1 })
      await nextFrame()
    })

    expect(startSessionDrag).not.toHaveBeenCalled()
    expect(row.style.opacity).not.toBe(LIFTED_OPACITY)
    expect(container.querySelector('[data-glass-opaque]')).toBeNull()

    fireEvent.pointerUp(doc, { clientX: 80, clientY: 20, isPrimary: true, pointerId: 1 })

    // The modal backdrop portals out of the same React subtree: pressing it and
    // moving must not lift the row either.
    const overlay = doc.querySelector<HTMLElement>('[data-slot="dialog-overlay"]')!
    fireEvent.pointerDown(overlay, {
      button: 0,
      clientX: 10,
      clientY: 10,
      isPrimary: true,
      pointerId: 2,
      pointerType: 'mouse'
    })
    await act(async () => {
      fireEvent.pointerMove(doc, { clientX: 90, clientY: 10, isPrimary: true, pointerId: 2 })
      await nextFrame()
    })

    expect(startSessionDrag).not.toHaveBeenCalled()
    expect(row.style.opacity).not.toBe(LIFTED_OPACITY)

    fireEvent.pointerUp(doc, { clientX: 90, clientY: 10, isPrimary: true, pointerId: 2 })

    // Leave no modal behind in the document for the next test.
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  })

  it('still lifts the row from a press on its own body', async () => {
    const { container } = render(<Host session={makeSession({ title: 'Draggable' })} />)
    const row = rowShell(container)
    const doc = container.ownerDocument
    const body = screen.getByText('Draggable').closest('button')!

    fireEvent.pointerDown(body, {
      button: 0,
      clientX: 20,
      clientY: 100,
      isPrimary: true,
      pointerId: 3,
      pointerType: 'mouse'
    })
    await act(async () => {
      fireEvent.pointerMove(doc, { clientX: 20, clientY: 160, isPrimary: true, pointerId: 3 })
      await nextFrame()
    })

    // The row's OWN press still runs BOTH drags off one gesture: the shared
    // session engages (the row takes the lifted look the user sees) and the
    // dnd-kit reorder arms (the row goes opaque, the grabber reports pressed).
    expect(startSessionDrag).toHaveBeenCalledTimes(1)
    expect(row.style.opacity).toBe(LIFTED_OPACITY)
    expect(row.getAttribute('data-glass-opaque')).not.toBeNull()
    expect(container.querySelector('[data-reorder-handle][aria-pressed="true"]')).not.toBeNull()

    fireEvent.pointerUp(doc, { clientX: 20, clientY: 160, isPrimary: true, pointerId: 3 })
  })
})
