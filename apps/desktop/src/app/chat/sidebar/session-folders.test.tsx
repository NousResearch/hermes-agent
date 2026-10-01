import { KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core'
import { sortableKeyboardCoordinates, useSortable } from '@dnd-kit/sortable'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { $sidebarWorkspaceNodeOpen } from '@/store/layout'
import { clearNotifications } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $draggingSession,
  $sessionSections,
  createSessionSection,
  loadSessionSections,
  moveSessionsToSection,
  sessionSectionId,
  UNASSIGNED_SESSION_KEY
} from '@/store/session-sections'
import { makeSessionInfo } from '@/test/session-info'

import {
  $sessionFolderDialog,
  fileSessionFromDrop,
  SessionFolderDialog,
  sessionFolderDropId,
  sessionFolderDropTarget,
  SidebarSessionFolderList
} from './session-folders'
import { SidebarSessionsSection } from './sessions-section'

afterEach(() => {
  cleanup()
  // The delete toast schedules its own dismissal; leaving that timer armed
  // would hold the worker open for the whole 8s.
  clearNotifications()
  $sessionFolderDialog.set(null)
})

beforeEach(() => {
  window.localStorage.clear()
  $activeGatewayProfile.set('default')
  $sidebarWorkspaceNodeOpen.set({})
  loadSessionSections()
})

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel', delete: 'Delete', save: 'Save' },
      sidebar: {
        dateDivider: {
          earlierThisMonth: 'Earlier this month',
          lastMonth: 'Last month',
          lastWeek: 'Last week',
          older: 'Older',
          today: 'Today',
          yesterday: 'Yesterday'
        },
        folders: {
          create: 'Create',
          deleted: (label: string) => `Folder “${label}” deleted`,
          menu: 'Folder actions',
          menuRename: 'Rename…',
          nameLabel: 'Folder name',
          namePlaceholder: 'e.g. Billing work',
          newFolder: 'New folder',
          renameTitle: 'Rename folder',
          toggle: (label: string, open: boolean) => `${open ? 'Show' : 'Hide'} ${label} sessions`,
          undo: 'Undo'
        },
        statusDivider: { done: 'DONE', working: 'WORKING' }
      }
    }
  })
}))

// Rows are the sidebar's own concern (session-row.test.tsx proves them); this
// file is about where a row is DRAWN — under which folder heading — so the row
// is reduced to its identity.
vi.mock('./session-row', () => ({
  SidebarSessionRow: ({ session }: { session: SessionInfo }) => (
    <div data-testid={`session-row-${session.id}`}>{session.id}</div>
  )
}))

const SESSIONS = [
  makeSessionInfo({ id: 's1', last_active: 3_000, profile: 'default', started_at: 3_000 }),
  makeSessionInfo({ id: 's2', last_active: 2_000, profile: 'default', started_at: 2_000 }),
  makeSessionInfo({ id: 's3', last_active: 1_000, profile: 'default', started_at: 1_000 })
]

const noop = () => {}

/** The sidebar root's own shape: the section, plus the ONE name dialog it
 *  mounts beside it (index.tsx). */
function renderSection(sessions: SessionInfo[] = SESSIONS) {
  return render(
    <>
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={<div>Empty</div>}
        folders
        grouping="none"
        label="Sessions"
        onArchiveSession={noop}
        onDeleteSession={noop}
        onReorderSessions={noop}
        onResumeSession={noop}
        onToggle={noop}
        onTogglePin={noop}
        onToggleUnread={noop}
        open
        pinned={false}
        sessions={sessions}
        sortable
      />
      <SessionFolderDialog />
    </>
  )
}

/** The rendered block for a folder (null = the Unassigned bucket). */
const folderBlock = (root: ParentNode, sectionId: null | string): HTMLElement => {
  const block = root.querySelector<HTMLElement>(`[data-session-folder="${sectionId ?? UNASSIGNED_SESSION_KEY}"]`)

  if (!block) {
    throw new Error(`no folder block for ${sectionId ?? 'unassigned'}`)
  }

  return block
}

/** The ids of the rows drawn inside a block, in render order. */
const filedIds = (root: ParentNode, sectionId: null | string) =>
  [...folderBlock(root, sectionId).querySelectorAll('[data-testid^="session-row-"]')].map(
    node => node.getAttribute('data-testid')?.replace('session-row-', '') ?? ''
  )

/** Radix's dropdown trigger opens on pointerdown, so a synthetic click alone
 *  won't do it — the same sequence session-actions-menu.test.tsx uses. */
const openFolderMenu = (root: ParentNode, sectionId: string) => {
  const trigger = within(folderBlock(root, sectionId)).getByRole('button', { name: 'Folder actions' })

  fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.click(trigger)
}

describe('session folders', () => {
  it('draws the one flat list it always did while no folders are made', () => {
    const { container } = renderSection()

    // No folder chrome at all: no headings, no blocks, no extra level.
    expect(container.querySelectorAll('[data-session-folder]')).toHaveLength(0)
    expect(container.querySelectorAll('[data-session-folder-header]')).toHaveLength(0)

    // Every session still renders, as siblings in ONE list.
    const rows = SESSIONS.map(session => screen.getByTestId(`session-row-${session.id}`))

    expect(rows[0]?.parentElement).toBe(rows[1]?.parentElement)
    expect(rows[1]?.parentElement).toBe(rows[2]?.parentElement)
  })

  it('draws a filed session under its folder and out of the flat list', () => {
    const billing = createSessionSection('Billing work', ['s2'])!

    const { container } = renderSection()

    // s2 is drawn under Billing and nowhere else; s1 stays in the flat list.
    expect(filedIds(container, billing.id)).toContain('s2')
    expect(filedIds(container, null)).not.toContain('s2')
    expect(filedIds(container, billing.id)).not.toContain('s1')
    expect(filedIds(container, null)).toContain('s1')
    expect(within(folderBlock(container, null)).queryByTestId('session-row-s2')).toBeNull()
  })

  it('keeps an empty folder drawn, so it is there to drop into', () => {
    const empty = createSessionSection('Empty')!

    const { container } = renderSection()

    expect(within(folderBlock(container, empty.id)).getByText('Empty')).toBeTruthy()
    expect(filedIds(container, empty.id)).toEqual([])
  })

  it('files a session dropped on a folder heading, and un-files one dropped on Sessions', () => {
    const billing = createSessionSection('Billing work', ['s2'])!

    const { container } = renderSection()

    const rows = new Set(SESSIONS.map(session => session.id))

    // A folder heading's drop id names that folder; the Unassigned heading's
    // names no folder at all.
    expect(sessionFolderDropTarget(sessionFolderDropId(billing.id))).toEqual({ sectionId: billing.id })
    expect(sessionFolderDropTarget(sessionFolderDropId(null))).toEqual({ sectionId: null })
    expect(sessionFolderDropTarget('s1')).toBeUndefined()

    // The drop a heading receives, driven exactly as the drag monitor drives it
    // (jsdom has no layout for a sensor gesture to resolve a target against).
    act(() => fileSessionFromDrop('s1', sessionFolderDropId(billing.id), rows))

    expect(sessionSectionId('s1')).toBe(billing.id)
    expect(filedIds(container, billing.id)).toContain('s1')
    expect(filedIds(container, null)).not.toContain('s1')

    // Dropping it back on the Unassigned heading un-files it.
    act(() => fileSessionFromDrop('s1', sessionFolderDropId(null), rows))

    expect(sessionSectionId('s1')).toBeNull()
    expect(filedIds(container, null)).toContain('s1')
    expect(filedIds(container, billing.id)).not.toContain('s1')
  })

  it('deleting a folder leaves its sessions in the list, unfiled', async () => {
    const billing = createSessionSection('Billing work', ['s2'])!

    const { container } = renderSection()
    expect(filedIds(container, billing.id)).toContain('s2')

    openFolderMenu(container, billing.id)
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Delete' }))

    // The folder is gone; its session is NOT — it falls back to Unassigned.
    expect($sessionSections.get().some(section => section.id === billing.id)).toBe(false)
    expect(sessionSectionId('s2')).toBeNull()
    expect(screen.getByTestId('session-row-s2')).toBeTruthy()
    expect(container.querySelectorAll('[data-session-folder-header]')).toHaveLength(0)
  })

  it('renames a folder from its ⋯ menu, and the heading follows', async () => {
    const billing = createSessionSection('Billing work', ['s2'])!

    const { container } = renderSection()

    openFolderMenu(container, billing.id)
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Rename…' }))

    const input = screen.getByRole('textbox', { name: 'Folder name' }) as HTMLInputElement

    expect(input.value).toBe('Billing work')

    fireEvent.change(input, { target: { value: 'Invoices' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    expect(within(folderBlock(container, billing.id)).getByText('Invoices')).toBeTruthy()
    expect($sessionSections.get().find(section => section.id === billing.id)?.name).toBe('Invoices')
  })

  it('creates a folder from the header "+" through the one name dialog', () => {
    const { container } = renderSection()

    fireEvent.click(screen.getByRole('button', { name: 'New folder' }))

    fireEvent.change(screen.getByRole('textbox', { name: 'Folder name' }), { target: { value: 'Clients' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create' }))

    const created = $sessionSections.get()[0]!

    expect(created.name).toBe('Clients')
    expect(within(folderBlock(container, created.id)).getByText('Clients')).toBeTruthy()
    // The new folder is empty, so nothing moved out of the flat list.
    expect(filedIds(container, created.id)).toEqual([])
    expect(filedIds(container, null)).toContain('s1')
  })
})

describe('folder drag state', () => {
  /** A bare sortable row: enough to lift a REAL gesture (Space to arm, Space to
   *  drop) inside the folder list's own DndContext, which is the only place
   *  `$draggingSession` is written from. */
  function SortableProbe({ id }: { id: string }) {
    const { attributes, listeners, setNodeRef } = useSortable({ id })

    return (
      <button {...attributes} {...listeners} ref={setNodeRef} type="button">
        {id}
      </button>
    )
  }

  /** The sidebar's own sensors (index.tsx), so the gesture driven here is the
   *  gesture the app runs. */
  function FolderListWithSensors(props: React.ComponentProps<typeof SidebarSessionFolderList>) {
    const sensors = useSensors(
      useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
      useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
    )

    return <SidebarSessionFolderList {...props} sensors={sensors} />
  }

  it('files the dropped row and clears the in-flight session, from a real gesture', async () => {
    const billing = createSessionSection('Billing work')!
    // Two rows: the keyboard drag needs somewhere to move TO before it can drop
    // (a one-item list has no next coordinate to land on). s2 is already filed,
    // so the drop lands inside the folder whichever of the two the layout-less
    // collision detection picks.
    const nodes = (id: string) => [<SortableProbe id={id} key={id} />]

    moveSessionsToSection(['s2'], billing.id)

    render(
      <FolderListWithSensors
        groups={[
          {
            id: billing.id,
            name: billing.name,
            rows: [
              { id: 's1', nodes: nodes('s1') },
              { id: 's2', nodes: nodes('s2') }
            ]
          }
        ]}
        label="Sessions"
      />
    )

    const handle = screen.getByRole('button', { name: 's1' })

    await act(async () => {
      fireEvent.keyDown(handle, { code: 'Space', key: ' ' })
      // The keyboard sensor attaches its own keydown listener on the next
      // macrotask (KeyboardSensor.attach), so the drop keys have to wait for it
      // or the gesture never ends.
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect($draggingSession.get()).toBe('s1')

    await act(async () => {
      fireEvent.keyDown(handle, { code: 'ArrowDown', key: 'ArrowDown' })
      fireEvent.keyDown(handle, { code: 'Space', key: ' ' })
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    // The drag ended: the in-flight session is cleared (never left stuck on),
    // and the session it carried was filed into the folder it was dropped in.
    expect($draggingSession.get()).toBeNull()
    expect(sessionSectionId('s1')).toBe(billing.id)
  })
})
