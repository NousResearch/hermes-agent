import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SessionActionsMenu, SessionContextMenu } from './session-actions-menu'

afterEach(cleanup)

// Exercises the real SessionActionsMenu end-to-end (no DropdownMenu mock) so
// a broken asChild composition on the kebab trigger fails here — the menu
// must still open on click.

vi.mock('@/components/pane-shell/tree/store', () => ({
  closeAllTreeTabs: vi.fn(),
  closeOtherTreeTabs: vi.fn(),
  closeTreeTabsToRight: vi.fn(),
  treeTabCloseTargets: vi.fn(() => null)
}))
vi.mock('@/hermes', () => ({
  renameSession: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setSessionUnreadRemote: vi.fn(() => Promise.resolve({ ok: true }))
}))
vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
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
        projects: {
          menuAppearance: 'Appearance',
          moveFailed: 'Could not move session',
          moveNoProjects: 'No other projects',
          moveCreateNew: 'New project and move here…',
          movedTo: (name: string) => `Moved to ${name}`,
          moveToProject: 'Move to project',
          noColor: 'No color'
        },
        row: {
          archive: 'Archive',
          branchFrom: 'Branch from here',
          copyId: 'Copy ID',
          copyIdFailed: 'Failed to copy ID',
          deleteDesc: (title: string) => `Delete ${title}?`,
          deleteTitle: 'Delete session?',
          deleting: 'Deleting…',
          deleted: 'Session deleted',
          export: 'Export',
          hideTabBar: 'Hide tab bar',
          markRead: 'Mark as read',
          pin: 'Pin',
          rename: 'Rename',
          renameDesc: 'Leave empty to clear.',
          renameFailed: 'Rename failed',
          renameTitle: 'Rename session',
          renamed: 'Renamed',
          sessionActions: 'Session actions',
          unpin: 'Unpin',
          untitledPlaceholder: 'Untitled'
        }
      },
      zones: { closeAll: 'Close all', closeOthers: 'Close others', closeToRight: 'Close to the right' }
    }
  })
}))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/lib/profile-color', () => ({ PROFILE_SWATCHES: [] }))
vi.mock('@/lib/session-export', () => ({ exportSession: vi.fn() }))
vi.mock('@/store/gateway', () => ({ activeGateway: vi.fn(() => null) }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
// `vi.mock` factories hoist above the module's `import` statements, so any
// value they close over has to be created via `vi.hoisted`. We use a tiny
// re-implementation of the nanostores atom shape (just the `.get() / .set() /
// .subscribe()` bits the menu reads) so the hoisted factory stays purely
// synchronous and free of cross-module imports. The atoms back the menu's
// "look up this session" / "list the project tree" reads; the test body
// mutates them to drive the "Home-only" assertions.
//
// IMPORTANT: the `vi.mock` factories below must close over `hoistedMocks`
// directly (NOT over a destructured `const { ... } = hoistedMocks` at file
// scope), because the destructuring line would also be hoisted and could
// run before `vi.hoisted` resolves.
type HoistedAtom<T> = {
  get: () => T
  set: (value: T) => void
  subscribe: (listener: (value: T) => void) => () => void
  listen: (listener: (value: T) => void) => () => void
}

const hoistedMocks = vi.hoisted(() => {
  function atom<T>(initial: T): HoistedAtom<T> {
    let value = initial
    const listeners = new Set<(next: T) => void>()

    return {
      get: () => value,
      set: (next: T) => {
        value = next

        for (const listener of listeners) {
          listener(next)
        }
      },
      subscribe: (listener: (next: T) => void) => {
        listeners.add(listener)

        return () => {
          listeners.delete(listener)
        }
      },
      // @nanostores/react's `useStore` reads `.listen` (the legacy API the
      // newer `.subscribe` mirrors). Without it, every component that uses
      // `useStore($someAtom)` throws "store.listen is not a function" on
      // mount. Provide the same shape so the hook accepts the mock.
      listen: (listener: (next: T) => void) => {
        listeners.add(listener)

        return () => {
          listeners.delete(listener)
        }
      }
    }
  }

  return {
    mockMoveSessionToProject: vi.fn(),
    mockOpenProjectCreate: vi.fn(),
    projectTree: atom<unknown[]>([]),
    sessionsAtom: atom<unknown[]>([]),
    connectionAtom: atom<null | { mode: string }>(null),
    nanoAtom: atom
  }
})

vi.mock('@/store/projects', () => ({
  $projectTree: hoistedMocks.projectTree,
  moveSessionToProject: hoistedMocks.mockMoveSessionToProject,
  openProjectCreate: hoistedMocks.mockOpenProjectCreate,
  projectIdForCwd: vi.fn(() => null),
  projectRootCwd: vi.fn(() => '')
}))
vi.mock('@/store/session', () => ({
  $activeSessionId: hoistedMocks.nanoAtom<null | string>(null),
  $connection: hoistedMocks.connectionAtom,
  $cronSessions: hoistedMocks.nanoAtom<unknown[]>([]),
  $messagingSessions: hoistedMocks.nanoAtom<unknown[]>([]),
  $selectedStoredSessionId: hoistedMocks.nanoAtom<null | string>(null),
  $sessions: hoistedMocks.sessionsAtom,
  $unreadFinishedSessionIds: hoistedMocks.nanoAtom<string[]>([]),
  markSessionRead: vi.fn(),
  sessionMatchesStoredId: vi.fn((session: { id: string }, id: string) => session.id === id),
  sessionPinId: vi.fn((s: { id: string }) => s.id),
  setSessions: vi.fn()
}))
vi.mock('@/store/session-color', () => ({
  $sessionColorOverrides: hoistedMocks.nanoAtom<Record<string, string>>({}),
  setSessionColorOverride: vi.fn()
}))
vi.mock('@/store/session-states', () => ({
  $sessionTiles: hoistedMocks.nanoAtom<unknown[]>([]),
  closeAllOpenSessionTiles: vi.fn(),
  openSessionTile: vi.fn()
}))
vi.mock('@/store/windows', () => ({
  canOpenSessionInTerminal: () => false,
  canOpenSessionWindow: () => false,
  isBrowserWindow: () => false,
  isSecondaryWindow: () => false,
  openSessionInNewWindow: vi.fn(),
  openSessionInTerminal: vi.fn()
}))
// File-scope aliases for test bodies (these run after vi.hoisted resolves,
// so destructuring is safe here).
const projectTree = hoistedMocks.projectTree
const sessionsAtom = hoistedMocks.sessionsAtom
const mockMoveSessionToProject = hoistedMocks.mockMoveSessionToProject
const mockOpenProjectCreate = hoistedMocks.mockOpenProjectCreate

function renderMenu() {
  return render(
    <SessionActionsMenu sessionId="s1" title="My session">
      <button aria-label="Session actions" type="button">
        ⋮
      </button>
    </SessionActionsMenu>
  )
}

async function openMenu() {
  const trigger = screen.getByRole('button', { name: 'Session actions' })

  // Radix's dropdown trigger opens on pointerdown (not on the synthetic
  // 'click' fireEvent alone would dispatch), so fire the full mouse
  // sequence a real click produces.
  fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.click(trigger)

  await screen.findByRole('menu')
}

describe('SessionActionsMenu', () => {
  it('opens the dropdown on click without a tooltip on the kebab', async () => {
    renderMenu()

    const trigger = screen.getByRole('button', { name: 'Session actions' })

    expect(trigger.closest('[data-slot="tooltip-trigger"]')).toBeNull()

    // Radix's dropdown trigger opens on pointerdown (not on the synthetic
    // 'click' fireEvent alone would dispatch), so fire the full mouse
    // sequence a real click produces.
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)

    expect(await screen.findByRole('menu')).toBeTruthy()
    expect(screen.getByRole('menuitem', { name: /rename/i })).toBeTruthy()
    expect(screen.getByRole('menuitem', { name: /archive/i })).toBeTruthy()
  })

  it('opens the rename dialog focused on its input, not the row trigger', async () => {
    renderMenu()

    const trigger = screen.getByRole('button', { name: 'Session actions' })

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)

    const rename = await screen.findByRole('menuitem', { name: /rename/i })
    fireEvent.click(rename)

    // The dialog opens and its textbox takes focus. If the menu's close restored
    // focus to the row trigger instead, Space would activate the row and the
    // arrow keys would move the list rather than the caret (the reported bug).
    const dialog = await screen.findByRole('dialog')
    const input = within(dialog).getByRole('textbox')

    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    await waitFor(() => expect(document.activeElement).toBe(input))
    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    expect(document.activeElement).not.toBe(trigger)
  })

  it('confirms before deleting — cancel keeps the session, confirm deletes it', async () => {
    const onDelete = vi.fn()
    render(
      <SessionActionsMenu onDelete={onDelete} sessionId="s1" title="My session">
        <button aria-label="Session actions" type="button">
          ⋮
        </button>
      </SessionActionsMenu>
    )

    const trigger = screen.getByRole('button', { name: 'Session actions' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)

    const deleteItem = await screen.findByRole('menuitem', { name: /delete/i })
    fireEvent.click(deleteItem)

    // The confirm dialog is up and names the session being deleted.
    expect(await screen.findByRole('dialog')).toBeTruthy()
    expect(screen.getByText(/My session/)).toBeTruthy()

    // Cancel: nothing is deleted.
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onDelete).not.toHaveBeenCalled()

    // Re-open the menu and confirm: only now does the delete call fire.
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)
    const deleteItemAgain = await screen.findByRole('menuitem', { name: /delete/i })
    fireEvent.click(deleteItemAgain)

    expect(await screen.findByRole('dialog')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }))
    // ConfirmDialog shows a done beat before auto-closing (600ms); awaiting it
    // also drains the async run() update inside act().
    expect(await screen.findByText('Session deleted')).toBeTruthy()
    expect(onDelete).toHaveBeenCalledTimes(1)
  })

  it('disables the delete item when no onDelete is provided', async () => {
    render(
      <SessionActionsMenu sessionId="s1" title="My session">
        <button aria-label="Session actions" type="button">
          ⋮
        </button>
      </SessionActionsMenu>
    )

    const trigger = screen.getByRole('button', { name: 'Session actions' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)

    const deleteItem = await screen.findByRole('menuitem', { name: /delete/i })
    expect(deleteItem.getAttribute('aria-disabled')).toBe('true')
  })

  it('confirms with the Enter key and cancels with Escape', async () => {
    const onDelete = vi.fn()
    render(
      <SessionActionsMenu onDelete={onDelete} sessionId="s1" title="My session">
        <button aria-label="Session actions" type="button">
          ⋮
        </button>
      </SessionActionsMenu>
    )

    const trigger = screen.getByRole('button', { name: 'Session actions' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)
    fireEvent.click(await screen.findByRole('menuitem', { name: /delete/i }))

    const dialog = await screen.findByRole('dialog')
    expect(dialog).toBeTruthy()

    // Escape cancels: dialog closes, nothing is deleted.
    fireEvent.keyDown(window.document, { key: 'Escape' })
    expect(await screen.queryByRole('dialog')).toBeNull()
    expect(onDelete).not.toHaveBeenCalled()

    // Re-open and confirm with Enter at wherever focus actually is. Firing on
    // the dialog node would pass even when the menu leaves focus on the row
    // trigger — where Enter re-activates the row instead of confirming.
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)
    fireEvent.click(await screen.findByRole('menuitem', { name: /delete/i }))

    const reopened = await screen.findByRole('dialog')
    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    await waitFor(() => expect(reopened.contains(document.activeElement)).toBe(true))
    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    fireEvent.keyDown(document.activeElement!, { key: 'Enter' })

    expect(await screen.findByText('Session deleted')).toBeTruthy()
    expect(onDelete).toHaveBeenCalledTimes(1)
  })

  it('routes the same confirm guard through the context menu', async () => {
    const onDelete = vi.fn()
    render(
      <SessionContextMenu onDelete={onDelete} sessionId="s1" title="My session">
        <button aria-label="Session row" type="button">
          Row
        </button>
      </SessionContextMenu>
    )

    const row = screen.getByRole('button', { name: 'Session row' })
    fireEvent.contextMenu(row)

    fireEvent.click(await screen.findByRole('menuitem', { name: /delete/i }))
    expect(await screen.findByRole('dialog')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Delete' }))
    expect(await screen.findByText('Session deleted')).toBeTruthy()
    expect(onDelete).toHaveBeenCalledTimes(1)
  })

  it('offers "New project and move here…" when the sidebar has only the Home bucket', async () => {
    // The sidebar currently exposes only the synthetic Home (NO_PROJECT_ID)
    // node — every other project the user might move into is filtered out
    // by `!node.isNoProject`. The submenu used to dead-end on a disabled
    // "No other projects" item; now it surfaces a real action that opens
    // the project-create dialog with the session's cwd pre-filled, and
    // chains back to moveSessionToProject on success.
    projectTree.set([{ id: '__no_project__', isNoProject: true, label: 'Home', path: null, repos: [] }])
    sessionsAtom.set([{ id: 's1', cwd: '/Users/me/proj' }])

    renderMenu()
    await openMenu()

    // The submenu trigger is the "Move to project" item. Radix renders it
    // as a menuitem with a sub-indicator; hovering/clicking it opens the
    // nested menu. We assert the trigger text is present and the
    // "create-and-move" affordance is reachable by clicking the trigger.
    const moveTrigger = screen.getByRole('menuitem', { name: /move to project/i })

    fireEvent.pointerDown(moveTrigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(moveTrigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(moveTrigger)

    const newProjectItem = await screen.findByRole('menuitem', { name: /new project and move here/i })

    expect(newProjectItem).toBeTruthy()
    expect(newProjectItem.getAttribute('aria-disabled')).not.toBe('true')

    // Clicking the new-project affordance should call openProjectCreate
    // with the session's cwd prefilled and an onCreated callback. We do
    // not assert the callback here — the dialog driver is exercised in
    // its own tests — only the entry point, which is the fix the user
    // asked for.
    fireEvent.pointerDown(newProjectItem, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(newProjectItem, { button: 0, pointerType: 'mouse' })
    fireEvent.click(newProjectItem)

    expect(mockOpenProjectCreate).toHaveBeenCalledTimes(1)
    expect(mockOpenProjectCreate).toHaveBeenCalledWith(
      expect.objectContaining({ prefillFolder: '/Users/me/proj' })
    )
  })
})
