import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $dismissedAutoProjectIds, $sidebarProjectFilter } from '@/store/layout'
import { $projectTree } from '@/store/projects'

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
vi.mock('@/store/projects', () => ({
  $projectTree: atom<unknown[]>([]),
  moveSessionToProject: vi.fn(),
  projectIdForCwd: vi.fn(() => null),
  // Mirror the real store's rule: primary folder, else first repo with one.
  projectRootCwd: vi.fn(
    (node: { path?: null | string; repos?: { path?: null | string }[] }) =>
      (node?.path || node?.repos?.find(repo => repo.path)?.path || '').trim()
  )
}))
vi.mock('@/store/session', () => ({
  $activeSessionId: atom<null | string>(null),
  $connection: atom<null | { mode: string }>(null),
  $cronSessions: atom<unknown[]>([]),
  $messagingSessions: atom<unknown[]>([]),
  $selectedStoredSessionId: atom<null | string>(null),
  $sessions: atom<unknown[]>([]),
  $unreadFinishedSessionIds: atom<string[]>([]),
  markSessionRead: vi.fn(),
  sessionMatchesStoredId: vi.fn(() => false),
  sessionPinId: vi.fn((s: { id: string }) => s.id),
  setSessions: vi.fn()
}))
vi.mock('@/store/session-color', () => ({
  $sessionColorOverrides: atom<Record<string, string>>({}),
  setSessionColorOverride: vi.fn()
}))
vi.mock('@/store/session-states', () => ({
  $sessionTiles: atom<unknown[]>([]),
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

function renderMenu() {
  return render(
    <SessionActionsMenu sessionId="s1" title="My session">
      <button aria-label="Session actions" type="button">
        ⋮
      </button>
    </SessionActionsMenu>
  )
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
})

// The "Move to project" submenu must offer the SAME project set the sidebar
// overview shows: a dismissed auto-project and a project excluded by the
// persisted sidebar filter must not be move targets. Drive the real
// interaction: open the kebab menu, then click the submenu trigger (Radix
// opens subs on click; the parent menu stays open). The real layout store is
// loaded (no mock) so the real filterVisibleProjects + persistent atoms run.
// Store mutations are wrapped in act() so the menu re-renders before asserting.
describe('Move to project mirrors the sidebar view filters', () => {
  async function openMoveToProjectSubmenu() {
    renderMenu()

    const trigger = screen.getByRole('button', { name: 'Session actions' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)

    await screen.findByRole('menu')
    const subTrigger = screen.getByRole('menuitem', { name: 'Move to project' })
    fireEvent.click(subTrigger)
  }

  it('offers visible projects and hides dismissed auto-projects', async () => {
    $projectTree.set([
      { id: 'p-ghost', isAuto: false, isNoProject: false, label: 'Ghost Recon', path: 'H:/workspace/Projects/Ghost Recon Stage 2/.re', repos: [], sessionCount: 1 },
      { id: 'p-x4', isAuto: true, isNoProject: false, label: 'x4-projects', path: 'C:/Users/gmkon/x4-projects', repos: [], sessionCount: 0 }
    ])
    $dismissedAutoProjectIds.set([])
    $sidebarProjectFilter.set([])

    await openMoveToProjectSubmenu()

    // Both visible projects are move targets.
    expect(await screen.findByRole('menuitem', { name: 'Ghost Recon' })).toBeTruthy()
    expect(screen.getByRole('menuitem', { name: 'x4-projects' })).toBeTruthy()

    // Dismiss the auto-project in the sidebar: the menu must follow (the
    // reported bug was that it did not).
    act(() => $dismissedAutoProjectIds.set(['p-x4']))
    expect(screen.queryByRole('menuitem', { name: 'x4-projects' })).toBeNull()
    expect(screen.getByRole('menuitem', { name: 'Ghost Recon' })).toBeTruthy()

    // Re-showing it restores the target.
    act(() => $dismissedAutoProjectIds.set([]))
    expect(await screen.findByRole('menuitem', { name: 'x4-projects' })).toBeTruthy()
  })

  it('hides projects excluded by the persisted sidebar project filter', async () => {
    $projectTree.set([
      { id: 'p-ghost', isAuto: false, isNoProject: false, label: 'Ghost Recon', path: 'H:/workspace/Projects/Ghost Recon Stage 2/.re', repos: [], sessionCount: 1 },
      { id: 'p-ue5', isAuto: false, isNoProject: false, label: 'UE5 Tactical Shooter', path: 'G:/Projects/Unreal Projects/test2', repos: [], sessionCount: 2 }
    ])
    $dismissedAutoProjectIds.set([])
    $sidebarProjectFilter.set([])

    await openMoveToProjectSubmenu()

    expect(await screen.findByRole('menuitem', { name: 'UE5 Tactical Shooter' })).toBeTruthy()

    // Narrow the sidebar to Ghost Recon only: the menu must follow.
    act(() => $sidebarProjectFilter.set(['p-ghost']))
    expect(screen.queryByRole('menuitem', { name: 'UE5 Tactical Shooter' })).toBeNull()
    expect(screen.getByRole('menuitem', { name: 'Ghost Recon' })).toBeTruthy()

    // Clearing the filter restores the target.
    act(() => $sidebarProjectFilter.set([]))
    expect(await screen.findByRole('menuitem', { name: 'UE5 Tactical Shooter' })).toBeTruthy()
  })
})
