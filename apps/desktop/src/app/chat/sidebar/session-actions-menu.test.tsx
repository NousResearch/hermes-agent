import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as SessionStampModule from '@/store/session-stamp'
import {
  applySessionStamp,
  normalizeSessionStamp,
  SESSION_STAMP_MAX_LENGTH,
  SESSION_STAMP_PRESETS
} from '@/store/session-stamp'

import { SessionActionsMenu, SessionContextMenu } from './session-actions-menu'

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
})

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
  listAllProfileSessions: vi.fn(() => Promise.resolve({ sessions: [] })),
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
          stamp: 'Stamp',
          stampClear: 'Clear stamp',
          stampCustom: 'Custom stamp…',
          stampCustomHint: 'Shown beside the title in the session list and tabs.',
          stampCustomPlaceholder: 'e.g. Merged or Waiting on CI',
          stampCustomTitle: 'Stamp this session',
          stampCleared: 'Stamp cleared',
          stampSaved: (label: string) => `Stamped ${label}`,
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
  projectRootCwd: vi.fn(() => '')
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
// Keep the REAL presets / normalizer / cap (the menu must offer exactly what the
// store module offers) and stub only the write, so a test can see precisely what
// the menu handed the one write path.
vi.mock('@/store/session-stamp', async importOriginal => {
  const actual = await importOriginal<typeof SessionStampModule>()

  return {
    ...actual,
    applySessionStamp: vi.fn(() => Promise.resolve(true))
  }
})
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

// Radix's SubTrigger opens on click (MenuItemImpl's own onClick), so a plain
// click is the whole gesture — no hover timer to wait out.
async function openKebab() {
  const trigger = screen.getByRole('button', { name: 'Session actions' })
  fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.click(trigger)
  await screen.findByRole('menu')
}

async function openStampSubmenu() {
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Stamp' }))
}

function renderStampMenu(stamp?: null | string) {
  return render(
    <SessionActionsMenu profile="p1" sessionId="s1" stamp={stamp} title="My session">
      <button aria-label="Session actions" type="button">
        ⋮
      </button>
    </SessionActionsMenu>
  )
}

describe('session menu — Stamp submenu', () => {
  it('offers every preset plus the custom door on a session with no stamp', async () => {
    renderStampMenu()
    await openKebab()
    await openStampSubmenu()

    for (const preset of SESSION_STAMP_PRESETS) {
      expect(await screen.findByRole('menuitem', { name: preset })).toBeTruthy()
    }

    expect(screen.getByRole('menuitem', { name: 'Custom stamp…' })).toBeTruthy()
    // Nothing to clear yet — the row must not be there.
    expect(screen.queryByRole('menuitem', { name: 'Clear stamp' })).toBeNull()
  })

  it('marks the stamp the session already carries and offers to clear it', async () => {
    renderStampMenu('WIP')
    await openKebab()
    await openStampSubmenu()

    const current = await screen.findByRole('menuitem', { name: 'WIP' })

    // The selected row is the one wearing the check — and only that one.
    expect(current.querySelector('.codicon-check')).toBeTruthy()
    expect(screen.getByRole('menuitem', { name: 'Merged' }).querySelector('.codicon-check')).toBeNull()

    fireEvent.click(screen.getByRole('menuitem', { name: 'Clear stamp' }))

    await waitFor(() => expect(applySessionStamp).toHaveBeenCalledWith('s1', 'p1', null))
  })

  it('writes a picked preset through applySessionStamp for this session and profile', async () => {
    renderStampMenu()
    await openKebab()
    await openStampSubmenu()

    fireEvent.click(await screen.findByRole('menuitem', { name: 'Handoff' }))

    await waitFor(() => expect(applySessionStamp).toHaveBeenCalledWith('s1', 'p1', 'Handoff'))
  })

  it('clamps a custom stamp to the shared cap, and collapses its whitespace', async () => {
    renderStampMenu()
    await openKebab()
    await openStampSubmenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Custom stamp…' }))

    const dialog = await screen.findByRole('dialog')
    const input = within(dialog).getByRole('textbox')
    const typed = 'Waiting on CI and then some more words'

    fireEvent.change(input, { target: { value: typed } })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(applySessionStamp).toHaveBeenCalledTimes(1))

    const [, , written] = vi.mocked(applySessionStamp).mock.calls[0]!

    // Exactly what the store module's own normalizer would produce from what was
    // typed, and never longer than the cap it (and the backend) enforce.
    expect(written).toBe(normalizeSessionStamp(typed))
    expect(written!.length).toBeLessThanOrEqual(SESSION_STAMP_MAX_LENGTH)

    // Re-open (a successful write closes the dialog). The row still carries no
    // stamp, so the dialog is seeded empty — and submitting that must CLEAR,
    // not write an empty string.
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    await openKebab()
    await openStampSubmenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Custom stamp…' }))

    const reopened = await screen.findByRole('dialog')
    const reopenedInput = within(reopened).getByRole('textbox')

    // Whitespace runs collapse rather than reaching the backend verbatim.
    fireEvent.change(reopenedInput, { target: { value: '  Review   now  ' } })
    fireEvent.click(within(reopened).getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(applySessionStamp).toHaveBeenLastCalledWith('s1', 'p1', 'Review now'))
  })

  it('clears the stamp when the custom dialog is submitted empty', async () => {
    renderStampMenu('Hold')
    await openKebab()
    await openStampSubmenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Custom stamp…' }))

    const dialog = await screen.findByRole('dialog')
    const input = within(dialog).getByRole('textbox')

    // Seeded from the row's own stamp, then emptied — an empty submit is a
    // CLEAR (null), never an empty string the backend would have to interpret.
    expect((input as HTMLInputElement).value).toBe('Hold')
    fireEvent.change(input, { target: { value: '   ' } })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(applySessionStamp).toHaveBeenCalledWith('s1', 'p1', null))
  })

  it('routes the Stamp submenu through the right-click menu too', async () => {
    render(
      <SessionContextMenu sessionId="s1" title="My session">
        <button aria-label="Session row" type="button">
          Row
        </button>
      </SessionContextMenu>
    )

    fireEvent.contextMenu(screen.getByRole('button', { name: 'Session row' }))
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Stamp' }))

    expect(await screen.findByRole('menuitem', { name: 'Merged' })).toBeTruthy()
    fireEvent.click(screen.getByRole('menuitem', { name: 'Review' }))

    await waitFor(() => expect(applySessionStamp).toHaveBeenCalledWith('s1', undefined, 'Review'))
  })
})
