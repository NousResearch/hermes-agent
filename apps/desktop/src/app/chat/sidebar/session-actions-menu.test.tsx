import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PROFILE_SWATCHES } from '@/lib/profile-color'
import type * as SessionStampModule from '@/store/session-stamp'
import {
  $stampColorOverrides,
  $stampPresets,
  $stampTitlePrefs,
  applySessionStamps,
  SESSION_STAMP_LIMIT,
  SESSION_STAMP_PRESETS,
  toggleSessionStamp
} from '@/store/session-stamp'

import { SessionActionsMenu, SessionContextMenu } from './session-actions-menu'

afterEach(cleanup)

// The stamp menu's emoji panel searches the bundled catalog, which in jsdom is
// neither fetched nor needed: the panel's own behaviour (when it searches, what
// it paints, what a tap writes) is what these cases are about.
const { searchEmojiMock } = vi.hoisted(() => ({ searchEmojiMock: vi.fn() }))

beforeEach(() => {
  vi.clearAllMocks()
  // The stamp menu's titles and colours are localStorage-backed, so they outlive
  // a single test in the same jsdom environment — reset rather than lean on
  // test order.
  $stampTitlePrefs.set({ added: [], deleted: [] })
  $stampColorOverrides.set({})
  window.localStorage.clear()
  searchEmojiMock.mockResolvedValue([
    { code: 'octopus', emoji: '🐙', haystack: ['octopus'] },
    { code: 'unicorn', emoji: '🦄', haystack: ['unicorn'] }
  ])
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
          stampClear: 'Clear stamps',
          stampCustom: 'Custom stamp…',
          stampCustomPlaceholder: 'e.g. Merged or Waiting on CI',
          stampLimit: (max: number) => `Up to ${max} stamps per session`,
          stampSaved: (label: string) => `Stamped ${label}`,
          stampRemoved: (label: string) => `Removed ${label}`,
          stampCleared: 'Stamp cleared',
          stampRemove: (label: string) => `Delete “${label}” from the stamp menu`,
          stampRestore: 'Restore deleted stamp titles',
          stampAdd: 'Add stamp title…',
          stampAddPlaceholder: 'e.g. Blocked',
          stampEmoji: 'Emoji…',
          stampEmojiEmpty: 'No emoji found',
          stampEmojiPick: (label: string) => `Stamp with ${label}`,
          stampEmojiSearch: 'Search emoji',
          stampColor: (label: string) => `Color of “${label}”`,
          stampColorReset: 'Default color',
          unpin: 'Unpin',
          untitledPlaceholder: 'Untitled'
        }
      },
      zones: { closeAll: 'Close all', closeOthers: 'Close others', closeToRight: 'Close to the right' }
    }
  })
}))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/lib/emoji-index', () => ({
  isEmojiIndexLoaded: () => true,
  searchEmoji: searchEmojiMock
}))
// A short palette, not an empty one: the title colour test has to click a
// swatch, and an empty list makes every button match a lookup by swatch name.
vi.mock('@/lib/profile-color', () => ({ PROFILE_SWATCHES: ['hsl(0 68% 58%)', 'hsl(120 68% 58%)'] }))
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
    // Both write doors the submenu uses: the toggle behind every title row, and
    // the list write behind Clear. Stubbed so a test can see exactly what the
    // menu handed over — the real ones would need a backend.
    applySessionStamps: vi.fn(() => Promise.resolve(true)),
    toggleSessionStamp: vi.fn(() => Promise.resolve(true))
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

function renderStampMenu(stamps?: string[]) {
  return render(
    <SessionActionsMenu profile="p1" sessionId="s1" stamps={stamps} title="My session">
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
    // Nothing to clear yet — the row must not be there, and no cap hint either.
    expect(screen.queryByRole('menuitem', { name: 'Clear stamps' })).toBeNull()
    expect(screen.queryByText(`Up to ${SESSION_STAMP_LIMIT} stamps per session`)).toBeNull()
  })

  it('marks every stamp the session carries and clears the whole list', async () => {
    renderStampMenu(['WIP', 'Hold'])
    await openKebab()
    await openStampSubmenu()

    // The rows the session carries wear the check — and only those.
    expect((await screen.findByRole('menuitem', { name: 'WIP' })).querySelector('.codicon-check')).toBeTruthy()
    expect(screen.getByRole('menuitem', { name: 'Hold' }).querySelector('.codicon-check')).toBeTruthy()
    expect(screen.getByRole('menuitem', { name: 'Merged' }).querySelector('.codicon-check')).toBeNull()

    fireEvent.click(screen.getByRole('menuitem', { name: 'Clear stamps' }))

    // Clearing is a LIST write: every label off, not just the first.
    await waitFor(() => expect(applySessionStamps).toHaveBeenCalledWith('s1', 'p1', []))
  })

  it('toggles a picked title ON, and keeps the menu open while it does', async () => {
    renderStampMenu()
    await openKebab()
    await openStampSubmenu()

    fireEvent.click(await screen.findByRole('menuitem', { name: 'Handoff' }))

    await waitFor(() => expect(toggleSessionStamp).toHaveBeenCalledWith('s1', 'p1', 'Handoff'))
    // Setting a stamp is not a reason to lose your place in the submenu.
    expect(screen.queryByRole('menuitem', { name: 'WIP' })).toBeTruthy()
  })

  it('toggles a second title ON beside the one already carried', async () => {
    renderStampMenu(['WIP'])
    await openKebab()
    await openStampSubmenu()

    fireEvent.click(await screen.findByRole('menuitem', { name: 'Hold' }))

    // The list write carries BOTH labels, in the order the session built them.
    await waitFor(() => expect(toggleSessionStamp).toHaveBeenCalledWith('s1', 'p1', 'Hold'))
    // ...and the write is a toggle, so the menu's own state (not a refetch) is
    // what moves the check marks.
    expect(applySessionStamps).not.toHaveBeenCalled()
  })

  it('is a real toggle: clicking a title the session carries takes it OFF', async () => {
    renderStampMenu(['WIP', 'Hold'])
    await openKebab()
    await openStampSubmenu()

    fireEvent.click(await screen.findByRole('menuitem', { name: 'Hold' }))

    await waitFor(() => expect(toggleSessionStamp).toHaveBeenCalledWith('s1', 'p1', 'Hold'))
  })

  it('says so and goes inert once the session is full', async () => {
    renderStampMenu(['WIP', 'Hold', 'Review'])
    await openKebab()
    await openStampSubmenu()

    // The hint explains the dead rows, in the menu rather than in a notification.
    expect(await screen.findByText(`Up to ${SESSION_STAMP_LIMIT} stamps per session`)).toBeTruthy()

    const free = screen.getByRole('menuitem', { name: 'Merged' })
    expect(free.getAttribute('aria-disabled')).toBe('true')
    fireEvent.click(free)
    // A fourth label is impossible, and a click that cannot land writes nothing.
    expect(toggleSessionStamp).not.toHaveBeenCalled()

    // A carried label still comes OFF — a full session is not a locked one.
    fireEvent.click(screen.getByRole('menuitem', { name: 'Hold' }))
    await waitFor(() => expect(toggleSessionStamp).toHaveBeenCalledWith('s1', 'p1', 'Hold'))
  })

  it('deletes a title without stamping, keeps the menu open, and offers a way back', async () => {
    // The session carries "Hold", so deleting the TITLE must be visibly a
    // different act from clearing the SESSION's stamp.
    renderStampMenu(['Hold'])
    await openKebab()
    await openStampSubmenu()

    const removeHold = await screen.findByRole('button', { name: 'Delete “Hold” from the stamp menu' })

    // The full pointer sequence a real click sends: Radix resolves the press on
    // pointerdown/up, so stopping `click` alone still selects the row underneath.
    fireEvent.pointerDown(removeHold, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(removeHold, { button: 0, pointerType: 'mouse' })
    fireEvent.click(removeHold)

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Delete “Hold” from the stamp menu' })).toBeNull())
    expect(toggleSessionStamp).not.toHaveBeenCalled()
    expect(applySessionStamps).not.toHaveBeenCalled()
    // The session's own stamp is untouched, so its clear row is still there.
    expect(screen.queryByRole('menuitem', { name: 'Clear stamps' })).toBeTruthy()

    fireEvent.click(screen.getByRole('menuitem', { name: 'Restore deleted stamp titles' }))

    // Restoring keeps the menu open too, so the title returns in place.
    expect(await screen.findByRole('menuitem', { name: 'Hold' })).toBeTruthy()
  })

  it('adds a title from an input INSIDE the menu, keeps it, and stamps the session', async () => {
    renderStampMenu()
    await openKebab()
    await openStampSubmenu()

    fireEvent.click(await screen.findByRole('menuitem', { name: 'Add stamp title…' }))

    // In place, never a dialog: a modal would take the submenu down with it.
    const input = await screen.findByPlaceholderText('e.g. Blocked')
    fireEvent.change(input, { target: { value: 'Blocked' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    await waitFor(() => expect(toggleSessionStamp).toHaveBeenCalledWith('s1', 'p1', 'Blocked'))
    expect($stampPresets.get()).toContain('Blocked')
    // Still open, and the new title is a row now.
    expect(await screen.findByRole('menuitem', { name: 'Blocked' })).toBeTruthy()
  })

  it('opens a title’s colours from the dot on its row, in place', async () => {
    renderStampMenu(['Merged'])
    await openKebab()
    await openStampSubmenu()

    fireEvent.click(await screen.findByRole('button', { name: 'Color of “Merged”' }))

    // The swatches appear under the row rather than in a popover or a dialog.
    const swatch = await screen.findByRole('button', { name: PROFILE_SWATCHES[0] })
    fireEvent.click(swatch)

    await waitFor(() => expect($stampColorOverrides.get().merged).toBe(PROFILE_SWATCHES[0]))
    // A colour is a menu preference: no stamp is written, and the menu lives on.
    expect(toggleSessionStamp).not.toHaveBeenCalled()
    expect(screen.queryByRole('menuitem', { name: 'WIP' })).toBeTruthy()
  })

  it('normalizes a custom label typed in place, and an empty submit writes nothing', async () => {
    renderStampMenu(['Hold'])
    await openKebab()
    await openStampSubmenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Custom stamp…' }))

    const input = await screen.findByPlaceholderText('e.g. Merged or Waiting on CI')

    // Empty on purpose: the panel ADDS a label now, so seeding it with one the
    // session already carries would make a submit toggle that one off instead.
    expect((input as HTMLInputElement).value).toBe('')
    fireEvent.change(input, { target: { value: '  Waiting   on CI  ' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    // Normalized on the way out: whitespace runs collapse rather than reaching
    // the backend verbatim.
    await waitFor(() => expect(toggleSessionStamp).toHaveBeenCalledWith('s1', 'p1', 'Waiting on CI'))

    // An empty submit is a no-op. With a list behind the menu, "take everything
    // off" is what the Clear row says, and guessing it from an empty input would
    // be a destructive default.
    fireEvent.click(screen.getByRole('menuitem', { name: 'Custom stamp…' }))
    const reopened = await screen.findByPlaceholderText('e.g. Merged or Waiting on CI')
    fireEvent.change(reopened, { target: { value: '   ' } })
    fireEvent.keyDown(reopened, { key: 'Enter' })

    expect(toggleSessionStamp).toHaveBeenCalledTimes(1)
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

    await waitFor(() => expect(toggleSessionStamp).toHaveBeenCalledWith('s1', undefined, 'Review'))
  })

  it('stamps with an emoji off the curated grid, and keeps the emoji as a title', async () => {
    renderStampMenu()
    await openKebab()
    await openStampSubmenu()

    fireEvent.click(await screen.findByRole('menuitem', { name: 'Emoji…' }))

    // The curated grid stands in before anything is typed, so reaching for the
    // catalog is not the price of a common emoji.
    await screen.findByLabelText('Search emoji')
    expect(searchEmojiMock).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Stamp with 🔥' }))

    await waitFor(() => expect(toggleSessionStamp).toHaveBeenCalledWith('s1', 'p1', '🔥'))
    // Tapping is "Add stamp title…" with a chooser: the emoji joins the menu's
    // own titles, so it is ONE tap next time, and the submenu stays open for a
    // second pick.
    expect($stampPresets.get()).toContain('🔥')
    expect(screen.getByRole('button', { name: 'Stamp with ✅' })).toBeTruthy()
  })

  it('searches the whole catalog for an emoji the grid does not carry', async () => {
    renderStampMenu()
    await openKebab()
    await openStampSubmenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Emoji…' }))

    const search = await screen.findByLabelText('Search emoji')
    fireEvent.change(search, { target: { value: 'octo' } })

    // Typing filters by the catalog's own shortcodes/labels, and the grid is
    // replaced by the hits (a bounded paint budget, not the whole catalog).
    await waitFor(() => expect(searchEmojiMock).toHaveBeenCalledWith('octo', expect.any(Number)))
    expect(await screen.findByRole('button', { name: 'Stamp with 🐙' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Stamp with 🐙' }))

    await waitFor(() => expect(toggleSessionStamp).toHaveBeenCalledWith('s1', 'p1', '🐙'))
  })

  it('says so when a search finds nothing, rather than painting an empty grid', async () => {
    searchEmojiMock.mockResolvedValue([])
    renderStampMenu()
    await openKebab()
    await openStampSubmenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Emoji…' }))

    fireEvent.change(await screen.findByLabelText('Search emoji'), { target: { value: 'zzzz' } })

    expect(await screen.findByText('No emoji found')).toBeTruthy()
  })

  it('leaves an emoji title no colour door, and still offers a way to take it off', async () => {
    // A colour emoji is drawn by the platform's own emoji font: a colour picker
    // for it would be a control that changes nothing visible.
    $stampTitlePrefs.set({ added: ['🔥'], deleted: [] })
    renderStampMenu(['🔥'])
    await openKebab()
    await openStampSubmenu()

    expect(await screen.findByRole('menuitem', { name: '🔥' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Color of “🔥”' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Delete “🔥” from the stamp menu' })).toBeTruthy()
  })

  it('goes inert at the cap like the other doors that ADD, while a carried emoji still toggles off', async () => {
    $stampTitlePrefs.set({ added: ['🔥'], deleted: [] })
    renderStampMenu(['🔥', 'WIP', 'Review'])
    await openKebab()
    await openStampSubmenu()

    // The panel exists to ADD an emoji, so a full session cannot open it ...
    expect((await screen.findByRole('menuitem', { name: 'Emoji…' })).getAttribute('aria-disabled')).toBe('true')

    // ... and the emoji the session wears is a row of its own, so it still comes
    // off (a full session is not a locked one).
    fireEvent.click(screen.getByRole('menuitem', { name: '🔥' }))

    await waitFor(() => expect(toggleSessionStamp).toHaveBeenCalledWith('s1', 'p1', '🔥'))
  })
})
