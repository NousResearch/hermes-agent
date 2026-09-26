import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
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
  setSessionSlackSyncRemote: vi.fn(() => Promise.resolve({ ok: true })),
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
          startSlackSync: 'Sync to Slack thread',
          stopSlackSync: 'Stop Slack sync',
          slackSyncFailed: 'Could not update Slack sync',
          pin: 'Pin',
          rename: 'Rename',
          renameDesc: 'Leave empty to clear.',
          renameFailed: 'Rename failed',
          renameTitle: 'Rename session',
          renamed: 'Renamed',
          sessionActions: 'Session actions',
          unarchive: 'Unarchive',
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
  sessionMatchesStoredId: vi.fn((row: { id: string }, id: string) => row.id === id),
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
  it('targets only the row with the matching profile and connection when IDs collide', async () => {
    const { $sessions, setSessions } = await import('@/store/session')
    const { setSessionSlackSyncRemote } = await import('@/hermes')
    vi.mocked(setSessionSlackSyncRemote).mockClear()
    vi.mocked(setSessions).mockClear()
    $sessions.set([
      { id: 's1', profile: 'other', connection_id: 'remote-a', slack_sync_available: true, slack_sync: true },
      { id: 's1', profile: 'poweronline', connection_id: 'remote-b', slack_sync_available: true, slack_sync: false },
      { id: 's1', profile: 'poweronline', connection_id: 'remote-a', slack_sync_available: true, slack_sync: false }
    ])
    render(
      <SessionActionsMenu connectionId="remote-a" profile="poweronline" sessionId="s1" title="Target">
        <button aria-label="Session actions" type="button">⋮</button>
      </SessionActionsMenu>
    )
    const trigger = screen.getByRole('button', { name: 'Session actions' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Sync to Slack thread' }))
    await waitFor(() => expect(setSessionSlackSyncRemote).toHaveBeenCalledWith('s1', true, {
      connectionId: 'remote-a', profile: 'poweronline'
    }))
    const update = vi.mocked(setSessions).mock.calls.at(-1)?.[0]
    expect(typeof update).toBe('function')
    expect((update as (rows: unknown[]) => unknown[])($sessions.get())).toEqual([
      $sessions.get()[0], $sessions.get()[1], { ...$sessions.get()[2], slack_sync: true }
    ])
    $sessions.set([])
  })

  it('fails closed when the menu does not identify a unique owner', async () => {
    const { $sessions } = await import('@/store/session')
    const { setSessionSlackSyncRemote } = await import('@/hermes')
    vi.mocked(setSessionSlackSyncRemote).mockClear()
    $sessions.set([
      { id: 's1', profile: 'default', connection_id: 'remote-a', slack_sync_available: true },
      { id: 's1', profile: 'default', connection_id: 'remote-b', slack_sync_available: true }
    ])
    renderMenu()
    const trigger = screen.getByRole('button', { name: 'Session actions' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)
    expect(screen.queryByRole('menuitem', { name: 'Sync to Slack thread' })).toBeNull()
    expect(setSessionSlackSyncRemote).not.toHaveBeenCalled()
    $sessions.set([])
  })

  it('hides consent even for a sole eligible ID match when the menu has no owner', async () => {
    const { $sessions } = await import('@/store/session')
    const { setSessionSlackSyncRemote } = await import('@/hermes')
    vi.mocked(setSessionSlackSyncRemote).mockClear()
    $sessions.set([{ id: 's1', profile: 'default', connection_id: 'local', slack_sync_available: true }])
    renderMenu()
    const trigger = screen.getByRole('button', { name: 'Session actions' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)
    expect(screen.queryByRole('menuitem', { name: 'Sync to Slack thread' })).toBeNull()
    expect(setSessionSlackSyncRemote).not.toHaveBeenCalled()
    $sessions.set([])
  })

  it('hides context-menu consent when its connection was omitted despite a sole match', async () => {
    const { $sessions } = await import('@/store/session')
    $sessions.set([{ id: 's1', profile: 'default', connection_id: 'local', slack_sync_available: true }])
    render(
      <SessionContextMenu profile="default" sessionId="s1" title="Thread">
        <button type="button">Thread</button>
      </SessionContextMenu>
    )
    fireEvent.contextMenu(screen.getByRole('button', { name: 'Thread' }))
    expect(screen.queryByRole('menuitem', { name: 'Sync to Slack thread' })).toBeNull()
    $sessions.set([])
  })

  it('never PATCHes a different owner when the row changes after the menu opens', async () => {
    const { $sessions } = await import('@/store/session')
    const { setSessionSlackSyncRemote } = await import('@/hermes')
    vi.mocked(setSessionSlackSyncRemote).mockClear()
    $sessions.set([{ id: 's1', profile: 'work', connection_id: 'remote-a', slack_sync_available: true }])
    render(
      <SessionActionsMenu connectionId="remote-a" profile="work" sessionId="s1" title="Remote">
        <button aria-label="Session actions" type="button">⋮</button>
      </SessionActionsMenu>
    )
    const trigger = screen.getByRole('button', { name: 'Session actions' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)
    const item = await screen.findByRole('menuitem', { name: 'Sync to Slack thread' })
    // Keep the stale item reference to exercise the click-time guard.
    $sessions.set([{ id: 's1', profile: 'default', connection_id: 'local', slack_sync_available: true }])
    fireEvent.click(item)
    expect(setSessionSlackSyncRemote).not.toHaveBeenCalled()
    $sessions.set([])
  })

  it('does not offer consent for a row whose owning profile is unknown', async () => {
    const { $sessions } = await import('@/store/session')
    $sessions.set([{ id: 's1', connection_id: 'remote-a', slack_sync_available: true }])
    renderMenu()
    const trigger = screen.getByRole('button', { name: 'Session actions' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)
    expect(screen.queryByRole('menuitem', { name: 'Sync to Slack thread' })).toBeNull()
    $sessions.set([])
  })

  it('pins an untagged local row to local even when the active connection is remote', async () => {
    const { $connection, $sessions } = await import('@/store/session')
    const { setSessionSlackSyncRemote } = await import('@/hermes')
    vi.mocked(setSessionSlackSyncRemote).mockClear()
    $connection.set({ mode: 'remote' })
    $sessions.set([
      { id: 's1', profile: 'default', connection_id: 'remote-active', slack_sync_available: true, slack_sync: true },
      { id: 's1', profile: 'default', slack_sync_available: true, slack_sync: false }
    ])
    render(
      <SessionActionsMenu connectionId="local" profile="default" sessionId="s1" title="Local">
        <button aria-label="Session actions" type="button">⋮</button>
      </SessionActionsMenu>
    )
    const trigger = screen.getByRole('button', { name: 'Session actions' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Sync to Slack thread' }))
    await waitFor(() => expect(setSessionSlackSyncRemote).toHaveBeenCalledWith('s1', true, {
      connectionId: 'local', profile: 'default'
    }))
    $connection.set(null)
    $sessions.set([])
  })

  it('only offers Slack sync on an eligible conversation and writes its owning profile', async () => {
    const { $sessions } = await import('@/store/session')
    const { setSessionSlackSyncRemote } = await import('@/hermes')
    $sessions.set([{ id: 's1', profile: 'poweronline', connection_id: 'remote-a', slack_sync_available: true, slack_sync: false }])
    render(
      <SessionActionsMenu connectionId="remote-a" profile="poweronline" sessionId="s1" title="Thread">
        <button aria-label="Session actions" type="button">⋮</button>
      </SessionActionsMenu>
    )

    const trigger = screen.getByRole('button', { name: 'Session actions' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Sync to Slack thread' }))
    await waitFor(() => expect(setSessionSlackSyncRemote).toHaveBeenCalledWith('s1', true, {
      connectionId: 'remote-a', profile: 'poweronline'
    }))

    cleanup()
    $sessions.set([{ id: 's1', profile: 'poweronline', slack_sync_available: false }])
    render(
      <SessionActionsMenu connectionId="local" profile="poweronline" sessionId="s1" title="Thread">
        <button aria-label="Session actions" type="button">⋮</button>
      </SessionActionsMenu>
    )
    const otherTrigger = screen.getByRole('button', { name: 'Session actions' })
    fireEvent.pointerDown(otherTrigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(otherTrigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(otherTrigger)
    expect(screen.queryByRole('menuitem', { name: 'Sync to Slack thread' })).toBeNull()
    cleanup()
    $sessions.set([{ id: 's1', profile: 'poweronline', connection_id: 'remote-a', slack_sync_available: true, slack_sync: true }])
    render(
      <SessionActionsMenu connectionId="remote-a" profile="poweronline" sessionId="s1" title="Thread">
        <button aria-label="Session actions" type="button">⋮</button>
      </SessionActionsMenu>
    )
    const enabledTrigger = screen.getByRole('button', { name: 'Session actions' })
    fireEvent.pointerDown(enabledTrigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(enabledTrigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(enabledTrigger)
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Stop Slack sync' }))
    await waitFor(() => expect(setSessionSlackSyncRemote).toHaveBeenCalledWith('s1', false, {
      connectionId: 'remote-a', profile: 'poweronline'
    }))
    $sessions.set([])
  })

  it('keeps opt-out available after an enabled thread loses bot eligibility', async () => {
    const { $sessions } = await import('@/store/session')
    const { setSessionSlackSyncRemote } = await import('@/hermes')
    vi.mocked(setSessionSlackSyncRemote).mockClear()
    $sessions.set([{
      id: 's1', profile: 'default', connection_id: 'local',
      slack_sync_available: false, slack_sync: true
    }])
    render(
      <SessionActionsMenu connectionId="local" profile="default" sessionId="s1" title="Thread">
        <button aria-label="Session actions" type="button">⋮</button>
      </SessionActionsMenu>
    )
    const trigger = screen.getByRole('button', { name: 'Session actions' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Stop Slack sync' }))
    await waitFor(() => expect(setSessionSlackSyncRemote).toHaveBeenCalledWith('s1', false, {
      connectionId: 'local', profile: 'default'
    }))
    $sessions.set([])
  })

  it('opens the dropdown on click', async () => {
    renderMenu()

    const trigger = screen.getByRole('button', { name: 'Session actions' })

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

  // The sidebar's Archived view reuses this menu; its rows must offer the
  // restore verb instead of a no-op re-archive (#98813). The item still fires
  // the shared onArchive callback — the wiring dispatches it to the restore
  // path based on the row's archived state.
  it('labels the archive verb Unarchive for an already-archived row and fires the shared callback', async () => {
    const onArchive = vi.fn()
    render(
      <SessionActionsMenu archived onArchive={onArchive} sessionId="s1" title="My session">
        <button aria-label="Session actions" type="button">
          ⋮
        </button>
      </SessionActionsMenu>
    )

    const trigger = screen.getByRole('button', { name: 'Session actions' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)

    const restoreItem = await screen.findByRole('menuitem', { name: /^unarchive$/i })
    expect(screen.queryByRole('menuitem', { name: /^archive$/i })).toBeNull()

    fireEvent.click(restoreItem)
    await waitFor(() => expect(onArchive).toHaveBeenCalledTimes(1))
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
