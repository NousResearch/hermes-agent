import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

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
      desktop: {
        sessionExportTitle: 'Export conversation',
        sessionExportFormat: 'File format',
        sessionExportRedact: 'Redact detected secrets',
        sessionExportRedactHint: 'Review before sharing',
        sessionExportAction: 'Export',
        sessionExported: 'Exported',
        sessionExportFailed: 'Export failed',
        sessionExportNeedsLive: 'Needs live',
        sessionExportUnsupported: 'Unsupported'
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
vi.mock('@/lib/session-export', () => ({ exportSession: vi.fn(), exportSessionFile: vi.fn() }))
vi.mock('@/store/gateway', () => ({ activeGateway: vi.fn(() => null) }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/projects', () => ({
  $projectTree: atom<unknown[]>([]),
  moveSessionToProject: vi.fn(),
  projectIdForCwd: vi.fn(() => null),
  projectRootCwd: vi.fn(() => ''),
  refreshProjectTree: vi.fn(() => Promise.resolve())
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
  $sessionStates: atom<Record<string, unknown>>({}),
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

it('opens an export chooser without directly exporting', async () => {
  const { exportSession } = await import('@/lib/session-export')
  render(
    <SessionActionsMenu sessionId="s1" title="My session">
      <button type="button">Actions</button>
    </SessionActionsMenu>
  )
  fireEvent.pointerDown(screen.getByRole('button', { name: 'Actions' }), { button: 0, pointerType: 'mouse' })
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Export' }))
  expect(exportSession).not.toHaveBeenCalled()
  expect(await screen.findByRole('dialog')).toBeTruthy()
})

it('retains the captured target when the row changes', async () => {
  const { exportSessionFile } = await import('@/lib/session-export')
  vi.mocked(exportSessionFile).mockResolvedValue({ kind: 'canceled' })

  const row = (id: string) => (
    <SessionActionsMenu profile="personal" sessionId={id} title={id}>
      <button type="button">Actions</button>
    </SessionActionsMenu>
  )

  const { rerender } = render(row('A'))
  fireEvent.pointerDown(screen.getByRole('button', { name: 'Actions' }), { button: 0, pointerType: 'mouse' })
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Export' }))
  rerender(row('B'))
  const dialog = screen.queryByRole('dialog')

  if (dialog) {
    fireEvent.click(within(dialog).getByRole('button', { name: 'Export' }))
    await waitFor(() =>
      expect(exportSessionFile).toHaveBeenCalledWith({
        sessionId: 'A',
        profile: 'personal',
        title: 'A',
        format: 'md',
        redact: false,
        options: {
          include_tools: false,
          include_reasoning: false,
          include_images: true,
          include_videos: true,
          include_files: true,
          include_links: true,
          include_asides: true,
          collapse_details: true
        }
      })
    )
  } else {
    expect(exportSessionFile).not.toHaveBeenCalled()
  }
})

it('opens from the context menu', async () => {
  render(
    <SessionContextMenu sessionId="s1" title="Context chat">
      <button type="button">Row</button>
    </SessionContextMenu>
  )
  fireEvent.contextMenu(screen.getByRole('button', { name: 'Row' }))
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Export' }))
  expect((await screen.findByRole('dialog')).textContent).toContain('Context chat')
})

it('resets choices on reopening and leaves focus inside the chooser', async () => {
  render(
    <SessionActionsMenu sessionId="s1" title="My session">
      <button type="button">Actions</button>
    </SessionActionsMenu>
  )
  const trigger = screen.getByRole('button', { name: 'Actions' })
  fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Export' }))
  let dialog = await screen.findByRole('dialog')
  await waitFor(() => expect(dialog.contains(window.document.activeElement)).toBe(true))
  fireEvent.click(within(dialog).getByRole('button', { name: 'HTML' }))
  fireEvent.click(within(dialog).getByRole('checkbox', { name: 'Redact detected secrets' }))
  fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }))
  fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Export' }))
  dialog = await screen.findByRole('dialog')
  expect(within(dialog).getByRole('button', { name: 'Markdown' }).getAttribute('aria-pressed')).toBe('true')
  expect(within(dialog).getByRole('checkbox', { name: 'Redact detected secrets' }).getAttribute('aria-checked')).toBe(
    'false'
  )
})
