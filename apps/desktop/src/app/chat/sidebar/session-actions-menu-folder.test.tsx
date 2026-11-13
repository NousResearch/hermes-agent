import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { copyFilePath, revealFile, shouldOfferLocalReveal } from '@/store/file-actions'
import { $sessions } from '@/store/session'

import { SessionContextMenu } from './session-actions-menu'

// A tab's right-click menu must answer "which folder is this chat in?" for
// THAT tab's session — not the focused one — so these drive the real menu
// against a session row carrying its own cwd.

vi.mock('@/components/pane-shell/tree/store', () => ({
  closeAllTreeTabs: vi.fn(),
  closeOtherTreeTabs: vi.fn(),
  closeTreeTabsToRight: vi.fn(),
  reloadTreePane: vi.fn(),
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
      common: { cancel: 'Cancel', close: 'Close', delete: 'Delete' },
      errors: { genericFailure: 'Something went wrong' },
      fileMenu: { copyPath: 'Copy path', revealFileManager: 'Open containing folder' },
      sidebar: {
        projects: { menuAppearance: 'Appearance', moveToProject: 'Move to project', noColor: 'No color' },
        row: {
          archive: 'Archive',
          branchFrom: 'Branch from here',
          copyId: 'Copy ID',
          copyIdFailed: 'Failed to copy ID',
          deleteDesc: (title: string) => `Delete ${title}?`,
          deleteTitle: 'Delete session?',
          deleted: 'Session deleted',
          deleting: 'Deleting…',
          export: 'Export',
          markRead: 'Mark as read',
          markUnread: 'Mark as unread',
          pin: 'Pin',
          rename: 'Rename',
          renameDesc: 'Leave empty to clear.',
          renameTitle: 'Rename session',
          untitledPlaceholder: 'Untitled'
        }
      },
      zones: {
        closeAll: 'Close all',
        closeOthers: 'Close others',
        closeToRight: 'Close to the right',
        reload: 'Reload'
      }
    }
  })
}))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/lib/profile-color', () => ({ PROFILE_SWATCHES: [] }))
vi.mock('@/lib/session-export', () => ({ exportSession: vi.fn() }))
vi.mock('@/store/file-actions', () => ({
  copyFilePath: vi.fn(() => Promise.resolve()),
  revealFile: vi.fn(() => Promise.resolve()),
  shouldOfferLocalReveal: vi.fn(() => true)
}))
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
  sessionMatchesStoredId: (s: { id: string }, id: string) => s.id === id,
  sessionPinId: (s: { id: string }) => s.id,
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

beforeEach(() => {
  vi.mocked(shouldOfferLocalReveal).mockReturnValue(true)
  vi.mocked(copyFilePath).mockClear()
  vi.mocked(revealFile).mockClear()
})

afterEach(() => {
  cleanup()
  ;($sessions as unknown as { set: (v: unknown[]) => void }).set([])
})

function setRows(rows: { connection_id?: string; cwd?: string; id: string }[]) {
  ;($sessions as unknown as { set: (v: unknown[]) => void }).set(rows)
}

async function openTabMenu(sessionId: string) {
  render(
    <SessionContextMenu sessionId={sessionId} surface="tab" title="Tab">
      <button aria-label="Tab" type="button">
        Tab
      </button>
    </SessionContextMenu>
  )

  fireEvent.contextMenu(screen.getByRole('button', { name: 'Tab' }))
  await screen.findByRole('menu')
}

describe('session menu workspace folder items', () => {
  it("copies and reveals the menu's own session cwd, not another session's", async () => {
    setRows([
      { cwd: '/work/other-project', id: 'focused' },
      { cwd: '/work/vault', id: 'tab-2' }
    ])
    await openTabMenu('tab-2')

    fireEvent.click(screen.getByRole('menuitem', { name: 'Copy path' }))
    expect(copyFilePath).toHaveBeenCalledWith('/work/vault')

    fireEvent.contextMenu(screen.getByRole('button', { name: 'Tab' }))
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Open containing folder' }))
    expect(revealFile).toHaveBeenCalledWith('/work/vault')
  })

  it('omits reveal for a workspace that is not on this computer, but still copies its path', async () => {
    vi.mocked(shouldOfferLocalReveal).mockReturnValue(false)
    setRows([{ connection_id: 'ssh-box', cwd: '/srv/app', id: 'remote' }])
    await openTabMenu('remote')

    expect(shouldOfferLocalReveal).toHaveBeenCalledWith('ssh-box')
    expect(screen.queryByRole('menuitem', { name: 'Open containing folder' })).toBeNull()

    fireEvent.click(screen.getByRole('menuitem', { name: 'Copy path' }))
    expect(copyFilePath).toHaveBeenCalledWith('/srv/app')
  })

  it('disables both items for a session with no known cwd', async () => {
    setRows([{ id: 'draft' }])
    await openTabMenu('draft')

    expect(screen.getByRole('menuitem', { name: 'Copy path' }).getAttribute('data-disabled')).not.toBeNull()
    expect(
      screen.getByRole('menuitem', { name: 'Open containing folder' }).getAttribute('data-disabled')
    ).not.toBeNull()
  })
})
