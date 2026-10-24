import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as HermesModule from '@/hermes'
import type { SessionInfo } from '@/hermes'
import type * as SessionStore from '@/store/session'
import type * as SessionStatesStore from '@/store/session-states'

import { SidebarSessionRow } from './session-row'

afterEach(cleanup)

// Exercises the REAL SessionActionsMenu inside the REAL row (no menu stub, no
// DropdownMenu mock) so a menu-item click that bubbles into the card row's
// resume onClick fails here — Radix portals the menu content, but React still
// propagates synthetic events through the logical parent (#85163).

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel', close: 'Close', delete: 'Delete', save: 'Save' },
      assistant: {
        thread: {
          today: (time: string) => `Today, ${time}`,
          yesterday: (time: string) => `Yesterday, ${time}`
        }
      },
      sidebar: {
        messageCount: (count: number) => `${count} messages`,
        toolCallCount: (count: number) => `${count} tool calls`,
        projects: {
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
          deleting: 'Deleting…',
          deleted: 'Session deleted',
          export: 'Export',
          finishedUnread: 'Finished',
          handoffOrigin: (platform: string) => `Started on ${platform}`,
          hideTabBar: 'Hide tab bar',
          messageCount: (count: number) => `${count} messages`,
          needsInput: 'Needs input',
          pin: 'Pin',
          rename: 'Rename',
          renameDesc: 'Leave empty to clear.',
          renameFailed: 'Rename failed',
          renameTitle: 'Rename session',
          renamed: 'Renamed',
          sessionActions: 'Session actions',
          sessionRunning: 'Running',
          unpin: 'Unpin',
          untitledPlaceholder: 'Untitled',
          waitingForAnswer: 'Waiting for answer'
        }
      },
      zones: { closeAll: 'Close all', closeOthers: 'Close others', closeToRight: 'Close to the right' }
    }
  })
}))
vi.mock('@/app/chat/profile-tag', () => ({ ProfileTag: () => null }))
vi.mock('@/app/chat/session-drag', () => ({ startSessionDrag: vi.fn() }))
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesModule>()),
  renameSession: vi.fn(),
  setSessionUnreadRemote: vi.fn(() => Promise.resolve({ ok: true }))
}))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/lib/session-source', () => ({
  handoffOriginSource: () => null,
  sessionSourceLabel: () => ''
}))
vi.mock('@/lib/session-export', () => ({ exportSession: vi.fn() }))
vi.mock('@/lib/time', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  coarseElapsed: () => ({ unit: 'minute' as const, value: 5 })
}))
vi.mock('@/lib/profile-color', () => ({ PROFILE_SWATCHES: [] }))
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  activeGateway: vi.fn(() => null)
}))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/projects', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  moveSessionToProject: vi.fn(),
  projectIdForCwd: vi.fn(() => null),
  projectRootCwd: vi.fn(() => '')
}))
vi.mock('@/store/session', async importOriginal => {
  const actual = await importOriginal<typeof SessionStore>()

  return { ...actual, $unreadFinishedSessionIds: atom<string[]>([]) }
})
vi.mock('@/store/session-states', async importOriginal => {
  const actual = await importOriginal<typeof SessionStatesStore>()

  return { ...actual, openSessionTile: vi.fn() }
})
vi.mock('@/store/windows', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  canOpenSessionWindow: () => false,
  openSessionInNewWindow: vi.fn()
}))
vi.mock('./use-profile-prewarm', () => ({
  useProfilePrewarm: () => ({ cancelPrewarm: vi.fn(), notePointerMove: vi.fn(), startPrewarm: vi.fn() })
}))

const session = {
  cwd: '/tmp/project',
  handoff_platform: null,
  handoff_state: null,
  id: 's1',
  last_active: 0,
  message_count: 1,
  profile: 'default',
  started_at: 0,
  title: 'Archive me'
} as SessionInfo

describe('SidebarSessionRow actions', () => {
  it('archives an Inbox card without also resuming it (#85163)', async () => {
    const onArchive = vi.fn()
    const onResume = vi.fn()

    render(
      <SidebarSessionRow
        card
        isPinned={false}
        isSelected={false}
        onArchive={onArchive}
        onDelete={vi.fn()}
        onPin={vi.fn()}
        onResume={onResume}
        onToggleUnread={vi.fn()}
        session={session}
        unread={false}
      />
    )

    // Full mouse gesture (pointerDown/up + click) — the same sequence Radix
    // listens for on the trigger and the menu items.
    const trigger = screen.getByRole('button', { name: 'Session actions' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)

    const archive = await screen.findByRole('menuitem', { name: 'Archive' })
    fireEvent.pointerDown(archive, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(archive, { button: 0, pointerType: 'mouse' })
    fireEvent.click(archive)

    expect(onArchive).toHaveBeenCalledTimes(1)
    expect(onResume).not.toHaveBeenCalled()
  })
})
