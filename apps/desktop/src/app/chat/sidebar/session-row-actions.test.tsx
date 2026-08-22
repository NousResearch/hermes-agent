import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { SidebarSessionRow } from './session-row'

afterEach(cleanup)

vi.mock('@/app/chat/profile-tag', () => ({ ProfileTag: () => null }))
vi.mock('@/app/chat/session-drag', () => ({ startSessionDrag: vi.fn() }))
vi.mock('@/components/pane-shell/tree/store', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  closeAllTreeTabs: vi.fn(),
  closeOtherTreeTabs: vi.fn(),
  closeTreeTabsToRight: vi.fn(),
  reloadTreePane: vi.fn(),
  treeTabCloseTargets: vi.fn(() => null)
}))
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  renameSession: vi.fn()
}))
vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      assistant: {
        thread: {
          today: (time: string) => `Today, ${time}`,
          yesterday: (time: string) => `Yesterday, ${time}`
        }
      },
      common: { cancel: 'Cancel', close: 'Close', delete: 'Delete', home: 'Home', save: 'Save' },
      sidebar: {
        messageCount: (count: number) => `${count} messages`,
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
        },
        toolCallCount: (count: number) => `${count} tool calls`
      },
      zones: { closeAll: 'Close all', closeOthers: 'Close others', closeToRight: 'Close to the right' }
    }
  })
}))
vi.mock('@/lib/chat-runtime', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  sessionTitle: (session: SessionInfo) => session.title ?? 'Untitled'
}))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/lib/profile-color', () => ({ PROFILE_SWATCHES: [] }))
vi.mock('@/lib/session-export', () => ({ exportSession: vi.fn() }))
vi.mock('@/lib/session-source', () => ({ handoffOriginSource: () => null, sessionSourceLabel: () => '' }))
vi.mock('@/lib/time', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  coarseElapsed: () => ({ unit: 'minute' as const, value: 5 })
}))
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  activeGateway: vi.fn(() => null)
}))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/projects', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  $projectTree: atom<unknown[]>([]),
  $projects: atom<unknown[]>([]),
  moveSessionToProject: vi.fn(),
  projectIdForCwd: vi.fn(() => null),
  projectRootCwd: vi.fn(() => '')
}))
vi.mock('@/store/windows', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  canOpenSessionWindow: () => false,
  openSessionInNewWindow: vi.fn()
}))
vi.mock('./use-profile-prewarm', () => ({
  useProfilePrewarm: () => ({ cancelPrewarm: vi.fn(), startPrewarm: vi.fn() })
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
  it('archives an Inbox card without also resuming it', async () => {
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
