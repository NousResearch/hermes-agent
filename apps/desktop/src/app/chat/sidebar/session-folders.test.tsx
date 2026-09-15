import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { $activeGatewayProfile, $showAllProfiles } from '@/store/profile'
import { $sessionFolderStore } from '@/store/session-folders'

import { type ReorderDropTarget, resolveReorderDrop } from './reorderable-list'
import { folderDropTarget, UNFILED_DROP_TARGET } from './session-folders'
import { SidebarSessionsSection } from './sessions-section'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
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
          delete: 'Delete folder',
          deleteHint: 'Delete this folder?',
          label: 'Folders',
          newFolder: 'New folder',
          rename: 'Rename folder',
          unfiled: 'Not in a folder',
          unfiledHint: 'Drop a session here'
        }
      }
    }
  })
}))

// Flat-list tests only care about WHERE a row landed: keep rows inert so they
// can't collide on accessible names or pull in row-level store dependencies.
vi.mock('./session-row', () => ({
  SidebarSessionRow: ({ session }: { session: SessionInfo }) => (
    <div data-testid={`session-row-${session.id}`}>{session.id}</div>
  )
}))

const noop = () => {}

const session = (id: string, startedAt = 1000): SessionInfo =>
  ({ id, last_active: startedAt, profile: 'default', started_at: startedAt }) as unknown as SessionInfo

const folderStore = (filed: Record<string, string>, collapsed = false) => ({
  default: {
    filed,
    folders: [{ collapsed, id: 'f_work', name: 'Work' }]
  }
})

const renderSection = (sessions: SessionInfo[], enableFolders = true) =>
  render(
    <SidebarSessionsSection
      activeSessionId={null}
      emptyState={<div>Empty</div>}
      enableFolders={enableFolders}
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
  )

beforeEach(() => {
  window.localStorage.clear()
  $showAllProfiles.set(false)
  $activeGatewayProfile.set('default')
  $sessionFolderStore.set({})
})

describe('sidebar session folders', () => {
  it('lists a filed session under its folder and out of the flat list', () => {
    $sessionFolderStore.set(folderStore({ s1: 'f_work' }))

    const { container } = renderSection([session('s1'), session('s2')])
    const folder = container.querySelector('[data-session-folder-id="f_work"]') as HTMLElement

    // The folder header is the drop target for its own id, so a row dropped on
    // it routes to this folder.
    expect(folder.querySelector('[data-folder-drop="f_work"]')).not.toBeNull()
    expect(within(folder).getByText('Work')).toBeTruthy()

    // Filed once, in the folder — never also in the main list below it.
    expect(container.querySelectorAll('[data-testid="session-row-s1"]')).toHaveLength(1)
    expect(folder.contains(screen.getByTestId('session-row-s1'))).toBe(true)

    // Its unfiled sibling stays in the main list.
    expect(folder.contains(screen.getByTestId('session-row-s2'))).toBe(false)
  })

  it('hides a collapsed folder\u2019s sessions', () => {
    $sessionFolderStore.set(folderStore({ s1: 'f_work' }, true))

    renderSection([session('s1'), session('s2')])

    expect(screen.queryByTestId('session-row-s1')).toBeNull()
    expect(screen.getByTestId('session-row-s2')).toBeTruthy()
  })

  it('leaves the list alone where folders are not enabled', () => {
    $sessionFolderStore.set(folderStore({ s1: 'f_work' }))

    const { container } = renderSection([session('s1'), session('s2')], false)

    expect(container.querySelector('[data-session-folders]')).toBeNull()
    expect(screen.getByTestId('session-row-s1')).toBeTruthy()
  })

  it('creates, renames and deletes a folder from the strip', () => {
    $sessionFolderStore.set(folderStore({ s1: 'f_work' }))

    const { container } = renderSection([session('s1'), session('s2')])

    fireEvent.click(screen.getByRole('button', { name: 'New folder' }))
    expect($sessionFolderStore.get().default.folders.map(folder => folder.name)).toEqual(['Work', 'New folder'])

    fireEvent.click(screen.getByRole('button', { name: 'Rename folder: Work' }))
    const input = screen.getByLabelText('Rename folder')
    fireEvent.change(input, { target: { value: 'Deep work' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect($sessionFolderStore.get().default.folders[0].name).toBe('Deep work')
    // Renaming keeps the membership: the row is still inside the folder.
    expect(
      (container.querySelector('[data-session-folder-id="f_work"]') as HTMLElement).contains(
        screen.getByTestId('session-row-s1')
      )
    ).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: 'Delete folder: Deep work' }))

    // The folder is gone and its session is back in the main list — the session
    // itself was never touched. The folder made earlier in this test stays.
    expect($sessionFolderStore.get().default.folders.map(folder => folder.name)).toEqual(['New folder'])
    expect(screen.getByTestId('session-row-s1')).toBeTruthy()
  })

  it('routes a drop on a folder to that folder, and a drop on a row to the list', () => {
    const fileInFolder = vi.fn<ReorderDropTarget['onDrop']>()
    const unfiled = vi.fn<ReorderDropTarget['onDrop']>()

    const targets: ReorderDropTarget[] = [
      { id: folderDropTarget('f_work'), onDrop: fileInFolder },
      { id: UNFILED_DROP_TARGET, onDrop: unfiled }
    ]

    const ids = ['s1', 's2']

    expect(resolveReorderDrop('s1', folderDropTarget('f_work'), ids, targets)).toEqual({
      kind: 'drop-target',
      targetId: folderDropTarget('f_work')
    })
    expect(resolveReorderDrop('s1', UNFILED_DROP_TARGET, ids, targets)).toEqual({
      kind: 'drop-target',
      targetId: UNFILED_DROP_TARGET
    })

    // A drop on another row is still an ordinary reorder.
    expect(resolveReorderDrop('s1', 's2', ids, targets)).toEqual({ ids: ['s2', 's1'], kind: 'reorder' })

    // A folder's own row is not in the reorder order: dropping it on a row does
    // neither (it is moved by dropping it on a folder).
    expect(resolveReorderDrop('s9', 's2', ids, targets)).toEqual({ kind: 'none' })
    expect(resolveReorderDrop('s1', null, ids, targets)).toEqual({ kind: 'none' })
    expect(resolveReorderDrop('s1', 's1', ids, targets)).toEqual({ kind: 'none' })
  })
})
