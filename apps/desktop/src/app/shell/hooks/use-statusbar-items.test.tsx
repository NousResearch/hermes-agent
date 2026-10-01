import { renderHook } from '@testing-library/react'
import type { WritableAtom } from 'nanostores'
import { isValidElement, type ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $connection,
  $currentCwd,
  $selectedStoredSessionId,
  $sessions,
  $sessionStartedAt,
  $tileSessionFocusStartedAt,
  setActiveSessionId
} from '@/store/session'
import { $focusedTreePaneId as $focusedTreePaneIdMock } from '@/store/session-focus'
import { $sessionStates, $sessionTiles } from '@/store/session-states'

import { useStatusbarItems } from './use-statusbar-items'

// The mock above replaces the computed store with a writable atom.
const $focusedTreePaneId = $focusedTreePaneIdMock as unknown as WritableAtom<null | string>

// The focused pane is derived from the layout tree; a settable atom stands in
// so a test can focus a tile without building a pane tree.
vi.mock('@/store/session-focus', async () => {
  const { atom, computed } = await import('nanostores')
  const { $selectedStoredSessionId } = await import('@/store/session')

  // The focused pane is derived from the layout tree; a settable atom stands
  // in so a test can focus a tile without building a pane tree.
  const $focusedTreePaneId = atom<null | string>(null)

  // The preview store (reached via session-states) derives $visiblePreviewTabs
  // from $focusedStoredSessionId at import time, and the timer tests depend on
  // the REAL derivation (tile focus overrides the primary selection), so the
  // mock mirrors session-focus's shape instead of stubbing a static atom.
  const TILE_PANE_PREFIX = 'session-tile:'

  return {
    $focusedTreePaneId,
    $focusedStoredSessionId: computed([$focusedTreePaneId, $selectedStoredSessionId], (active, selected) =>
      active?.startsWith(TILE_PANE_PREFIX) ? active.slice(TILE_PANE_PREFIX.length) : selected
    )
  }
})

// $focusedStoredSessionId derives from the LAYOUT TREE ($activeTreeGroup +
// $layoutTree), not from session-focus's pane atom: focusing a tile means the
// main zone's active pane IS that tile. Drive the tree the same way.
const MAIN_GROUP_ID = 'statusbar-test-main'

function focusPane(storedId: null | string): void {
  const tilePane = storedId ? `session-tile:${storedId}` : null

  $layoutTree.set(
    tilePane
      ? group(['workspace', tilePane], { active: tilePane, id: MAIN_GROUP_ID })
      : group(['workspace'], { active: 'workspace', id: MAIN_GROUP_ID })
  )
  noteActiveTreeGroup(MAIN_GROUP_ID)
  $focusedTreePaneId.set(tilePane)
}

const wrapper = ({ children }: { children: ReactNode }) => <MemoryRouter>{children}</MemoryRouter>

function workspaceMenuIds(): string[] {
  const { result } = renderHook(
    () =>
      useStatusbarItems({
        agentsOpen: false,
        chatOpen: true,
        commandCenterOpen: false,
        extraLeftItems: [],
        extraRightItems: [],
        freshDraftReady: false,
        gatewayState: 'ready',
        inferenceStatus: null,
        openAgents: () => {},
        openCommandCenterSection: () => {},
        requestGateway: async () => undefined as never,
        statusSnapshot: null,
        toggleCommandCenter: () => {}
      }),
    { wrapper }
  )

  const workspace = result.current.leftStatusbarItems.find(item => item.id === 'workspace-cwd')

  return (workspace?.menuItems ?? []).map(item => item.id)
}

afterEach(() => {
  $connection.set(null)
  $currentCwd.set('')
  $sessionTiles.set([])
  focusPane(null)
  $selectedStoredSessionId.set(null)
  $sessions.set([])
  $sessionStartedAt.set(null)
  $tileSessionFocusStartedAt.set(null)
})

describe('statusbar workspace menu — "Open containing folder"', () => {
  it('offers reveal for a local workspace and hides it when the connection is remote', () => {
    $currentCwd.set('/home/me/project')
    $connection.set({ mode: 'local' } as never)
    expect(workspaceMenuIds()).toContain('reveal-workspace-finder')

    $connection.set({ mode: 'remote' } as never)
    expect(workspaceMenuIds()).not.toContain('reveal-workspace-finder')
  })

  // The reporter's topology (#115167): the window's primary is local but the
  // focused tile is a Connections-tagged session on a remote gateway. The gate
  // follows the FOCUSED session's owner, not the window's primary.
  it('hides reveal for a focused remote tile inside a local-primary window', () => {
    $connection.set({ mode: 'local' } as never)
    $selectedStoredSessionId.set('primary-local')
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'conn-remote', mode: 'remote', profile: 'default' },
        storedSessionId: 'tile-remote'
      }
    ])
    focusPane('tile-remote')
    // A focused tile never inherits the primary's cwd; its stored row carries it.
    $sessions.set([{ cwd: '/srv/bot/workspace', id: 'tile-remote' }] as never)

    expect(workspaceMenuIds()).toContain('copy-workspace-path')
    expect(workspaceMenuIds()).not.toContain('reveal-workspace-finder')
  })
})

const statusbarOptions = {
  agentsOpen: false,
  chatOpen: true,
  commandCenterOpen: false,
  extraLeftItems: [],
  extraRightItems: [],
  freshDraftReady: false,
  gatewayState: 'ready' as const,
  inferenceStatus: null,
  openAgents: () => {},
  openCommandCenterSection: () => {},
  requestGateway: async () => undefined as never,
  statusSnapshot: null,
  toggleCommandCenter: () => {}
}

function sessionTimerItem() {
  const { result } = renderHook(() => useStatusbarItems(statusbarOptions), { wrapper })

  return result.current.statusbarItems.find(item => item.id === 'session-timer')
}

function timerSince(item: ReturnType<typeof sessionTimerItem>): number | null {
  return isValidElement<{ since: number | null }>(item?.detail) ? item.detail.props.since : null
}

describe('statusbar session timer — focused since (#103123)', () => {
  const dayOldRowSeconds = 1_700_000_000

  it('uses the primary focus stamp, not the row age, and labels it so a long value is not a turn', () => {
    $selectedStoredSessionId.set('primary')
    $sessionStartedAt.set(4_000)
    $sessions.set([{ id: 'primary', started_at: dayOldRowSeconds }] as never)
    focusPane(null)

    const item = sessionTimerItem()

    expect(timerSince(item)).toBe(4_000)
    expect(item?.label).toBe('Focused since')
    expect(item?.title).toMatch(/not how long a turn/)
  })

  it('stamps tile focus instead of the day-old row, on the same labeled item', () => {
    const before = Date.now()

    $selectedStoredSessionId.set('primary')
    $sessionStartedAt.set(4_000)
    $sessions.set([{ id: 'tile-old', started_at: dayOldRowSeconds }] as never)
    focusPane('tile-old')

    const item = sessionTimerItem()
    const since = timerSince(item)

    expect(since).toBeGreaterThanOrEqual(before)
    expect(since).toBeLessThanOrEqual(Date.now())
    expect(since).not.toBe(dayOldRowSeconds * 1000)
    expect(since).not.toBe(4_000)
    expect(item?.label).toBe('Focused since')
    expect(item?.hidden).toBeFalsy()
  })

  it('re-stamps when the same tile is focused again after primary', () => {
    const now = vi.spyOn(Date, 'now')

    $selectedStoredSessionId.set('primary')
    now.mockReturnValue(10_000)
    focusPane('tile-old')
    expect($tileSessionFocusStartedAt.get()).toEqual({ since: 10_000, storedId: 'tile-old' })

    now.mockReturnValue(20_000)
    focusPane(null)
    expect($tileSessionFocusStartedAt.get()?.since).toBe(10_000)

    now.mockReturnValue(30_000)
    focusPane('tile-old')
    expect($tileSessionFocusStartedAt.get()).toEqual({ since: 30_000, storedId: 'tile-old' })

    now.mockRestore()
  })

  it('hides the item when a focused tile has no focus stamp', () => {
    $selectedStoredSessionId.set('primary')
    $sessionStartedAt.set(4_000)
    $sessions.set([{ id: 'tile-old', started_at: dayOldRowSeconds }] as never)
    focusPane('tile-old')
    $tileSessionFocusStartedAt.set(null)

    const item = sessionTimerItem()

    expect(timerSince(item)).toBeNull()
    expect(item?.hidden).toBe(true)
  })
})

describe('useStatusbarItems session timer — runtime cache anchor', () => {
  it("anchors a focused branch tile to its runtime cache instead of the parent's stored age", () => {
    const parentRowStartedAt = 1_600_000_000
    const branchRuntimeStartedAt = 1_800_000_000_000

    // The branch is a live runtime on the focused tile; its slice carries the
    // runtime's own start, which must win over the parent row's stored age.
    setActiveSessionId('branch-runtime')
    $selectedStoredSessionId.set('parent-stored')
    $sessionStartedAt.set(1_700_000_000_000)
    $sessions.set([
      { id: 'parent-stored', started_at: parentRowStartedAt },
      { id: 'branch-stored', parent_session_id: 'parent-stored', started_at: parentRowStartedAt }
    ] as never)
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: null, mode: 'local', profile: 'default' },
        runtimeId: 'branch-runtime',
        storedSessionId: 'branch-stored'
      }
    ] as never)
    $sessionStates.set({
      'branch-runtime': { ...createClientSessionState('branch-stored'), runtimeStartedAt: branchRuntimeStartedAt }
    } as never)
    focusPane('branch-stored')

    const item = sessionTimerItem()
    const since = timerSince(item)

    expect(since).toBe(branchRuntimeStartedAt)
    expect(since).not.toBe(parentRowStartedAt * 1000)
  })
})
