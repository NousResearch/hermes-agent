import { renderHook } from '@testing-library/react'
import type { WritableAtom } from 'nanostores'
import { isValidElement, type ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $freeTierStatus } from '@/store/free-tier'
import { INTERFACE_MODES, setInterfaceMode } from '@/store/interface-mode'
import { markQuestionnaireDecided, setOnboardingSurfaceActive } from '@/store/onboarding-presence'
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
import { $statusbarVisible } from '@/store/statusbar-prefs'
import type { FreeTierStatus } from '@/types/hermes'

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
    $focusedSessionIsTile: computed($focusedTreePaneId, active => Boolean(active?.startsWith(TILE_PANE_PREFIX))),
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

function renderStatusbarItems() {
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

  return result.current
}

function workspaceMenuIds(): string[] {
  const workspace = renderStatusbarItems().leftStatusbarItems.find(item => item.id === 'workspace-cwd')

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
    expect(item?.label).toBe('Focused for')
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
    expect(item?.label).toBe('Focused for')
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

// The statusbar chip is a free-tier user's standing way in to a sign-in, and
// an interface mode decides whether the bar is mounted at all. Whatever a mode
// rests, a signed-out free-tier user keeps one visible Sign in.
describe('free-tier Sign in in every interface mode', () => {
  const freeTier = (available: boolean): FreeTierStatus => ({
    available,
    enabled: true,
    has_guest: available,
    label: 'Nous · free tier',
    model: 'nous/welcome',
    notice_pending: false
  })

  function signInChipVisible(): boolean {
    const { leftStatusbarItems, statusbarItems } = renderStatusbarItems()
    const chip = [...leftStatusbarItems, ...statusbarItems].find(item => item.id === 'free-tier')

    return $statusbarVisible.get() && chip !== undefined && !chip.hidden
  }

  beforeEach(() => markQuestionnaireDecided())

  afterEach(() => {
    $freeTierStatus.set(null)
    setOnboardingSurfaceActive('questionnaire', false)
    setInterfaceMode('advanced')
  })

  it.each(INTERFACE_MODES)('shows the Sign in chip to a signed-out free-tier user in %s mode', mode => {
    setInterfaceMode(mode)
    $freeTierStatus.set(freeTier(true))

    expect(signInChipVisible()).toBe(true)
  })

  it('lets Simple rest the bar again once the user has signed in', () => {
    setInterfaceMode('simple')
    $freeTierStatus.set(freeTier(true))
    expect($statusbarVisible.get()).toBe(true)

    $freeTierStatus.set(freeTier(false))
    expect($statusbarVisible.get()).toBe(false)
  })

  it('holds the chip while the questionnaire is open and brings it back after', () => {
    $freeTierStatus.set(freeTier(true))
    setOnboardingSurfaceActive('questionnaire', true)
    expect(signInChipVisible()).toBe(false)

    setOnboardingSurfaceActive('questionnaire', false)
    expect(signInChipVisible()).toBe(true)
  })
})
