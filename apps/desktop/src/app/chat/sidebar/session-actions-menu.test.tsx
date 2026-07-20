import { atom } from 'nanostores'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SessionActionsMenu } from './session-actions-menu'

afterEach(cleanup)

// This file exists specifically to catch the regression flagged in #67500:
// SessionActionsMenu used to be composed as
//   <DropdownMenuTrigger asChild>{children}</DropdownMenuTrigger>
// with the caller wrapping ITS children in <Tip>. Radix's `asChild` clones
// its single child and injects onClick/aria-haspopup/ref onto it — but Tip
// doesn't forward those extra props to whatever it wraps, so they were
// silently dropped and the menu could stop opening. Tip has since moved
// inside this component (wrapping DropdownMenuTrigger itself, not the other
// way around) — these tests exercise the REAL component end-to-end (no mock
// of DropdownMenu/Tip) so a future regression of this composition fails here.

vi.mock('@/components/pane-shell/tree/store', () => ({
  closeAllTreeTabs: vi.fn(),
  closeOtherTreeTabs: vi.fn(),
  closeTreeTabsToRight: vi.fn(),
  treeTabCloseTargets: vi.fn(() => null)
}))
vi.mock('@/hermes', () => ({ renameSession: vi.fn() }))
vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel', close: 'Close', delete: 'Delete', save: 'Save' },
      sidebar: {
        projects: { menuAppearance: 'Appearance', noColor: 'No color' },
        row: {
          actionsFor: (title: string) => `Actions for ${title}`,
          archive: 'Archive',
          branchFrom: 'Branch from here',
          copyId: 'Copy ID',
          copyIdFailed: 'Failed to copy ID',
          export: 'Export',
          hideTabBar: 'Hide tab bar',
          pin: 'Pin',
          rename: 'Rename',
          renameDesc: 'Rename this session',
          renameFailed: 'Rename failed',
          renameTitle: 'Rename session',
          renamed: 'Renamed',
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
vi.mock('@/store/session', () => ({
  $activeSessionId: atom<null | string>(null),
  $selectedStoredSessionId: atom<null | string>(null),
  $sessions: atom<unknown[]>([]),
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
  openSessionTile: vi.fn()
}))
vi.mock('@/store/windows', () => ({
  canOpenSessionWindow: () => false,
  openSessionInNewWindow: vi.fn()
}))

function renderMenu() {
  return render(
    <SessionActionsMenu sessionId="s1" title="My session" tooltip="Actions for My session">
      <button aria-label="Actions for My session" type="button">
        ⋮
      </button>
    </SessionActionsMenu>
  )
}

describe('SessionActionsMenu', () => {
  it('shows the tooltip label wired to the real trigger button', () => {
    renderMenu()

    const trigger = screen.getByRole('button', { name: 'Actions for My session' })

    expect(trigger.closest('[data-slot="tooltip-trigger"]')).toBeTruthy()
  })

  it('still opens the dropdown on click with the trigger wrapped in a Tip (#67500)', async () => {
    renderMenu()

    const trigger = screen.getByRole('button', { name: 'Actions for My session' })

    // Radix's dropdown trigger opens on pointerdown (not on the synthetic
    // 'click' fireEvent alone would dispatch), so fire the full mouse
    // sequence a real click produces.
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(trigger)

    // If Tip (now composed around DropdownMenuTrigger, not the other way
    // round) ever stopped forwarding the asChild-injected props again, this
    // menu would never open and these queries would throw instead of
    // resolving.
    expect(await screen.findByRole('menu')).toBeTruthy()
    expect(screen.getByRole('menuitem', { name: /rename/i })).toBeTruthy()
    expect(screen.getByRole('menuitem', { name: /archive/i })).toBeTruthy()
  })
})
