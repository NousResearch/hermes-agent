import { act, cleanup, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { StableNavigateProvider } from '@/app/stable-navigate'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { PROFILE_SWATCHES } from '@/lib/profile-color'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $selectedStoredSessionId, $sessions, sessionPinId } from '@/store/session'
import { $sessionColorOverrides, setSessionColorOverride } from '@/store/session-color'
import { $sessionTiles } from '@/store/session-states'
import { makeCwdSession } from '@/test/session-info'

import { SimplifiedSessionList } from './simplified-session-list'

const noop = () => {}

// A real picker value (the row menu's Appearance submenu writes PROFILE_SWATCHES).
const SWATCH = PROFILE_SWATCHES[0]

function mount(sessions: SessionInfo[]) {
  return render(
    <MemoryRouter><StableNavigateProvider><RootTooltipProvider>
      <SimplifiedSessionList
        activeSessionId={null}
        ambiguousIds={new Set()}
        onArchiveSession={noop}
        onBranchSession={noop}
        onDeleteSession={noop}
        onResumeSession={noop}
        onUpdate={noop}
        pinnedIds={new Set()}
        sessions={sessions}
      />
    </RootTooltipProvider></StableNavigateProvider></MemoryRouter>
  )
}

/** The innermost span of the settled mark — the element the stylesheet paints. */
const idleInk = (container: HTMLElement) =>
  container.querySelector<HTMLElement>('.portal-row-mark[data-state="idle"] > span > span')!

/** jsdom normalizes whatever color it is handed, so compare against that same form. */
function painted(color: string) {
  const probe = document.createElement('span')

  probe.style.backgroundColor = color

  return probe.style.backgroundColor
}

beforeEach(() => {
  $apiRequestScope.set({ connectionId: 'local', profile: 'default' })
  $activeGatewayProfile.set('default')
  $selectedStoredSessionId.set(null)
  $activeSessionId.set(null)
  $sessions.set([])
  $sessionTiles.set([])
  $sessionColorOverrides.set({})
})

afterEach(() => {
  cleanup()
  $sessions.set([])
  $sessionTiles.set([])
  $sessionColorOverrides.set({})
  $apiRequestScope.set({ connectionId: null, profile: null })
})

describe('the settled row mark', () => {
  it('paints the session color where the user picked it, and restores the default ink when cleared', () => {
    const session = makeCwdSession('/www/app', {
      connection_id: 'local',
      id: 'chosen',
      profile: 'default',
      title: 'Colored row'
    })

    $sessions.set([session])
    const { container } = mount([session])

    // No color chosen: the row keeps the stylesheet's neutral ink, no inline paint.
    expect(idleInk(container).style.backgroundColor).toBe('')

    // Picking a color repaints the mark in place (no remount, no stale subscription).
    // The non-empty guard keeps the comparison honest if jsdom ever drops the value.
    const expected = painted(SWATCH)

    expect(expected).not.toBe('')

    act(() => setSessionColorOverride(sessionPinId(session), SWATCH))
    expect(idleInk(container).style.backgroundColor).toBe(expected)

    // Clearing falls back to that same neutral ink.
    act(() => setSessionColorOverride(sessionPinId(session), null))
    expect(idleInk(container).style.backgroundColor).toBe('')
  })

  it('colors a row whose color was picked under its durable lineage id', () => {
    // The row menu writes `sessionPinId(row)`, which is the lineage ROOT once
    // auto-compression has rotated the live id. The paint must resolve through
    // that key, not the row's current id.
    const session = makeCwdSession('/www/app', {
      _lineage_root_id: 'root',
      connection_id: 'local',
      id: 'compressed',
      profile: 'default',
      title: 'Colored row'
    })

    $sessions.set([session])
    setSessionColorOverride(sessionPinId(session), SWATCH)
    const { container } = mount([session])

    expect(idleInk(container).style.backgroundColor).toBe(painted(SWATCH))
  })

  it('yields to a live semantic mark: an unread row keeps its own dot, not the session color', () => {
    const session: SessionInfo = {
      ...makeCwdSession('/www/app', {
        connection_id: 'local',
        id: 'chosen',
        profile: 'default',
        title: 'Colored row'
      }),
      unread: true
    }

    $sessions.set([session])
    setSessionColorOverride(sessionPinId(session), SWATCH)
    const { container } = mount([session])

    const mark = container.querySelector('.portal-row-mark')!

    expect(mark.getAttribute('data-state')).toBe('unread')
    // SessionStatusDot owns this dot; the session color never overpaints it.
    expect(mark.querySelector<HTMLElement>('[style*="background"]')).toBeNull()
  })
})
