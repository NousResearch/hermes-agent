/**
 * Route resume vs late disk plugins. A plugin page's path (`/html-gallery`) reads
 * as a session id until its route registers, and disk plugins register async.
 * A reload or deep link on that page used to resume the "session" as soon as
 * the gateway opened; the miss opened a fresh draft and replaced the route
 * with `/`, so the page was lost.
 *
 * Two parts: the resume waits for the disk door's first scan (as the
 * remembered-route restore already does), and the routed id and view are
 * re-derived from the routes snapshot when the page registers. Under React
 * Compiler a bare `routeSessionId(pathname)` stays memoized on the unchanged
 * path, so the late registration alone never reached the resume.
 */
import { act, cleanup, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { useRouteContributions } from '@/app/hooks/use-route-contributions'
import { $workspaceIsPage, ROUTES_AREA, routeSessionId, workspaceHeaderVetoed } from '@/app/routes'
import { useOverlayRouting } from '@/app/shell/hooks/use-overlay-routing'
import { $diskPluginsScanPending } from '@/contrib/disk-scan'
import { registry } from '@/contrib/registry'

import { useRouteResume } from './use-route-resume'

vi.mock('@/store/session-states', async importOriginal => ({
  ...(await importOriginal<object>()),
  markSelectionRestore: vi.fn()
}))

afterEach(() => {
  cleanup()
  $diskPluginsScanPending.set(false)
  $workspaceIsPage.set(false)
})

function mountAt(pathname: string) {
  const resumeSession = vi.fn(async () => undefined)
  const startFreshSessionDraft = vi.fn()
  const views: string[] = []

  const refs = {
    activeSessionIdRef: { current: null },
    creatingSessionRef: { current: false },
    runtimeIdByStoredSessionIdRef: { current: new Map<string, string>() },
    selectedStoredSessionIdRef: { current: null }
  }

  // The view and the routed id come from the same hooks the app wiring uses.
  function Harness() {
    const location = useLocation()
    const { currentView } = useOverlayRouting()
    const routedSessionId = useRouteContributions(location.pathname)

    views.push(currentView)
    useRouteResume({
      ...refs,
      activeSessionId: null,
      currentView,
      freshDraftReady: false,
      gatewayState: 'open',
      locationPathname: location.pathname,
      resumeExhaustedSessionId: null,
      resumeFailedSessionId: null,
      resumeSession,
      routedSessionId,
      selectedStoredSessionId: null,
      sessionResumeRequest: null,
      startFreshSessionDraft
    })

    return null
  }

  const wrap = (children: ReactNode) => <MemoryRouter initialEntries={[pathname]}>{children}</MemoryRouter>

  render(wrap(<Harness />))

  return { resumeSession, startFreshSessionDraft, views }
}

it('keeps a plugin page whose route registers after the gateway opened', () => {
  $diskPluginsScanPending.set(true)

  const { resumeSession, startFreshSessionDraft, views } = mountAt('/html-gallery')

  // Not registered yet: the path still reads as a session id, but nothing resumes it.
  expect(routeSessionId('/html-gallery')).toBe('html-gallery')
  expect(resumeSession).not.toHaveBeenCalled()
  // Undecided route: the zone tab bar stands down instead of showing a "New session" tab.
  expect(workspaceHeaderVetoed('/html-gallery')).toBe(true)

  let dispose = () => {}

  // The disk scan registers the page, then settles.
  act(() => {
    dispose = registry.register({
      area: ROUTES_AREA,
      id: 'gallery:page',
      data: { path: '/html-gallery' },
      render: () => null
    })
  })
  act(() => $diskPluginsScanPending.set(false))

  expect(views.at(-1)).toBe('extension')
  // A page now: the zone tab bar stays down, as on any page.
  expect($workspaceIsPage.get()).toBe(true)
  expect(workspaceHeaderVetoed('/html-gallery')).toBe(true)
  expect(resumeSession).not.toHaveBeenCalled()
  expect(startFreshSessionDraft).not.toHaveBeenCalled()
  dispose()
})

it('resumes a real session route once the disk scan has settled', () => {
  $diskPluginsScanPending.set(true)

  const { resumeSession } = mountAt('/sess-1')

  expect(resumeSession).not.toHaveBeenCalled()
  expect(workspaceHeaderVetoed('/sess-1')).toBe(true)

  act(() => $diskPluginsScanPending.set(false))

  // A real session: its tab bar comes back.
  expect(workspaceHeaderVetoed('/sess-1')).toBe(false)
  expect(resumeSession).toHaveBeenCalledTimes(1)
  expect(resumeSession).toHaveBeenCalledWith('sess-1', true)
})

it('resumes a session route at once when no disk scan is pending', () => {
  const { resumeSession } = mountAt('/sess-1')

  expect(resumeSession).toHaveBeenCalledTimes(1)
  expect(resumeSession).toHaveBeenCalledWith('sess-1', true)
})
