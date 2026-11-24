import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { closeFindBar, openFindBar } from '@/store/find-in-page'
import { clearNotifications, notify } from '@/store/notifications'
import { $shellChoice } from '@/store/shell-variant'
import { stubResizeObserver } from '@/test/jsdom'

import { FindBar } from './find-bar'
import { NotificationStack } from './notifications'
import { overlayStyle } from './pane-shell/tree/renderer/narrow-overlays'
import { zoneFrameAttrs, zoneFrameStyle } from './pane-shell/tree/renderer/zone-frame'

// The redesign's chrome additions must leave classic's DOM as main renders it.

beforeAll(stubResizeObserver)

afterEach(() => {
  cleanup()
  act(() => closeFindBar())
  clearNotifications()
  $shellChoice.set(null)
})

const NO_DRAG = '[-webkit-app-region:no-drag]'

it.each([false, true])('the find bar opts out of window drag only in the redesign (redesign=%s)', redesign => {
  $shellChoice.set(redesign)
  render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <FindBar />
      </I18nProvider>
    </MemoryRouter>
  )
  act(() => openFindBar())

  expect(screen.getByRole('search').className.includes(NO_DRAG)).toBe(redesign)
})

it('the classic toast stack keeps main’s drag behaviour', () => {
  $shellChoice.set(false)
  notify({ kind: 'error', message: 'Failed', durationMs: 0 })
  render(<NotificationStack />)

  for (const region of screen.getAllByRole('region')) {
    expect(region.className).not.toContain(NO_DRAG)
  }
})

it('a classic zone carries only main’s frame attributes and style', () => {
  const edges = { left: true, right: true, top: true }
  const input = { active: undefined, activeId: 'workspace', edges, headerVisible: true, isEmpty: false }
  const style = { titlebarHeight: 40, topEdge: true, verticalCollapse: false, wcOverlap: null }

  expect(zoneFrameAttrs({ ...input, portal: false })).toEqual({ 'data-window-top': true, 'data-zone-header': true })
  expect(zoneFrameStyle({ ...style, portal: false })).toBeUndefined()
  expect(zoneFrameAttrs({ ...input, portal: true })).toMatchObject({ 'data-window-start': true, 'data-zone-pane': 'workspace' })
  expect(zoneFrameStyle({ ...style, portal: true })).toEqual({ '--zone-band-height': '40px' })
})

it('a classic narrow overlay keeps main’s width style', () => {
  const input = { chatSidebar: false, overlayWidth: '16rem', side: 'right' as const, titlebarHeight: 40, wcOverlap: null }

  expect(overlayStyle({ ...input, portal: false })).toEqual({ paddingTop: undefined, width: 'min(16rem, 85vw)' })
  expect(overlayStyle({ ...input, portal: true })).toMatchObject({ maxWidth: 'var(--portal-sheet-end-max-width, 100%)' })
})
