import { act, cleanup, render, screen } from '@testing-library/react'
import { useEffect } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { I18nProvider } from '@/i18n'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { group, split } from '../model'
import { $hiddenTreePanes, $layoutTree, setTreePaneHidden } from '../store'
import { $routeCollapsedPanes, setPaneRouteCollapsed } from '../store-route-collapse'

import { LayoutTreeRoot } from '.'

// A page that takes the chat sidebar off screen must not tear it down: returning to the chat has to
// be a visibility flip, not a rebuild of every row (the redesign's rail navigation). The user's own
// hide keeps unmounting, exactly as before.

let mounts = 0
let disposers: (() => void)[] = []

function Sidebar() {
  useEffect(() => {
    mounts += 1
  }, [])

  return <button type="button">Sidebar row</button>
}

function mount() {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <LayoutTreeRoot />
      </I18nProvider>
    </MemoryRouter>
  )
}

const sidebarTrack = () => screen.queryByText('Sidebar row')?.closest('[data-tree-group]')?.parentElement

beforeEach(() => {
  stubResizeObserver()
  stubMenuDomApis()
  mounts = 0
  disposers = [
    registry.register({
      area: 'panes',
      data: { placement: 'left', width: '320px' },
      id: 'sessions',
      render: () => <Sidebar />,
      title: 'Sessions'
    }),
    registry.register({
      area: 'panes',
      data: { placement: 'main', uncloseable: true },
      id: 'workspace',
      render: () => <p>Stage</p>,
      title: 'Workspace'
    })
  ]
  $layoutTree.set(split('row', [group(['sessions'], { id: 'side' }), group(['workspace'], { id: 'main' })]))
})

afterEach(() => {
  cleanup()
  disposers.forEach(dispose => dispose())
  $layoutTree.set(null)
  $routeCollapsedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())
})

it('collapses a route-hidden pane out of layout while keeping it mounted, and shows it again without a remount', () => {
  mount()
  expect(mounts).toBe(1)

  act(() => setPaneRouteCollapsed('sessions', true))
  // Off screen the way a hidden zone is (display:none: no layout, focus or accessibility tree)…
  expect(sidebarTrack()?.style.display).toBe('none')
  // …but still the same mounted instance.
  expect(screen.getByText('Sidebar row', { selector: 'button' })).toBeTruthy()

  act(() => setPaneRouteCollapsed('sessions', false))
  expect(sidebarTrack()?.style.display).not.toBe('none')
  expect(mounts).toBe(1)
})

it('still unmounts a pane the user hides', () => {
  mount()
  act(() => setTreePaneHidden('sessions', true))
  expect(screen.queryByText('Sidebar row')).toBeNull()
  act(() => setTreePaneHidden('sessions', false))
  expect(screen.getByText('Sidebar row')).toBeTruthy()
  expect(mounts).toBe(2)
})
