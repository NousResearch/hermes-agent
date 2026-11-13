import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it } from 'vitest'

import { $workspaceIsPage } from '@/app/routes'
import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { setTabStripDefault } from '@/store/tabstrip-prefs'

import { $bandPillPane, PortalSessionTab } from './portal-session-tab'

const disposers: (() => void)[] = []

function chatPane(id: string, title: string) {
  disposers.push(registry.register({ area: 'panes', id, title, data: { placement: 'main' }, render: () => null }))
}

beforeEach(() => {
  $workspaceIsPage.set(false)
  chatPane('workspace', 'Funnel review')
  chatPane('session-tile:a', 'Hiring loop')
  chatPane('session-tile:b', 'Investor update')
})

afterEach(() => {
  cleanup()
  act(() => {
    disposers.splice(0).forEach(dispose => dispose())
    setTabStripDefault('auto')
    $layoutTree.set(null)
  })
})

function band() {
  return render(
    <MemoryRouter>
      <PortalSessionTab />
    </MemoryRouter>
  )
}

it('stands for the lone main chat, and yields to the zone strip once a second chat tab opens', () => {
  act(() => $layoutTree.set(group(['workspace'], { id: 'main', active: 'workspace' })))
  band()
  expect(screen.getByText('Funnel review')).toBeTruthy()

  // "Open in new tab": the zone now holds three chats, so its strip (in the band) lists them all and
  // the pill would only repeat one title.
  act(() =>
    $layoutTree.set(group(['workspace', 'session-tile:a', 'session-tile:b'], { id: 'main', active: 'session-tile:a' }))
  )
  expect($bandPillPane.get()).toBeNull()
  expect(screen.queryByText('Funnel review')).toBeNull()
  expect(screen.queryByText('Hiring loop')).toBeNull()

  // Closing back down to the lone chat brings the pill back with that chat's title.
  act(() => $layoutTree.set(group(['workspace'], { id: 'main', active: 'workspace' })))
  expect(screen.getByText('Funnel review')).toBeTruthy()
})

it('follows the Tab Strip setting: hidden strips leave the pill naming the front tab', () => {
  act(() =>
    $layoutTree.set(group(['workspace', 'session-tile:a', 'session-tile:b'], { id: 'main', active: 'session-tile:b' }))
  )
  band()
  expect(screen.queryByText('Investor update')).toBeNull()

  act(() => setTabStripDefault('never'))
  expect(screen.getByText('Investor update')).toBeTruthy()

  // Switching tabs (⌃Tab / ⌘N) retitles the pill.
  act(() =>
    $layoutTree.set(group(['workspace', 'session-tile:a', 'session-tile:b'], { id: 'main', active: 'session-tile:a' }))
  )
  expect(screen.getByText('Hiring loop')).toBeTruthy()

  // A zone's own choice outranks the app default, as in every other zone.
  act(() =>
    $layoutTree.set(
      group(['workspace', 'session-tile:a'], { id: 'main', active: 'session-tile:a', tabStrip: 'always' })
    )
  )
  expect($bandPillPane.get()).toBeNull()
})
