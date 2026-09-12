import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { group, split } from '../model'
import { $layoutTree, $newSessionTabAction, registerPaneCloser } from '../store'

import { TreeGroup } from './tree-group'

/**
 * A LONE MAIN zone inside a TILED layout — the multi-session view.
 *
 * Reported: "with several sessions open, one session window has no title bar,
 * so I can't close it or add tabs to it". That window is the zone holding only
 * the workspace: sitting among sibling session zones, every one of which shows
 * a tab strip (chips + ✕ + "+"), while it alone renders none — no chip to grab,
 * no Close, no "+", nothing to click to get any of them back.
 *
 * The auto rung ("a lone pane is not a tab") is about ONE CHAT IN ONE WINDOW.
 * In a tiled layout the zone is not the app — it is one window of several, and
 * its siblings all carry the strip's handles.
 */

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => undefined
  Element.prototype.releasePointerCapture ??= () => undefined
  HTMLElement.prototype.scrollIntoView ??= () => undefined
})

const disposers: (() => void)[] = []

beforeEach(() => {
  window.localStorage.clear()
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  // jsdom lacks CSS.escape, which the tab-strip scroll effect uses. Stubbed per
  // test (not in beforeAll): afterEach restores the globals.
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })

  disposers.push(
    registry.register({
      area: 'panes',
      data: { minWidth: '20rem', placement: 'main', uncloseable: true },
      id: 'workspace',
      render: () => <div>Chat</div>,
      title: 'Hermes'
    }),
    registry.register({
      area: 'panes',
      data: { minWidth: '20rem', placement: 'main' },
      id: 'session-tile:a',
      render: () => <div>Tile</div>,
      title: 'Tile A'
    })
  )

  // The workspace tab closes (to a draft) even though the pane itself cannot
  // leave the tree — that closer is what puts the ✕ on its tab.
  registerPaneCloser('workspace', () => undefined)
  $newSessionTabAction.set(() => undefined)
})

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  $layoutTree.set(null)
  $newSessionTabAction.set(null)
  vi.unstubAllGlobals()
})

const zone = (id: string) => globalThis.document.querySelector(`[data-tree-group="${id}"]`)

describe('a lone workspace in a tiled layout', () => {
  /** The reported arrangement: the main chat zone beside session-tile zones. */
  const tiledTree = () =>
    split('row', [
      group(['workspace'], { id: 'grp-main' }),
      group(['session-tile:a'], { id: 'g-tile' }),
      group(['workspace'], { id: 'grp-other' })
    ])

  it('keeps its title bar, so Close and + stay reachable', () => {
    const tree = tiledTree()

    $layoutTree.set(tree)

    render(<TreeGroup node={group(['workspace'], { active: 'workspace', id: 'grp-main' })} parentAxis="row" />)

    const main = zone('grp-main')!

    expect(main.querySelector('[role="tablist"]')).toBeTruthy()
    expect(main.querySelector('[data-tree-tab="workspace"]')).toBeTruthy()
    expect(main.querySelector('button[aria-label="Close"]')).toBeTruthy()
    expect(main.querySelector('button[aria-label="New session tab"]')).toBeTruthy()
  })

  it('keeps its title bar even when the zone was told to hide tabs', () => {
    $layoutTree.set(tiledTree())

    render(
      <TreeGroup
        node={group(['workspace'], { active: 'workspace', id: 'grp-main', tabStrip: 'never' })}
        parentAxis="row"
      />
    )

    expect(zone('grp-main')!.querySelector('[role="tablist"]')).toBeTruthy()
    expect(zone('grp-main')!.querySelector('button[aria-label="New session tab"]')).toBeTruthy()
  })

  it('leaves a lone SIDE-CHROME zone chromeless in the same layout', () => {
    $layoutTree.set(tiledTree())

    disposers.push(
      registry.register({
        area: 'panes',
        data: { placement: 'right', width: '20rem' },
        id: 'files',
        render: () => <div>Files</div>,
        title: 'Files'
      })
    )

    render(<TreeGroup node={group(['files'], { active: 'files', id: 'grp-files' })} parentAxis="row" />)

    expect(zone('grp-files')!.querySelector('[role="tablist"]')).toBeNull()
  })
})

// The behaviour the auto rung was written for, which must survive: ONE chat in
// ONE window is not a tab, so the lone workspace stays chromeless.
describe('a lone workspace as the whole layout', () => {
  it('renders no strip at all', () => {
    const solo = split('row', [group(['workspace'], { id: 'grp-main' })])

    $layoutTree.set(solo)

    render(<TreeGroup node={group(['workspace'], { active: 'workspace', id: 'grp-main' })} parentAxis="row" />)

    expect(zone('grp-main')!.querySelector('[role="tablist"]')).toBeNull()
  })
})
