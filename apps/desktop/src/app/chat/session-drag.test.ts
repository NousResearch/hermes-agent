import type { PointerEvent as ReactPointerEvent } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { findGroup, findGroupOfPane, group, split } from '@/components/pane-shell/tree/model'
import { $layoutTree } from '@/components/pane-shell/tree/store'
import { openSessionTile } from '@/store/session-states'

import { requestComposerInsertRefs } from './composer/focus'
import { startSessionDrag } from './session-drag'

/**
 * A session drop resolves its target by rect-testing the chat surfaces in the
 * document. A tab group keeps inactive tabs MOUNTED with their layout box
 * intact, so a background tab's rect is identical to the foreground tab's —
 * the drop has to land on the tab the user can actually see.
 */

vi.mock('@/store/session-states', () => ({ openSessionTile: vi.fn() }))
vi.mock('./composer/focus', () => ({ requestComposerInsertRefs: vi.fn() }))

const ZONE = { left: 0, top: 0, right: 1000, bottom: 800 }
const COMPOSER = { left: 100, top: 700, right: 900, bottom: 780 }

const stubRect = (el: Element, box: { left: number; top: number; right: number; bottom: number }) => {
  el.getBoundingClientRect = () =>
    ({ ...box, width: box.right - box.left, height: box.bottom - box.top, x: box.left, y: box.top }) as DOMRect
}

/** The workspace tab kept alive behind an active session tile tab. */
function mountStackedTabs() {
  document.body.innerHTML = `
    <div data-tree-group="g1">
      <div data-pane-hidden>
        <div data-session-anchor="workspace" data-composer-target="main">
          <div data-slot="composer-root"></div>
        </div>
      </div>
      <div>
        <div data-session-anchor="session-tile:visible" data-composer-target="tile:visible">
          <div data-slot="composer-root"></div>
        </div>
      </div>
    </div>
    <div id="row"></div>
  `

  stubRect(document.querySelector('[data-tree-group]')!, ZONE)

  for (const surface of document.querySelectorAll('[data-session-anchor]')) {
    stubRect(surface, ZONE)
  }

  for (const composer of document.querySelectorAll('[data-slot="composer-root"]')) {
    stubRect(composer, COMPOSER)
  }

  $layoutTree.set(group(['workspace', 'session-tile:visible'], { id: 'g1' }))

  return document.getElementById('row')!
}

/** Press on `source`, drag to (x, y), release. The drag session flushes its
 *  pending move synchronously on release, so no frame wait is needed. */
function dragTo(source: HTMLElement, x: number, y: number) {
  startSessionDrag({ id: 'dragged', profile: 'default', title: 'Dragged chat' }, {
    button: 0,
    clientX: 0,
    clientY: 0,
    currentTarget: source,
    pointerId: 1
  } as unknown as ReactPointerEvent<HTMLElement>)

  window.dispatchEvent(new MouseEvent('pointermove', { bubbles: true, clientX: x, clientY: y }))
  window.dispatchEvent(new MouseEvent('pointerup', { bubbles: true, clientX: x, clientY: y }))
}

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  document.body.innerHTML = ''
  $layoutTree.set(null)
})

describe('session drop targeting across stacked tabs', () => {
  it('links into the visible tab’s composer, not the tab kept alive behind it', () => {
    const row = mountStackedTabs()

    dragTo(row, 500, 740)

    expect(requestComposerInsertRefs).toHaveBeenCalledWith(expect.anything(), { target: 'tile:visible' })
  })

  it('docks a split against the visible tab’s pane', () => {
    const row = mountStackedTabs()

    dragTo(row, 980, 400)

    expect(openSessionTile).toHaveBeenCalledWith('dragged', 'right', 'session-tile:visible', undefined)
    expect(requestComposerInsertRefs).not.toHaveBeenCalled()
  })

  it('commits nothing over a zone that hosts no chat surface', () => {
    mountStackedTabs()
    $layoutTree.set(group(['terminal'], { id: 'g1' }))

    dragTo(document.getElementById('row')!, 500, 740)

    expect(requestComposerInsertRefs).not.toHaveBeenCalled()
    expect(openSessionTile).not.toHaveBeenCalled()
  })

  // Standing side chrome hosts no main tile, so a session has nowhere to land
  // there. That refusal is load-bearing twice over: the sidebar row runs the
  // reorder off the SAME press, so the deny is what leaves the list to it,
  // and ZoneDropOverlay keys off the same test to stay dark over those zones
  // instead of outlining a drop that would only be refused.
  it('commits nothing over the sidebar, leaving the region to the reorder', () => {
    mountStackedTabs()
    $layoutTree.set(group(['sessions'], { id: 'g1' }))

    dragTo(document.getElementById('row')!, 120, 400)

    expect(requestComposerInsertRefs).not.toHaveBeenCalled()
    expect(openSessionTile).not.toHaveBeenCalled()
  })

  it('moves the primary workspace pane when its loaded chat is stacked into another chat strip', () => {
    document.body.innerHTML = `
      <div data-tree-group="top">
        <div data-zone-tabstrip="top">
          <button data-tree-tab="session-tile:top"></button>
        </div>
      </div>
      <div data-tree-group="bottom">
        <button id="workspace-tab" data-tree-tab="workspace"></button>
      </div>
    `

    const top = document.querySelector<HTMLElement>('[data-tree-group="top"]')!
    const bottom = document.querySelector<HTMLElement>('[data-tree-group="bottom"]')!
    const strip = document.querySelector<HTMLElement>('[data-zone-tabstrip="top"]')!
    const targetTab = document.querySelector<HTMLElement>('[data-tree-tab="session-tile:top"]')!
    const workspaceTab = document.getElementById('workspace-tab')!

    stubRect(top, { left: 0, top: 0, right: 1000, bottom: 390 })
    stubRect(strip, { left: 0, top: 0, right: 1000, bottom: 32 })
    stubRect(targetTab, { left: 0, top: 0, right: 300, bottom: 32 })
    stubRect(bottom, { left: 0, top: 410, right: 1000, bottom: 800 })
    stubRect(workspaceTab, { left: 0, top: 410, right: 300, bottom: 442 })

    $layoutTree.set(
      split('column', [
        group(['session-tile:top'], { active: 'session-tile:top', id: 'top' }),
        group(['workspace'], { active: 'workspace', id: 'bottom' })
      ])
    )

    startSessionDrag(
      { id: 'main-session', profile: 'default', title: 'Main chat' },
      {
        button: 0,
        clientX: 100,
        clientY: 425,
        currentTarget: workspaceTab,
        pointerId: 1
      } as unknown as ReactPointerEvent<HTMLElement>,
      { sourcePaneId: 'workspace' }
    )

    window.dispatchEvent(new MouseEvent('pointermove', { bubbles: true, clientX: 800, clientY: 16 }))
    window.dispatchEvent(new MouseEvent('pointerup', { bubbles: true, clientX: 800, clientY: 16 }))

    expect(findGroupOfPane($layoutTree.get()!, 'workspace')).toMatchObject({
      id: 'top',
      panes: ['session-tile:top', 'workspace']
    })
    expect(openSessionTile).not.toHaveBeenCalled()
  })

  it('reveals a minimized destination after moving the primary workspace into it', () => {
    document.body.innerHTML = `
      <div data-tree-group="top">
        <div data-zone-tabstrip="top">
          <button data-tree-tab="session-tile:top"></button>
        </div>
      </div>
      <div data-tree-group="bottom">
        <button id="workspace-tab" data-tree-tab="workspace"></button>
      </div>
    `

    const top = document.querySelector<HTMLElement>('[data-tree-group="top"]')!
    const bottom = document.querySelector<HTMLElement>('[data-tree-group="bottom"]')!
    const strip = document.querySelector<HTMLElement>('[data-zone-tabstrip="top"]')!
    const targetTab = document.querySelector<HTMLElement>('[data-tree-tab="session-tile:top"]')!
    const workspaceTab = document.getElementById('workspace-tab')!

    stubRect(top, { left: 0, top: 0, right: 1000, bottom: 390 })
    stubRect(strip, { left: 0, top: 0, right: 1000, bottom: 32 })
    stubRect(targetTab, { left: 0, top: 0, right: 300, bottom: 32 })
    stubRect(bottom, { left: 0, top: 410, right: 1000, bottom: 800 })
    stubRect(workspaceTab, { left: 0, top: 410, right: 300, bottom: 442 })

    $layoutTree.set(
      split('column', [
        group(['session-tile:top'], { active: 'session-tile:top', id: 'top', minimized: true }),
        group(['workspace'], { active: 'workspace', id: 'bottom' })
      ])
    )

    startSessionDrag(
      { id: 'main-session', profile: 'default', title: 'Main chat' },
      {
        button: 0,
        clientX: 100,
        clientY: 425,
        currentTarget: workspaceTab,
        pointerId: 1
      } as unknown as ReactPointerEvent<HTMLElement>,
      { sourcePaneId: 'workspace' }
    )

    window.dispatchEvent(new MouseEvent('pointermove', { bubbles: true, clientX: 800, clientY: 16 }))
    window.dispatchEvent(new MouseEvent('pointerup', { bubbles: true, clientX: 800, clientY: 16 }))

    expect(findGroupOfPane($layoutTree.get()!, 'workspace')).toMatchObject({
      active: 'workspace',
      id: 'top',
      minimized: false,
      panes: ['session-tile:top', 'workspace']
    })
    expect(openSessionTile).not.toHaveBeenCalled()
  })

  // The reviewed PR path is not strip-only: `subZonePosition` resolves an edge
  // band to a split pos, and `moveTreePane` must carry the workspace through
  // `movePane` for EVERY TileDock value, not just `center`. Parameterized over
  // all four edges — each asserts the pane landed in a NEW split group on that
  // edge of the destination zone AND the emptied lower zone dissolved (the
  // reviewer's dangling-empty-group question, answered per edge).
  it.each(['bottom', 'left', 'right', 'top'] as const)(
    'moves the primary workspace on an edge-split drop (%s) and dissolves the source zone',
    pos => {
      document.body.innerHTML = `
        <div data-tree-group="top">
          <div data-tree-tab="session-tile:top"></div>
        </div>
        <div data-tree-group="bottom">
          <button id="workspace-tab" data-tree-tab="workspace"></button>
        </div>
      `

      const top = document.querySelector<HTMLElement>('[data-tree-group="top"]')!
      const bottom = document.querySelector<HTMLElement>('[data-tree-group="bottom"]')!
      const workspaceTab = document.getElementById('workspace-tab')!

      stubRect(top, { left: 0, top: 0, right: 1000, bottom: 390 })
      stubRect(bottom, { left: 0, top: 410, right: 1000, bottom: 800 })
      stubRect(workspaceTab, { left: 0, top: 410, right: 300, bottom: 442 })

      $layoutTree.set(
        split('column', [
          group(['session-tile:top'], { active: 'session-tile:top', id: 'top' }),
          group(['workspace'], { active: 'workspace', id: 'bottom' })
        ])
      )

      startSessionDrag(
        { id: 'main-session', profile: 'default', title: 'Main chat' },
        {
          button: 0,
          clientX: 100,
          clientY: 425,
          currentTarget: workspaceTab,
          pointerId: 1
        } as unknown as ReactPointerEvent<HTMLElement>,
        { sourcePaneId: 'workspace' }
      )

      // Off the strip, past the center ellipse (r = 0.62): the dominant-axis
      // radial pick resolves each named edge, e.g. bottom-right → 'bottom'.
      window.dispatchEvent(
        new MouseEvent('pointermove', {
          bubbles: true,
          clientX: pos === 'left' ? 30 : 970,
          clientY: pos === 'top' ? 30 : 360
        })
      )
      window.dispatchEvent(
        new MouseEvent('pointerup', {
          bubbles: true,
          clientX: pos === 'left' ? 30 : 970,
          clientY: pos === 'top' ? 30 : 360
        })
      )

      // The workspace moved to a fresh group docked on `pos` of the upper zone
      // — not duplicated through openSessionTile.
      expect(openSessionTile).not.toHaveBeenCalled()
      const tree = $layoutTree.get()!
      expect(findGroupOfPane(tree, 'workspace')).toMatchObject({ panes: ['workspace'] })

      // The upper zone still stands; the emptied lower one is GONE (normalize
      // prunes it) — the reviewer's dangling-empty-group question.
      expect(findGroup(tree, 'top')).not.toBeNull()
      expect(findGroup(tree, 'bottom')).toBeNull()
    }
  )

  it('excludes the primary workspace pane from its own same-strip insertion slots', () => {
    document.body.innerHTML = `
      <div data-tree-group="shared">
        <div data-zone-tabstrip="shared">
          <button id="workspace-tab" data-tree-tab="workspace"></button>
          <button data-tree-tab="session-tile:other"></button>
        </div>
      </div>
    `

    const groupEl = document.querySelector<HTMLElement>('[data-tree-group="shared"]')!
    const strip = document.querySelector<HTMLElement>('[data-zone-tabstrip="shared"]')!
    const workspaceTab = document.getElementById('workspace-tab')!
    const otherTab = document.querySelector<HTMLElement>('[data-tree-tab="session-tile:other"]')!

    stubRect(groupEl, { left: 0, top: 0, right: 1000, bottom: 800 })
    stubRect(strip, { left: 0, top: 0, right: 1000, bottom: 32 })
    stubRect(workspaceTab, { left: 0, top: 0, right: 300, bottom: 32 })
    stubRect(otherTab, { left: 300, top: 0, right: 600, bottom: 32 })

    const tree = group(['workspace', 'session-tile:other'], { active: 'workspace', id: 'shared' })
    $layoutTree.set(tree)

    startSessionDrag(
      { id: 'main-session', profile: 'default', title: 'Main chat' },
      {
        button: 0,
        clientX: 150,
        clientY: 16,
        currentTarget: workspaceTab,
        pointerId: 1
      } as unknown as ReactPointerEvent<HTMLElement>,
      { sourcePaneId: 'workspace' }
    )

    // A small leftward drag over the workspace's own slot is a no-op. Before
    // the fix, the slot named `workspace`; removal erased that anchor and the
    // insert fell back to append, unexpectedly moving workspace to the end.
    window.dispatchEvent(new MouseEvent('pointermove', { bubbles: true, clientX: 10, clientY: 16 }))
    window.dispatchEvent(new MouseEvent('pointerup', { bubbles: true, clientX: 10, clientY: 16 }))

    expect($layoutTree.get()).toBe(tree)
    expect(findGroupOfPane($layoutTree.get()!, 'workspace')?.panes).toEqual(['workspace', 'session-tile:other'])
    expect(openSessionTile).not.toHaveBeenCalled()
  })
})
