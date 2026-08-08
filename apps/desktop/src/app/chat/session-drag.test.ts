import type { PointerEvent as ReactPointerEvent } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree } from '@/components/pane-shell/tree/store'
import { assignSessionConversationGroup } from '@/store/projects'
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
vi.mock('@/store/projects', () => ({ assignSessionConversationGroup: vi.fn() }))
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
})

describe('session drop targeting for project conversation groups', () => {
  it('moves a session into the conversation group under the pointer', () => {
    document.body.innerHTML = `
      <div id="row"></div>
      <div data-conversation-group-drop data-project-id="p_sk" data-group-id="cg_ito"></div>
    `
    stubRect(document.querySelector('[data-conversation-group-drop]')!, {
      left: 100,
      top: 100,
      right: 400,
      bottom: 220
    })

    dragTo(document.getElementById('row')!, 200, 160)

    expect(assignSessionConversationGroup).toHaveBeenCalledWith('dragged', 'p_sk', 'cg_ito')
  })

  it('moves a session into a project when its overview header is the drop target', () => {
    document.body.innerHTML = `
      <div id="row"></div>
      <div data-conversation-group-drop data-project-id="p_wedding" data-group-id=""></div>
    `
    stubRect(document.querySelector('[data-conversation-group-drop]')!, {
      left: 100,
      top: 100,
      right: 400,
      bottom: 160
    })

    dragTo(document.getElementById('row')!, 200, 130)

    expect(assignSessionConversationGroup).toHaveBeenCalledWith('dragged', 'p_wedding', null)
  })
})
