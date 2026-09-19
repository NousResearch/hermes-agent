import type { PointerEvent as ReactPointerEvent } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree } from '@/components/pane-shell/tree/store'
import { openSessionTile } from '@/store/session-states'

import { startBotChatDrag } from './bot-drag'
import { requestComposerInsertRefs } from './composer/focus'

/**
 * A bot roster drop resolves its target by rect-testing the chat surfaces in
 * the document, exactly like a session drop (see session-drag.test.ts). The
 * minted tile must land in the BOTS workspace: the payload carries the
 * plugin-computed scope, and the commit forwards it to openSessionTile — a
 * roster drag never re-buckets the bot's chat into Sessions.
 */

vi.mock('@/store/session-states', () => ({ openSessionTile: vi.fn() }))
vi.mock('./composer/focus', () => ({ requestComposerInsertRefs: vi.fn() }))

const ZONE = { left: 0, top: 0, right: 1000, bottom: 800 }
const COMPOSER = { left: 100, top: 700, right: 900, bottom: 780 }

const stubRect = (el: Element, box: { left: number; top: number; right: number; bottom: number }) => {
  el.getBoundingClientRect = () =>
    ({ ...box, width: box.right - box.left, height: box.bottom - box.top, x: box.left, y: box.top }) as DOMRect
}

const SCOPE = {
  workspaceMode: 'bots' as const,
  workspaceOwnerKey: 'bot:local::pavi',
  workspaceTabTitle: 'Pavi'
}

const PAYLOAD = {
  id: 'botchat-1',
  profile: 'pavi',
  scope: SCOPE,
  title: 'Pavi — Bot Chat'
}

function mountWorkspace() {
  document.body.innerHTML = `
    <div data-tree-group="g1">
      <div>
        <div data-session-anchor="workspace" data-composer-target="main">
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

  $layoutTree.set(group(['workspace'], { id: 'g1' }))

  return document.getElementById('row')!
}

/** Press on `source`, drag to (x, y), release. The drag session flushes its
 *  pending move synchronously on release, so no frame wait is needed. */
function dragTo(source: HTMLElement, x: number, y: number) {
  startBotChatDrag(PAYLOAD, {
    button: 0,
    clientX: 0,
    clientB: 0,
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

describe('bot roster drop targeting', () => {
  it('docks the bot chat as a split and forwards the plugin scope', () => {
    const row = mountWorkspace()

    dragTo(row, 980, 400)

    expect(openSessionTile).toHaveBeenCalledWith('botchat-1', 'right', 'workspace', undefined, SCOPE)
    expect(requestComposerInsertRefs).not.toHaveBeenCalled()
  })

  it('links into the composer under the pointer, like a session drag', () => {
    const row = mountWorkspace()

    dragTo(row, 500, 740)

    expect(requestComposerInsertRefs).toHaveBeenCalledWith(expect.anything(), { target: 'main' })
    expect(openSessionTile).not.toHaveBeenCalled()
  })

  it('commits nothing over a zone that hosts no chat surface', () => {
    mountWorkspace()
    $layoutTree.set(group(['terminal'], { id: 'g1' }))

    dragTo(document.getElementById('row')!, 500, 740)

    expect(requestComposerInsertRefs).not.toHaveBeenCalled()
    expect(openSessionTile).not.toHaveBeenCalled()
  })

  it('commits nothing over the roster side chrome, leaving the region to the section drag', () => {
    mountWorkspace()
    $layoutTree.set(group(['bots'], { id: 'g1' }))

    dragTo(document.getElementById('row')!, 120, 400)

    expect(requestComposerInsertRefs).not.toHaveBeenCalled()
    expect(openSessionTile).not.toHaveBeenCalled()
  })
})
