import type { PointerEvent as ReactPointerEvent } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree, closeTreePane } from '@/components/pane-shell/tree/store'
import { requestFreshSession } from '@/store/profile'
import { moveSessionToProject, projectIdForCwd } from '@/store/projects'
import { $selectedStoredSessionId, $sessions } from '@/store/session'
import { $sessionTiles, nextSessionTileForWorkspace, openSessionTile } from '@/store/session-states'

import { requestComposerInsertRefs } from './composer/focus'
import { startSessionDrag } from './session-drag'
import { NO_PROJECT_ID } from './sidebar/projects/workspace-groups'

/**
 * A session drop resolves its target by rect-testing the chat surfaces in the
 * document. A tab group keeps inactive tabs MOUNTED with their layout box
 * intact, so a background tab's rect is identical to the foreground tab's —
 * the drop has to land on the tab the user can actually see.
 */

vi.mock('@/store/session-states', async () => {
  const { atom } = await import('nanostores')

  return { $sessionTiles: atom([]), nextSessionTileForWorkspace: vi.fn(() => null), openSessionTile: vi.fn() }
})
vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestFreshSession: vi.fn()
}))
vi.mock('@/components/pane-shell/tree/store', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  closeTreePane: vi.fn()
}))
vi.mock('./composer/focus', () => ({ requestComposerInsertRefs: vi.fn() }))
vi.mock('@/store/projects', () => ({
  moveSessionToProject: vi.fn().mockResolvedValue(undefined),
  projectIdForCwd: vi.fn(() => null)
}))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))

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

  for (const type of ['pointermove', 'pointerup']) {
    const event = new MouseEvent(type, {
      bubbles: true,
      clientX: x,
      clientY: y,
      buttons: type === 'pointermove' ? 1 : 0
    })

    Object.defineProperty(event, 'pointerId', { value: 1 })
    window.dispatchEvent(event)
  }
}

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  document.body.innerHTML = ''
  $layoutTree.set(null)
  $selectedStoredSessionId.set(null)
  $sessionTiles.set([])
  $sessions.set([])
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

  // Dragging MAIN's own tab is a move: once the tile exists, main hands the
  // chat over instead of keeping a second copy of the same transcript.
  it('vacates main after its own tab is dropped as a split', () => {
    const row = mountStackedTabs()
    $selectedStoredSessionId.set('dragged')
    vi.mocked(openSessionTile).mockImplementation(id => {
      $sessionTiles.set([{ dir: 'right', storedSessionId: id, workspaceMode: 'bots' }] as never)
    })
    vi.mocked(nextSessionTileForWorkspace).mockReturnValue('visible')

    dragTo(row, 980, 400)

    expect(closeTreePane).toHaveBeenCalledWith('workspace')
    expect(requestFreshSession).not.toHaveBeenCalled()
  })

  it('drops main to a fresh draft when nothing else can shift in', () => {
    const row = mountStackedTabs()
    $selectedStoredSessionId.set('dragged')
    vi.mocked(openSessionTile).mockImplementation(id => {
      $sessionTiles.set([{ dir: 'right', storedSessionId: id, workspaceMode: 'bots' }] as never)
    })
    vi.mocked(nextSessionTileForWorkspace).mockReturnValue('dragged')

    dragTo(row, 980, 400)

    expect(requestFreshSession).toHaveBeenCalled()
    expect(closeTreePane).not.toHaveBeenCalled()
  })
})

/** A sidebar project row, exactly as `ProjectOverviewRow` tags it. */
function mountProjectRow(id: string, box = { left: 0, top: 0, right: 200, bottom: 40 }) {
  const row = document.createElement('div')
  row.dataset.sessionsProject = id
  document.body.appendChild(row)
  stubRect(row, box)

  return row
}

describe('session drop targeting onto a sidebar project row', () => {
  beforeEach(() => {
    $sessions.set([{ id: 'dragged', cwd: '', _lineage_root_id: null } as never])
  })

  it('moves the dragged session to the project row it is dropped on', () => {
    mountStackedTabs()
    $layoutTree.set(group(['sessions'], { id: 'g1' }))
    mountProjectRow('proj-a')

    dragTo(document.getElementById('row')!, 100, 20)

    expect(moveSessionToProject).toHaveBeenCalledWith('dragged', 'proj-a', 'default')
    // A resolved project-row target takes priority: nothing else commits.
    expect(openSessionTile).not.toHaveBeenCalled()
    expect(requestComposerInsertRefs).not.toHaveBeenCalled()
  })

  it('is not a target for Home — there is no folder to move into', () => {
    mountStackedTabs()
    $layoutTree.set(group(['sessions'], { id: 'g1' }))
    mountProjectRow(NO_PROJECT_ID)

    dragTo(document.getElementById('row')!, 100, 20)

    expect(moveSessionToProject).not.toHaveBeenCalled()
  })

  it('is not a target for the session’s own current project', () => {
    mountStackedTabs()
    $layoutTree.set(group(['sessions'], { id: 'g1' }))
    mountProjectRow('proj-a')
    vi.mocked(projectIdForCwd).mockReturnValue('proj-a')
    $sessions.set([{ id: 'dragged', cwd: '/repos/proj-a', _lineage_root_id: null } as never])

    dragTo(document.getElementById('row')!, 100, 20)

    expect(moveSessionToProject).not.toHaveBeenCalled()
  })
})
