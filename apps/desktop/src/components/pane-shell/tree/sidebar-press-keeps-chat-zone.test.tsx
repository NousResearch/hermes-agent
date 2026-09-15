import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { TreeGroup } from './renderer/tree-group'

// The zone trackers (`store.ts`: `trackActiveTreeGroup`, installed once by the
// renderer root) resolve the `[data-tree-group]` ancestor of every pointerdown.
// The sidebar is a zone in the tree like any other, so the press that opens a
// session from a session row reports the SIDEBAR's own zone — the store-level
// tests call `noteActiveTreeGroup('grp-sessions')` directly, which ASSUMES that
// precondition instead of asserting it. This file asserts it against the DOM,
// and with it the invariant it protects: reaching for a session row must not
// move "the chat zone I was working in" back to main.

/** jsdom has no real pointer events; PointerEvent falls back to MouseEvent. */
function pointer(target: Element, type: string) {
  const event = new MouseEvent(type, { bubbles: true, cancelable: true, button: 0 })
  Object.defineProperty(event, 'pointerId', { value: 1 })

  act(() => {
    target.dispatchEvent(event)
  })
}

let root: null | Root = null
let container: HTMLDivElement | null = null
const disposers: Array<() => void> = []

function render(ui: ReactNode) {
  if (!container) {
    container = globalThis.document.createElement('div')
    globalThis.document.body.append(container)
    root = createRoot(container)
  }

  act(() => {
    root!.render(ui)
  })
}

beforeEach(() => {
  window.localStorage.clear()
  vi.resetModules()
})

afterEach(() => {
  if (root) {
    act(() => root!.unmount())
  }

  container?.remove()

  for (const dispose of disposers.splice(0)) {
    dispose()
  }

  root = null
  container = null
  vi.resetModules()
})

async function setup() {
  const tree = await import('@/components/pane-shell/tree/store')
  const model = await import('@/components/pane-shell/tree/model')

  for (const id of ['workspace', 'session-tile:a', 'session-tile:b']) {
    disposers.push(
      registry.register({
        area: 'panes',
        data: id === 'workspace' ? { placement: 'main', uncloseable: true } : { placement: 'main' },
        id,
        render: () => null,
        title: id
      })
    )
  }

  // A stand-in for the sidebar: the real one is a whole app view, and all this
  // test needs from it is a session row inside the zone's DOM subtree.
  disposers.push(
    registry.register({
      area: 'panes',
      data: { placement: 'left', uncloseable: true },
      id: 'sessions',
      render: () => <button data-slot="row-button">session row</button>,
      title: 'sessions'
    })
  )

  const main = model.group(['workspace'], { active: 'workspace', id: 'grp-main' })
  const side = model.group(['session-tile:a', 'session-tile:b'], { active: 'session-tile:b', id: 'grp-side' })
  const sessions = model.group(['sessions'], { active: 'sessions', id: 'grp-sessions' })

  tree.$layoutTree.set(model.split('row', [main, side, sessions]))

  // Same wiring the renderer root does (`useEffect(trackActiveTreeGroup, [])`).
  disposers.push(tree.trackActiveTreeGroup())

  return { sessions, side, tree }
}

describe('a press inside the sidebar zone keeps the chat-zone memory', () => {
  it('reports the sidebar zone for a press on a session row, and keeps the chat target', async () => {
    const { sessions, tree } = await setup()

    // The user is working in the side chat zone — this is what a press in it
    // reports, through the same tracker.
    tree.noteActiveTreeGroup('grp-side')
    expect(tree.focusedChatZoneIsSidePane()).toBe(true)
    expect(tree.lastChatZoneSessionAnchor()).toBe('session-tile:b')

    render(<TreeGroup node={sessions} parentAxis="column" />)
    const row = container!.querySelector('[data-slot="row-button"]')!
    expect(row).not.toBeNull()

    pointer(row, 'pointerdown')

    // The precondition: the press reports the SIDEBAR's zone, not the chat zone.
    expect(tree.$activeTreeGroup.get()).toBe('grp-sessions')

    // The invariant: opening a session from that row still targets the chat zone
    // the user was working in — the sidebar's zone is not a chat strip, however
    // recently it was pressed.
    expect(tree.focusedChatZoneIsSidePane()).toBe(true)
    expect(tree.lastChatZoneSessionAnchor()).toBe('session-tile:b')
  })

  it('a pointer crossing a chat zone on its way to the sidebar does not move the memory', async () => {
    const { tree } = await setup()

    // The user is working in the side chat zone.
    tree.noteActiveTreeGroup('grp-side')
    expect(tree.lastChatZoneSessionAnchor()).toBe('session-tile:b')

    // The tracker reports EVERY zone the pointer crosses (pointerover), and the
    // path from a side pane to the session list usually crosses main. That pass
    // must not rewrite "the chat zone I was working in" to main — which is the
    // exact answer that sends a session opened from a row into the initial pane.
    for (const id of ['grp-main', 'grp-sessions']) {
      const el = globalThis.document.createElement('div')
      el.dataset.treeGroup = id
      globalThis.document.body.append(el)
      disposers.push(() => el.remove())
      pointer(el, 'pointerover')
    }

    expect(tree.$hoveredTreeGroup.get()).toBe('grp-sessions')
    expect(tree.lastChatZoneSessionAnchor()).toBe('session-tile:b')
    expect(tree.focusedChatZoneIsSidePane()).toBe(true)
  })

  it('still moves the target when the press lands in a chat zone (control)', async () => {
    const { tree } = await setup()

    // No chat zone touched yet: the memory is empty rather than pointing at main
    // — callers fall back to the focused-zone anchor.
    tree.noteActiveTreeGroup(null)
    expect(tree.lastChatZoneSessionAnchor()).toBeNull()

    // The tracker resolves the `[data-tree-group]` ancestor of the press, so this
    // stands in for a press anywhere in that zone; rendering the whole strip is
    // not what this control is about.
    const zone = globalThis.document.createElement('div')
    zone.dataset.treeGroup = 'grp-side'
    globalThis.document.body.append(zone)
    disposers.push(() => zone.remove())

    pointer(zone, 'pointerdown')

    expect(tree.$activeTreeGroup.get()).toBe('grp-side')
    expect(tree.lastChatZoneSessionAnchor()).toBe('session-tile:b')
  })
})
