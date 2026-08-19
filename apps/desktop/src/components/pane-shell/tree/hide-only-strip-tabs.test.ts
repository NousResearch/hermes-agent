import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { allPaneIds, group, split } from './model'
import {
  $hiddenStripTabs,
  $hiddenTreePanes,
  $layoutTree,
  closeAllTreeTabs,
  hideOnlyZoneTabs,
  isHideOnlyPane,
  revealTreePane,
  setStripTabHidden,
  treeTabCloseTargets
} from './store'

vi.mock('@/store/notifications', () => ({ notify: vi.fn() }))

import { notify } from '@/store/notifications'

const disposers: (() => void)[] = []

function registerPane(id: string, data: Record<string, unknown>) {
  disposers.push(registry.register({ area: 'panes', data, id, render: () => null, title: id }))
}

/** The SESSIONS | BOTS shape: both hide-only chrome tabs stacked in one zone. */
function sessionsBotsTree() {
  registerPane('sessions', { placement: 'left', hideOnly: true })
  registerPane('hermes-bots:pane', { placement: 'left', hideOnly: true })
  registerPane('workspace', { placement: 'main', uncloseable: true })
  $layoutTree.set(
    split('row', [
      group(['sessions', 'hermes-bots:pane'], { active: 'sessions', id: 'g-side' }),
      group(['workspace'], { active: 'workspace', id: 'g-main' })
    ])
  )
}

beforeEach(() => {
  window.localStorage.clear()
  $hiddenStripTabs.set(new Set())
  $hiddenTreePanes.set(new Set())
  vi.mocked(notify).mockReset()
})

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
})

describe('hide-only strip tabs', () => {
  it('hides and shows a chrome tab, keeping the pane in the tree', () => {
    sessionsBotsTree()

    expect(setStripTabHidden('hermes-bots:pane', true)).toBe(true)
    expect($hiddenTreePanes.get()).toContain('hermes-bots:pane')
    expect($hiddenStripTabs.get()).toContain('hermes-bots:pane')
    // Hidden, not dismissed: the pane stays in the layout tree.
    expect(allPaneIds($layoutTree.get()!)).toContain('hermes-bots:pane')

    expect(setStripTabHidden('hermes-bots:pane', false)).toBe(true)
    expect($hiddenTreePanes.get()).not.toContain('hermes-bots:pane')
    expect($hiddenStripTabs.get()).not.toContain('hermes-bots:pane')
  })

  it('refuses to hide the zone last visible tab', () => {
    sessionsBotsTree()
    setStripTabHidden('hermes-bots:pane', true)

    // Sessions is now the only visible tab in the zone — the hide is refused
    // and the user is told, so the zone can never become an empty dead strip.
    expect(setStripTabHidden('sessions', true)).toBe(false)
    expect($hiddenTreePanes.get()).not.toContain('sessions')
    expect(vi.mocked(notify)).toHaveBeenCalledTimes(1)
  })

  it('persists hides and clears them on reveal', () => {
    sessionsBotsTree()
    setStripTabHidden('hermes-bots:pane', true)

    const persisted = JSON.parse(window.localStorage.getItem('hermes.desktop.hiddenStripTabs.v1') ?? '[]')

    expect(persisted).toContain('hermes-bots:pane')

    // Reveal intent (⌘K toggle on, a programmatic reveal) beats the hide —
    // including the persisted record, so the tab can't pop back hidden on the
    // next launch while visibly on screen now.
    revealTreePane('hermes-bots:pane')
    expect($hiddenTreePanes.get()).not.toContain('hermes-bots:pane')
    expect(window.localStorage.getItem('hermes.desktop.hiddenStripTabs.v1')).toBeNull()
  })

  it('lists the zone hide-only tabs with live hidden state for the menu', () => {
    sessionsBotsTree()
    setStripTabHidden('hermes-bots:pane', true)

    expect(hideOnlyZoneTabs('g-side')).toEqual([
      { hidden: false, id: 'sessions', title: 'sessions' },
      { hidden: true, id: 'hermes-bots:pane', title: 'hermes-bots:pane' }
    ])
    expect(hideOnlyZoneTabs('g-main')).toEqual([])
  })

  it('excludes hide-only tabs from every close verb', () => {
    registerPane('sessions', { placement: 'left', hideOnly: true })
    registerPane('hermes-bots:pane', { placement: 'left', hideOnly: true })
    registerPane('session-tile:x', { placement: 'main' })
    $layoutTree.set(group(['sessions', 'hermes-bots:pane', 'session-tile:x'], { active: 'sessions', id: 'g-mixed' }))

    expect(isHideOnlyPane('sessions')).toBe(true)
    // Close-others measured from the tile must not sweep standing chrome.
    expect(treeTabCloseTargets('session-tile:x')).toEqual({ all: 1, others: 0, right: 0 })
    // Close-all leaves both chrome tabs in the tree.
    closeAllTreeTabs('sessions')
    expect(allPaneIds($layoutTree.get()!)).toContain('sessions')
    expect(allPaneIds($layoutTree.get()!)).toContain('hermes-bots:pane')
    expect(allPaneIds($layoutTree.get()!)).not.toContain('session-tile:x')
  })
})
