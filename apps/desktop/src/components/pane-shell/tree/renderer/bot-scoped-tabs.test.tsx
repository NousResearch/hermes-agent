import { useStore } from '@nanostores/react'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $botTabsFollowSelectedBot } from '@/store/tabstrip-prefs'

import { resetRememberedActivePanes, setWorkspaceScope } from '../../workspace-scope'
import { findGroup, group, type GroupNode } from '../model'
import {
  $hiddenTreePanes,
  $layoutTree,
  activateTreePane,
  activateTreeTabSlot,
  closeOtherTreeTabs,
  noteActiveTreeGroup,
  registerPaneCloser,
  removeTreePane,
  shownPanesInGroup
} from '../store'

import { TreeGroup } from './tree-group'

// Bot Mode's opt-in per-bot strip (#115140). The first workspace scoping was
// taken out (35b53af10d) because it decided whether panes RENDERED: close the
// last bot chat and the main zone had nothing left, so it collapsed. This one
// only drops chips, and these tests pin that it can never strand a zone, never
// close or unmount what it hides, and is inert unless the user turns it on.

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })
  HTMLElement.prototype.scrollIntoView ??= () => undefined
})

const BOT_A = 'conn-1:alpha'
const BOT_B = 'conn-1:beta'
const ZONE = 'grp-main'

// Pane → Bot Mode owner: a bot key, null = a Sessions tab, absent = not a
// workspace tab (the terminal), which is never scoped.
const OWNERS: Record<string, null | string | undefined> = {
  workspace: BOT_A,
  'session-tile:a1': BOT_A,
  'session-tile:b1': BOT_B,
  'session-tile:s1': null,
  terminal: undefined
}

const disposers = new Map<string, () => void>()
const closed: string[] = []

/** TreeGroup reads its node from props; render it from the subscribed tree. */
function LiveZone() {
  const tree = useStore($layoutTree)

  return <TreeGroup node={findGroup(tree!, ZONE) as GroupNode} />
}

const chips = () =>
  [...globalThis.document.querySelectorAll('[data-tree-tab]')].map(el => el.getAttribute('data-tree-tab'))

const zone = () => findGroup($layoutTree.get()!, ZONE) as GroupNode

beforeEach(() => {
  window.localStorage.clear()
  resetRememberedActivePanes()
  $hiddenTreePanes.set(new Set())
  closed.length = 0

  for (const [id, owner] of Object.entries(OWNERS)) {
    const data = {
      placement: id === 'terminal' ? 'bottom' : 'main',
      uncloseable: id === 'workspace' || undefined,
      ...(id === 'terminal' ? {} : { workspaceOwner: () => owner ?? null })
    }

    disposers.set(id, registry.register({ area: 'panes', data, id, render: () => <p>body {id}</p>, title: id }))

    if (id !== 'workspace') {
      registerPaneCloser(id, () => void closed.push(id))
    }
  }

  $layoutTree.set(group(Object.keys(OWNERS), { active: 'session-tile:a1', id: ZONE }))
  noteActiveTreeGroup(ZONE)
  $botTabsFollowSelectedBot.set(true)
  setWorkspaceScope('bots', BOT_A)
})

afterEach(() => {
  cleanup()
  disposers.forEach(dispose => dispose())
  disposers.clear()
  $botTabsFollowSelectedBot.set(false)
  setWorkspaceScope('sessions')
})

describe('Bot Mode per-bot tab strip', () => {
  it('is inert by default and in Sessions: every tab stays in the strip', async () => {
    $botTabsFollowSelectedBot.set(false)
    render(<LiveZone />)
    expect(chips()).toEqual(Object.keys(OWNERS))

    await act(async () => {
      $botTabsFollowSelectedBot.set(true)
      setWorkspaceScope('sessions')
    })
    expect(chips()).toEqual(Object.keys(OWNERS))
  })

  it("shows only the selected bot's tabs and brings the others back, still open", async () => {
    render(<LiveZone />)

    // Bot A: its chat + side tab, plus the unowned terminal. B's tab and the
    // Sessions tab leave the strip.
    expect(chips()).toEqual(['workspace', 'session-tile:a1', 'terminal'])

    // Switch to B the way a roster click does: front B's tab, then scope.
    await act(async () => {
      activateTreePane(ZONE, 'session-tile:b1')
      setWorkspaceScope('bots', BOT_B)
    })
    expect(chips()).toEqual(['session-tile:b1', 'terminal'])

    // Back to A: its tabs return. Nothing was closed, nothing left the tree.
    await act(async () => {
      activateTreePane(ZONE, 'session-tile:a1')
      setWorkspaceScope('bots', BOT_A)
    })
    expect(chips()).toEqual(['workspace', 'session-tile:a1', 'terminal'])
    expect(zone().panes).toEqual(Object.keys(OWNERS))
    expect(closed).toEqual([])

    // ⌘3 indexes the chips the strip paints, not the hidden tabs between them.
    expect(activateTreeTabSlot(3)).toBe('terminal')
  })

  it('never hides the active tab, and Close Others never reaches a hidden one', async () => {
    // B selected but A's tab is the one on screen (e.g. fronted in a split).
    setWorkspaceScope('bots', BOT_B)
    render(<LiveZone />)
    expect(chips()).toEqual(['session-tile:a1', 'session-tile:b1', 'terminal'])

    closeOtherTreeTabs('session-tile:a1')
    expect(closed.sort()).toEqual(['session-tile:b1', 'terminal'])
  })

  // The regression that reverted the first scoping: the last bot chat closes
  // and the zone must still have something to show and a way to reach it.
  it("keeps a zone reachable when the selected bot's last tab closes", async () => {
    setWorkspaceScope('bots', BOT_B)
    activateTreePane(ZONE, 'session-tile:b1')
    render(<LiveZone />)
    expect(chips()).toEqual(['session-tile:b1', 'terminal'])

    await act(async () => {
      disposers.get('session-tile:b1')!()
      removeTreePane('session-tile:b1')
    })

    // B has no tab left here, so the strip falls back to every tab instead
    // of leaving an empty zone (and no HERMES placeholder body).
    expect(chips()).toEqual(['workspace', 'session-tile:a1', 'session-tile:s1', 'terminal'])
    expect(shownPanesInGroup(zone())).toHaveLength(4)
    expect(globalThis.document.body.textContent).not.toContain('HERMES')
  })
})
