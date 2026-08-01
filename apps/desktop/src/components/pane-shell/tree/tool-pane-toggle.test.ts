import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'

import { allPaneIds, group, split } from './model'
import {
  $dismissedPanes,
  $hiddenTreePanes,
  $layoutTree,
  bindToolPaneCollapse,
  closeToolPane,
  isToolPaneVisible,
  setTreeGroupHeaderHidden,
  toggleToolPane
} from './store'

// Ground truth for "toggle terminal broke — ⌘J/⌘B work fine, but once I move
// the terminal around it just doesn't open/close anymore", and for "I have the
// header hidden and it keeps coming back".
//
// Both symptoms need the terminal STACKED with logs, which is the layout you
// get by dragging the terminal to the bottom: adoption stacks logs into the
// terminal's zone. A lone terminal never reproduced either bug, which is why
// they read as "randomly broken".

const disposers: (() => void)[] = []

/** Bind through the REAL production function, with the ergonomics the
 *  controller gives it (closer/opener writing the same store). */
function bindPaneCollapse(paneId: string, $open: ReturnType<typeof atom<boolean>>) {
  bindToolPaneCollapse(
    paneId,
    $open,
    () => $open.set(false),
    () => $open.set(true)
  )
}

beforeEach(() => {
  window.localStorage.clear()
  $dismissedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())

  for (const [id, data] of [
    ['workspace', { placement: 'main', uncloseable: true }],
    ['terminal', { placement: 'bottom' }],
    ['logs', { placement: 'bottom' }]
  ] as const) {
    disposers.push(registry.register({ area: 'panes', data, id, render: () => null, title: id }))
  }
})

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
})

/** The bottom tool zone, as the tree currently holds it. */
const toolZone = () => {
  const tree = $layoutTree.get()!

  const found =
    tree.type === 'split'
      ? tree.children.find(c => c.type === 'group' && (c.panes.includes('terminal') || c.panes.includes('logs')))
      : null

  return found as { active?: string; headerHidden?: boolean; minimized?: boolean; panes: string[] } | null
}

/** Terminal dragged to the bottom; logs adopted into the same zone.
 *
 *  Set via `$layoutTree.set`, NOT `declareDefaultTree` — that only adopts into
 *  an existing tree, and `$layoutTree` is module state that survives between
 *  tests, so the second case would silently assert against the first's shape. */
const stackTree = (options?: { active?: string; headerHidden?: boolean }) => {
  $layoutTree.set(
    split('column', [
      group(['workspace'], { active: 'workspace', id: 'grp-main' }),
      group(['terminal', 'logs'], {
        active: options?.active ?? 'terminal',
        headerHidden: options?.headerHidden,
        id: 'g-tools'
      })
    ])
  )
}

describe('binding a tool panel on boot', () => {
  it('does not front itself over the tab the persisted tree had active', () => {
    stackTree({ active: 'terminal' })

    const $terminal = atom(true)
    const $logs = atom(true)
    bindPaneCollapse('terminal', $terminal)
    bindPaneCollapse('logs', $logs)

    // Binding logs LAST must not steal the active slot. It used to, because
    // boot called setPaneCollapsed(id, false), whose shared-zone branch
    // reveals.
    expect(toolZone()?.active).toBe('terminal')
  })

  it('leaves ⌃` able to collapse the zone straight after boot', () => {
    stackTree({ active: 'terminal' })

    const $terminal = atom(true)
    bindPaneCollapse('terminal', $terminal)
    bindPaneCollapse('logs', atom(true))

    // THE reported symptom, driven the way the keybind drives it. When boot
    // fronted logs, this asked setPaneCollapsed to fold a terminal that wasn't
    // the active tab; the shared-zone branch declines that, so the first press
    // did nothing at all and ⌃` read as a dead key.
    $terminal.set(false)

    expect(toolZone()?.minimized).toBe(true)
    expect(isToolPaneVisible('terminal')).toBe(false)
  })

  it('still collapses a tool panel whose store says it is off', () => {
    stackTree()

    bindPaneCollapse('terminal', atom(false))

    expect(toolZone()?.minimized).toBe(true)
  })
})

describe('toggling the terminal while it is stacked with logs', () => {
  it('closes and reopens on every press', () => {
    stackTree({ active: 'terminal' })
    bindPaneCollapse('terminal', atom(true))
    bindPaneCollapse('logs', atom(true))

    toggleToolPane('terminal')
    expect(isToolPaneVisible('terminal')).toBe(false)

    toggleToolPane('terminal')
    expect(isToolPaneVisible('terminal')).toBe(true)
  })

  it('brings the terminal forward when logs holds the active tab', () => {
    stackTree({ active: 'logs' })
    bindPaneCollapse('terminal', atom(true))
    bindPaneCollapse('logs', atom(true))

    // The zone is open but showing LOGS, so the terminal is not on screen —
    // the first press must reveal it rather than collapse the whole zone.
    expect(isToolPaneVisible('terminal')).toBe(false)

    toggleToolPane('terminal')

    expect(isToolPaneVisible('terminal')).toBe(true)
    expect(toolZone()?.minimized).toBeFalsy()
  })

  it('reopens after the terminal tab was closed outright', () => {
    stackTree()
    const $terminal = atom(true)
    bindPaneCollapse('terminal', $terminal)
    bindPaneCollapse('logs', atom(true))

    closeToolPane('terminal')
    expect(allPaneIds($layoutTree.get()!)).not.toContain('terminal')

    toggleToolPane('terminal')

    expect(allPaneIds($layoutTree.get()!)).toContain('terminal')
    expect(isToolPaneVisible('terminal')).toBe(true)
  })
})

describe('a zone whose header the user hid', () => {
  it('keeps it hidden after a stacked sibling is closed and toggled back', () => {
    stackTree({ headerHidden: true })
    bindPaneCollapse('terminal', atom(true))
    const $logs = atom(true)
    bindPaneCollapse('logs', $logs)

    setTreeGroupHeaderHidden('g-tools', true)

    // Close logs: the zone drops to one pane. normalize used to DISCARD the
    // hidden flag here ("a lone zone is headerless anyway"), so the bar
    // reappeared the moment logs was toggled back in.
    closeToolPane('logs')
    expect(toolZone()?.headerHidden).toBe(true)

    $logs.set(true)

    expect(toolZone()?.panes).toContain('logs')
    expect(toolZone()?.headerHidden).toBe(true)
  })

  it('keeps it hidden when a closed pane is re-adopted into it', () => {
    stackTree()
    const $terminal = atom(true)
    bindPaneCollapse('terminal', $terminal)
    bindPaneCollapse('logs', atom(true))

    setTreeGroupHeaderHidden('g-tools', true)

    // Re-adoption pins headerHidden:false so a surprise pane always has a
    // handle — correct for a new pane, wrong for a zone the user hid.
    closeToolPane('terminal')
    $terminal.set(true)

    expect(toolZone()?.panes).toContain('terminal')
    expect(toolZone()?.headerHidden).toBe(true)
  })
})
