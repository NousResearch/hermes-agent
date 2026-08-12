import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $terminalTakeover, setTerminalTakeover } from '@/app/right-sidebar/store'
import { registry } from '@/contrib/registry'

import { allPaneIds, group, split } from './model'
import {
  $dismissedPanes,
  $hiddenTreePanes,
  $layoutTree,
  activateTreePane,
  bindToolPaneCollapse,
  closeToolPane,
  isPaneVisible,
  revealTreePane,
  setTreeGroupHeaderHidden,
  togglePaneVisible
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
    ['files', { placement: 'right' }],
    ['review', { placement: 'right' }],
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
    expect(isPaneVisible('terminal')).toBe(false)
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

    togglePaneVisible('terminal')
    expect(isPaneVisible('terminal')).toBe(false)

    togglePaneVisible('terminal')
    expect(isPaneVisible('terminal')).toBe(true)
  })

  it('brings the terminal forward when logs holds the active tab', () => {
    stackTree({ active: 'logs' })
    bindPaneCollapse('terminal', atom(true))
    bindPaneCollapse('logs', atom(true))

    // The zone is open but showing LOGS, so the terminal is not on screen —
    // the first press must reveal it rather than collapse the whole zone.
    expect(isPaneVisible('terminal')).toBe(false)

    togglePaneVisible('terminal')

    expect(isPaneVisible('terminal')).toBe(true)
    expect(toolZone()?.minimized).toBeFalsy()
  })

  it('reopens after the terminal tab was closed outright', () => {
    stackTree()
    const $terminal = atom(true)
    bindPaneCollapse('terminal', $terminal)
    bindPaneCollapse('logs', atom(true))

    closeToolPane('terminal')
    expect(allPaneIds($layoutTree.get()!)).not.toContain('terminal')

    togglePaneVisible('terminal')

    expect(allPaneIds($layoutTree.get()!)).toContain('terminal')
    expect(isPaneVisible('terminal')).toBe(true)
  })
})

describe('collapsing the active terminal in a shared group with the workspace', () => {
  // The terminal is a TAB in the workspace's own group under the Focus preset:
  // [workspace, files, review, terminal]. Collapsing it used to hand the slot
  // to `panes[at - 1]` — 'review' — so ⌃` / the rail / the statusbar toggle
  // dropped the user on a diff pane they never opened, and with the overlay
  // still painting it read as "the terminal came back". The slot belongs to
  // the uncloseable workspace, the one member guaranteed to be a real
  // destination.

  function focusGroup() {
    $layoutTree.set(group(['workspace', 'files', 'review', 'terminal'], { active: 'terminal', id: 'g-focus' }))
  }

  const focusActive = () => {
    const tree = $layoutTree.get()

    return tree?.type === 'group' ? tree.active : undefined
  }

  it('⌃` / the rail toggle lands on workspace, not review', () => {
    focusGroup()

    // Production wiring (controller.tsx): the takeover atom IS the terminal's
    // toggle store; the closer/opener write it back.
    bindToolPaneCollapse(
      'terminal',
      $terminalTakeover,
      () => setTerminalTakeover(false),
      () => setTerminalTakeover(true)
    )

    // User had the terminal open and active.
    setTerminalTakeover(true)
    expect($terminalTakeover.get()).toBe(true)
    expect(focusActive()).toBe('terminal')
    expect(isPaneVisible('terminal')).toBe(true)

    // The user presses ⌃` (or clicks the rail / statusbar toggle) to put the
    // terminal away. bindToolPaneCollapse routes a false store through
    // setPaneCollapsed.
    setTerminalTakeover(false)

    expect($terminalTakeover.get()).toBe(false)
    expect(isPaneVisible('terminal')).toBe(false)
    // THE regression: used to be 'review' (group.panes[at - 1]).
    expect(focusActive()).toBe('workspace')
    expect(isPaneVisible('workspace')).toBe(true)
  })

  it('openNewSessionTile (+ / ⌘T): the new session tab fronts and the terminal tab goes hidden', () => {
    const stored = 'stored-1'
    const tilePaneId = `session-tile:${stored}`

    const dispose = registry.register({
      area: 'panes',
      data: { placement: 'main' },
      id: tilePaneId,
      render: () => null,
      title: 'new session'
    })

    disposers.push(dispose)

    // The tile lands in the same shared zone as the workspace + terminal
    // (Focus preset: docked 'center' into the focused chat zone).
    $layoutTree.set(
      group(['workspace', 'files', 'review', 'terminal', tilePaneId], { active: 'terminal', id: 'g-focus' })
    )

    bindToolPaneCollapse(
      'terminal',
      $terminalTakeover,
      () => setTerminalTakeover(false),
      () => setTerminalTakeover(true)
    )

    setTerminalTakeover(true)
    expect(focusActive()).toBe('terminal')
    expect(isPaneVisible('terminal')).toBe(true)

    // Exactly what openNewSessionTile does after openSessionTile + patch:
    // `revealTreePane(`session-tile:${stored}`)` (use-session-actions:521).
    revealTreePane(tilePaneId)

    expect(focusActive()).toBe(tilePaneId)
    expect(isPaneVisible('terminal')).toBe(false)
    // The terminal layer gets data-pane-hidden (tree-group spreads
    // hiddenPaneProps(!isActive)), which the PersistentTerminal overlay now
    // measures — so the terminal surface stops covering the new session tab.
    expect(isPaneVisible(tilePaneId)).toBe(true)
    expect($terminalTakeover.get()).toBe(true) // toggle store stays truthful
  })

  it('startFreshSessionDraft (⌘N): fronts the workspace and leaves the terminal open behind its tab', () => {
    focusGroup()

    bindToolPaneCollapse(
      'terminal',
      $terminalTakeover,
      () => setTerminalTakeover(false),
      () => setTerminalTakeover(true)
    )

    setTerminalTakeover(true)
    expect(focusActive()).toBe('terminal')

    // What startFreshSessionDraft does: reveal, and nothing else. It must NOT
    // clear the takeover atom — that is the terminal's open/closed state, not
    // a fronting flag.
    revealTreePane('workspace')

    expect(focusActive()).toBe('workspace')
    expect(isPaneVisible('workspace')).toBe(true)
    // Hidden behind the workspace tab, so the overlay stops painting (the
    // actual cause of the chat being covered) …
    expect(isPaneVisible('terminal')).toBe(false)
    // … while the terminal stays OPEN: its PTYs live and clicking the tab
    // brings back a mounted workspace.
    expect($terminalTakeover.get()).toBe(true)
  })
})

describe('a terminal that owns its own zone (Default / Terminal deck / Quad)', () => {
  // Only the Focus preset stacks the terminal with the workspace. Everywhere
  // else it sits in a group of its own, beside the chat rather than over it —
  // so a fresh chat has no reason to touch it.
  it('stays open and visible when a fresh chat fronts the workspace', () => {
    $layoutTree.set(
      split('row', [group(['workspace'], { id: 'grp-main' }), group(['terminal'], { id: 'grp-terminal' })], [3, 1])
    )

    bindToolPaneCollapse(
      'terminal',
      $terminalTakeover,
      () => setTerminalTakeover(false),
      () => setTerminalTakeover(true)
    )

    setTerminalTakeover(true)
    expect(isPaneVisible('terminal')).toBe(true)

    revealTreePane('workspace')

    expect(isPaneVisible('workspace')).toBe(true)
    // Regression: clearing takeover inside startFreshSessionDraft minimized
    // this zone, folding a terminal that was never obscuring anything.
    expect(isPaneVisible('terminal')).toBe(true)
    expect($terminalTakeover.get()).toBe(true)
  })

  it('survives a restart with a clickable, mountable terminal tab', () => {
    $layoutTree.set(group(['workspace', 'files', 'review', 'terminal'], { active: 'terminal', id: 'g-focus' }))

    bindToolPaneCollapse(
      'terminal',
      $terminalTakeover,
      () => setTerminalTakeover(false),
      () => setTerminalTakeover(true)
    )

    setTerminalTakeover(true)
    revealTreePane('workspace')

    // ── restart: the tree and the takeover atom are both persisted ──
    const persistedTakeover = $terminalTakeover.get()

    bindToolPaneCollapse(
      'terminal',
      $terminalTakeover,
      () => setTerminalTakeover(false),
      () => setTerminalTakeover(true)
    )

    // Regression: a persisted `false` left the tab in the strip and
    // un-minimized, but PersistentTerminal mounts its workspace only while
    // takeover is true — so clicking the tab (a bare activateTreePane) fronted
    // an empty pane.
    expect(persistedTakeover).toBe(true)

    activateTreePane('g-focus', 'terminal')

    expect(isPaneVisible('terminal')).toBe(true)
    expect($terminalTakeover.get()).toBe(true)
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
