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
  isPaneVisible,
  togglePaneVisible
} from './store'

// Ground truth for the reported bug: the bottom tool pane (terminal) ended up
// stacked as a tab inside the left hide-style sessions column (a user drag —
// `userPlacedPanes: ["terminal"]`), with the terminal's toggle store persisted
// as OFF. Persisted evidence from the field: layoutTree put terminal in
// `grp-sessions` next to `sessions`, `terminalTakeover` stayed false, and the
// ⌃`/⌘K/statusbar toggle appeared to do nothing — no pane, no shell, no tab.
// Recovery was only possible by resetting the layout state.

const disposers: (() => void)[] = []

/** Bind through the REAL production function (controller.tsx wiring). */
function bindTerminalToggle() {
  bindToolPaneCollapse(
    'terminal',
    $terminalTakeover,
    () => setTerminalTakeover(false),
    () => setTerminalTakeover(true)
  )
}

beforeEach(() => {
  window.localStorage.clear()
  $dismissedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())
  setTerminalTakeover(false)

  for (const [id, data] of [
    ['workspace', { placement: 'main', uncloseable: true }],
    ['sessions', { placement: 'left', hideOnly: true, collapsible: true }],
    ['terminal', { placement: 'bottom' }]
  ] as const) {
    disposers.push(registry.register({ area: 'panes', data, id, render: () => null, title: id }))
  }
})

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
})

/** The field layout: terminal dragged into the sessions column, sessions active. */
function sessionsColumnWithTerminal() {
  $layoutTree.set(
    split('row', [
      group(['sessions', 'terminal'], { active: 'sessions', id: 'grp-sessions' }),
      group(['workspace'], { active: 'workspace', id: 'grp-main' })
    ])
  )
}

describe('a bottom tool pane stranded in the sessions column (field layout)', () => {
  it('fronting its tab via activate opens its toggle store, so the shell can mount', () => {
    sessionsColumnWithTerminal()
    bindTerminalToggle()

    // Persisted state after the reported bug: takeover is OFF while the pane
    // still sits in the tree (un-minimized). Clicking the terminal's tab only
    // runs activateTreePane — PersistentTerminal mounts its workspace ONLY
    // while the takeover store is true, so a bare front leaves an EMPTY pane:
    // no shell is ever spawned and the pane reads as "not showing".
    expect($terminalTakeover.get()).toBe(false)
    expect(isPaneVisible('terminal')).toBe(false)

    activateTreePane('grp-sessions', 'terminal')

    expect(isPaneVisible('terminal')).toBe(true)
    // THE regression: the store must open with the front, or the pane is an
    // unmountable shell (empty surface, no PTY spawn, dead toggle afterwards).
    expect($terminalTakeover.get()).toBe(true)
  })

  it('⌃` / the statusbar toggle surfaces the terminal from the sessions column and keeps it open', () => {
    sessionsColumnWithTerminal()
    bindTerminalToggle()

    togglePaneVisible('terminal')

    expect(isPaneVisible('terminal')).toBe(true)
    expect($terminalTakeover.get()).toBe(true)

    // A second press collapses it again (plain toggle round-trip).
    togglePaneVisible('terminal')
    expect(isPaneVisible('terminal')).toBe(false)
  })

  it('closing the stranded terminal hands the slot to sessions instead of folding the whole column', () => {
    sessionsColumnWithTerminal()
    bindTerminalToggle()

    togglePaneVisible('terminal')
    expect(isPaneVisible('terminal')).toBe(true)

    // THE second half of the field bug: with the terminal active in the
    // sessions column, a closing ⌃` used to minimize the WHOLE group — the
    // sessions list collapsed along with the terminal it was stacked in.
    togglePaneVisible('terminal')

    expect(isPaneVisible('terminal')).toBe(false)
    expect($terminalTakeover.get()).toBe(false)
    // The column survives; sessions takes back the slot.
    const tree = $layoutTree.get()!
    const column = tree.type === 'split' ? tree.children[0] : null
    expect(column).toMatchObject({ type: 'group', id: 'grp-sessions', active: 'sessions' })
    expect(column?.type === 'group' && column.minimized).not.toBe(true)
    expect(isPaneVisible('sessions')).toBe(true)
  })

  it('the terminal tab remains clickable and mountable after a simulated restart', () => {
    sessionsColumnWithTerminal()
    bindTerminalToggle()

    // ── restart: tree + takeover atom are both persisted ──
    // Terminal is a tab in the sessions column again; takeover is OFF.
    expect($terminalTakeover.get()).toBe(false)

    activateTreePane('grp-sessions', 'terminal')

    expect(isPaneVisible('terminal')).toBe(true)
    expect($terminalTakeover.get()).toBe(true)
    // The pane must stay in the tree (no dismissal from fronting a tab).
    expect(allPaneIds($layoutTree.get()!)).toContain('terminal')
  })
})
