import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Re-fronting the Bots home is a CLOSE followed by an OPEN of the workspace
// tab, which tears down and rebuilds the whole Bots view. The re-front branch
// runs whenever a passive reconcile finds the tab open but not holding its
// zone's active slot — and it is completely unbounded: it re-fronts again on
// every pass, for as long as that condition holds.
//
// The condition is not always transient. `revealTreePane` returns early for a
// pane in `$hiddenTreePanes` WITHOUT activating it, `isPaneVisible` is false
// for a minimized zone, and a pane the tree never adopted has no group at all.
// Pinned in that state, every surface sync the plugin runs — sidebar
// visibility flips, focus churn, group changes all call syncWorkspaceSurfaces
// — costs one full remount of the Bots view. That is the strobe in the bug
// report's screen recording: successive frames differ only by text
// re-rasterization across the whole view, the signature of a remount rather
// than a repaint.
//
// The invariant: a PASSIVE reconcile gets one re-front attempt. If the shell
// does not grant visibility, the plugin stops rather than remounting on every
// subsequent signal. An explicit user gesture is still always honored.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

/** A shell whose reveal does NOT hand the tab its zone's active slot, modelled
 *  on `revealTreePane`'s hidden-pane early return. `grantsVisibility: true`
 *  restores the cooperative shell so the same harness covers both directions. */
function load({ grantsVisibility = false } = {}) {
  const values = new Map()
  const atom = initial => {
    const slot = {
      get: () => values.get(slot),
      set: value => {
        values.set(slot, value)
        for (const fn of slot.__listeners || []) {
          fn(value)
        }
      },
      listen: fn => {
        slot.__listeners = [...(slot.__listeners || []), fn]
        return () => {
          slot.__listeners = (slot.__listeners || []).filter(entry => entry !== fn)
        }
      }
    }
    values.set(slot, initial)
    return slot
  }

  const opened = []
  const closed = []
  const visible = new Map()
  const watchers = new Map()

  // Synchronous notification, like a nanostores computed over $layoutTree.
  const setVisible = (paneId, value) => {
    if (visible.get(paneId) === value) {
      return
    }

    visible.set(paneId, value)
    for (const fn of watchers.get(paneId) || []) {
      fn(value)
    }
  }

  const host = {
    state: {
      profile: { get: () => 'default', listen: () => undefined },
      gateway: { get: () => 'open', listen: () => undefined },
      focusedStoredSessionId: atom(null)
    },
    request: async () => ({}),
    requestProfile: async () => ({}),
    openSession: async () => undefined,
    setWorkspaceScope: () => true,
    notify: () => undefined,
    notifyError: () => undefined,
    ensureAgent: async () => undefined,
    activeConnectionId: () => 'local',
    openWorkspace: (id, options) => {
      const paneId = `plugin-workspace:${id}`
      const entry = { id, options }
      opened.push(entry)
      setVisible(paneId, grantsVisibility)

      return () => {
        closed.push(entry)
        setVisible(paneId, false)
        options.onClose?.()
      }
    },
    paneVisibility: paneId => ({
      get: () => visible.get(paneId) ?? false,
      listen: fn => {
        watchers.set(paneId, [...(watchers.get(paneId) || []), fn])
        return () => watchers.set(paneId, (watchers.get(paneId) || []).filter(entry => entry !== fn))
      }
    })
  }

  const context = {
    atom,
    haptic: () => undefined,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host,
    queryClient: { invalidateQueries: () => undefined },
    navigator: { clipboard: { writeText: async () => undefined } },
    sdk: new Proxy({}, { get: () => undefined })
  }

  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(`
globalThis.__refront = {
  syncBotsHomeWorkspace,
  openBotsHomeWorkspace,
  closeBotsHomeWorkspace,
  botsHomeVisible,
  BOTS_HOME_PANE_ID,
  $botsPaneVisible,
  $openBotChat
};
`)

  vm.runInNewContext(source, context, { filename: 'plugin.js' })

  return { ...context.__refront, closed, host, opened, setVisible }
}

test('a passive reconcile re-fronts the home once, not forever', () => {
  const t = load({ grantsVisibility: false })
  t.$botsPaneVisible.set(true)

  // First pass opens the tab. The shell does not front it.
  t.syncBotsHomeWorkspace()
  assert.equal(t.opened.length, 1)
  assert.equal(t.botsHomeVisible(), false, 'the shell withheld the active slot')

  // Every later passive pass sees "open but not visible" and takes the
  // re-front branch. Before the fix each pass closed and re-opened the tab,
  // remounting the whole Bots view — 20 passes, 20 remounts.
  for (let i = 0; i < 20; i++) {
    t.syncBotsHomeWorkspace()
  }

  assert.equal(t.opened.length, 2, 'exactly one re-front attempt, then the plugin stops')
  assert.equal(t.closed.length, 1)
})

test('giving up on the re-front still leaves the tab open', () => {
  const t = load({ grantsVisibility: false })
  t.$botsPaneVisible.set(true)

  t.syncBotsHomeWorkspace()
  for (let i = 0; i < 5; i++) {
    t.syncBotsHomeWorkspace()
  }

  // Bounding the retries must not degrade into closing the home: the Bots tab
  // would fall through to the ownerless Sessions composer, which is the exact
  // hole the home exists to plug. Stop re-fronting, keep the surface.
  assert.equal(t.closed.length, 1, 'one close, from the single re-front — not a teardown')
  assert.equal(t.opened.length, 2)
})

test('a cooperative shell still gets its one legitimate re-front', () => {
  const t = load({ grantsVisibility: true })
  t.$botsPaneVisible.set(true)

  t.syncBotsHomeWorkspace()
  assert.equal(t.opened.length, 1)
  assert.equal(t.botsHomeVisible(), true)

  // A persisted layout can restore the tab BEHIND the core draft pane. The
  // home must still reclaim the active slot — the fix bounds retries, it does
  // not remove the re-front (bots-home.test.mjs pins this case too).
  t.setVisible(t.BOTS_HOME_PANE_ID, false)
  t.syncBotsHomeWorkspace()

  assert.equal(t.opened.length, 2, 're-opened to reclaim the active slot')
  assert.equal(t.botsHomeVisible(), true)

  // And having converged, the plugin is free to re-front again the next time
  // the tab is genuinely backgrounded: the bound is per-attempt, not one-shot.
  t.setVisible(t.BOTS_HOME_PANE_ID, false)
  t.syncBotsHomeWorkspace()

  assert.equal(t.opened.length, 3)
})

test('an explicit gesture is never blocked by the passive bound', () => {
  const t = load({ grantsVisibility: false })
  t.$botsPaneVisible.set(true)

  t.syncBotsHomeWorkspace()
  for (let i = 0; i < 5; i++) {
    t.syncBotsHomeWorkspace()
  }
  const passive = t.opened.length

  // The user clicked a bot whose owner is unavailable: the home IS the
  // destination, so the gesture must re-front even though passive reconciles
  // have given up on this shell.
  t.openBotsHomeWorkspace(true)

  assert.equal(t.opened.length, passive + 1, 'an explicit open always re-fronts')
})
