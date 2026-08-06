import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $gatewayState } from '@/store/session'
import { $sessionStates, dropSessionState, publishSessionState } from '@/store/session-states'

import { host } from './index'

// Plugins read app state exclusively through host.state — and before this
// contract existed, only the PRIMARY workspace tab was reachable
// ($activeSessionId). Clicking a tile never moved any plugin-visible atom,
// and tile focus is pure renderer state that gateway RPC can never see, so
// no plugin-side workaround was possible. These atoms are the plugin door to
// the same focused-session signals the core statusbar reads
// (use-statusbar-items.tsx).

describe('host.state focused-session atoms', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    vi.resetModules()
  })

  async function setup() {
    const { host } = await import('@/sdk/index')
    const states = await import('@/store/session-states')
    const session = await import('@/store/session')

    return { host, states, session }
  }

  it('exposes readonly atoms for the focused session (runtime id, stored id, usage)', async () => {
    const { host } = await setup()

    for (const key of ['focusedSessionId', 'focusedStoredSessionId', 'focusedUsage'] as const) {
      const store = host.state[key]
      expect(store, key).toBeDefined()
      expect(typeof store.get, key).toBe('function')
      expect(typeof store.listen, key).toBe('function')
      expect(typeof store.subscribe, key).toBe('function')
    }
  })

  it('mirrors the primary session while no tile is focused', async () => {
    const { host, states } = await setup()

    expect(host.state.focusedSessionId.get()).toBe(states.$focusedRuntimeId.get())
    expect(host.state.focusedStoredSessionId.get()).toBe(states.$focusedStoredSessionId.get())
  })

  it('focusedUsage projects the focused session usage, null while unresolved', async () => {
    const { host, states } = await setup()

    const focused = states.$focusedSessionState.get()
    expect(host.state.focusedUsage.get()).toBe(focused?.usage ?? null)
  })

  it('follows the interacted tile while the primary-only atom stays put', async () => {
    const { host, session } = await setup()
    const tree = await import('@/components/pane-shell/tree/store')
    const model = await import('@/components/pane-shell/tree/model')
    const { registry } = await import('@/contrib/registry')

    // A second chat zone holding a session tile, next to the main workspace.
    for (const id of ['workspace', 'session-tile:tile-a']) {
      registry.register({
        area: 'panes',
        data: id === 'workspace' ? { placement: 'main', uncloseable: true } : { placement: 'main' },
        id,
        render: () => null,
        title: id
      })
    }

    tree.declareDefaultTree(
      model.split('row', [
        model.group(['workspace'], { active: 'workspace', id: 'grp-main' }),
        model.group(['session-tile:tile-a'], { active: 'session-tile:tile-a', id: 'grp-side' })
      ])
    )

    const primaryBefore = session.$activeSessionId.get()
    const primarySelection = session.$selectedStoredSessionId.get()

    // Focusing the tile zone moves the focused atoms onto the tile's session…
    tree.noteActiveTreeGroup('grp-side')
    expect(host.state.focusedStoredSessionId.get()).toBe('tile-a')
    // …while the primary-only atom a plugin used to rely on does not move.
    expect(host.state.activeSessionId.get()).toBe(primaryBefore)

    // Focusing back homes to the primary's selection, whatever it is.
    tree.noteActiveTreeGroup('grp-main')
    expect(host.state.focusedStoredSessionId.get()).toBe(primarySelection)
  })
})

describe('host.state busy vs gateway', () => {
  afterEach(() => {
    $sessionStates.set({})
    $gatewayState.set('idle')
  })

  it('exposes per-session turn-busy and does not treat gateway as busy', () => {
    const running = { ...createClientSessionState('stored-a'), busy: true }
    const idle = { ...createClientSessionState('stored-b'), busy: false }

    publishSessionState('runtime-a', running)
    publishSessionState('runtime-b', idle)
    $gatewayState.set('open')

    expect(host.state.busyBySession.get()).toEqual({ 'runtime-a': true, 'runtime-b': false })
    expect(host.state.gateway.get()).toBe('open')

    dropSessionState('runtime-a')
    expect(host.state.busyBySession.get()['runtime-a']).toBeUndefined()
    expect(host.state.gateway.get()).toBe('open')
  })
})
