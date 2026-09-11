import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Ephemeral pane persistence (#92818): preview/route tile panes are mirrored
// from live tab lists that die with the process. Persisting them 1:1 made the
// stored layout describe panes that could never come back, so every restart
// re-arranged a hand-built split. The LIVE tree must keep its tiles; only the
// STORED copy drops them.

const TREE_KEY = 'hermes.desktop.layoutTree.v2'

const treeWithTiles = {
  type: 'split',
  id: 'root',
  orientation: 'row',
  weights: [2, 1],
  children: [
    { type: 'group', id: 'g-main', panes: ['workspace', 'preview-tile:file:a'], active: 'preview-tile:file:a' },
    {
      type: 'group',
      id: 'g-right',
      panes: ['terminal', 'route-tile:/skills'],
      active: 'route-tile:/skills'
    }
  ]
}

describe('ephemeral panes are stripped from the persisted layout tree', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    vi.resetModules()
  })

  async function setup() {
    const store = await import('@/components/pane-shell/tree/store')

    return store
  }

  it('persist drops ephemeral panes but keeps the live tree intact', async () => {
    const store = await setup()

    store.$layoutTree.set(treeWithTiles as never)
    store.persistTree()

    const persisted = JSON.parse(window.localStorage.getItem(TREE_KEY)!) as {
      children?: Array<{ panes?: string[] }>
    }

    // Stored copy: tiles gone, real chrome stays.
    expect(persisted.children?.[0]?.panes).toEqual(['workspace'])
    expect(persisted.children?.[1]?.panes).toEqual(['terminal'])

    // Live tree: untouched — the preview is still on screen.
    const live = store.$layoutTree.get()
    expect(live).not.toBeNull()
    expect(JSON.stringify(live)).toContain('preview-tile:file:a')
  })

  it('a group holding ONLY ephemeral panes is pruned from the stored copy', async () => {
    const store = await setup()

    store.$layoutTree.set({
      type: 'split',
      id: 'root',
      orientation: 'row',
      weights: [1, 1],
      children: [
        { type: 'group', id: 'g-main', panes: ['workspace'], active: 'workspace' },
        { type: 'group', id: 'g-tiles', panes: ['preview-tile:url:x', 'route-tile:/y'], active: 'preview-tile:url:x' }
      ]
    } as never)
    store.persistTree()

    const persisted = JSON.parse(window.localStorage.getItem(TREE_KEY)!)

    // normalize collapses the now-empty right group into the root.
    expect(JSON.stringify(persisted)).not.toContain('g-tiles')
    expect(JSON.stringify(persisted)).toContain('workspace')
  })

  it('the active pane falls back to a surviving sibling when the active was ephemeral', async () => {
    const store = await setup()

    store.$layoutTree.set(treeWithTiles as never)
    store.persistTree()

    const persisted = JSON.parse(window.localStorage.getItem(TREE_KEY)!) as {
      children?: Array<{ active?: string; panes?: string[] }>
    }

    expect(persisted.children?.[0]?.active).toBe('workspace')
    expect(persisted.children?.[1]?.active).toBe('terminal')
  })

  it('stripEphemeralPanes leaves a tree without ephemeral panes unchanged (same reference)', () => {
    void import('@/components/pane-shell/tree/store').then(async () => {
      const { stripEphemeralPanes } = await import('@/components/pane-shell/tree/store')

      const plain = {
        type: 'group',
        id: 'g',
        panes: ['workspace', 'terminal'],
        active: 'workspace'
      }

      expect(stripEphemeralPanes(plain as never)).toBe(plain)
    })
  })
})

// #94260: user-saved layout presets are a SECOND persistence path for the
// tree. Baking in live tile ids (session-tile:/preview-tile:/route-tile:) made
// applying a preset remount dead conversations — session.resume against
// vanished runtimes, ws_orphan_reap, RPCs to nothing. The strip must cover
// both paths: saveLayoutPresetTree (save time) and applyLayoutPreset
// (apply time, so presets saved by older builds heal too).
describe('layout presets never carry ephemeral panes (#94260)', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    window.localStorage.clear()
    vi.restoreAllMocks()
  })

  it('saveLayoutPresetTree strips session/preview/route tiles at save time', async () => {
    const { saveLayoutPresetTree } = await import('./presets')

    const live = {
      type: 'split',
      id: 'root',
      orientation: 'row',
      weights: [1, 1],
      children: [
        {
          type: 'group',
          id: 'g-a',
          panes: ['workspace', 'session-tile:20260825_000852_e08305'],
          active: 'session-tile:20260825_000852_e08305'
        },
        { type: 'group', id: 'g-b', panes: ['preview-tile:url:x', 'route-tile:/y'] }
      ]
    } as never

    expect(saveLayoutPresetTree('Work', live)).toBeTruthy()

    const stored = JSON.parse(window.localStorage.getItem('hermes.desktop.layoutPresets.v2') ?? '{}')
    const serialized = JSON.stringify(stored)

    expect(serialized).not.toContain('session-tile:')
    expect(serialized).not.toContain('preview-tile:')
    expect(serialized).not.toContain('route-tile:')
    // The durable pane survives.
    expect(serialized).toContain('workspace')
  })

  it('applyLayoutPreset heals legacy presets that baked in tiles', async () => {
    // Simulate a preset saved by an older build: baked-in tile ids in storage.
    window.localStorage.setItem(
      'hermes.desktop.layoutPresets.v2',
      JSON.stringify({
        legacy: {
          name: 'Legacy',
          tree: {
            type: 'group',
            id: 'g',
            panes: ['workspace', 'session-tile:dead-id', 'preview-tile:undefined'],
            active: 'session-tile:dead-id'
          }
        }
      })
    )

    const applyTree = vi.fn()
    vi.doMock('./store', async importOriginal => {
      const actual = await importOriginal<Record<string, unknown>>()

      return { ...actual, applyTree }
    })

    const { applyLayoutPreset } = await import('./presets')
    const stored = JSON.parse(window.localStorage.getItem('hermes.desktop.layoutPresets.v2') ?? '{}')
    applyLayoutPreset('legacy', stored.legacy.tree)

    expect(applyTree).toHaveBeenCalledTimes(1)
    const applied = applyTree.mock.calls[0][0] as { children?: Array<{ panes: string[] }> }
    const serialized = JSON.stringify(applied)
    expect(serialized).not.toContain('session-tile:')
    expect(serialized).not.toContain('preview-tile:')
    expect(JSON.stringify(applied)).toContain('workspace')
  })
})
