import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { allPaneIds, group, type LayoutNode, split } from './model'
import { applyLayoutPreset, deleteUserPreset, saveLayoutPresetTree } from './presets'
import { $dismissedPanes, $hiddenTreePanes, $layoutTree } from './store'

// #94260: a named layout preset is GEOMETRY. Saving cloned the live tree, so a
// preset carried `session-tile:<storedSessionId>` (from ANY profile — the
// layout tree and its presets are global) and `preview-tile:*` /
// `route-tile:*`. Applying it remounted those conversations (`session.resume` →
// agent init), the tile WS dropped, the gateway `ws_orphan_reap`ed the runtime
// and the UI kept RPCing the dead id. These are the save / apply / load-heal
// paths for that defect.

const USER_KEY = 'hermes.desktop.layoutPresets.v2'

const LIVE_TILE = 'session-tile:20260823_233759_0fc103'
const FOREIGN_TILE = 'session-tile:20260823_193634_728a24'
const OTHER_FOREIGN_TILE = 'session-tile:20260823_193532_a7dbd9'

/** The shape `user-jrl` had in the field report: live tiles cloned into a
 *  saved deck, two of them from named profiles. */
function dirtyDeck() {
  return split(
    'row',
    [
      group(['sessions', 'preview'], { active: 'sessions', id: 'grp-sessions' }),
      group(['workspace', FOREIGN_TILE, OTHER_FOREIGN_TILE], { active: FOREIGN_TILE, id: 'grp-main' }),
      group(['preview-tile:undefined', 'route-tile:skills'], { active: 'preview-tile:undefined', id: 'grp-ghost' })
    ],
    [1, 3.4, 1],
    'spl-root'
  )
}

const storedPresets = () =>
  JSON.parse(window.localStorage.getItem(USER_KEY) ?? 'null') as Record<string, { resting?: string[]; tree: LayoutNode }>

beforeEach(() => {
  window.localStorage.clear()
  $dismissedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())
})

afterEach(() => {
  deleteUserPreset('user-jrl')
  deleteUserPreset('user-rest')
  vi.resetModules()
})

describe('user layout presets stay geometry-only (#94260)', () => {
  it('saves geometry, not the session tabs open at save time', () => {
    const id = saveLayoutPresetTree('JRL', dirtyDeck())

    expect(id).toBe('user-jrl')

    expect(allPaneIds(storedPresets()['user-jrl'].tree)).toEqual(['sessions', 'preview', 'workspace'])
  })

  it('drops resting records for the live tiles it strips', () => {
    saveLayoutPresetTree('Rest', dirtyDeck(), [FOREIGN_TILE, 'sessions'])

    expect(storedPresets()['user-rest'].resting).toEqual(['sessions'])
  })

  it('applies a dirty snapshot without resurrecting its baked-in tiles', () => {
    $layoutTree.set(group(['workspace', LIVE_TILE], { active: 'workspace', id: 'grp-live' }))

    applyLayoutPreset('user-jrl', dirtyDeck())

    const ids = allPaneIds($layoutTree.get()!)

    // Nothing from the snapshot's tiles comes back…
    expect(ids).not.toContain(FOREIGN_TILE)
    expect(ids).not.toContain(OTHER_FOREIGN_TILE)
    expect(ids).not.toContain('preview-tile:undefined')
    expect(ids).not.toContain('route-tile:skills')
    // …the preset's geometry does…
    expect(ids).toContain('sessions')
    // …and the tile that was ALREADY open is adopted, so no tab is lost.
    expect(ids).toContain(LIVE_TILE)
  })

  it('leaves the live layout alone when a preset is only a tile snapshot', () => {
    const live = group(['workspace'], { active: 'workspace', id: 'grp-live' })

    $layoutTree.set(live)
    applyLayoutPreset('user-tiles-only', group([FOREIGN_TILE, 'preview-tile:undefined'], { id: 'only-tiles' }))

    expect($layoutTree.get()).toBe(live)
  })

  it('heals a preset stored by an older build and rewrites the healed copy', async () => {
    window.localStorage.setItem(USER_KEY, JSON.stringify({ 'user-jrl': { name: 'JRL', tree: dirtyDeck() } }))

    vi.resetModules()

    await import('./presets')

    const healed = storedPresets()['user-jrl'] as { name: string; tree: LayoutNode }

    expect(healed.name).toBe('JRL')
    expect(allPaneIds(healed.tree)).toEqual(['sessions', 'preview', 'workspace'])
  })

  it('drops a stored preset that was nothing but tiles', async () => {
    window.localStorage.setItem(
      USER_KEY,
      JSON.stringify({
        'user-ghosts': { name: 'Ghosts', tree: group([FOREIGN_TILE, 'preview-tile:undefined'], { id: 'only-tiles' }) }
      })
    )

    vi.resetModules()

    await import('./presets')

    expect(window.localStorage.getItem(USER_KEY)).toBe('{}')
  })
})
