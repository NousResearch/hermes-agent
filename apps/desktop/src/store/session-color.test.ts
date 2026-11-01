import { afterEach, describe, expect, it } from 'vitest'

import type { ProjectInfo } from '@/types/hermes'

import { makeCwdSession } from '../test/session-info'

import { $projects } from './projects'
import { $sessions } from './session'
import {
  $sessionColorById,
  $sessionColorOverrides,
  $sessionTabTintById,
  $sessionTileColors,
  sessionColorFor,
  sessionTabTintFor,
  setSessionColorOverride,
  setSessionTileColor
} from './session-color'

function makeProject(id: string, folders: string[], color: null | string): ProjectInfo {
  return {
    archived: false,
    board_slug: null,
    color,
    created_at: 0,
    description: null,
    folders: folders.map((path, i) => ({ added_at: 0, is_primary: i === 0, label: null, path })),
    icon: null,
    id,
    name: id,
    primary_path: folders[0] ?? null,
    slug: id
  }
}

afterEach(() => {
  $sessions.set([])
  $projects.set([])
  $sessionColorOverrides.set({})
  $sessionTileColors.set({})
})

describe('$sessionColorById', () => {
  it('maps each session under a colored project to that color, keyed by live id', () => {
    const a = makeCwdSession('/www/app/src', { git_repo_root: '/www/app' })
    const b = makeCwdSession('/other/place')

    $projects.set([makeProject('p_app', ['/www/app'], '#4a9eff')])
    $sessions.set([a, b])

    const map = $sessionColorById.get()

    expect(map[a.id]).toBe('#4a9eff')
    // Sessions with no colored project are absent (a sparse map, not null-filled).
    expect(b.id in map).toBe(false)
  })

  it('recomputes when the projects list changes (color applied later)', () => {
    const a = makeCwdSession('/www/app', { git_repo_root: '/www/app' })

    $sessions.set([a])
    $projects.set([makeProject('p_app', ['/www/app'], null)])
    expect($sessionColorById.get()[a.id]).toBeUndefined()

    $projects.set([makeProject('p_app', ['/www/app'], '#7bc86c')])
    expect($sessionColorById.get()[a.id]).toBe('#7bc86c')
  })
})

describe('$sessionColorOverrides', () => {
  it('an override wins over the inherited project color', () => {
    const a = makeCwdSession('/www/app', { git_repo_root: '/www/app' })

    $projects.set([makeProject('p_app', ['/www/app'], '#4a9eff')])
    $sessions.set([a])
    setSessionColorOverride(a.id, '#ff0000')

    expect($sessionColorById.get()[a.id]).toBe('#ff0000')
  })

  it('clearing an override falls back to the project color', () => {
    const a = makeCwdSession('/www/app', { git_repo_root: '/www/app' })

    $projects.set([makeProject('p_app', ['/www/app'], '#4a9eff')])
    $sessions.set([a])

    setSessionColorOverride(a.id, '#ff0000')
    expect($sessionColorById.get()[a.id]).toBe('#ff0000')

    setSessionColorOverride(a.id, null)
    expect($sessionColorById.get()[a.id]).toBe('#4a9eff')
  })

  it('keys on the durable lineage id so a color survives compression', () => {
    // The live id rotates on auto-compression; the override is stored against the
    // lineage root, so the continuation tip still resolves to the same color.
    const root = makeCwdSession('/x', { id: 'root' })
    const tip = makeCwdSession('/x', { id: 'tip', _lineage_root_id: 'root' })

    setSessionColorOverride('root', '#abcdef')

    $sessions.set([tip])
    expect($sessionColorById.get().tip).toBe('#abcdef')
  })
})

describe('sessionColorFor', () => {
  it('reads a single session through the same shared map', () => {
    const a = makeCwdSession('/www/app', { git_repo_root: '/www/app' })

    $projects.set([makeProject('p_app', ['/www/app'], '#5865f2')])
    $sessions.set([a])

    expect(sessionColorFor(a)).toBe('#5865f2')
  })

  it('returns undefined for a null/absent session', () => {
    expect(sessionColorFor(null)).toBeUndefined()
    expect(sessionColorFor(undefined)).toBeUndefined()
  })
})

describe('the tile tier ($sessionTileColors)', () => {
  it('tints the TAB and leaves the session colour (the row dot) alone', () => {
    const a = makeCwdSession('/other/place')

    $sessions.set([a])
    setSessionTileColor(a.id, '#ff8800')

    expect($sessionTabTintById.get()[a.id]).toBe('#ff8800')
    // The dot says what the session IS; a group tint says where it is filed.
    // A tile tier that moved the dot would repaint the sidebar behind a verb
    // the user ran in another pane.
    expect($sessionColorById.get()[a.id]).toBeUndefined()
  })

  it('a hand-picked session colour wins over the tile tier', () => {
    const a = makeCwdSession('/other/place')

    $sessions.set([a])
    setSessionTileColor(a.id, '#ff8800')
    setSessionColorOverride(a.id, '#0000ff')

    expect($sessionTabTintById.get()[a.id]).toBe('#0000ff')
    expect(sessionTabTintFor(a)).toBe('#0000ff')
  })

  it('sits above the inherited project colour', () => {
    const a = makeCwdSession('/www/app', { git_repo_root: '/www/app' })

    $projects.set([makeProject('p_app', ['/www/app'], '#4a9eff')])
    $sessions.set([a])

    expect($sessionTabTintById.get()[a.id]).toBe('#4a9eff')

    setSessionTileColor(a.id, '#7bc86c')
    expect($sessionTabTintById.get()[a.id]).toBe('#7bc86c')
  })

  it('clears by DROPPING the key, not by writing an empty string', () => {
    const a = makeCwdSession('/other/place')

    setSessionTileColor(a.id, '#ff8800')
    setSessionTileColor(a.id, null)

    expect(a.id in $sessionTileColors.get()).toBe(false)
  })

  it('keys on the durable lineage id so a tint survives compression', () => {
    const tip = makeCwdSession('/x', { id: 'tip', _lineage_root_id: 'root' })

    setSessionTileColor('root', '#abcdef')
    $sessions.set([tip])

    expect($sessionTabTintById.get().tip).toBe('#abcdef')
  })

  it('resolves a session outside the recents page through the same tiers', () => {
    // A project-tree tile older than the paginated page has no map entry; the
    // resolver is what keeps its colour.
    const a = makeCwdSession('/other/place')

    setSessionTileColor(a.id, '#123456')

    expect($sessionTabTintById.get()[a.id]).toBeUndefined()
    expect(sessionTabTintFor(a)).toBe('#123456')
  })
})
