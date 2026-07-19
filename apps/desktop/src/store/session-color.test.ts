import { afterEach, describe, expect, it } from 'vitest'

import type { ProjectInfo, SessionInfo } from '@/types/hermes'

import { $projects } from './projects'
import { $sessions } from './session'
import { $sessionColorById, sessionColorFor } from './session-color'

let nextId = 0

function makeSession(cwd: null | string, overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: false,
    cwd,
    ended_at: null,
    id: `s${nextId++}`,
    input_tokens: 0,
    is_active: false,
    last_active: 1_000,
    message_count: 1,
    model: 'claude',
    output_tokens: 0,
    preview: null,
    source: 'cli',
    started_at: 1_000,
    title: null,
    tool_call_count: 0,
    ...overrides
  }
}

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
})

describe('$sessionColorById', () => {
  it('maps each session under a colored project to that color, keyed by live id', () => {
    const a = makeSession('/www/app/src', { git_repo_root: '/www/app' })
    const b = makeSession('/other/place')

    $projects.set([makeProject('p_app', ['/www/app'], '#4a9eff')])
    $sessions.set([a, b])

    const map = $sessionColorById.get()

    expect(map[a.id]).toBe('#4a9eff')
    // Sessions with no colored project are absent (a sparse map, not null-filled).
    expect(b.id in map).toBe(false)
  })

  it('omits a session whose project has no color', () => {
    const a = makeSession('/www/app', { git_repo_root: '/www/app' })

    $projects.set([makeProject('p_app', ['/www/app'], null)])
    $sessions.set([a])

    expect(a.id in $sessionColorById.get()).toBe(false)
  })

  it('recomputes when the projects list changes (color applied later)', () => {
    const a = makeSession('/www/app', { git_repo_root: '/www/app' })

    $sessions.set([a])
    $projects.set([makeProject('p_app', ['/www/app'], null)])
    expect($sessionColorById.get()[a.id]).toBeUndefined()

    $projects.set([makeProject('p_app', ['/www/app'], '#7bc86c')])
    expect($sessionColorById.get()[a.id]).toBe('#7bc86c')
  })
})

describe('sessionColorFor', () => {
  it('reads a single session through the same shared map', () => {
    const a = makeSession('/www/app', { git_repo_root: '/www/app' })

    $projects.set([makeProject('p_app', ['/www/app'], '#5865f2')])
    $sessions.set([a])

    expect(sessionColorFor(a)).toBe('#5865f2')
  })

  it('returns undefined for a null/absent session', () => {
    expect(sessionColorFor(null)).toBeUndefined()
    expect(sessionColorFor(undefined)).toBeUndefined()
  })
})
