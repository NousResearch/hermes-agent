import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import {
  filterRecentProjects,
  mergeRecentProjects,
  normalizeProjectPath,
  parseRecentProjects,
  projectDisplayName,
  type RecentProject,
  sessionProjectEntries,
  upsertRecentProject
} from './recent-projects'

const project = (path: string, lastUsedAt: number, name?: string): RecentProject => ({
  path,
  name: name ?? projectDisplayName(path),
  lastUsedAt
})

const session = (cwd: null | string | undefined, startedAt: number): SessionInfo =>
  ({ id: `s-${cwd}-${startedAt}`, cwd, started_at: startedAt }) as unknown as SessionInfo

describe('normalizeProjectPath', () => {
  it('strips trailing separators but keeps the path stable', () => {
    expect(normalizeProjectPath('/Users/x/proj/')).toBe('/Users/x/proj')
    expect(normalizeProjectPath('/Users/x/proj')).toBe('/Users/x/proj')
    expect(normalizeProjectPath('C:\\code\\app\\')).toBe('C:\\code\\app')
  })

  it('returns empty for blank input', () => {
    expect(normalizeProjectPath('')).toBe('')
    expect(normalizeProjectPath('   ')).toBe('')
    expect(normalizeProjectPath(null)).toBe('')
    expect(normalizeProjectPath(undefined)).toBe('')
  })

  it('collapses paths that differ only by a trailing slash to one key', () => {
    expect(normalizeProjectPath('/a/b/')).toBe(normalizeProjectPath('/a/b'))
  })
})

describe('projectDisplayName', () => {
  it('uses the final path segment', () => {
    expect(projectDisplayName('/Users/x/hermes-agent')).toBe('hermes-agent')
    expect(projectDisplayName('C:\\code\\my-app')).toBe('my-app')
  })
})

describe('upsertRecentProject', () => {
  it('inserts a new project at the top with the given time', () => {
    const next = upsertRecentProject([], '/a/proj', 100)
    expect(next).toEqual([project('/a/proj', 100)])
  })

  it('dedupes by normalized path and bumps the timestamp', () => {
    const start = [project('/a/proj', 100), project('/b/other', 200)]
    const next = upsertRecentProject(start, '/a/proj/', 300)

    expect(next).toHaveLength(2)
    expect(next[0]).toEqual(project('/a/proj', 300))
    expect(next.filter(p => p.path === '/a/proj')).toHaveLength(1)
  })

  it('prefers an explicit name over the derived basename', () => {
    const next = upsertRecentProject([], '/a/proj', 100, 'Custom Name')
    expect(next[0]?.name).toBe('Custom Name')
  })

  it('caps the list to the MRU limit, dropping the oldest', () => {
    let list: RecentProject[] = []

    for (let i = 0; i < 30; i += 1) {
      list = upsertRecentProject(list, `/p/${i}`, i)
    }

    expect(list).toHaveLength(24)
    // Newest first, oldest 6 dropped.
    expect(list[0]?.path).toBe('/p/29')
    expect(list.some(p => p.path === '/p/5')).toBe(false)
    expect(list.some(p => p.path === '/p/6')).toBe(true)
  })

  it('ignores blank paths', () => {
    const start = [project('/a/proj', 100)]
    expect(upsertRecentProject(start, '   ', 200)).toBe(start)
  })
})

describe('sessionProjectEntries', () => {
  it('keeps project cwds and drops home-dir / empty chats', () => {
    const entries = sessionProjectEntries([
      session('/Users/kael/hermes-agent', 500),
      session('/Users/kael', 400), // home dir → plain chat, skipped
      session('', 300), // no cwd → skipped
      session(null, 200)
    ])

    expect(entries.map(e => e.path)).toEqual(['/Users/kael/hermes-agent'])
  })

  it('collapses multiple sessions in one project to its freshest start time', () => {
    const entries = sessionProjectEntries([
      session('/code/app', 100),
      session('/code/app/', 900),
      session('/code/app', 400)
    ])

    expect(entries).toHaveLength(1)
    expect(entries[0]).toEqual(project('/code/app', 900))
  })
})

describe('mergeRecentProjects', () => {
  it('unions persisted + derived, dedupes by path, keeps freshest time', () => {
    const persisted = [project('/code/app', 100, 'My App')]
    const derived = [project('/code/app', 900), project('/code/other', 300)]

    const merged = mergeRecentProjects(persisted, derived)

    expect(merged).toHaveLength(2)
    const app = merged.find(p => p.path === '/code/app')!
    expect(app.name).toBe('My App') // explicit persisted name wins
    expect(app.lastUsedAt).toBe(900) // freshest time wins
    // Sorted most-recent-first.
    expect(merged[0]?.path).toBe('/code/app')
  })
})

describe('filterRecentProjects', () => {
  const list = [
    project('/Users/kael/hermes-agent', 300),
    project('/Users/kael/hermes-cloud', 200),
    project('/work/jewelry-store', 100)
  ]

  it('returns everything for a blank query', () => {
    expect(filterRecentProjects(list, '')).toEqual(list)
    expect(filterRecentProjects(list, '   ')).toEqual(list)
  })

  it('matches on name', () => {
    expect(filterRecentProjects(list, 'jewelry').map(p => p.path)).toEqual(['/work/jewelry-store'])
  })

  it('matches on path segments', () => {
    expect(filterRecentProjects(list, 'kael').map(p => p.path)).toEqual([
      '/Users/kael/hermes-agent',
      '/Users/kael/hermes-cloud'
    ])
  })

  it('requires every whitespace-separated term to match', () => {
    expect(filterRecentProjects(list, 'kael cloud').map(p => p.path)).toEqual(['/Users/kael/hermes-cloud'])
    expect(filterRecentProjects(list, 'kael nope')).toEqual([])
  })
})

describe('parseRecentProjects', () => {
  it('returns empty for null / malformed input', () => {
    expect(parseRecentProjects(null)).toEqual([])
    expect(parseRecentProjects('not json')).toEqual([])
    expect(parseRecentProjects('{"path":"/a"}')).toEqual([]) // not an array
  })

  it('drops rows without a usable path and dedupes', () => {
    const raw = JSON.stringify([
      { path: '/a/proj', name: 'Proj', lastUsedAt: 100 },
      { path: '  ' },
      { name: 'no path' },
      { path: '/a/proj/', lastUsedAt: 200 } // dup of first after normalize
    ])

    const parsed = parseRecentProjects(raw)
    expect(parsed).toEqual([project('/a/proj', 100, 'Proj')])
  })

  it('backfills a missing name from the path basename', () => {
    const parsed = parseRecentProjects(JSON.stringify([{ path: '/x/y/cool-proj', lastUsedAt: 5 }]))
    expect(parsed[0]?.name).toBe('cool-proj')
  })
})
