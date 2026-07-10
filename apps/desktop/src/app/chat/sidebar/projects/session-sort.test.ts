import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { projectPreviewLimit } from '@/lib/project-session-sort'

import { flattenProjectSessions, pageProjectSessions, sortProjectSessions } from './session-sort'

let nextId = 0

function session(title: null | string, overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: false,
    cwd: '/repo',
    ended_at: null,
    id: `session-${nextId++}`,
    input_tokens: 0,
    is_active: false,
    last_active: 1_000,
    message_count: 1,
    model: 'claude',
    output_tokens: 0,
    preview: null,
    source: 'cli',
    started_at: 1_000,
    title,
    tool_call_count: 0,
    ...overrides
  }
}

describe('sortProjectSessions', () => {
  it('requests all gateway candidates only for title modes', () => {
    expect(projectPreviewLimit('recent')).toBe(3)
    expect(projectPreviewLimit('title-asc')).toBe(2_000)
    expect(projectPreviewLimit('title-desc')).toBe(2_000)
  })

  it('keeps backend recent-activity order unchanged by default', () => {
    const recent = session('00_Personal-2', { id: 'recent', last_active: 300 })
    const older = session('00_Personal-1', { id: 'older', last_active: 200 })
    const input = [recent, older]

    expect(sortProjectSessions(input, 'recent')).toBe(input)
  })

  it('sorts titles naturally and case-insensitively in ascending order', () => {
    const input = [
      session('00_Personal-10', { id: 'ten' }),
      session('beta', { id: 'beta' }),
      session('00_Personal-2', { id: 'two' }),
      session('00_Personal-1', { id: 'one' }),
      session('Alpha', { id: 'alpha' })
    ]

    expect(sortProjectSessions(input, 'title-asc').map(item => item.id)).toEqual(['one', 'two', 'ten', 'alpha', 'beta'])
  })

  it('sorts titles naturally in descending order and keeps untitled sessions after named rows', () => {
    const input = [
      session(null, { id: 'untitled-old', last_active: 100 }),
      session('00_Personal-1', { id: 'one' }),
      session('00_Personal-2', { id: 'two' }),
      session('  ', { id: 'untitled-new', last_active: 200 })
    ]

    expect(sortProjectSessions(input, 'title-desc').map(item => item.id)).toEqual([
      'two',
      'one',
      'untitled-new',
      'untitled-old'
    ])
  })

  it('uses activity then id as deterministic title-sort tie-breakers', () => {
    const input = [
      session('Alpha', { id: 'z', last_active: 100 }),
      session('alpha', { id: 'b', last_active: 200 }),
      session('ALPHA', { id: 'a', last_active: 200 })
    ]

    expect(sortProjectSessions(input, 'title-asc').map(item => item.id)).toEqual(['a', 'b', 'z'])
  })

  it('keeps branch children adjacent to and after their parent before pagination', () => {
    const input = [
      session('Alpha child', { id: 'child', parent_session_id: 'parent' }),
      session('Beta root', { id: 'beta' }),
      session('Zulu parent', { id: 'parent' })
    ]

    const ordered = sortProjectSessions(input, 'title-asc')
    const entries = flattenProjectSessions(ordered.slice(0, 3), 'title-asc')

    expect(ordered.map(item => item.id)).toEqual(['beta', 'parent', 'child'])
    expect(entries.map(entry => [entry.session.id, entry.branchStem ?? null])).toEqual([
      ['beta', null],
      ['parent', null],
      ['child', '└─ ']
    ])
  })

  it('extends a title-sorted page through an entire branch subtree', () => {
    const parent = session('Alpha parent', { id: 'parent' })

    const children = Array.from({ length: 6 }, (_, index) =>
      session(`Alpha child ${index + 1}`, { id: `child-${index + 1}`, parent_session_id: parent.id })
    )

    const input = [session('Beta root', { id: 'beta' }), ...children.reverse(), parent]

    const ordered = sortProjectSessions(input, 'title-asc')
    const firstPage = pageProjectSessions(ordered, 5)

    expect(firstPage.map(item => item.id)).toEqual([
      'parent',
      'child-1',
      'child-2',
      'child-3',
      'child-4',
      'child-5',
      'child-6'
    ])
    expect(flattenProjectSessions(firstPage, 'title-asc').every(entry => entry.session.id === 'parent' || entry.branchStem)).toBe(true)
    expect(pageProjectSessions(ordered, 10).map(item => item.id)).toEqual([...firstPage.map(item => item.id), 'beta'])
  })

  it('extends a title-sorted page through a compressed lineage root alias', () => {
    const continuation = session('Alpha continuation', { id: 'continuation', _lineage_root_id: 'original-root' })
    const child = session('Alpha child', { id: 'child', parent_session_id: 'original-root' })
    const input = [session('Beta root', { id: 'beta' }), child, continuation]

    const ordered = sortProjectSessions(input, 'title-asc')

    expect(pageProjectSessions(ordered, 1).map(item => item.id)).toEqual(['continuation', 'child'])
  })

  it('sorts branch clusters and sibling branches by title without breaking nesting', () => {
    const input = [
      session('Zulu branch', { id: 'beta-zulu', parent_session_id: 'beta-parent' }),
      session('Gamma parent', { id: 'gamma-parent' }),
      session('Beta parent', { id: 'beta-parent' }),
      session('Alpha branch', { id: 'beta-alpha', parent_session_id: 'beta-parent' }),
      session('Delta branch', { id: 'gamma-delta', parent_session_id: 'gamma-parent' })
    ]

    expect(flattenProjectSessions(input, 'title-asc').map(entry => entry.session.id)).toEqual([
      'beta-parent',
      'beta-alpha',
      'beta-zulu',
      'gamma-parent',
      'gamma-delta'
    ])
    expect(flattenProjectSessions(input, 'title-desc').map(entry => entry.session.id)).toEqual([
      'gamma-parent',
      'gamma-delta',
      'beta-parent',
      'beta-zulu',
      'beta-alpha'
    ])
  })

  it('uses the active UI locale for non-Latin title collation', () => {
    const input = [
      session('张', { id: 'zhang' }),
      session('李', { id: 'li' }),
      session('波', { id: 'bo' }),
      session('阿', { id: 'a' })
    ]

    expect(sortProjectSessions(input, 'title-asc', 'zh').map(item => item.id)).toEqual(['a', 'bo', 'li', 'zhang'])
  })
})
