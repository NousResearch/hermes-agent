import { describe, expect, it } from 'vitest'

import { sessionPinId } from '@/store/session'
import type { SidebarFolder } from '@/store/sidebar-folders'
import type { SessionInfo } from '@/types/hermes'

import { filterUnfiledSessions, resolveFolderSessions } from './folders'

const session = (over: Partial<SessionInfo>): SessionInfo => ({
  archived: false,
  cwd: null,
  ended_at: null,
  id: 'live',
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 0,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 0,
  title: null,
  tool_call_count: 0,
  ...over
})

const folder = (over: Partial<SidebarFolder>): SidebarFolder => ({
  id: 'f1',
  name: 'Folder',
  open: true,
  sessionIds: [],
  ...over
})

// Mirror index.tsx's sessionByAnyId: index by both live id and lineage-root id.
function indexBy(sessions: SessionInfo[]): Map<string, SessionInfo> {
  const map = new Map<string, SessionInfo>()

  for (const s of sessions) {
    map.set(s.id, s)

    if (s._lineage_root_id && !map.has(s._lineage_root_id)) {
      map.set(s._lineage_root_id, s)
    }
  }

  return map
}

describe('resolveFolderSessions', () => {
  it('resolves keys to sessions in folder order', () => {
    const a = session({ id: 'a' })
    const b = session({ id: 'b' })
    const map = indexBy([a, b])

    const resolved = resolveFolderSessions(folder({ sessionIds: ['b', 'a'] }), map)

    expect(resolved.map(s => s.id)).toEqual(['b', 'a'])
  })

  it('drops stale / non-loaded keys without crashing', () => {
    const a = session({ id: 'a' })
    const map = indexBy([a])

    const resolved = resolveFolderSessions(folder({ sessionIds: ['ghost', 'a', 'also-gone'] }), map)

    expect(resolved.map(s => s.id)).toEqual(['a'])
  })

  it('resolves a lineage-root key to the live continuation', () => {
    const tip = session({ id: 'tip', _lineage_root_id: 'root' })
    const map = indexBy([tip])

    const resolved = resolveFolderSessions(folder({ sessionIds: ['root'] }), map)

    expect(resolved.map(s => s.id)).toEqual(['tip'])
  })

  it('does not duplicate when both live and root keys are stored', () => {
    const tip = session({ id: 'tip', _lineage_root_id: 'root' })
    const map = indexBy([tip])

    const resolved = resolveFolderSessions(folder({ sessionIds: ['root', 'tip'] }), map)

    expect(resolved.map(s => s.id)).toEqual(['tip'])
  })
})

describe('filterUnfiledSessions', () => {
  it('returns the input untouched when no keys are foldered', () => {
    const sessions = [session({ id: 'a' }), session({ id: 'b' })]

    expect(filterUnfiledSessions(sessions, new Set(), sessionPinId)).toBe(sessions)
  })

  it('removes foldered sessions by durable key', () => {
    const a = session({ id: 'a' })
    const b = session({ id: 'tip', _lineage_root_id: 'root' })
    const c = session({ id: 'c' })

    const result = filterUnfiledSessions([a, b, c], new Set(['root']), sessionPinId)

    expect(result.map(s => s.id)).toEqual(['a', 'c'])
  })
})
