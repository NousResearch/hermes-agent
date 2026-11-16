import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { listTileSessionRow } from './session-tile-actions'

const STORED = 'stored-tab'
const RUNTIME = 'runtime-tab'

function row(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    cwd: null,
    ended_at: null,
    id: STORED,
    input_tokens: 0,
    is_active: true,
    last_active: 1,
    message_count: 1,
    model: null,
    output_tokens: 0,
    parent_session_id: null,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: null,
    tool_call_count: 0,
    ...overrides
  }
}

function seed(preview: string, sessions: SessionInfo[] = $sessions.get()) {
  return listTileSessionRow({
    cwd: '/work/repo',
    model: 'claude-opus-5',
    preview,
    runtimeId: RUNTIME,
    sessions,
    storedSessionId: STORED
  })
}

describe('listTileSessionRow', () => {
  beforeEach(() => $sessions.set([]))
  afterEach(() => $sessions.set([]))

  it("lists an unlisted ⌘T tab from the user's first message", () => {
    expect(seed('fix the tab titles')).toBe(true)

    const listed = $sessions.get().find(session => session.id === STORED)

    expect(listed?.preview).toBe('fix the tab titles')
    expect(listed?.cwd).toBe('/work/repo')
    // Title stays null so the server's auto-title is what eventually wins.
    expect(listed?.title).toBeNull()
  })

  it('leaves an already-listed session alone so a real title survives re-sends', () => {
    const titled = row({ title: 'Tab title fix' })

    expect(seed('second message', [titled])).toBe(false)
    expect($sessions.get()).toHaveLength(0)
  })

  it('matches a listed session across compression by lineage root', () => {
    const rotated = row({ _lineage_root_id: STORED, id: 'stored-tab-compressed-2' })

    expect(seed('after compression', [rotated])).toBe(false)
  })

  it('does not list a session on an empty or whitespace-only send', () => {
    expect(seed('')).toBe(false)
    expect(seed('   ')).toBe(false)
    expect($sessions.get()).toHaveLength(0)
  })

  it('trims the seeded preview', () => {
    seed('  padded prompt  ')

    expect($sessions.get()[0]?.preview).toBe('padded prompt')
  })
})
