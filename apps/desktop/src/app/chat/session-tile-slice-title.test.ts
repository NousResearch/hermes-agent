import { afterEach, describe, expect, it } from 'vitest'

import { $projectTree } from '@/store/projects'
import { $cronSessions, $messagingSessions, $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { tileStoredRow } from './session-tile'

const STORED = 'tg-chat-1'
const TITLE = 'Debug the office sensor'

function row(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    cwd: null,
    ended_at: null,
    id: STORED,
    input_tokens: 0,
    is_active: true,
    last_active: 1,
    message_count: 466,
    model: null,
    output_tokens: 0,
    parent_session_id: null,
    preview: null,
    source: 'telegram',
    started_at: 1,
    title: TITLE,
    tool_call_count: 0,
    ...overrides
  }
}

/** A tab for a gateway conversation used to read "New session" forever.
 *  The sidebar fetch splits its rows into three source-scoped slices, and
 *  recents EXCLUDES every messaging/cron source — so a telegram tab's row is
 *  only ever in `$messagingSessions`. `tileStoredRow` searched recents alone,
 *  missed it, and `tileTitle` fell through to NEW_SESSION_TITLE. No amount of
 *  activity could fix it: the row is never eligible for recents. */
describe('tileStoredRow resolves across every sidebar slice', () => {
  afterEach(() => {
    $sessions.set([])
    $cronSessions.set([])
    $messagingSessions.set([])
    $projectTree.set([])
  })

  it('resolves a local recents row', () => {
    $sessions.set([row({ source: 'desktop' })])

    expect(tileStoredRow(STORED)?.title).toBe(TITLE)
  })

  it('resolves a telegram row listed only in the messaging slice', () => {
    $messagingSessions.set([row()])

    expect(tileStoredRow(STORED)?.title).toBe(TITLE)
  })

  it('resolves a cron row listed only in the cron slice', () => {
    $cronSessions.set([row({ source: 'cron', title: 'nightly-report' })])

    expect(tileStoredRow(STORED)?.title).toBe('nightly-report')
  })

  it('matches a messaging row across compression by lineage root', () => {
    $messagingSessions.set([row({ _lineage_root_id: STORED, id: 'tg-chat-1-compressed-2' })])

    expect(tileStoredRow(STORED)?.title).toBe(TITLE)
  })

  it('still misses an id no slice holds, so the placeholder path survives', () => {
    $messagingSessions.set([row({ id: 'someone-else' })])

    expect(tileStoredRow(STORED)).toBeUndefined()
  })
})
