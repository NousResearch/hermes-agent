import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $cronSessions, $messagingSessions, $sessions, $unlistedSessionOwnerRows } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { shouldShowIntro, useBlankTileDraft } from './intro-visibility'
import { tileCaption } from './session-tile'

// The "+" / ⌘T tab, and "New session" into a focused tab, both mint a tile
// whose session already has a row: the owner stub recorded at creation so its
// resume routes (#102792). That stub has no title, preview or messages. The tab
// read it like a real untitled conversation — captioned "Untitled session" and
// opened as an empty pane — instead of as the new session it is.

const DRAFT = 'draft-1'

function row(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    cwd: null,
    ended_at: null,
    id: DRAFT,
    input_tokens: 0,
    is_active: true,
    last_active: 1,
    message_count: 0,
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

const tile = (storedSessionId: string) => ({ dir: 'center', storedSessionId }) as never

afterEach(() => {
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $unlistedSessionOwnerRows.set([])
  $sessionTiles.set([])
})

describe('a new-session tab reads as a new session', () => {
  it('captions a fresh draft "New session", not "Untitled session"', () => {
    $sessionTiles.set([tile(DRAFT)])
    $unlistedSessionOwnerRows.set([row()])

    expect(tileCaption(DRAFT)).toBe('New session')
  })

  it('captions the same draft "New session" after a relaunch probed its blank row back', () => {
    $sessionTiles.set([tile(DRAFT)])
    $sessions.set([row()])

    expect(tileCaption(DRAFT)).toBe('New session')
  })

  it('still names a real conversation by its own title, or "Untitled session" once it has messages', () => {
    $sessionTiles.set([tile(DRAFT)])
    $sessions.set([row({ message_count: 2, title: 'Plan the launch' })])
    expect(tileCaption(DRAFT)).toBe('Plan the launch')

    $sessions.set([row({ message_count: 2 })])
    expect(tileCaption(DRAFT)).toBe('Untitled session')
  })

  it('shows the new-session intro in a draft tab, and not in a tab whose session has started', () => {
    $unlistedSessionOwnerRows.set([row()])
    const draft = renderHook(() => useBlankTileDraft(false, DRAFT))
    expect(draft.result.current).toBe(true)

    $unlistedSessionOwnerRows.set([])
    $sessions.set([row({ message_count: 4, preview: 'hello' })])
    draft.rerender()
    expect(draft.result.current).toBe(false)
  })

  it('does not treat a restored tab whose row has not resolved yet as a draft', () => {
    const { result } = renderHook(() => useBlankTileDraft(false, DRAFT))

    expect(result.current).toBe(false)
  })

  it('renders the intro for a draft tab under the same rules as main', () => {
    const tab = {
      activeSessionId: 'runtime-1',
      auxiliaryWindow: false,
      enabled: true,
      freshDraftReady: false,
      messagesEmpty: true,
      primary: false,
      routedSessionView: false,
      selectedSessionId: DRAFT,
      tileDraft: true
    }

    expect(shouldShowIntro(tab)).toBe(true)
    expect(shouldShowIntro({ ...tab, enabled: false })).toBe(false)
    expect(shouldShowIntro({ ...tab, messagesEmpty: false })).toBe(false)
    expect(shouldShowIntro({ ...tab, auxiliaryWindow: true })).toBe(false)
    expect(shouldShowIntro({ ...tab, tileDraft: false })).toBe(false)
  })
})
