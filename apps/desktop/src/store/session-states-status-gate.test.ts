import { beforeEach, describe, expect, it } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { setSessions } from '@/store/session'

import {
  $sessionStates,
  clearAllSessionStates,
  dropSessionState,
  publishSessionState
} from './session-states'
import { $attentionSessionIds, $draftSessionIds, $workingSessionIds } from './session-states-status-sets'

const state = (overrides: Partial<ClientSessionState> = {}): ClientSessionState =>
  ({
    storedSessionId: 'stored-a',
    messages: [],
    busy: true,
    awaitingResponse: true,
    needsInput: false,
    turnLive: false,
    ...overrides
  }) as unknown as ClientSessionState

beforeEach(() => {
  clearAllSessionStates()
  setSessions([])
})

describe('status-version gate', () => {
  it('does not recompute derived sets on a transcript-only publish', () => {
    publishSessionState('rt-a', state({ messages: [{ id: 'm1' } as never] }))

    let workingNotifications = 0
    let attentionNotifications = 0
    let draftNotifications = 0

    const unsubs = [
      $workingSessionIds.listen(() => {
        workingNotifications += 1
      }),
      $attentionSessionIds.listen(() => {
        attentionNotifications += 1
      }),
      $draftSessionIds.listen(() => {
        draftNotifications += 1
      })
    ]

    try {
      // Same length, new content: a streaming delta with no status edge.
      publishSessionState('rt-a', state({ messages: [{ id: 'm1', text: 'longer' } as never] }))

      expect(workingNotifications).toBe(0)
      expect(attentionNotifications).toBe(0)
      expect(draftNotifications).toBe(0)
      // And the cached arrays keep their references.
      expect($workingSessionIds.get()).toContain('stored-a')
    } finally {
      for (const unsub of unsubs) {
        unsub()
      }
    }
  })

  it('recomputes on a busy edge and on a message-count change', () => {
    publishSessionState('rt-a', state({ messages: [{ id: 'm1' } as never] }))
    expect($workingSessionIds.get()).toContain('stored-a')

    publishSessionState('rt-a', state({ busy: false, awaitingResponse: false, messages: [] }))
    expect($workingSessionIds.get()).not.toContain('stored-a')
    // Idle with no messages and no row: a draft.
    expect($draftSessionIds.get()).toContain('stored-a')
  })

  it('stays correct for direct writes (no producer accounting)', () => {
    publishSessionState('rt-a', state())
    expect($workingSessionIds.get()).toContain('stored-a')

    // A direct write flips busy without going through publishSessionState:
    // the watchdog invalidates the versions, so the sets still follow.
    $sessionStates.set({ 'rt-a': state({ busy: false, awaitingResponse: false }) })
    expect($workingSessionIds.get()).not.toContain('stored-a')

    dropSessionState('rt-a')
    publishSessionState('rt-b', state({ storedSessionId: 'stored-b', busy: false, awaitingResponse: false }))
    expect($draftSessionIds.get()).toContain('stored-b')
    dropSessionState('rt-b')
    expect($draftSessionIds.get()).not.toContain('stored-b')
  })

  it('tracks needsInput through the attention set', () => {
    publishSessionState('rt-a', state())
    expect($attentionSessionIds.get()).toHaveLength(0)

    publishSessionState('rt-a', state({ needsInput: true }))
    expect($attentionSessionIds.get()).toContain('stored-a')

    // Transcript-only traffic while waiting on the user: no recompute.
    let attentionNotifications = 0

    const unsub = $attentionSessionIds.listen(() => {
      attentionNotifications += 1
    })

    try {
      publishSessionState('rt-a', state({ needsInput: true, messages: [{ id: 'm1' } as never] }))
      expect(attentionNotifications).toBe(0)
      expect($attentionSessionIds.get()).toContain('stored-a')
    } finally {
      unsub()
    }
  })
})

describe('draft row map', () => {
  it('resolves rows by live id, lineage root, and rotated tip', () => {
    setSessions([
      {
        id: 'tip-2',
        _lineage_root_id: 'root-1',
        _lineage_ids: ['root-1', 'tip-1', 'tip-2'],
        message_count: 0
      } as never,
      { id: 'other', message_count: 0 } as never
    ])

    // Empty transcript, unlisted stored id: draft by definition.
    publishSessionState('rt-a', state({ storedSessionId: 'tip-2', busy: false, awaitingResponse: false }))
    expect($draftSessionIds.get()).toContain('tip-2')

    // A middle-segment id resolves to the same row (and its zero count).
    publishSessionState('rt-b', state({ storedSessionId: 'tip-1', busy: false, awaitingResponse: false }))
    expect($draftSessionIds.get()).toContain('tip-1')

    // Non-empty history is not a draft even with a zero-count row… (resume
    // window: messages arrive a beat after binding; the row tiebreak needs
    // the messages check first).
    publishSessionState('rt-c', state({ storedSessionId: 'other', busy: false, awaitingResponse: false }))
    expect($draftSessionIds.get()).toContain('other')
  })

  it('drops draft status once the row reports messages', () => {
    setSessions([{ id: 'stored-a', message_count: 4 } as never])

    publishSessionState('rt-a', state({ busy: false, awaitingResponse: false }))
    expect($draftSessionIds.get()).not.toContain('stored-a')
  })
})
