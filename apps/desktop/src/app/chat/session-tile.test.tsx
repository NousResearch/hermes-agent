import { afterEach, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { $activeSessionId } from '@/store/session'
import { $sessionTiles, clearAllSessionStates, publishSessionState } from '@/store/session-states'
import { holdTranscriptView } from '@/store/session-transcript-view'

import { buildTileView } from './session-tile'
import { PRIMARY_SESSION_VIEW } from './session-view'

vi.mock('./index', () => ({ ChatView: () => null }))
afterEach(() => { clearAllSessionStates(); clearClarifyRequest(); $sessionTiles.set([]); $activeSessionId.set(null) })

it('tiles share the primary runtime projection while another runtime and request stay isolated', () => {
  const a = { ...createClientSessionState('stored-A'), messages: [{ id: 'raw-A', role: 'assistant' as const, parts: [] }] }
  const b = { ...createClientSessionState('stored-B'), messages: [{ id: 'raw-B', role: 'assistant' as const, parts: [] }] }
  publishSessionState('A', a)
  publishSessionState('B', b)
  $sessionTiles.set([{ storedSessionId: 'stored-A', runtimeId: 'A' }, { storedSessionId: 'stored-B', runtimeId: 'B' }])
  $activeSessionId.set('A')
  const tileA = buildTileView('stored-A')
  const tileB = buildTileView('stored-B')
  const stopA = tileA.$messages.listen(() => {})
  const stopB = tileB.$messages.listen(() => {})

  try {
    const requestB = { sessionId: 'B', requestId: 'RB', question: 'B?', choices: null, multiSelect: false }
    setClarifyRequest(requestB)
    holdTranscriptView('A')
    setClarifyRequest({ ...requestB, sessionId: 'A', requestId: 'RA', question: 'A?' })
    expect(tileA.$messages.get()).toEqual(PRIMARY_SESSION_VIEW.$messages.get())
    expect(tileA.$messages.get()[0].id).toBe('pending-clarify:A:RA')
    expect(tileB.$messages.get()).toBe(b.messages)
    const projected = tileA.$messages.get()
    publishSessionState('A', { ...a, busy: true })
    setClarifyRequest({ ...requestB, question: 'B changed?' })
    expect(tileA.$messages.get()).toBe(projected)
    $sessionTiles.set([{ storedSessionId: 'stored-A', runtimeId: 'B' }])
    expect(tileA.$messages.get()).toBe(b.messages)
    expect(tileB.$messages.get()).toEqual([])
  } finally { stopA(); stopB() }
})
