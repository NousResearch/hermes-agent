import { afterEach, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { $activeSessionId } from '@/store/session'
import { $sessionTiles, clearAllSessionStates, publishSessionState } from '@/store/session-states'
import { holdTranscriptView } from '@/store/session-transcript-view'

import { buildTileView } from './session-tile'
import { PRIMARY_SESSION_VIEW } from './session-view'

vi.mock('./index', () => ({ ChatView: () => null }))

afterEach(() => {
  clearAllSessionStates()
  clearClarifyRequest()
  $sessionTiles.set([])
  $activeSessionId.set(null)
})

it('PRIMARY and the real tile reader share the gate without leaking across runtime switches', () => {
  const a = {
    ...createClientSessionState('stored-A'),
    messages: [{ id: 'old-A', role: 'assistant' as const, parts: [] }]
  }

  const b = { ...createClientSessionState('stored-B'), messages: [{ id: 'B', role: 'user' as const, parts: [] }] }
  publishSessionState('A', a)
  publishSessionState('B', b)
  $sessionTiles.set([
    { storedSessionId: 'stored-A', runtimeId: 'A' },
    { storedSessionId: 'stored-B', runtimeId: 'B' }
  ])
  $activeSessionId.set('A')
  const tileA = buildTileView('stored-A')
  const tileB = buildTileView('stored-B')
  const stop = tileA.$messages.listen(() => {})
  const release = holdTranscriptView('A', Symbol('test'), a.messages)

  try {
    setClarifyRequest({ sessionId: 'A', requestId: 'req-A', question: 'A?', choices: null, multiSelect: false })
    expect(tileA.$messages.get()).toEqual(PRIMARY_SESSION_VIEW.$messages.get())
    expect(tileA.$messages.get().flatMap(message => message.parts)).toContainEqual(
      expect.objectContaining({ toolName: 'clarify' })
    )
    expect(tileB.$messages.get()).toBe(b.messages)
    const projected = tileA.$messages.get()
    publishSessionState('A', { ...a, busy: true })
    setClarifyRequest({ sessionId: 'B', requestId: 'req-B', question: 'B?', choices: null, multiSelect: false })
    expect(tileA.$messages.get()).toBe(projected)
    $activeSessionId.set('B')
    expect(PRIMARY_SESSION_VIEW.$messages.get()).toBe(b.messages)
    expect(tileA.$messages.get()).toBe(projected)
    clearClarifyRequest('req-A', 'A')
    expect(tileA.$messages.get()).toEqual([])
    release()
    expect(tileA.$messages.get()).toBe(a.messages)
  } finally {
    stop()
  }
})
