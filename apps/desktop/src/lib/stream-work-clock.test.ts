import { afterEach, beforeEach, expect, test, vi } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'
import {
  clearInFlightTurnJournal,
  type JournalableSessionState,
  persistInFlightTurnState,
  readInFlightTurnJournal,
  resetInFlightTurnJournalStateForTests
} from '@/lib/inflight-turn-journal'

let now = 0
const requests: { token: string; delayMs: number | null }[] = []
const listeners = new Set<(token: string) => void>()
const currentToken = () => requests.at(-1)?.token ?? ''

const wake = (token = currentToken()) => {
  for (const listener of listeners) {
    listener(token)
  }
}

const recoveredText = () =>
  readInFlightTurnJournal('native-clock')
    ?.messages.map(message => chatMessageText(message))
    .join('\n')

function state(text: string): JournalableSessionState {
  return {
    storedSessionId: 'native-clock',
    busy: true,
    awaitingResponse: false,
    streamId: 'assistant-stream-native',
    turnStartedAt: 1000,
    messages: [
      { id: 'u1', role: 'user', parts: [{ type: 'text', text: 'request' }] },
      { id: 'assistant-stream-native', role: 'assistant', pending: true, parts: [{ type: 'text', text }] }
    ]
  }
}

beforeEach(() => {
  resetInFlightTurnJournalStateForTests()
  vi.useFakeTimers()
  now = 0
  requests.length = 0
  listeners.clear()
  window.localStorage.clear()
  vi.spyOn(performance, 'now').mockImplementation(() => now)
  vi.stubGlobal('window', {
    localStorage: window.localStorage,
    hermesDesktop: {
      scheduleStreamWorkWake: (token: string, delayMs: number | null) => requests.push({ token, delayMs }),
      onStreamWorkWake: (callback: (token: string) => void) => {
        listeners.add(callback)

        return () => listeners.delete(callback)
      }
    }
  })
})

afterEach(() => {
  resetInFlightTurnJournalStateForTests()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

test('a quiet tail reaches the real journal on the native deadline with browser timers parked', () => {
  persistInFlightTurnState(state('first chunk'))
  now = 100
  persistInFlightTurnState(state('quiet tail'))
  now = 399
  wake()
  expect(recoveredText()).toBeUndefined()
  now = 401
  wake()
  expect(recoveredText()).toBe('request\nquiet tail')
  vi.advanceTimersByTime(1000)
  expect(recoveredText()).toBe('request\nquiet tail')
})

test('a cleared turn cannot be resurrected or prematurely checkpoint a replacement by a late native wake', () => {
  persistInFlightTurnState(state('cancelled turn'))
  const staleToken = currentToken()
  clearInFlightTurnJournal('native-clock')
  persistInFlightTurnState(state('replacement turn'))
  wake(staleToken)
  expect(recoveredText()).toBeUndefined()
  now = 401
  wake()
  expect(recoveredText()).toBe('request\nreplacement turn')
})
