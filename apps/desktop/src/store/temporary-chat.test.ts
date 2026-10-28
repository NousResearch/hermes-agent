import { beforeEach, expect, test } from 'vitest'

import { $activeSessionStoredIdRotation } from './session'
import {
  $temporaryChatEndRequests,
  $temporaryChats,
  $temporaryDraft,
  endTemporaryChat,
  expiredTemporaryChats,
  forgetTemporaryChat,
  isTemporarySession,
  registerTemporaryChat,
  takeTemporaryChatEndRequests,
  temporaryChatOnScreen,
  toggleTemporaryDraft,
  updateTemporaryIdleClocks
} from './temporary-chat'

beforeEach(() => {
  $temporaryDraft.set(false)
  $temporaryChats.set({})
  $temporaryChatEndRequests.set([])
  $activeSessionStoredIdRotation.set(null)
})

test('the draft toggle flips the next-send intent', () => {
  toggleTemporaryDraft()
  expect($temporaryDraft.get()).toBe(true)
  toggleTemporaryDraft()
  expect($temporaryDraft.get()).toBe(false)
})

test('a registered chat is temporary by stored AND runtime id until forgotten', () => {
  registerTemporaryChat('stored-1', 'rt-1')
  expect(isTemporarySession('stored-1')).toBe(true)
  expect(isTemporarySession('rt-1')).toBe(true)
  expect(isTemporarySession('other')).toBe(false)
  expect(isTemporarySession(null)).toBe(false)

  forgetTemporaryChat('stored-1')
  expect(isTemporarySession('stored-1')).toBe(false)
  expect(isTemporarySession('rt-1')).toBe(false)
})

test('a chat covered by a workspace page is not on screen', () => {
  registerTemporaryChat('temp', 'rt')
  expect(temporaryChatOnScreen('temp', false)).toBe('temp')
  expect(temporaryChatOnScreen('temp', true)).toBeNull()
  expect(temporaryChatOnScreen('saved', false)).toBeNull()
})

test('the idle clock runs only while a chat is off screen and not running, and expires at the TTL', () => {
  registerTemporaryChat('a', 'rt-a')
  registerTemporaryChat('b', 'rt-b')
  const ttl = 1_000

  // a is on screen, b is off screen and idle.
  let chats = updateTemporaryIdleClocks($temporaryChats.get(), chat => chat.storedId === 'a', 100)
  expect(chats.a.idleSince).toBeNull()
  expect(chats.b.idleSince).toBe(100)
  expect(expiredTemporaryChats(chats, 100 + ttl - 1, ttl)).toEqual([])
  expect(expiredTemporaryChats(chats, 100 + ttl, ttl)).toEqual(['b'])

  // Coming back (or a turn running) stops the clock; leaving again restarts it from then.
  chats = updateTemporaryIdleClocks(chats, () => true, 500)
  expect(chats.b.idleSince).toBeNull()
  chats = updateTemporaryIdleClocks(chats, () => false, 900)
  expect(chats.b.idleSince).toBe(900)
  expect(updateTemporaryIdleClocks(chats, () => false, 950)).toBe(chats) // no change → same object
})

test('End requests are queued once and drained once', () => {
  registerTemporaryChat('temp')
  endTemporaryChat('temp')
  endTemporaryChat('temp')
  endTemporaryChat('saved') // not a temporary chat: ignored
  expect(takeTemporaryChatEndRequests()).toEqual(['temp'])
  expect(takeTemporaryChatEndRequests()).toEqual([])
})

test('compression rotation re-keys the chat to the new tip', () => {
  registerTemporaryChat('root', 'rt')
  $activeSessionStoredIdRotation.set({
    nextStoredSessionId: 'tip',
    previousStoredSessionId: 'root',
    runtimeSessionId: 'rt'
  })

  expect(isTemporarySession('tip')).toBe(true)
  expect(isTemporarySession('root')).toBe(false)
  expect($temporaryChats.get().tip.runtimeId).toBe('rt')
})

test('a saved chat rotating is not made temporary', () => {
  $activeSessionStoredIdRotation.set({
    nextStoredSessionId: 'tip',
    previousStoredSessionId: 'root',
    runtimeSessionId: 'rt'
  })
  expect(isTemporarySession('tip')).toBe(false)
})
