import { beforeEach, describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import {
  createFallbackNotice,
  deferFallbackNotice,
  FALLBACK_NOTICE_STORAGE_KEY,
  flushPendingFallbackNotices,
  persistFallbackNotice,
  pruneFallbackNoticesAfter,
  restoreFallbackNotices
} from './fallback-notices'

const message = (id: string, role: ChatMessage['role'], text: string): ChatMessage => ({
  id,
  role,
  parts: [{ type: 'text', text }]
})

describe('fallback notice persistence', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('restores a decision at its transcript boundary without changing model history', () => {
    const storedMessages = [
      message('u1', 'user', 'first'),
      message('a1', 'assistant', 'first answer'),
      message('u2', 'user', 'second'),
      message('a2', 'assistant', 'fallback answer')
    ]

    const notice = createFallbackNotice('Fallback policy local-only: switching to local/model', 1_700_000_000_000)

    persistFallbackNotice('stored-1', notice, 3, 1_700_000_000_001)
    const restored = restoreFallbackNotices('stored-1', storedMessages)

    expect(restored.map(item => item.id)).toEqual(['u1', 'a1', 'u2', notice.id, 'a2'])
    expect(restored[3]).toMatchObject({ role: 'system', timestamp: 1_700_000_000 })
    expect(storedMessages.map(item => item.id)).toEqual(['u1', 'a1', 'u2', 'a2'])
  })

  it('is idempotent when a live notice is already present', () => {
    const notice = createFallbackNotice('switching', 1_700_000_000_000)
    persistFallbackNotice('stored-1', notice, 1)

    const messages = [message('u1', 'user', 'hello'), notice, message('a1', 'assistant', 'answer')]
    expect(restoreFallbackNotices('stored-1', messages)).toBe(messages)
  })

  it('uses collision-safe ids for rapid fallback chain switches', () => {
    const first = createFallbackNotice('first switch', 1_700_000_000_000)
    const second = createFallbackNotice('second switch', 1_700_000_000_000)

    expect(first.id).not.toBe(second.id)
  })

  it('backfills a decision emitted before the stored session id is assigned', () => {
    const notice = createFallbackNotice('init-time switch', 1_700_000_000_000)

    deferFallbackNotice('runtime-before-bind', notice, 1)
    expect(window.localStorage.getItem(FALLBACK_NOTICE_STORAGE_KEY)).toBeNull()

    flushPendingFallbackNotices('runtime-before-bind', 'stored-after-bind')

    const restored = restoreFallbackNotices('stored-after-bind', [
      message('u1', 'user', 'hello'),
      message('a1', 'assistant', 'answer')
    ])

    expect(restored.map(item => item.id)).toEqual(['u1', notice.id, 'a1'])
  })

  it('fails open on malformed renderer storage', () => {
    window.localStorage.setItem(FALLBACK_NOTICE_STORAGE_KEY, '{bad json')
    const messages = [message('u1', 'user', 'hello')]

    expect(restoreFallbackNotices('stored-1', messages)).toBe(messages)
  })

  it('drops notices from an abandoned timeline after a rewind boundary', () => {
    const retained = createFallbackNotice('retained switch', 1_700_000_000_000)
    const abandoned = createFallbackNotice('abandoned switch', 1_700_000_000_001)
    persistFallbackNotice('stored-1', retained, 2)
    persistFallbackNotice('stored-1', abandoned, 3)

    pruneFallbackNoticesAfter('stored-1', 2)

    const restored = restoreFallbackNotices('stored-1', [
      message('u1', 'user', 'first'),
      message('a1', 'assistant', 'first answer'),
      message('u2', 'user', 'rewritten prompt'),
      message('a2', 'assistant', 'new answer')
    ])

    expect(restored.map(item => item.id)).toContain(retained.id)
    expect(restored.map(item => item.id)).not.toContain(abandoned.id)
  })
})
