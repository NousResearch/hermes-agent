import { describe, expect, it } from 'vitest'

import { $gateway } from '@/store/gateway'

import { $startChatRetries, markLiveStartChat, retryStartChat, startChatRetry, takeLiveStartChat } from './start-chat'

describe('start_chat per-call state', () => {
  it("keeps one chat's retry off another chat's call with the same tool call id", async () => {
    const started = { profile: 'default', session_id: 'child-a', status: 'started', title: 'Task A' }

    // SAFETY: retryStartChat only calls gateway.request, which this stub provides.
    $gateway.set({ request: async () => started } as never)

    await retryStartChat('chat-a', 'call_0', 'runtime-a', { message: 'do A' })

    expect(startChatRetry($startChatRetries.get(), 'chat-a', 'call_0')).toMatchObject({ sessionId: 'child-a' })
    expect(startChatRetry($startChatRetries.get(), 'chat-b', 'call_0')).toBeUndefined()
  })

  it("does not auto-open another chat's call that reused the same tool call id", () => {
    markLiveStartChat('chat-a', 'call_0')

    expect(takeLiveStartChat('chat-b', 'call_0')).toBe(false)
    expect(takeLiveStartChat('chat-a', 'call_0')).toBe(true)
  })
})
