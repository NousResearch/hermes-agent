import { describe, expect, it, vi } from 'vitest'

import { type UpdateContinuation, updateContinuationToken } from '@/store/update-continuation'

import { deliverUpdateContinuation } from './use-update-continuation'

const continuation: UpdateContinuation = {
  armedAt: 1_000,
  requestId: 'a'.repeat(32),
  sessionId: 'stored-1'
}

describe('deliverUpdateContinuation', () => {
  it('submits once only after the durable transcript check', async () => {
    const loadMessages = vi.fn().mockResolvedValue({ messages: [] })
    const submitText = vi.fn().mockResolvedValue(true)
    const markAttempt = vi.fn()

    await expect(
      deliverUpdateContinuation(continuation, loadMessages, submitText, { markAttempt })
    ).resolves.toBe('submitted')
    expect(loadMessages).toHaveBeenCalledWith('stored-1')
    expect(markAttempt).toHaveBeenCalledOnce()
    expect(submitText).toHaveBeenCalledOnce()
    expect(submitText.mock.calls[0][0]).toContain(updateContinuationToken(continuation.requestId))
  })

  it('treats an existing request token as the idempotent gateway acknowledgement', async () => {
    const submitText = vi.fn()

    const loadMessages = vi.fn().mockResolvedValue({
      messages: [{ role: 'user', text: `done ${updateContinuationToken(continuation.requestId)}` }]
    })

    await expect(deliverUpdateContinuation(continuation, loadMessages, submitText)).resolves.toBe('already-present')
    expect(submitText).not.toHaveBeenCalled()
  })

  it('never retries within the same delivery run after an ambiguous submit timeout', async () => {
    const loadMessages = vi.fn().mockResolvedValue({ messages: [] })
    const submitText = vi.fn().mockRejectedValueOnce(new Error('request timed out'))

    await expect(
      deliverUpdateContinuation(continuation, loadMessages, submitText, {
        sleep: vi.fn().mockResolvedValue(undefined)
      })
    ).resolves.toBe(false)
    expect(loadMessages).toHaveBeenCalledOnce()
    expect(submitText).toHaveBeenCalledOnce()
  })

  it('keeps an explicitly rejected marker for a later confirmed idle phase', async () => {
    const noWait = vi.fn().mockResolvedValue(undefined)
    const submitText = vi.fn().mockResolvedValue(false)

    await expect(
      deliverUpdateContinuation(continuation, vi.fn().mockResolvedValue({ messages: [] }), submitText, {
        sleep: noWait
      })
    ).resolves.toBe(false)
    expect(submitText).toHaveBeenCalledOnce()
  })
})
