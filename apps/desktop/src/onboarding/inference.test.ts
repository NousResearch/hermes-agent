import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { OnboardingRequester } from './due'
import { type InferenceWait, START_WAIT_MS, waitForInference } from './inference'

const READY_STATUS = { free_tier_account: true, free_tier_route: true, provider_configured: true, ready: true }

/** A backend whose `method` answers `reply` after `delayMs`, and whose other methods never answer. */
function slowBackend(answers: Record<string, { delayMs: number; reply: object }>): OnboardingRequester {
  return async <T>(method: string) => {
    const answer = answers[method]

    if (!answer) {
      return new Promise<T>(() => {})
    }

    await new Promise(resolve => setTimeout(resolve, answer.delayMs))

    // SAFETY: each method answers the shape waitForInference reads back.
    return answer.reply as T
  }
}

function track(wait: Promise<InferenceWait>) {
  const seen: { result?: InferenceWait } = {}

  void wait.then(result => {
    seen.result = result
  })

  return seen
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('waitForInference', () => {
  it('times out at the Start wait while setup.status never answers', async () => {
    const seen = track(waitForInference(slowBackend({})))

    await vi.advanceTimersByTimeAsync(START_WAIT_MS + 1)

    expect(seen.result).toEqual({ ok: false, reason: 'timeout' })
  })

  it('does not take a runtime check that answers after the Start wait', async () => {
    const seen = track(
      waitForInference(
        slowBackend({
          'setup.runtime_check': { delayMs: START_WAIT_MS, reply: { ok: true } },
          'setup.status': { delayMs: 1_000, reply: READY_STATUS }
        })
      )
    )

    await vi.advanceTimersByTimeAsync(START_WAIT_MS + 1)

    expect(seen.result).toEqual({ ok: false, reason: 'timeout' })
  })
})
