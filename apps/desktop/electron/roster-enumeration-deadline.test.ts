import { afterEach, describe, expect, it, vi } from 'vitest'

import { withRosterEnumerationDeadline } from './remote-oauth-ticket'

afterEach(() => vi.useRealTimers())

describe('roster enumeration deadline', () => {
  it('keeps the existing source deadline and clears its timer on timeout', async () => {
    vi.useFakeTimers()
    const pending = withRosterEnumerationDeadline(new Promise<never>(() => undefined), 10_000)
    const rejected = expect(pending).rejects.toThrow('roster enumeration timed out')
    await vi.advanceTimersByTimeAsync(10_000)
    await rejected
    expect(vi.getTimerCount()).toBe(0)
  })

  it('returns completed work and clears its deadline immediately', async () => {
    vi.useFakeTimers()
    await expect(withRosterEnumerationDeadline(Promise.resolve('ready'), 10_000)).resolves.toBe('ready')
    expect(vi.getTimerCount()).toBe(0)
  })
})
