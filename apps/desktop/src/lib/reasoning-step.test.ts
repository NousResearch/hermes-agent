import { describe, expect, it, vi } from 'vitest'

import { stepReasoningEffort, writeSessionReasoningEffort } from '@/lib/reasoning-step'

describe('stepReasoningEffort', () => {
  it('steps up and down through off and the normal levels, clamping at both ends', () => {
    expect(stepReasoningEffort('none', 1)).toBe('minimal')
    expect(stepReasoningEffort('minimal', 1)).toBe('low')
    expect(stepReasoningEffort('low', 1)).toBe('medium')
    expect(stepReasoningEffort('medium', 1)).toBe('high')
    expect(stepReasoningEffort('high', 1)).toBe('xhigh')
    expect(stepReasoningEffort('xhigh', 1)).toBe('xhigh')

    expect(stepReasoningEffort('xhigh', -1)).toBe('high')
    expect(stepReasoningEffort('high', -1)).toBe('medium')
    expect(stepReasoningEffort('medium', -1)).toBe('low')
    expect(stepReasoningEffort('low', -1)).toBe('minimal')
    expect(stepReasoningEffort('minimal', -1)).toBe('none')
    expect(stepReasoningEffort('none', -1)).toBe('none')
  })

  it('never steps into the expensive tiers (max/ultra stay behind the menu)', () => {
    // From an expensive tier, stepping down rejoins the ladder at xhigh; up is
    // a no-op (callers compare and skip the RPC).
    expect(stepReasoningEffort('max', -1)).toBe('xhigh')
    expect(stepReasoningEffort('ultra', -1)).toBe('xhigh')
    expect(stepReasoningEffort('max', 1)).toBe('max')
    // The ladder's top clamps — stepping up from xhigh never crosses into max.
    expect(stepReasoningEffort('xhigh', 1)).toBe('xhigh')
  })

  it('resolves an unset or stale value through the fallback before stepping', () => {
    expect(stepReasoningEffort('', 1, 'low')).toBe('medium')
    expect(stepReasoningEffort('', -1, 'low')).toBe('minimal')
    // An unrecognized value falls back to Hermes' own default (medium).
    expect(stepReasoningEffort('banana', 1, 'high')).toBe('high')
    expect(stepReasoningEffort('', 1)).toBe('high')
  })
})

describe('writeSessionReasoningEffort', () => {
  it('writes the session-scoped reasoning RPC and reports the gateway value', async () => {
    const request = vi.fn().mockResolvedValue({ value: 'xhigh' })

    await expect(writeSessionReasoningEffort(request, 'runtime-1', 'xhigh')).resolves.toBe('xhigh')

    expect(request).toHaveBeenCalledWith('config.set', { key: 'reasoning', session_id: 'runtime-1', value: 'xhigh' })
  })

  it('normalizes a gateway reply that is not a step level', async () => {
    const request = vi.fn().mockResolvedValue({ value: 'banana' })

    // The gateway echoed something unrecognized: settle on the default level
    // rather than passing the stale string back into the store.
    await expect(writeSessionReasoningEffort(request, 'runtime-1', 'high')).resolves.toBe('medium')
  })
})
