import { afterEach, describe, expect, it, vi } from 'vitest'

import { $currentReasoningEffort, setCurrentReasoningEffort } from '@/store/session'

import { stepReasoningEffort, writeSessionReasoningEffort } from './model-reasoning'

afterEach(() => {
  setCurrentReasoningEffort('')
})

describe('model reasoning shortcuts', () => {
  it('steps reasoning effort through off and Hermes effort levels', () => {
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

  it('treats empty or stale reasoning values as Hermes default medium before stepping', () => {
    expect(stepReasoningEffort('', 1)).toBe('high')
    expect(stepReasoningEffort('', -1)).toBe('low')
    expect(stepReasoningEffort('banana', 1)).toBe('high')
  })

  it('persists a session-scoped reasoning effort and returns the normalized value', async () => {
    setCurrentReasoningEffort('low')
    const requestGateway = vi.fn().mockResolvedValue({ value: 'high' })

    await expect(writeSessionReasoningEffort(requestGateway, 'session-1', 'xhigh')).resolves.toBe('high')

    expect(requestGateway).toHaveBeenCalledWith('config.set', {
      key: 'reasoning',
      session_id: 'session-1',
      value: 'xhigh'
    })
    expect($currentReasoningEffort.get()).toBe('low')
  })
})
