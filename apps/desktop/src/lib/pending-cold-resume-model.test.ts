import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  _resetPendingColdResumeModelChoiceForTests,
  applyPendingColdResumeModelChoice,
  recordPendingColdResumeModelChoice,
  takePendingColdResumeModelChoice
} from '@/lib/pending-cold-resume-model'
import { $currentModel, $currentModelSource, $currentProvider } from '@/store/session'

type RequestGateway = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

const choice = (overrides: Partial<Parameters<typeof recordPendingColdResumeModelChoice>[0]> = {}) => ({
  storedSessionId: 'stored-1',
  provider: 'anthropic',
  model: 'claude-opus-5-5',
  prevModel: 'gpt-6.1-sol',
  prevProvider: 'openai-codex',
  prevSource: 'default' as const,
  ...overrides
})

describe('pendingColdResumeModelChoice', () => {
  beforeEach(() => {
    _resetPendingColdResumeModelChoiceForTests()
    $currentModel.set('')
    $currentProvider.set('')
    $currentModelSource.set('')
  })

  afterEach(() => {
    _resetPendingColdResumeModelChoiceForTests()
    $currentModel.set('')
    $currentProvider.set('')
    $currentModelSource.set('')
  })

  it('consumes a recorded choice exactly once for the owning stored session', () => {
    recordPendingColdResumeModelChoice(choice())

    expect(takePendingColdResumeModelChoice('stored-1')?.model).toBe('claude-opus-5-5')
    expect(takePendingColdResumeModelChoice('stored-1')).toBeNull()
  })

  it('keeps a choice that targets a different stored session instead of leaking it', () => {
    recordPendingColdResumeModelChoice(choice())

    expect(takePendingColdResumeModelChoice('stored-2')).toBeNull()
    expect(takePendingColdResumeModelChoice('stored-1')?.model).toBe('claude-opus-5-5')
  })

  it('sends one config.set to the rebound runtime and clears the slot on success', async () => {
    recordPendingColdResumeModelChoice(choice())
    const requestGateway = vi.fn<Parameters<RequestGateway>>(() => Promise.resolve({}))

    await applyPendingColdResumeModelChoice('stored-1', 'runtime-9', requestGateway as unknown as RequestGateway)

    expect(requestGateway).toHaveBeenCalledTimes(1)
    expect(requestGateway).toHaveBeenCalledWith('config.set', {
      session_id: 'runtime-9',
      key: 'model',
      value: 'claude-opus-5-5 --provider anthropic'
    })
    expect(takePendingColdResumeModelChoice('stored-1')).toBeNull()
  })

  it('pins a moa preset with the session-only scope', async () => {
    recordPendingColdResumeModelChoice(choice({ provider: 'moa', model: 'balanced' }))
    const requestGateway = vi.fn<Parameters<RequestGateway>>(() => Promise.resolve({}))

    await applyPendingColdResumeModelChoice('stored-1', 'runtime-9', requestGateway as unknown as RequestGateway)

    expect(requestGateway).toHaveBeenCalledWith('config.set', {
      session_id: 'runtime-9',
      key: 'model',
      value: 'balanced --provider moa --session'
    })
  })

  it('rolls the composer back when the switch needs an expensive-model confirmation', async () => {
    recordPendingColdResumeModelChoice(choice())

    const requestGateway = vi.fn<Parameters<RequestGateway>>(() =>
      Promise.resolve({ confirm_required: true, confirm_message: 'expensive' })
    )

    const applied = await applyPendingColdResumeModelChoice(
      'stored-1',
      'runtime-9',
      requestGateway as unknown as RequestGateway
    )

    expect(applied).toBe(false)
    expect($currentModel.get()).toBe('gpt-6.1-sol')
    expect($currentProvider.get()).toBe('openai-codex')
    expect($currentModelSource.get()).toBe('default')
    expect(takePendingColdResumeModelChoice('stored-1')).toBeNull()
  })

  it('rolls the composer back when the switch call fails', async () => {
    recordPendingColdResumeModelChoice(choice())
    const requestGateway = vi.fn<Parameters<RequestGateway>>(() => Promise.reject(new Error('gateway down')))

    const applied = await applyPendingColdResumeModelChoice(
      'stored-1',
      'runtime-9',
      requestGateway as unknown as RequestGateway
    )

    expect(applied).toBe(false)
    expect($currentModel.get()).toBe('gpt-6.1-sol')
    expect($currentProvider.get()).toBe('openai-codex')
  })

  it('is a no-op without a recorded choice', async () => {
    const requestGateway = vi.fn<Parameters<RequestGateway>>(() => Promise.resolve({}))

    const applied = await applyPendingColdResumeModelChoice(
      'stored-1',
      'runtime-9',
      requestGateway as unknown as RequestGateway
    )

    expect(applied).toBe(false)
    expect(requestGateway).not.toHaveBeenCalled()
  })
})
