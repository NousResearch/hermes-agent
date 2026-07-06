import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ContextFull } from '@/types/hermes'

import {
  $activeBucket,
  $activeTab,
  $contextData,
  $contextInspectorOpen,
  $contextSource,
  closeContextInspector,
  openContextInspector
} from './context-inspector'

const readyPayload = (overrides: Partial<ContextFull> = {}): ContextFull => ({
  available: true,
  context_max: 200_000,
  context_used: 12_000,
  exact_capture_available: false,
  messages: [],
  model: 'gpt-test',
  raw_unredacted: true,
  slices: [],
  source: 'reconstructed_base',
  source_label: 'Reconstructed base',
  state: 'ready',
  ...overrides
})

describe('context inspector store', () => {
  beforeEach(() => {
    closeContextInspector()
    $contextData.set(null)
    $contextSource.set({ status: 'idle' })
    $activeBucket.set('tools')
    $activeTab.set('raw')
  })

  it('resolves a stored lineage id to a live runtime id before calling the injected gateway request', async () => {
    const requestGateway = vi.fn().mockResolvedValue(readyPayload())

    await openContextInspector('stored-1', requestGateway, {
      runtimeIdByStoredSessionId: new Map([['stored-1', 'runtime-1']])
    })

    expect(requestGateway).toHaveBeenCalledTimes(1)
    expect(requestGateway).toHaveBeenCalledWith('session.context_full', { session_id: 'runtime-1' })
    expect($contextInspectorOpen.get()).toBe(true)
    expect($contextSource.get()).toMatchObject({ runtimeSessionId: 'runtime-1', sessionId: 'stored-1', status: 'ready' })
    expect($contextData.get()?.available).toBe(true)
    expect($activeBucket.get()).toBe('all')
    expect($activeTab.get()).toBe('transcript')
  })

  it('opens an agent_not_built empty state without fetching when a stored id has no live runtime', async () => {
    const requestGateway = vi.fn()

    await openContextInspector('stored-1', requestGateway, {
      runtimeIdByStoredSessionId: new Map()
    })

    expect(requestGateway).not.toHaveBeenCalled()
    expect($contextInspectorOpen.get()).toBe(true)
    expect($contextSource.get()).toMatchObject({ status: 'empty', sessionId: 'stored-1' })
    expect($contextData.get()).toMatchObject({ available: false, messages: [], slices: [], state: 'agent_not_built' })
  })

  it('stores the backend agent_not_built payload as an empty state', async () => {
    const requestGateway = vi.fn().mockResolvedValue(
      readyPayload({ available: false, messages: [], slices: [], state: 'agent_not_built' })
    )

    await openContextInspector('runtime-1', requestGateway)

    expect(requestGateway).toHaveBeenCalledTimes(1)
    expect($contextSource.get()).toMatchObject({ runtimeSessionId: 'runtime-1', status: 'empty' })
    expect($contextData.get()?.state).toBe('agent_not_built')
  })

  it('keeps an inline failure state when the gateway request fails', async () => {
    const requestGateway = vi.fn().mockRejectedValue(new Error('gateway offline'))

    await openContextInspector('runtime-1', requestGateway)

    expect(requestGateway).toHaveBeenCalledTimes(1)
    expect($contextInspectorOpen.get()).toBe(true)
    expect($contextSource.get()).toEqual({ error: 'gateway offline', runtimeSessionId: 'runtime-1', sessionId: 'runtime-1', status: 'error' })
    expect($contextData.get()).toBeNull()
  })

  it('does not let a slower earlier open overwrite a newer open (race guard)', async () => {
    // First open (runtime-1) resolves LATE; second open (runtime-2) resolves first.
    let resolveFirst: (value: ContextFull) => void = () => undefined

    const firstGateway = vi.fn().mockImplementation(
      () => new Promise<ContextFull>(resolve => { resolveFirst = resolve })
    )

    const secondGateway = vi.fn().mockResolvedValue(readyPayload({ model: 'session-2' }))

    const firstOpen = openContextInspector('runtime-1', firstGateway)
    await openContextInspector('runtime-2', secondGateway)

    // Newer open (session-2) already committed.
    expect($contextData.get()?.model).toBe('session-2')
    expect($contextSource.get()).toMatchObject({ runtimeSessionId: 'runtime-2', status: 'ready' })

    // Now the stale earlier request resolves — it must be dropped, not committed.
    resolveFirst(readyPayload({ model: 'session-1-STALE' }))
    await firstOpen

    expect($contextData.get()?.model).toBe('session-2')
    expect($contextSource.get()).toMatchObject({ runtimeSessionId: 'runtime-2', status: 'ready' })
  })
})
