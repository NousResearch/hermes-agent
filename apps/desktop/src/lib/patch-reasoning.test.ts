import { beforeEach, describe, expect, it, vi } from 'vitest'

import { applyReasoningPatch } from './patch-reasoning'

// Mock the imports the helper touches so we can assert what was written and
// drive the success / failure paths deterministically.
const setModelPreset = vi.fn()
const notifyError = vi.fn()
const setCurrentReasoningEffort = vi.fn()

vi.mock('@/store/model-presets', () => ({ setModelPreset: (...args: unknown[]) => setModelPreset(...args) }))
vi.mock('@/store/notifications', () => ({ notifyError: (...args: unknown[]) => notifyError(...args) }))
vi.mock('@/store/session', () => ({
  setCurrentReasoningEffort: (...args: unknown[]) => setCurrentReasoningEffort(...args)
}))

/** The helper's request signature is `<T>(method, params?) => Promise<T>`. To
 *  avoid generic plumbing in tests, build a function that can be success or
 *  failure driven from outside, cast to the expected shape once. */
type RequestFn = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

function asRequest(handler: (method: string, params?: Record<string, unknown>) => Promise<unknown>): RequestFn {
  return handler as unknown as RequestFn
}

describe('applyReasoningPatch', () => {
  beforeEach(() => {
    setModelPreset.mockReset()
    notifyError.mockReset()
    setCurrentReasoningEffort.mockReset()
  })

  it('writes the preset and the active-session atom on the happy path', async () => {
    let generation = 0
    const latestGeneration = () => generation

    await applyReasoningPatch({
      failMessage: 'reasoning failed',
      generation: ++generation,
      isActive: true,
      latestGeneration,
      model: 'opus',
      next: 'high',
      prev: 'medium',
      provider: 'anthropic',
      request: asRequest(async () => undefined),
      sessionId: 'session-1'
    })

    expect(setModelPreset).toHaveBeenCalledWith('anthropic', 'opus', { effort: 'high' })
    expect(setCurrentReasoningEffort).toHaveBeenCalledWith('high')
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('reverts the preset and atom when the RPC fails and this is still the latest call', async () => {
    let generation = 0
    const latestGeneration = () => generation

    await applyReasoningPatch({
      failMessage: 'reasoning failed',
      generation: ++generation,
      isActive: true,
      latestGeneration,
      model: 'opus',
      next: 'high',
      prev: 'medium',
      provider: 'anthropic',
      request: asRequest(async () => {
        throw new Error('gateway 503')
      }),
      sessionId: 'session-1'
    })

    expect(setModelPreset).toHaveBeenNthCalledWith(1, 'anthropic', 'opus', { effort: 'high' })
    expect(setModelPreset).toHaveBeenNthCalledWith(2, 'anthropic', 'opus', { effort: 'medium' })
    expect(setCurrentReasoningEffort).toHaveBeenNthCalledWith(1, 'high')
    expect(setCurrentReasoningEffort).toHaveBeenNthCalledWith(2, 'medium')
    expect(notifyError).toHaveBeenCalledWith(expect.any(Error), 'reasoning failed')
  })

  it('skips the revert (A→B race) when a newer call has bumped generation before A fails', async () => {
    // Callers A and B each capture their own generation snapshot. Inside this
    // helper the `latestGeneration` accessor is the caller's ref at the time
    // of the check — bumping it (simulating a newer click) must suppress A's
    // revert so B's optimistic write stands.
    let generation = 0
    const latestGeneration = () => generation

    // First call (A) hangs and resolves later with an error.
    let rejectA!: (err: Error) => void
    const aPromise = new Promise<unknown>((_, reject) => {
      rejectA = reject
    })
    const handlerA = vi.fn(async () => aPromise)

    const callA = applyReasoningPatch({
      failMessage: 'A failed',
      generation: ++generation,
      isActive: true,
      latestGeneration,
      model: 'opus',
      next: 'high',
      prev: 'medium',
      provider: 'anthropic',
      request: asRequest(handlerA),
      sessionId: 'session-1'
    })

    // Second call (B) succeeds and bumps the generation ref to simulate a
    // newer user click.
    const handlerB = vi.fn(async () => undefined)
    const callB = applyReasoningPatch({
      failMessage: 'B failed',
      generation: ++generation,
      isActive: true,
      latestGeneration,
      model: 'opus',
      next: 'max',
      prev: 'high',
      provider: 'anthropic',
      request: asRequest(handlerB),
      sessionId: 'session-1'
    })
    await callB

    // Now A's RPC fails. The catch path must check the generation and skip
    // both the preset revert and the atom revert — B is still the user's
    // intent.
    //
    // Microtask ordering (deterministic): callA's RPC resolves A's `await`
    // only after rejectA() runs. Between `await callB` resolving and the
    // body calling rejectA(), A's await is still pending (its request
    // promise hasn't settled). Once rejectA() fires:
    //   1. A's `await` re-enters its catch block.
    //   2. The catch checks latestGeneration() === 2 vs generation === 1.
    //   3. Mismatch → early return before revert/notify.
    rejectA(new Error('A: gateway 503'))
    await callA

    // The presets: B's optimistic write happens after A's. The critical
    // assertion is the LAST preset call — it must be B's `max`, not A's
    // revert-to-`medium`.
    const presetCalls = setModelPreset.mock.calls
    expect(presetCalls).toContainEqual(['anthropic', 'opus', { effort: 'high' }])
    expect(presetCalls).toContainEqual(['anthropic', 'opus', { effort: 'max' }])
    expect(presetCalls).not.toContainEqual(['anthropic', 'opus', { effort: 'medium' }])

    // Atoms: same — last atom write is B's `max`, not A's revert to `medium`.
    const atomCalls = setCurrentReasoningEffort.mock.calls
    expect(atomCalls).toContainEqual(['high'])
    expect(atomCalls).toContainEqual(['max'])
    expect(atomCalls).not.toContainEqual(['medium'])

    // Only B's failure (which never happened) would have notified — A's
    // suppressed revert means A's notifyError must NOT fire. (B succeeded.)
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('skips the optimistic write entirely when model or provider is empty', async () => {
    let generation = 0
    const latestGeneration = () => generation

    // Empty model — the preset store would otherwise get a `provider::` key.
    await applyReasoningPatch({
      failMessage: 'should not fire',
      generation: ++generation,
      isActive: true,
      latestGeneration,
      model: '',
      next: 'high',
      prev: 'medium',
      provider: 'anthropic',
      request: asRequest(async () => undefined),
      sessionId: 'session-1'
    })

    // Empty provider — symmetric guard.
    await applyReasoningPatch({
      failMessage: 'should not fire',
      generation: ++generation,
      isActive: true,
      latestGeneration,
      model: 'opus',
      next: 'high',
      prev: 'medium',
      provider: '  ',
      request: asRequest(async () => undefined),
      sessionId: 'session-1'
    })

    expect(setModelPreset).not.toHaveBeenCalled()
    expect(setCurrentReasoningEffort).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('writes only the preset (not the live session atom) for inactive rows', async () => {
    let generation = 0
    const latestGeneration = () => generation

    await applyReasoningPatch({
      failMessage: 'should not fire',
      generation: ++generation,
      isActive: false,
      latestGeneration,
      model: 'opus',
      next: 'high',
      prev: 'medium',
      provider: 'anthropic',
      request: asRequest(async () => {
        throw new Error('would be ignored')
      }),
      sessionId: 'session-1'
    })

    expect(setModelPreset).toHaveBeenCalledWith('anthropic', 'opus', { effort: 'high' })
    expect(setCurrentReasoningEffort).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('writes only the preset and atom when there is no live session', async () => {
    let generation = 0
    const latestGeneration = () => generation

    await applyReasoningPatch({
      failMessage: 'should not fire',
      generation: ++generation,
      isActive: true,
      latestGeneration,
      model: 'opus',
      next: 'high',
      prev: 'medium',
      provider: 'anthropic',
      request: null,
      sessionId: null
    })

    expect(setModelPreset).toHaveBeenCalledWith('anthropic', 'opus', { effort: 'high' })
    expect(setCurrentReasoningEffort).toHaveBeenCalledWith('high')
    // No request, so no failure, so no notify.
    expect(notifyError).not.toHaveBeenCalled()
  })
})
