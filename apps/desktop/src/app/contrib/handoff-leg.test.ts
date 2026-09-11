import { describe, expect, it, vi } from 'vitest'

import { type HandoffDeps, type HandoffReceipt, startHandoff } from './handoff-leg'

function harness() {
  let receipt: HandoffReceipt | null = null

  const request = vi.fn(async (_owner: HandoffReceipt['owner'], method: string): Promise<Record<string, unknown>> => {
    if (method === 'session.resume') {
      return { session_id: 'runtime-2', session_key: 'stored', running: false, messages: [] }
    }

    if (method === 'prompt.submit') {
      throw Object.assign(new Error('Provider unavailable'), { code: 4090 })
    }

    return {}
  })

  const deps: HandoffDeps = {
    create: vi.fn(async () => ({
      runtimeId: 'runtime-1',
      storedId: 'stored',
      owner: { connectionId: 'source-a', profile: 'default' as const }
    })),
    personalize: vi.fn(async () => undefined),
    request: request as HandoffDeps['request'],
    read: () => receipt,
    save: value => {
      receipt = structuredClone(value)
    },
    bind: vi.fn()
  }

  return { deps, request, receipt: () => receipt }
}

const task = { brief: 'Build my tracker', task: 'Tracker', plan: 'build' as const }

describe('first-build handoff', () => {
  it.each([5070, 5071, 5072])(
    'retries storage refusal %i after repair without creating a second session',
    async code => {
      const h = harness()
      h.request.mockRejectedValueOnce(Object.assign(new Error('Storage unavailable'), { code }))
      await expect(startHandoff(h.deps, task)).rejects.toThrow('Storage unavailable')
      expect(h.receipt()).toMatchObject({ storedId: 'stored', status: 'created' })

      h.request.mockImplementation(async (_owner, method) =>
        method === 'session.resume'
          ? { session_id: 'runtime-1', session_key: 'stored', running: false, messages: [] }
          : { status: 'streaming' }
      )
      await expect(startHandoff(h.deps, task)).resolves.toMatchObject({ status: 'accepted', storedId: 'stored' })
      expect(h.deps.create).toHaveBeenCalledTimes(1)
      expect(h.request.mock.calls.filter(([, method]) => method === 'prompt.submit')).toHaveLength(2)
    }
  )

  it.each([4090, 5070, 5072])(
    'does not accept refusal %i from stale running alone or repeat it while busy',
    async code => {
      const h = harness()
      h.request.mockRejectedValueOnce(Object.assign(new Error('Preflight refused'), { code }))
      await expect(startHandoff(h.deps, task)).rejects.toThrow('Preflight refused')
      h.request.mockImplementation(async () => ({
        session_id: 'runtime-1',
        session_key: 'stored',
        running: true,
        messages: [{ role: 'user', text: 'hidden setup seed', display_kind: 'hidden' }]
      }))
      await expect(startHandoff(h.deps, task)).rejects.toThrow('no duplicate was sent')
      expect(h.receipt()?.status).toBe('created')
      expect(h.request.mock.calls.filter(([, method]) => method === 'prompt.submit')).toHaveLength(1)

      h.request.mockImplementation(async (_owner, method) =>
        method === 'session.resume'
          ? { session_id: 'runtime-1', session_key: 'stored', running: false, messages: [] }
          : { status: 'streaming' }
      )
      await expect(startHandoff(h.deps, task)).resolves.toMatchObject({ status: 'accepted' })
      expect(h.deps.create).toHaveBeenCalledTimes(1)
      expect(h.request.mock.calls.filter(([, method]) => method === 'prompt.submit')).toHaveLength(2)
    }
  )

  it('keeps a rejected submit retryable on the same stored session, never reports acceptance early', async () => {
    const h = harness()
    await expect(startHandoff(h.deps, task)).rejects.toThrow('Provider unavailable')
    expect(h.receipt()).toMatchObject({ storedId: 'stored', status: 'created' })
    expect(h.deps.personalize).toHaveBeenCalledBefore(vi.mocked(h.deps.create))

    h.request.mockImplementation(async (_profile, method) =>
      method === 'session.resume'
        ? { session_id: 'runtime-2', session_key: 'stored', running: false, messages: [] }
        : { status: 'streaming' }
    )
    await expect(startHandoff(h.deps, task)).resolves.toMatchObject({
      storedId: 'stored',
      runtimeId: 'runtime-2',
      status: 'accepted'
    })
    expect(h.deps.create).toHaveBeenCalledTimes(1)
    expect(h.request.mock.calls.filter(([, method]) => method === 'prompt.submit')).toHaveLength(2)
    expect(h.request.mock.calls.some(([, method]) => method === 'session.close' || method === 'session.list')).toBe(
      false
    )

    // Internal errors can arrive after side effects too; only documented
    // preflight refusals authorize a repeat of the go signal.
    h.deps.save({ ...h.receipt()!, status: 'created' })
    // A new attempt after a known refusal must first finish the resume read.
    h.request.mockImplementation(async (_owner, method) => {
      if (method === 'session.resume') {
        return { session_id: 'runtime-2', session_key: 'stored', running: false, messages: [] }
      }

      throw Object.assign(new Error('Internal error'), { code: -32603 })
    })
    await expect(startHandoff(h.deps, task)).rejects.toThrow('Internal error')
    expect(h.receipt()?.status).toBe('submitting')
  })

  it('recovers a reaped runtime once and reconciles a lost ACK without resubmitting or minting a duplicate', async () => {
    const h = harness()
    h.request.mockImplementation(async (_profile, method) => {
      if (method === 'session.resume') {
        return { session_id: 'runtime-2', session_key: 'stored', running: false, messages: [] }
      }

      if (method === 'prompt.submit') {
        const submits = h.request.mock.calls.filter(([, name]) => name === 'prompt.submit').length

        if (submits === 1) {
          throw Object.assign(new Error('session not found'), { code: 4001 })
        }

        throw new Error('connection closed after send')
      }

      return {}
    })
    await expect(startHandoff(h.deps, task)).rejects.toThrow('connection closed after send')
    expect(h.receipt()).toMatchObject({ runtimeId: 'runtime-2', storedId: 'stored', status: 'submitting' })
    const submits = h.request.mock.calls.filter(([, method]) => method === 'prompt.submit').length

    // Partial hydration cannot establish whether the server already accepted.
    h.request.mockImplementation(async () => ({
      session_id: 'runtime-2',
      session_key: 'stored',
      running: false,
      messages: [],
      hydrating: true
    }))
    await expect(startHandoff(h.deps, task)).rejects.toThrow('Could not verify')
    h.request.mockImplementation(async () => ({
      session_id: 'runtime-2',
      session_key: 'stored',
      running: false,
      messages: []
    }))
    // Even an idle/empty read cannot disprove an in-flight request with a lost ACK.
    await expect(startHandoff(h.deps, task)).rejects.toThrow('no duplicate was sent')
    h.request.mockImplementation(async () => ({
      session_id: 'runtime-3',
      session_key: 'stored',
      running: true,
      messages: []
    }))
    await expect(startHandoff(h.deps, task)).resolves.toMatchObject({ runtimeId: 'runtime-3', status: 'accepted' })
    expect(h.deps.create).toHaveBeenCalledTimes(1)
    expect(h.request.mock.calls.filter(([, method]) => method === 'prompt.submit')).toHaveLength(submits)
  })
})
