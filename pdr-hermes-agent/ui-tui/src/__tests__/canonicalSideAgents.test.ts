import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

// The shared (canonical) gateway serves prompt.background / prompt.btw: /bg and /btw take main's
// path there too instead of refusing with canonical.controls.notAvailable.

const flush = () => new Promise(resolve => setImmediate(resolve))

function harness() {
  const request = vi.fn(async (method: string, params: any) => {
    if (method === 'prompt.background') {
      return { task_id: 'bg_abc123' }
    }

    if (method === 'prompt.btw') {
      return { task_id: 'btw_def456' }
    }

    throw new Error(`unknown_method: ${method} ${JSON.stringify(params)}`)
  })

  const sys = vi.fn()

  const ctx = {
    slashFlightRef: { current: 0 },
    gateway: { gw: { isCanonical: true, request }, rpc: request },
    transcript: { page: vi.fn(), panel: vi.fn(), send: vi.fn(), sys, setHistoryItems: vi.fn(), trimLastExchange: (items: unknown[]) => items },
    local: { getHistoryItems: () => [], getLastUserMsg: () => '', maybeWarn: vi.fn() },
    session: { resumeById: vi.fn() },
    composer: { enqueue: vi.fn() }
  }

  return { request, slash: createSlashHandler(ctx as any), sys }
}

beforeEach(() => {
  resetUiState()
  patchUiState({ sid: 'owner', info: { model: 'm', skills: {}, tools: {}, execution_epoch: '1', execution_generation: 3 } as any })
})

it('/bg starts prompt.background on the shared gateway and tracks the task', async () => {
  const { request, slash, sys } = harness()

  slash('/bg summarize the repo')
  await flush()

  expect(request).toHaveBeenCalledWith('prompt.background', { session_id: 'owner', text: 'summarize the repo' })
  expect(sys).toHaveBeenCalledWith('bg bg_abc123 started')
  expect(sys.mock.calls.flat().join('\n')).not.toMatch(/not available/i)
  expect(getUiState().bgTasks.has('bg_abc123')).toBe(true)
})

it('/btw asks prompt.btw on the shared gateway and announces the task', async () => {
  const { request, slash, sys } = harness()

  slash('/btw what was that error')
  await flush()

  expect(request).toHaveBeenCalledWith('prompt.btw', { session_id: 'owner', text: 'what was that error' })
  expect(sys).toHaveBeenCalledWith('btw btw_def456 — answering from a conversation snapshot')
  expect(sys.mock.calls.flat().join('\n')).not.toMatch(/not available/i)
})
