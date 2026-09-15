import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'

import { refreshBackgroundProcesses, resetBackgroundPollingGuard } from './composer-status'
import { $gateway } from './gateway'
import { replayPendingApproval } from './prompts'
import { clearSingleFlightSessionResumeState } from '@/app/session/hooks/use-prompt-actions/single-flight-resume'
import { resetRuntimeGoneHealing } from './runtime-gone'
import {
  beginScopedRpcEpoch,
  ensureLiveSessionIdForScopedRpc,
  idsFromSessionReclaimed,
  resetScopedRpcResume
} from './scoped-rpc-resume'
import { $activeSessionId, $sessionResumeRequest } from './session'
import { $sessionStates, $sessionTiles } from './session-states'

const STORED = '20260814_094231_4e849d'
const RUNTIME = 'a9b07c82'
const LIVE = 'live-after-resume'

const cachedState = (storedSessionId: string) => createClientSessionState(storedSessionId)

beforeEach(() => {
  resetScopedRpcResume()
  clearSingleFlightSessionResumeState()
  resetRuntimeGoneHealing()
  resetBackgroundPollingGuard()
  $sessionStates.set({})
  $sessionTiles.set([])
  $activeSessionId.set(null)
  $sessionResumeRequest.set(null)
  $gateway.set(null as never)
})

afterEach(() => {
  resetScopedRpcResume()
  clearSingleFlightSessionResumeState()
  resetRuntimeGoneHealing()
  resetBackgroundPollingGuard()
  $sessionStates.set({})
  $sessionTiles.set([])
  $activeSessionId.set(null)
  $sessionResumeRequest.set(null)
  $gateway.set(null as never)
})

describe('idsFromSessionReclaimed', () => {
  it('reads payload ids when the envelope session_id is empty', () => {
    expect(
      idsFromSessionReclaimed({ reason: 'ws_orphan_reap', session_id: RUNTIME, stored_session_id: STORED }, '')
    ).toEqual({ runtimeId: RUNTIME, storedSessionId: STORED })
  })

  it('falls back to the cached stored id when payload omits it', () => {
    $sessionStates.set({ [RUNTIME]: cachedState(STORED) })

    expect(idsFromSessionReclaimed({ session_id: RUNTIME }, '')).toEqual({
      runtimeId: RUNTIME,
      storedSessionId: STORED
    })
  })
})

describe('ensureLiveSessionIdForScopedRpc', () => {
  it('resumes the stored row after gateway.ready before returning a live id', async () => {
    $sessionStates.set({ [RUNTIME]: cachedState(STORED) })
    beginScopedRpcEpoch()

    const methods: string[] = []
    const gateway = {
      request: vi.fn(async (method: string, params?: Record<string, unknown>) => {
        methods.push(method)

        if (method === 'session.resume') {
          expect(params?.session_id).toBe(STORED)

          return { session_id: LIVE }
        }

        throw new Error(`unexpected ${method}`)
      })
    }

    await expect(ensureLiveSessionIdForScopedRpc(gateway, RUNTIME)).resolves.toBe(LIVE)
    expect(methods).toEqual(['session.resume'])

    await expect(ensureLiveSessionIdForScopedRpc(gateway, RUNTIME)).resolves.toBe(LIVE)
    expect(methods).toEqual(['session.resume'])
  })

  it('fails closed after one resume miss instead of retrying', async () => {
    beginScopedRpcEpoch()
    const gateway = {
      request: vi.fn(async () => {
        throw new Error('session not found')
      })
    }

    await expect(ensureLiveSessionIdForScopedRpc(gateway, STORED)).resolves.toBeNull()
    await expect(ensureLiveSessionIdForScopedRpc(gateway, STORED)).resolves.toBeNull()
    expect(gateway.request).toHaveBeenCalledTimes(1)
  })
})

describe('restart → resume before scoped poll', () => {
  it('calls session.resume before approval.pending', async () => {
    $sessionStates.set({ [RUNTIME]: cachedState(STORED) })
    beginScopedRpcEpoch()

    const methods: string[] = []
    const gateway = {
      request: async (method: string, params?: Record<string, unknown>) => {
        methods.push(`${method}:${String(params?.session_id ?? '')}`)

        if (method === 'session.resume') {
          return { session_id: LIVE }
        }

        if (method === 'approval.pending') {
          return { approvals: [] }
        }

        throw new Error(method)
      }
    }

    await replayPendingApproval(gateway, RUNTIME)

    expect(methods).toEqual([`session.resume:${STORED}`, `approval.pending:${LIVE}`])
  })

  it('calls session.resume before process.list', async () => {
    $sessionStates.set({ [RUNTIME]: cachedState(STORED) })
    beginScopedRpcEpoch()

    const methods: string[] = []
    $gateway.set({
      request: async (method: string, params?: Record<string, unknown>) => {
        methods.push(`${method}:${String(params?.session_id ?? '')}`)

        if (method === 'session.resume') {
          return { session_id: LIVE }
        }

        if (method === 'process.list') {
          return { processes: [] }
        }

        throw new Error(method)
      }
    } as never)

    await refreshBackgroundProcesses(RUNTIME)

    expect(methods).toEqual([`session.resume:${STORED}`, `process.list:${LIVE}`])
  })

  it('does not keep polling approval.pending when resume never succeeds', async () => {
    beginScopedRpcEpoch()
    const methods: string[] = []
    const gateway = {
      request: async (method: string) => {
        methods.push(method)
        throw new Error('session not found')
      }
    }

    for (let i = 0; i < 8; i += 1) {
      await replayPendingApproval(gateway, STORED)
    }

    expect(methods).toEqual(['session.resume'])
  })
})
