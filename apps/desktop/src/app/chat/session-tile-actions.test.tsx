import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { type SessionTileDelegate, setSessionTileDelegate } from '@/store/session-states'
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { MAIN_COMPOSER_SCOPE } from './composer/scope'
import { useSessionTileActions } from './session-tile-actions'

const { requestGateway } = vi.hoisted(() => ({ requestGateway: vi.fn() }))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({ t: { desktop: new Proxy({}, { get: (_target, key) => String(key) }) } })
}))

const SID = 'tile-runtime'

function installDelegate() {
  let state = createClientSessionState()
  const noop = vi.fn(async () => undefined)

  setSessionTileDelegate({
    archiveSession: noop,
    branchSession: noop,
    deleteSession: noop,
    executeSlash: noop,
    interruptSession: noop,
    resumeTile: vi.fn(async () => SID),
    submitToSession: noop,
    updateSession: (_runtimeId, updater) => (state = updater(state))
  } satisfies SessionTileDelegate)
}

describe('useSessionTileActions Stop', () => {
  beforeEach(() => {
    $subagentsBySession.set({})
    installDelegate()
    requestGateway.mockReset().mockResolvedValue({})
  })

  it('terminalizes live rows without deleting a late-completion target', async () => {
    upsertSubagent(SID, { goal: 'child', status: 'running', subagent_id: 'child', task_index: 0 })
    const { result } = renderHook(() =>
      useSessionTileActions({ requestGateway, runtimeId: SID, scope: MAIN_COMPOSER_SCOPE, storedSessionId: 'tile-stored' })
    )

    await act(async () => result.current.cancelRun())

    expect($subagentsBySession.get()[SID]?.[0]?.status).toBe('interrupted')

    upsertSubagent(
      SID,
      { status: 'completed', subagent_id: 'child', summary: 'finished after tile stop', task_index: 0 },
      false,
      'subagent.complete'
    )

    expect($subagentsBySession.get()[SID]?.[0]).toMatchObject({
      status: 'completed',
      summary: 'finished after tile stop'
    })
    expect(requestGateway).toHaveBeenCalledWith('session.interrupt', { session_id: SID })
  })
})
