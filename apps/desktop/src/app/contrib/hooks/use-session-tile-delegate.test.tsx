import { cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { sessionTileDelegate } from '@/store/session-states'

import { useSessionTileDelegate } from './use-session-tile-delegate'

vi.mock('@/hermes', () => ({
  getSessionMessages: vi.fn(async () => ({ messages: [] })),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn()
}))

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('useSessionTileDelegate prompt binding', () => {
  it('sends the caller-provided durable tile id on direct delegate submits', async () => {
    const requestGateway = vi.fn(async () => ({}) as never)

    renderHook(() =>
      useSessionTileDelegate({
        archiveSession: async () => undefined,
        branchStoredSession: async () => undefined,
        executeSlashCommand: async () => undefined,
        removeSession: async () => undefined,
        requestGateway,
        runtimeIdByStoredSessionIdRef: { current: new Map() },
        sessionStateByRuntimeIdRef: { current: new Map() },
        updateSessionState: vi.fn()
      } as unknown as Parameters<typeof useSessionTileDelegate>[0])
    )

    const submit = sessionTileDelegate()!.submitToSession as unknown as (
      runtimeId: string,
      storedSessionId: string,
      text: string
    ) => Promise<void>

    await submit('tile-runtime', 'tile-stored', 'delegate prompt')

    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        expected_stored_session_id: 'tile-stored',
        session_id: 'tile-runtime',
        text: 'delegate prompt'
      },
      1_800_000
    )

    await sessionTileDelegate()!.interruptSession('tile-runtime', 'tile-stored')

    expect(requestGateway).toHaveBeenCalledWith('session.interrupt', {
      expected_stored_session_id: 'tile-stored',
      session_id: 'tile-runtime'
    })
  })
})
