import type { AppendMessage } from '@assistant-ui/react'
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { clearAllSessionStates, publishSessionState, setSessionTileDelegate } from '@/store/session-states'

import { MAIN_COMPOSER_SCOPE } from './composer/scope'
import { useSessionTileActions } from './session-tile-actions'

const requestGateway = vi.hoisted(() => vi.fn(async () => ({}) as never))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

vi.mock('@/hermes', () => ({
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn()
}))

const RUNTIME_ID = 'tile-runtime'
const STORED_ID = 'tile-stored'

const messages = [
  { id: 'u1', role: 'user' as const, parts: [textPart('first prompt')] },
  { id: 'a1', role: 'assistant' as const, parts: [textPart('first answer')] },
  { id: 'u2', role: 'user' as const, parts: [textPart('second prompt')] },
  { id: 'a2', role: 'assistant' as const, parts: [textPart('second answer')] }
]

function renderActions() {
  let state = createClientSessionState(STORED_ID, messages)
  publishSessionState(RUNTIME_ID, state)

  setSessionTileDelegate({
    archiveSession: async () => undefined,
    branchSession: async () => undefined,
    deleteSession: async () => undefined,
    executeSlash: async () => undefined,
    interruptSession: async () => undefined,
    resumeTile: async () => RUNTIME_ID,
    submitToSession: async () => undefined,
    updateSession: (_runtimeId, updater) => {
      state = updater(state)
      publishSessionState(RUNTIME_ID, state)

      return state
    }
  })

  return renderHook(() =>
    useSessionTileActions({ runtimeId: RUNTIME_ID, scope: MAIN_COMPOSER_SCOPE, storedSessionId: STORED_ID })
  )
}

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  requestGateway.mockClear()
})

describe('session tile durable prompt binding', () => {
  it('binds stop to the tile stored session id', async () => {
    const { result } = renderActions()

    await act(async () => result.current.cancelRun())

    expect(requestGateway).toHaveBeenCalledWith('session.interrupt', {
      expected_stored_session_id: STORED_ID,
      session_id: RUNTIME_ID
    })
  })

  it('binds reload to the tile stored session id', async () => {
    const { result } = renderActions()

    await act(async () => result.current.reloadFromMessage(null))

    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        expected_stored_session_id: STORED_ID,
        session_id: RUNTIME_ID,
        text: 'second prompt',
        truncate_before_user_ordinal: 1
      },
      1_800_000
    )
  })

  it('binds restore rewind to the tile stored session id', async () => {
    const { result } = renderActions()

    await act(async () => result.current.restoreToMessage('u1'))

    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        expected_stored_session_id: STORED_ID,
        session_id: RUNTIME_ID,
        text: 'first prompt',
        truncate_before_user_ordinal: 0
      },
      1_800_000
    )
  })

  it('binds edit rewind to the tile stored session id', async () => {
    const { result } = renderActions()

    await act(async () =>
      result.current.editMessage({
        parentId: null,
        role: 'user',
        sourceId: 'u2',
        content: [{ type: 'text', text: 'edited second prompt' }]
      } as unknown as AppendMessage)
    )

    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        expected_stored_session_id: STORED_ID,
        session_id: RUNTIME_ID,
        text: 'edited second prompt',
        truncate_before_user_ordinal: 1
      },
      1_800_000
    )
  })
})
