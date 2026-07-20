import { act, cleanup, render, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { createComposerAttachmentScope } from '@/store/composer'
import {
  clearAllSessionStates,
  publishSessionState,
  type SessionTileDelegate,
  setSessionTileDelegate
} from '@/store/session-states'
import { $todoHistoryBySession, clearAllSessionTodoState, rebuildSessionTodoHistory } from '@/store/todos'

import type { ComposerScope } from './composer/scope'
import { useSessionTileActions } from './session-tile-actions'

const requestGateway = vi.hoisted(() => vi.fn())

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      desktop: {
        editFailed: 'edit failed',
        regenerateFailed: 'reload failed',
        stopFailed: 'stop failed'
      }
    }
  })
}))

const RUNTIME_ID = 'runtime-tile'
const STORED_ID = 'stored-tile'

const todoAssistant = (id: string, content: string) => ({
  id,
  parts: [
    {
      args: { todos: [{ content, id, status: 'completed' as const }] },
      toolCallId: `call-${id}`,
      toolName: 'todo',
      type: 'tool-call' as const
    }
  ],
  role: 'assistant' as const
})

const originalMessages = [
  { id: 'u1', parts: [textPart('first prompt')], role: 'user' as const },
  todoAssistant('a1', 'first task'),
  { id: 'u2', parts: [textPart('second prompt')], role: 'user' as const },
  todoAssistant('a2', 'tail task')
]

interface Handle {
  editMessage: ReturnType<typeof useSessionTileActions>['editMessage']
  reloadFromMessage: ReturnType<typeof useSessionTileActions>['reloadFromMessage']
  restoreToMessage: ReturnType<typeof useSessionTileActions>['restoreToMessage']
}

function Harness({ onReady }: { onReady: (handle: Handle) => void }) {
  const scope: ComposerScope = {
    $awaitingInput: atom(false),
    attachments: createComposerAttachmentScope(),
    popoutAllowed: false,
    readMessages: () => originalMessages,
    target: `tile:${STORED_ID}`
  }

  const actions = useSessionTileActions({ runtimeId: RUNTIME_ID, scope, storedSessionId: STORED_ID })

  useEffect(() => {
    onReady(actions)
  }, [actions, onReady])

  return null
}

function installDelegate(stateRef: { current: ClientSessionState }) {
  const delegate: SessionTileDelegate = {
    archiveSession: async () => undefined,
    branchSession: async () => undefined,
    deleteSession: async () => undefined,
    executeSlash: async () => undefined,
    interruptSession: async () => undefined,
    resumeTile: async () => RUNTIME_ID,
    submitToSession: async () => undefined,
    updateSession: (_runtimeId, updater) => {
      stateRef.current = updater(stateRef.current)
      publishSessionState(RUNTIME_ID, stateRef.current)

      return stateRef.current
    }
  }

  setSessionTileDelegate(delegate)
}

describe('useSessionTileActions task-history rollback', () => {
  let stateRef: { current: ClientSessionState }
  let handle: Handle | null

  beforeEach(async () => {
    handle = null
    requestGateway.mockReset()
    requestGateway.mockRejectedValue(new Error('gateway rejected rewind'))
    stateRef = {
      current: {
        ...createClientSessionState(STORED_ID),
        messages: originalMessages
      }
    }
    publishSessionState(RUNTIME_ID, stateRef.current)
    rebuildSessionTodoHistory(RUNTIME_ID, originalMessages)
    installDelegate(stateRef)
    render(<Harness onReady={value => (handle = value)} />)
    await waitFor(() => expect(handle).not.toBeNull())
  })

  afterEach(() => {
    cleanup()
    clearAllSessionStates()
    clearAllSessionTodoState()
    vi.restoreAllMocks()
  })

  const expectOriginalHistory = () => {
    expect(stateRef.current.messages).toEqual(originalMessages)
    expect($todoHistoryBySession.get()[RUNTIME_ID]?.map(snapshot => snapshot.id)).toEqual(['a2', 'a1'])
  }

  it('restores both transcript and task history when tile reload fails', async () => {
    await act(async () => handle!.reloadFromMessage('a1'))

    expectOriginalHistory()
  })

  it('restores both transcript and task history when tile restore fails', async () => {
    await expect(act(async () => handle!.restoreToMessage('u1'))).rejects.toThrow('gateway rejected rewind')

    expectOriginalHistory()
  })

  it('restores both transcript and task history when tile edit fails', async () => {
    await act(async () =>
      handle!.editMessage({
        attachments: [],
        content: [{ text: 'edited first prompt', type: 'text' }],
        createdAt: new Date(0),
        metadata: { custom: {} },
        parentId: null,
        role: 'user',
        runConfig: undefined,
        sourceId: 'u1'
      })
    )

    expectOriginalHistory()
  })
})
