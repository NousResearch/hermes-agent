import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getSessionMessages } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import { sessionTileDelegate } from '@/store/session-states'
import { $todoHistoryBySession, clearAllSessionTodoState, rebuildSessionTodoHistory } from '@/store/todos'

import type { ClientSessionState } from '../../types'

import { useSessionTileDelegate } from './use-session-tile-delegate'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getSessionMessages: vi.fn()
}))

describe('useSessionTileDelegate task-history hydration', () => {
  afterEach(() => {
    cleanup()
    clearAllSessionTodoState()
    vi.restoreAllMocks()
  })

  it('hydrates a cold tile under its runtime id without touching another tile', async () => {
    const runtimeByStored: MutableRefObject<Map<string, string>> = { current: new Map() }
    const states: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return { session_id: 'runtime-tile-a', messages: [], info: {} } as never
      }

      return {} as never
    })

    const updateSessionState = (runtimeId: string, updater: (state: ClientSessionState) => ClientSessionState) => {
      const next = updater(states.current.get(runtimeId) ?? createClientSessionState('stored-tile-a'))
      states.current.set(runtimeId, next)

      return next
    }

    vi.mocked(getSessionMessages).mockResolvedValue({
      session_id: 'stored-tile-a',
      messages: [
        {
          content: '',
          role: 'assistant',
          timestamp: 2,
          tool_calls: [
            {
              id: 'todo-tile',
              function: {
                name: 'todo',
                arguments: JSON.stringify({
                  todos: [{ content: 'Tile A task', id: 'same', status: 'completed' }]
                })
              }
            }
          ]
        }
      ]
    } as never)
    $todoHistoryBySession.set({
      'runtime-tile-b': [
        { id: 'todo-tile', state: 'completed', todos: [{ content: 'Tile B task', id: 'same', status: 'completed' }] }
      ]
    })

    function Harness() {
      useSessionTileDelegate({
        archiveSession: async () => undefined,
        branchStoredSession: async () => undefined,
        executeSlashCommand: async () => undefined,
        removeSession: async () => undefined,
        requestGateway,
        runtimeIdByStoredSessionIdRef: runtimeByStored,
        sessionStateByRuntimeIdRef: states,
        updateSessionState: updateSessionState as never
      })

      return null
    }

    render(<Harness />)
    await waitFor(() => expect(sessionTileDelegate()).not.toBeNull())
    await act(async () => sessionTileDelegate()!.resumeTile('stored-tile-a'))

    expect($todoHistoryBySession.get()['runtime-tile-a']?.[0]?.todos[0]?.content).toBe('Tile A task')
    expect($todoHistoryBySession.get()['runtime-tile-b']?.[0]?.todos[0]?.content).toBe('Tile B task')
  })

  it('rebuilds from newer runtime messages and removes the temporary stored-id history key', async () => {
    const runtimeByStored: MutableRefObject<Map<string, string>> = { current: new Map() }

    const runtimeMessages = [
      {
        id: 'runtime-newer',
        parts: [
          {
            args: { todos: [{ content: 'Runtime task', id: 'runtime', status: 'completed' }] },
            toolCallId: 'runtime-todo',
            toolName: 'todo',
            type: 'tool-call' as const
          }
        ],
        role: 'assistant' as const,
        timestamp: 3
      }
    ]

    const states: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['runtime-tile-a', { ...createClientSessionState(), messages: runtimeMessages }]])
    }

    const requestGateway = vi.fn(async () => ({ session_id: 'runtime-tile-a', messages: [], info: {} }) as never)

    const updateSessionState = (
      runtimeId: string,
      updater: (state: ClientSessionState) => ClientSessionState,
      storedSessionId?: string | null
    ) => {
      const current = states.current.get(runtimeId) ?? createClientSessionState(storedSessionId)
      const next = updater({ ...current, storedSessionId: storedSessionId ?? current.storedSessionId })
      states.current.set(runtimeId, next)

      return next
    }

    vi.mocked(getSessionMessages).mockResolvedValue({
      session_id: 'stored-tile-a',
      messages: [
        {
          content: '',
          role: 'assistant',
          timestamp: 2,
          tool_calls: [
            {
              id: 'prefetch-todo',
              function: {
                name: 'todo',
                arguments: JSON.stringify({
                  todos: [{ content: 'Prefetch task', id: 'prefetch', status: 'completed' }]
                })
              }
            }
          ]
        }
      ]
    } as never)
    rebuildSessionTodoHistory('stored-tile-a', runtimeMessages)

    function Harness() {
      useSessionTileDelegate({
        archiveSession: async () => undefined,
        branchStoredSession: async () => undefined,
        executeSlashCommand: async () => undefined,
        removeSession: async () => undefined,
        requestGateway,
        runtimeIdByStoredSessionIdRef: runtimeByStored,
        sessionStateByRuntimeIdRef: states,
        updateSessionState: updateSessionState as never
      })

      return null
    }

    render(<Harness />)
    await waitFor(() => expect(sessionTileDelegate()).not.toBeNull())
    await act(async () => sessionTileDelegate()!.resumeTile('stored-tile-a'))

    expect($todoHistoryBySession.get()['runtime-tile-a']?.[0]?.id).toBe('runtime-newer:runtime-todo')
    expect($todoHistoryBySession.get()['stored-tile-a']).toBeUndefined()
  })
})
