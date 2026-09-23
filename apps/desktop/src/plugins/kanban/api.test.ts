import { type PluginRestOptions, type PluginStorage, queryClient } from '@hermes/plugin-sdk'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { bindApi } from './api'

const BOARD = { assignees: [], columns: [], latest_event_id: 14_386, now: 0, tenants: [] }

afterEach(() => {
  queryClient.clear()
  vi.clearAllMocks()
})

describe('kanban event stream', () => {
  it('starts after the board snapshot instead of replaying event history', async () => {
    let resolveBoard: (board: typeof BOARD) => void = () => undefined

    const board = new Promise<typeof BOARD>(resolve => {
      resolveBoard = resolve
    })

    const rest = async <T>(path: string, _opts?: PluginRestOptions): Promise<T> => {
      if (path === '/board') {
        return board as Promise<T>
      }

      throw new Error(`unexpected REST path: ${path}`)
    }

    const storage: PluginStorage = {
      get: (_key, fallback) => fallback,
      remove: vi.fn(),
      set: vi.fn()
    }

    const socket = vi.fn(() => vi.fn())
    const dispose = bindApi(rest, storage, socket)

    expect(socket).not.toHaveBeenCalled()
    resolveBoard(BOARD)

    await vi.waitFor(() => {
      expect(socket).toHaveBeenCalledWith('/events?since=14386', expect.any(Function))
    })

    dispose()
  })
})
