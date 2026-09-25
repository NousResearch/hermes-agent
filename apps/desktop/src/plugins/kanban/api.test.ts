import { host, type PluginRestOptions, type PluginStorage, queryClient } from '@hermes/plugin-sdk'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $boardSlug, bindApi, boardKey } from './api'

// The events socket reopens on every board switch and reconnect. Opened with no
// cursor it used to replay the board's whole `task_events` history, 200 rows per
// frame, and every frame invalidates the full board (#81537).

type Rest = <T>(path: string, _opts?: PluginRestOptions) => Promise<T>

const noRest: Rest = async path => {
  throw new Error(`unexpected REST path: ${path}`)
}

const storage: PluginStorage = {
  get: (_key, fallback) => fallback,
  remove: vi.fn(),
  set: vi.fn()
}

type OnMessage = (data: unknown) => void

function bind(rest: Rest = noRest) {
  const frames: OnMessage[] = []

  const socket = vi.fn((_path: string, onMessage: OnMessage) => {
    frames.push(onMessage)

    return vi.fn()
  })

  const dispose = bindApi(rest, storage, socket)
  const paths = () => socket.mock.calls.map(([path]) => path)

  return { dispose, frames, paths }
}

afterEach(() => {
  $boardSlug.set('')
  queryClient.clear()
})

describe('kanban event stream cursor', () => {
  it('the first event after a connect notifies: stream and baseline both start at the snapshot', async () => {
    const notify = vi.spyOn(host, 'notify').mockImplementation(() => '')
    let latest = 450

    const rest: Rest = async <T>(path: string) => {
      if (path.startsWith('/board')) {
        return { assignees: [], columns: [], latest_event_id: latest, now: 0, tenants: [] } as T
      }

      throw new Error(`unexpected REST path: ${path}`)
    }

    const { dispose, frames, paths } = bind(rest)

    $boardSlug.set('first-toast')
    await vi.waitFor(() => expect(paths().at(-1)).toBe('/events?board=first-toast&since=450'))

    // A task completes. By the time its frame arrives the board's tail is past it.
    latest = 451
    frames.at(-1)!({ cursor: 451, events: [{ id: 451, kind: 'completed', payload: null, task_id: 't_1' }] })

    await vi.waitFor(() => expect(notify).toHaveBeenCalledTimes(1))
    notify.mockRestore()
    dispose()
  })

  it('starts after the cached board snapshot instead of replaying event history', () => {
    queryClient.setQueryData(boardKey('local', 'ops', false), {
      assignees: [],
      columns: [],
      latest_event_id: 14_386,
      now: 0,
      tenants: []
    })

    const { dispose, paths } = bind()

    $boardSlug.set('ops')

    expect(paths().at(-1)).toBe('/events?board=ops&since=14386')
    dispose()
  })

  it('resumes a board from the last frame it saw this session, not from its snapshot', () => {
    queryClient.setQueryData(boardKey('local', 'ship', false), {
      assignees: [],
      columns: [],
      latest_event_id: 10,
      now: 0,
      tenants: []
    })

    const { dispose, frames, paths } = bind()

    $boardSlug.set('ship')
    frames.at(-1)!({ cursor: 25, events: [{ id: 25, kind: 'spawned', task_id: 't_1' }] })

    // Switch away and back: the socket for `ship` reopens where its stream left off.
    $boardSlug.set('ops')
    $boardSlug.set('ship')

    expect(paths().at(-1)).toBe('/events?board=ship&since=25')
    dispose()
  })
})
