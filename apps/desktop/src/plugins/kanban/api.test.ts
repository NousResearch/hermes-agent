import type { PluginStorage } from '@hermes/plugin-sdk'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $boardSlug, bindApi, decomposeTask, specifyTask } from './api'

const rest = vi.fn()
const storageGet = vi.fn()
const storageRemove = vi.fn()
const storageSet = vi.fn()
const storage: PluginStorage = {
  get<T>(key: string, fallback: T): T {
    storageGet(key, fallback)

    return fallback
  },
  remove: (key: string) => storageRemove(key),
  set: (key: string, value: unknown) => storageSet(key, value)
}
const socket = vi.fn(() => vi.fn())
let dispose: () => void

beforeEach(() => {
  vi.useFakeTimers()
  rest.mockReset()
  storageGet.mockClear()
  storageRemove.mockClear()
  storageSet.mockClear()
  socket.mockClear()
  dispose = bindApi(rest, storage, socket)
  $boardSlug.set('product')
})

afterEach(() => {
  dispose()
  vi.useRealTimers()
})

describe('triage actions', () => {
  it('posts Specify through the selected board route', async () => {
    const outcome = { ok: true, task_id: 't_1', reason: null, new_title: 'Specified task' }
    rest.mockResolvedValue(outcome)

    await expect(specifyTask('t_1')).resolves.toBe(outcome)
    expect(rest).toHaveBeenCalledWith('/tasks/t_1/specify?board=product', { method: 'POST', body: {} })
  })

  it('posts Decompose through the selected board route', async () => {
    const outcome = {
      ok: true,
      task_id: 't_1',
      reason: null,
      fanout: true,
      child_ids: ['t_2'],
      new_title: 'Root task'
    }
    rest.mockResolvedValue(outcome)

    await expect(decomposeTask('t_1')).resolves.toBe(outcome)
    expect(rest).toHaveBeenCalledWith('/tasks/t_1/decompose?board=product', { method: 'POST', body: {} })
  })

  it('wakes the dispatcher after triage actions change runnable work', async () => {
    rest.mockResolvedValue({ ok: true, task_id: 't_1' })

    await specifyTask('t_1')
    await decomposeTask('t_1')
    await vi.advanceTimersByTimeAsync(400)

    expect(rest).toHaveBeenLastCalledWith('/dispatch?board=product', { method: 'POST', body: {} })
  })
})
