import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $notifications } from '@/store/notifications'

import { $installsRemoval, removeInstall } from './installs-removal'

describe('removeInstall', () => {
  beforeEach(() => {
    $installsRemoval.set({ removingId: null, finished: 0, outcome: null })
    $notifications.set([])
  })

  it('keeps the result and toasts it even when nothing is subscribed to the page', async () => {
    let finish: (result: { ok: boolean; message?: string }) => void = () => undefined
    const pending = removeInstall(() => new Promise(resolve => (finish = resolve)), 'abc123')

    expect($installsRemoval.get().removingId).toBe('abc123')

    finish({ ok: false, message: 'not removed: it has uncommitted changes' })
    await pending

    const state = $installsRemoval.get()

    expect(state.removingId).toBeNull()
    expect(state.finished).toBe(1)
    expect(state.outcome).toEqual({ id: 'abc123', ok: false, message: 'not removed: it has uncommitted changes' })
    expect($notifications.get().map(n => [n.kind, n.message])).toEqual([
      ['error', 'not removed: it has uncommitted changes']
    ])
  })

  it('runs one removal at a time', async () => {
    const remove = vi.fn(() => new Promise<{ ok: boolean }>(() => undefined))

    void removeInstall(remove, 'one')
    await removeInstall(remove, 'two')

    expect(remove).toHaveBeenCalledTimes(1)
    expect($installsRemoval.get().removingId).toBe('one')
  })
})
