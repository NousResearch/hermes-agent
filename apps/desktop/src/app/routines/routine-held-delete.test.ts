// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $heldRoutine,
  ABANDONED_AFTER_MS,
  commitAbandonedRoutineDeletes,
  commitHeldRoutine,
  type HeldDeleteBackend,
  type HeldRoutine,
  holdRoutineDelete,
  undoHeldRoutine
} from './routine-held-delete'

const KEY = 'hermes.desktop.routines.heldDeletes.v1'

const brief: HeldRoutine = { id: 'brief', name: 'Daily brief', resume: true, scope: { connectionId: null, profile: 'ops' } }
const review: HeldRoutine = { id: 'review', name: 'Weekly review', resume: false, scope: { connectionId: 'box', profile: 'default' } }

/** The owner's routines, by id: what each call did to them. */
function fakeOwner() {
  const state = new Map<string, 'deleted' | 'on' | 'paused'>([
    ['brief', 'on'],
    ['review', 'paused']
  ])

  const calls: string[] = []

  const backend: HeldDeleteBackend = {
    pause: vi.fn(async (id, scope) => {
      calls.push(`pause ${id} @${scope.connectionId ?? 'local'}/${scope.profile}`)
      state.set(id, 'paused')
    }),
    remove: vi.fn(async (id, scope) => {
      calls.push(`delete ${id} @${scope.connectionId ?? 'local'}/${scope.profile}`)

      if (state.get(id) === 'deleted') {
        throw new Error('404: Job not found')
      }

      state.set(id, 'deleted')
    }),
    resume: vi.fn(async (id, scope) => {
      calls.push(`resume ${id} @${scope.connectionId ?? 'local'}/${scope.profile}`)
      state.set(id, 'on')
    })
  }

  return { backend, calls, state }
}

const markers = () => JSON.parse(localStorage.getItem(KEY) ?? '[]') as { id: string }[]

beforeEach(() => {
  localStorage.clear()
  $heldRoutine.set(null)
})

afterEach(() => {
  vi.useRealTimers()
})

describe('holding a routine’s delete', () => {
  it('pauses it on its owner and hides it, without deleting anything yet', async () => {
    const owner = fakeOwner()

    await holdRoutineDelete(brief, owner.backend)

    expect(owner.calls).toEqual(['pause brief @local/ops'])
    expect(owner.state.get('brief')).toBe('paused')
    expect($heldRoutine.get()).toEqual(brief)
    expect(markers().map(marker => marker.id)).toEqual(['brief'])
  })

  it('leaves a routine that was already off alone', async () => {
    const owner = fakeOwner()

    await holdRoutineDelete(review, owner.backend)

    expect(owner.calls).toEqual([])
    expect($heldRoutine.get()).toEqual(review)
  })

  it('undo restores it exactly: resumed only if it was on, and never deleted later', async () => {
    const owner = fakeOwner()

    await holdRoutineDelete(brief, owner.backend)
    expect(await undoHeldRoutine(owner.backend)).toEqual(brief)
    expect(owner.state.get('brief')).toBe('on')

    await holdRoutineDelete(review, owner.backend)
    await undoHeldRoutine(owner.backend)
    expect(owner.state.get('review')).toBe('paused')

    expect(markers()).toEqual([])
    await commitHeldRoutine(owner.backend)
    await commitAbandonedRoutineDeletes(owner.backend, Date.now() + 10 * ABANDONED_AFTER_MS)
    expect(owner.calls.filter(call => call.startsWith('delete'))).toEqual([])
  })

  it('undo waits for an in-flight pause before resuming, so the restored routine is never paused', async () => {
    const owner = fakeOwner()
    const entered = Promise.withResolvers<void>()
    const release = Promise.withResolvers<void>()
    const order: string[] = []

    owner.backend.pause = vi.fn(async (id: string) => {
      entered.resolve()
      await release.promise
      order.push('pause resolved')
      owner.state.set(id, 'paused')
    })
    owner.backend.resume = vi.fn(async (id: string) => {
      order.push('resume')
      owner.state.set(id, 'on')
    })

    const holding = holdRoutineDelete(brief, owner.backend)
    await entered.promise
    const undoing = undoHeldRoutine(owner.backend)
    expect(order).toEqual([])

    release.resolve()
    await Promise.all([holding, undoing])

    expect(order).toEqual(['pause resolved', 'resume'])
    expect(owner.state.get('brief')).toBe('on')
  })

  it('dismissing makes it final on its own owner', async () => {
    const owner = fakeOwner()

    await holdRoutineDelete(review, owner.backend)
    expect(await commitHeldRoutine(owner.backend)).toEqual(review)

    expect(owner.calls).toEqual(['delete review @box/default'])
    expect($heldRoutine.get()).toBeNull()
    expect(markers()).toEqual([])
  })

  it('keeps one banner: deleting a second routine makes the first final', async () => {
    const owner = fakeOwner()

    await holdRoutineDelete(brief, owner.backend)
    await holdRoutineDelete(review, owner.backend)

    expect(owner.state.get('brief')).toBe('deleted')
    expect($heldRoutine.get()).toEqual(review)
    expect(markers().map(marker => marker.id)).toEqual(['review'])
  })

  it('puts everything back when the owner cannot pause it', async () => {
    const owner = fakeOwner()
    vi.mocked(owner.backend.pause).mockRejectedValueOnce(new Error('machine offline'))

    await expect(holdRoutineDelete(brief, owner.backend)).rejects.toThrow('machine offline')

    expect($heldRoutine.get()).toBeNull()
    expect(markers()).toEqual([])
    expect(owner.state.get('brief')).toBe('on')
  })

  it('shows a routine whose delete failed again, paused, instead of hiding it alive', async () => {
    const owner = fakeOwner()
    vi.mocked(owner.backend.remove).mockRejectedValueOnce(new Error('machine offline'))

    await holdRoutineDelete(brief, owner.backend)
    await expect(commitHeldRoutine(owner.backend)).rejects.toThrow('machine offline')

    expect($heldRoutine.get()).toBeNull()
    expect(markers()).toEqual([])
    expect(owner.state.get('brief')).toBe('paused')
  })

  it('counts a routine already gone as deleted', async () => {
    const owner = fakeOwner()
    owner.state.set('review', 'deleted')

    await holdRoutineDelete(review, owner.backend)

    await expect(commitHeldRoutine(owner.backend)).resolves.toEqual(review)
    expect(markers()).toEqual([])
  })
})

describe('a delete held by a window that went away', () => {
  it('is made final on the next launch, on its owner', async () => {
    const owner = fakeOwner()
    const left = Date.now()
    localStorage.setItem(KEY, JSON.stringify([{ ...brief, beat: left, holder: 'closed-window' }]))

    await commitAbandonedRoutineDeletes(owner.backend, left + ABANDONED_AFTER_MS)

    expect(owner.calls).toEqual(['delete brief @local/ops'])
    expect(markers()).toEqual([])
  })

  it('is left alone while its window still beats', async () => {
    vi.useFakeTimers()
    const owner = fakeOwner()

    await holdRoutineDelete(brief, owner.backend)
    const other = JSON.parse(localStorage.getItem(KEY)!) as { holder: string }[]
    // Another window's live hold, stamped a moment ago.
    localStorage.setItem(KEY, JSON.stringify([...other, { ...review, beat: Date.now(), holder: 'other-window' }]))

    vi.advanceTimersByTime(5 * ABANDONED_AFTER_MS)
    await commitAbandonedRoutineDeletes(owner.backend, Date.now())

    // This window's own hold kept beating, so it is still held; the other window's went quiet.
    expect(owner.calls.filter(call => call.startsWith('delete'))).toEqual(['delete review @box/default'])
    expect(markers().map(marker => marker.id)).toEqual(['brief'])
    expect($heldRoutine.get()).toEqual(brief)
  })

  it('is dropped rather than retried when its owner cannot be reached: the routine stays, paused', async () => {
    const owner = fakeOwner()
    vi.mocked(owner.backend.remove).mockRejectedValueOnce(new Error('machine offline'))
    localStorage.setItem(KEY, JSON.stringify([{ ...brief, beat: 0, holder: 'closed-window' }]))

    await commitAbandonedRoutineDeletes(owner.backend, ABANDONED_AFTER_MS)

    expect(markers()).toEqual([])
    expect(owner.state.get('brief')).toBe('on')
  })
})
