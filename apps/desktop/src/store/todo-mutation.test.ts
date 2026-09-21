import { describe, expect, it, vi } from 'vitest'

import type { TodoSnapshot } from '../lib/todos'

import { createTodoMutationController, humanTodoTarget, TodoMutationFailure } from './todo-mutation'

const snapshot = (over: Partial<TodoSnapshot> = {}): TodoSnapshot => ({
  generation: 2,
  revision: 2,
  session_id: 's1',
  todos: [
    { content: 'Build tray', id: 'build', status: 'in_progress' },
    { content: 'Ship it', id: 'ship', status: 'pending' }
  ],
  ...over
})

const mutationInput = (over: Partial<Parameters<ReturnType<typeof createTodoMutationController>['run']>[0]> = {}) => ({
  content: 'Build tray',
  expectedGeneration: 2,
  expectedRevision: 2,
  itemId: 'build',
  sessionId: 's1',
  status: 'completed' as const,
  ...over
})

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void

  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, reject, resolve }
}

function harness(initial: TodoSnapshot | null = snapshot()) {
  let current: TodoSnapshot | null = initial
  const applied: TodoSnapshot[] = []
  const optimistic: Array<{ itemId: string; sessionId: string; status: string }> = []
  const request = vi.fn()

  const controller = createTodoMutationController({
    applySnapshot(next) {
      if (!current || next.generation >= current.generation) {
        current = next
        applied.push(next)
      }
    },
    getSnapshot: () => current,
    optimisticStatus(sessionId, itemId, status) {
      optimistic.push({ itemId, sessionId, status })

      return Boolean(current?.todos.some(todo => todo.id === itemId))
    },
    request
  })

  return {
    applied,
    controller,
    current: () => current,
    optimistic,
    request,
    setCurrent(next: TodoSnapshot) {
      current = next
    }
  }
}

describe('humanTodoTarget', () => {
  it('maps open work to completed and terminal work to pending', () => {
    expect(humanTodoTarget('pending')).toBe('completed')
    expect(humanTodoTarget('in_progress')).toBe('completed')
    expect(humanTodoTarget('completed')).toBe('pending')
    expect(humanTodoTarget('cancelled')).toBe('pending')
  })
})

describe('createTodoMutationController', () => {
  it('optimistically updates after run and sends the exact CAS payload', async () => {
    const h = harness()

    const result = snapshot({
      generation: 3,
      revision: 3,
      todos: [{ content: 'Build tray', id: 'build', status: 'completed' }]
    })

    h.request.mockResolvedValueOnce(result)

    await expect(
      h.controller.run(mutationInput())
    ).resolves.toEqual(result)

    expect(h.optimistic).toEqual([{ itemId: 'build', sessionId: 's1', status: 'completed' }])
    expect(h.request).toHaveBeenCalledWith('todo.update_status', {
      actor: 'user',
      expected_revision: 2,
      item_id: 'build',
      session_id: 's1',
      status: 'completed'
    })
    expect(h.applied).toEqual([result])
  })

  it('deduplicates duplicate activation while one task mutation is pending', async () => {
    const h = harness()
    const pending = deferred<TodoSnapshot>()
    h.request.mockReturnValueOnce(pending.promise)

    const first = h.controller.run(mutationInput())
    const second = h.controller.run(mutationInput())

    expect(h.controller.isPending('s1', 'build')).toBe(true)
    expect(h.request).toHaveBeenCalledTimes(1)
    pending.resolve(snapshot({ generation: 3, revision: 3 }))
    await expect(Promise.all([first, second])).resolves.toHaveLength(2)
    expect(h.controller.isPending('s1', 'build')).toBe(false)
  })

  it('adopts a stale-error snapshot and reports a machine-readable failure', async () => {
    const h = harness()

    const latest = snapshot({
      generation: 4,
      revision: 4,
      todos: [{ content: 'New plan', id: 'new', status: 'pending' }]
    })

    h.request.mockRejectedValueOnce(Object.assign(new Error('stale'), { code: 4096, data: latest }))

    await expect(
      h.controller.run(mutationInput())
    ).rejects.toMatchObject({
      kind: 'stale'
    })
    expect(h.applied).toEqual([latest])
    expect(h.request).toHaveBeenCalledTimes(1)
  })

  it('adopts a missing-item snapshot and reports missing', async () => {
    const h = harness()
    const latest = snapshot({ generation: 3, revision: 3, todos: [] })
    h.request.mockRejectedValueOnce(Object.assign(new Error('missing'), { code: 4044, data: latest }))

    await expect(
      h.controller.run(mutationInput())
    ).rejects.toMatchObject({
      kind: 'missing'
    })
    expect(h.applied).toEqual([latest])
  })

  it('refreshes once after a transport failure and restores authoritative display', async () => {
    const h = harness()
    const refreshed = snapshot({ generation: 5, revision: 5 })
    h.request.mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce(refreshed)

    await expect(
      h.controller.run(mutationInput())
    ).rejects.toBeInstanceOf(TodoMutationFailure)
    expect(h.request).toHaveBeenNthCalledWith(2, 'todo.snapshot', { session_id: 's1' })
    expect(h.applied).toEqual([refreshed])
  })

  it('does not let a lower-generation response overwrite a newer event', async () => {
    const h = harness()
    const pending = deferred<TodoSnapshot>()
    h.request.mockReturnValueOnce(pending.promise)
    const run = h.controller.run(mutationInput())

    const newer = snapshot({
      generation: 8,
      revision: 8,
      todos: [{ content: 'Model update', id: 'model', status: 'pending' }]
    })

    h.setCurrent(newer)
    pending.resolve(snapshot({ generation: 3, revision: 3 }))

    await run
    expect(h.current()?.generation).toBe(8)
  })

  it('fails closed when no authoritative revision is loaded', async () => {
    const h = harness(null)

    await expect(
      h.controller.run(mutationInput())
    ).rejects.toMatchObject({
      kind: 'unavailable'
    })
    expect(h.request).not.toHaveBeenCalled()
  })

  it('does not retarget a confirmation when the same id now has different content', async () => {
    const h = harness(
      snapshot({ todos: [{ content: 'Different task', id: 'build', status: 'pending' }] })
    )

    await expect(
      h.controller.run(mutationInput())
    ).rejects.toMatchObject({
      kind: 'missing'
    })
    expect(h.request).not.toHaveBeenCalled()
  })

  it('rejects a confirmation when authority changed after the dialog opened', async () => {
    const h = harness()

    h.setCurrent(snapshot({ generation: 3, revision: 3 }))

    await expect(
      h.controller.run(mutationInput({ expectedGeneration: 2, expectedRevision: 2 }))
    ).rejects.toMatchObject({ kind: 'stale' })
    expect(h.request).not.toHaveBeenCalled()
    expect(h.optimistic).toEqual([])
  })
})
