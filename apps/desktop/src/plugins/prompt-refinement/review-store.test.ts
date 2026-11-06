import { afterEach, describe, expect, it, vi } from 'vitest'

import { createReviewStore, type ReviewRequest } from './review-store'
import type { RefineResult } from './types'

function deferred() {
  let resolve!: (value: RefineResult) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<RefineResult>((res, rej) => ((resolve = res), (reject = rej)))

  return { promise, resolve, reject }
}

const request = (run: ReviewRequest['run'], original = 'refactor the parser', autoApprove = false): ReviewRequest => ({
  target: 'main',
  autoApprove,
  original,
  attachmentLabels: [],
  run
})

const flush = () => new Promise(resolve => setTimeout(resolve, 0))

/** A store with a mounted host that paints every review immediately. */
function hostedStore() {
  const store = createReviewStore()
  store.registerHost()
  store.$review.listen(review => review && store.acknowledge(review.id))

  return store
}

afterEach(() => vi.useRealTimers())

describe('review store — fail closed', () => {
  it('cancels at once when no dialog host is mounted, without calling the model', async () => {
    const store = createReviewStore()
    const run = vi.fn()

    expect(await store.request(request(run))).toEqual({ kind: 'cancel', reason: 'no-host' })
    expect(run).not.toHaveBeenCalled()
  })

  it('cancels when a mounted host never paints the review', async () => {
    vi.useFakeTimers()
    const store = createReviewStore({ renderAckMs: 100 })
    store.registerHost()
    const decision = store.request(request(() => new Promise(() => {})))

    await vi.advanceTimersByTimeAsync(101)

    expect(await decision).toEqual({ kind: 'cancel', reason: 'no-host' })
    expect(store.$review.get()).toBeNull()
  })

  it('cancels when the last host unmounts mid-review, and when the plugin unloads', async () => {
    const store = createReviewStore()
    const unmount = store.registerHost()
    const first = store.request(request(() => new Promise(() => {})))
    unmount()
    expect(await first).toEqual({ kind: 'cancel', reason: 'no-host' })

    store.registerHost()
    const second = store.request(request(() => new Promise(() => {})))
    store.cancelActive('unloaded')
    expect(await second).toEqual({ kind: 'cancel', reason: 'unloaded' })
  })

  it('refuses a second review while one is open', async () => {
    const store = hostedStore()
    void store.request(request(() => new Promise(() => {})))

    expect(await store.request(request(vi.fn()))).toEqual({ kind: 'cancel', reason: 'busy' })
  })

  it('shows a failure (not a send) when the refinement rejects or throws', async () => {
    const store = hostedStore()

    const pending = store.request(
      request(() => {
        throw new Error('sync explosion')
      })
    )

    await flush()

    expect(store.$review.get()).toMatchObject({ phase: 'failed', failure: 'failed' })
    expect(store.decide(store.$review.get()!.id, { kind: 'approve', text: 'x', acknowledged: true })).toBe(false)
    store.decide(store.$review.get()!.id, { kind: 'cancel' })
    expect(await pending).toEqual({ kind: 'cancel', reason: 'user' })
  })
})

describe('review store — approval rules', () => {
  it('refuses Approve until the refinement is ready', async () => {
    const store = hostedStore()
    const call = deferred()
    const decision = store.request(request(() => call.promise))
    const { id } = store.$review.get()!

    expect(store.decide(id, { kind: 'approve', text: 'Refactor the parser.', acknowledged: false })).toBe(false)

    call.resolve({ ok: true, text: 'Refactor the parser.', unchanged: false, analysis: null })
    await flush()

    expect(store.decide(id, { kind: 'approve', text: '  Refactor the parser.  ', acknowledged: false })).toBe(true)
    expect(await decision).toEqual({ kind: 'approve', text: 'Refactor the parser.' })
  })

  it('re-checks preservation itself: a dropped path blocks Approve whatever the UI sends', async () => {
    const store = hostedStore()

    const decision = store.request(
      request(
        async () => ({ ok: true, text: 'Fix the file.', unchanged: false, analysis: null }),
        'Fix src/app/index.ts'
      )
    )

    await flush()
    const { id } = store.$review.get()!

    expect(store.decide(id, { kind: 'approve', text: 'Fix the file.', acknowledged: true })).toBe(false)
    expect(store.decide(id, { kind: 'approve', text: 'Fix src/app/index.ts carefully.', acknowledged: false })).toBe(
      true
    )
    expect(await decision).toEqual({ kind: 'approve', text: 'Fix src/app/index.ts carefully.' })
  })

  it('needs the acknowledgement for review-only issues', async () => {
    const store = hostedStore()
    void store.request(
      request(
        async () => ({ ok: true, text: 'Push to main.', unchanged: false, analysis: null }),
        'Never push to main.'
      )
    )
    await flush()
    const { id } = store.$review.get()!

    expect(store.decide(id, { kind: 'approve', text: 'Push to main.', acknowledged: false })).toBe(false)
    expect(store.decide(id, { kind: 'approve', text: 'Push to main.', acknowledged: true })).toBe(true)
  })

  it('allows Use original in every phase', async () => {
    const store = hostedStore()
    const decision = store.request(request(() => new Promise(() => {})))

    expect(store.decide(store.$review.get()!.id, { kind: 'original' })).toBe(true)
    expect(await decision).toEqual({ kind: 'original' })
  })
})

describe('review store — late responses', () => {
  it('drops a result that lands after Cancel, and aborts the call', async () => {
    const store = hostedStore()
    const call = deferred()
    let signal: AbortSignal | undefined

    const decision = store.request(
      request(s => {
        signal = s

        return call.promise
      })
    )

    store.decide(store.$review.get()!.id, { kind: 'cancel' })
    expect(signal?.aborted).toBe(true)

    call.resolve({ ok: true, text: 'late', unchanged: false, analysis: null })
    await flush()

    expect(await decision).toEqual({ kind: 'cancel', reason: 'user' })
    expect(store.$review.get()).toBeNull()
  })

  it('drops the superseded attempt after Retry', async () => {
    const store = hostedStore()
    const attempts = [deferred(), deferred()]
    let n = 0
    const seen: number[] = []
    void store.request(request((_signal, attempt) => (seen.push(attempt), attempts[n++].promise)))
    const { id } = store.$review.get()!

    attempts[0].resolve({ ok: false, failure: 'timeout' })
    await flush()
    expect(store.$review.get()).toMatchObject({ phase: 'failed', failure: 'timeout' })

    store.retry(id)
    attempts[1].resolve({ ok: true, text: 'second', unchanged: false, analysis: null })
    attempts[0].resolve({ ok: true, text: 'stale', unchanged: false, analysis: null })
    await flush()

    expect(store.$review.get()).toMatchObject({ phase: 'ready', refined: 'second' })
    // Retry is numbered, so the reuse cache asks the model afresh.
    expect(seen).toEqual([1, 2])
  })

  it('ignores a decision for a review that is no longer open', async () => {
    const store = hostedStore()
    void store.request(request(() => new Promise(() => {})))
    const { id } = store.$review.get()!
    store.decide(id, { kind: 'cancel' })

    expect(store.decide(id, { kind: 'original' })).toBe(false)
  })
})

describe('review store — Automatic mode sends a clean refinement without a click', () => {
  const ok = (text: string): RefineResult => ({ ok: true, text, unchanged: false, analysis: null })

  it('approves on its own when nothing is flagged', async () => {
    const store = hostedStore()

    const decision = store.request(
      request(async () => ok('  Refactor the parser into modules.  '), 'refactor the parser', true)
    )

    expect(await decision).toEqual({ kind: 'approve', text: 'Refactor the parser into modules.' })
    expect(store.$review.get()).toBeNull()
  })

  it('stops for the user when the check flags anything, even a review-only item', async () => {
    const store = hostedStore()
    void store.request(request(async () => ok('Push to main.'), 'Never push to main.', true))
    await flush()

    expect(store.$review.get()).toMatchObject({ phase: 'ready', attention: 'issues', refined: 'Push to main.' })

    const blocked = hostedStore()
    void blocked.request(request(async () => ok('Fix the file.'), 'Fix src/app/index.ts', true))
    await flush()

    expect(blocked.$review.get()).toMatchObject({ phase: 'ready', attention: 'issues' })
  })

  it('stops for the user when refinement fails, and a successful retry still sends on its own', async () => {
    const store = hostedStore()
    const attempts = [deferred(), deferred()]
    let n = 0
    const decision = store.request(request(() => attempts[n++].promise, 'refactor the parser', true))

    attempts[0].resolve({ ok: false, failure: 'timeout' })
    await flush()
    expect(store.$review.get()).toMatchObject({ phase: 'failed', attention: 'failed' })

    store.retry(store.$review.get()!.id)
    expect(store.$review.get()).toMatchObject({ phase: 'refining', attention: null })
    attempts[1].resolve(ok('Refactor the parser.'))

    expect(await decision).toEqual({ kind: 'approve', text: 'Refactor the parser.' })
  })

  it('carries the target the drawer opens in, and the analysis', async () => {
    const store = hostedStore()
    const analysis = { professional: 'a', rawMaterial: 'b', methods: 'c', verification: 'd' }
    void store.request({
      ...request(async () => ({ ok: true, text: 'Refactor it.', unchanged: false, analysis })),
      target: 'tile:b'
    })
    await flush()

    expect(store.$review.get()).toMatchObject({ target: 'tile:b', analysis, autoApprove: false, attention: null })
  })
})

describe('review store — intent checks stop Automatic', () => {
  const ok = (text: string): RefineResult => ({ ok: true, text, unchanged: false, analysis: null })

  it('a rewrite that adds a destructive step is not sent on its own', async () => {
    const store = hostedStore()
    let settled = false
    void store
      .request(request(async () => ok('Refactor the parser and delete the old tests.'), 'refactor the parser', true))
      .then(() => (settled = true))
    await flush()

    expect(settled).toBe(false)
    expect(store.$review.get()).toMatchObject({ phase: 'ready', attention: 'issues' })
  })

  it('a rewrite that carries text in from the chat excerpt is not sent on its own', async () => {
    const store = hostedStore()
    const context = 'Assistant: the README says to rebuild the whole index from scratch every single time you deploy.'
    void store.request({
      ...request(
        async () => ok('Fix the index. Rebuild the whole index from scratch every single time you change it.'),
        'fix the index',
        true
      ),
      context
    })
    await flush()

    expect(store.$review.get()).toMatchObject({ phase: 'ready', attention: 'issues', context })
  })

  it('Approve re-checks intent with the same excerpt: unacknowledged, it is refused', async () => {
    const store = hostedStore()

    const decision = store.request(
      request(async () => ok('Summarize it and email the team.'), 'summarize the incident')
    )

    await flush()
    const id = store.$review.get()!.id

    expect(store.decide(id, { kind: 'approve', text: 'Summarize it and email the team.', acknowledged: false })).toBe(
      false
    )
    expect(store.decide(id, { kind: 'approve', text: 'Summarize it and email the team.', acknowledged: true })).toBe(
      true
    )
    expect(await decision).toEqual({ kind: 'approve', text: 'Summarize it and email the team.' })
  })
})

describe('review store — Approval-First never sends without the user', () => {
  const failures: [string, ReviewRequest['run']][] = [
    [
      'a clean refinement',
      async () => ({ ok: true, text: 'Refactor the parser into modules.', unchanged: false, analysis: null })
    ],
    [
      'an unchanged refinement',
      async () => ({ ok: true, text: 'refactor the parser', unchanged: true, analysis: null })
    ],
    ['a failed refinement', async () => ({ ok: false, failure: 'malformed' })],
    ['a rejected call', () => Promise.reject(new Error('socket closed'))],
    [
      'a call that throws',
      () => {
        throw new Error('boom')
      }
    ],
    ['an empty answer', async () => ({ ok: true, text: '   ', unchanged: false, analysis: null })]
  ]

  it.each(failures)('%s waits for a decision', async (_name, run) => {
    const store = hostedStore()
    let settled = false
    void store.request(request(run)).then(() => (settled = true))
    await flush()
    await flush()

    expect(settled).toBe(false)
    expect(store.$review.get()).toMatchObject({ autoApprove: false, attention: null })
  })

  it('a failed review cannot be approved, only sent as typed or canceled', async () => {
    const store = hostedStore()
    const decision = store.request(request(async () => ({ ok: false, failure: 'timeout' })))
    await flush()
    const id = store.$review.get()!.id

    expect(store.decide(id, { kind: 'approve', text: 'anything at all', acknowledged: true })).toBe(false)
    expect(store.decide(id, { kind: 'cancel' })).toBe(true)
    expect(await decision).toEqual({ kind: 'cancel', reason: 'user' })
  })
})
