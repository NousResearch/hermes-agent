import type { GrammarState, ThemedToken } from 'shiki'
import { describe, expect, it, vi } from 'vitest'

import { DiffHighlightCache } from './diff-highlight-cache'

const TOKENS: ThemedToken[][] = [[{ content: 'const', offset: 0 }]]
const RESULT = { tokens: TOKENS }
const INPUT = { code: 'const answer = 42', language: 'typescript', theme: 'github-light-default' }

describe('DiffHighlightCache', () => {
  it('yields before starting highlight work', async () => {
    const highlight = vi.fn(async () => RESULT)
    const scheduled: Array<() => void> = []
    const cache = new DiffHighlightCache({ highlight, schedule: task => scheduled.push(task) })

    const pending = cache.request(INPUT)

    expect(highlight).not.toHaveBeenCalled()
    expect(scheduled).toHaveLength(1)

    scheduled.shift()?.()
    await expect(pending.promise).resolves.toBe(RESULT)
    expect(highlight).toHaveBeenCalledOnce()
  })

  it('gives each distinct expensive job its own yield point', async () => {
    const highlight = vi.fn(async () => RESULT)
    const scheduled: Array<() => void> = []
    const cache = new DiffHighlightCache({ highlight, schedule: task => scheduled.push(task) })

    const alpha = cache.request({ ...INPUT, code: 'alpha' })
    const beta = cache.request({ ...INPUT, code: 'beta' })

    expect(scheduled).toHaveLength(2)
    expect(highlight).not.toHaveBeenCalled()

    scheduled.shift()?.()
    await alpha.promise
    expect(highlight).toHaveBeenCalledTimes(1)
    expect(scheduled).toHaveLength(1)

    scheduled.shift()?.()
    await beta.promise
    expect(highlight).toHaveBeenCalledTimes(2)
  })

  it('deduplicates in-flight work and reuses its completed result', async () => {
    let finish!: (result: typeof RESULT) => void
    const highlight = vi.fn(() => new Promise<typeof RESULT>(resolve => (finish = resolve)))
    const scheduled: Array<() => void> = []
    const cache = new DiffHighlightCache({ highlight, schedule: task => scheduled.push(task) })

    const first = cache.request(INPUT)
    const duplicate = cache.request(INPUT)

    expect(scheduled).toHaveLength(1)
    scheduled.shift()?.()
    expect(highlight).toHaveBeenCalledOnce()

    finish(RESULT)
    await expect(first.promise).resolves.toBe(RESULT)
    await expect(duplicate.promise).resolves.toBe(RESULT)

    const completed = cache.request(INPUT)
    await expect(completed.promise).resolves.toBe(RESULT)
    expect(highlight).toHaveBeenCalledOnce()
  })

  it('bounds completed entries and evicts the least-recently-used result', async () => {
    const highlight = vi.fn(async ({ code }: { code: string }) => ({ tokens: [[{ content: code, offset: 0 }]] }))
    const scheduled: Array<() => void> = []

    const cache = new DiffHighlightCache({
      highlight,
      maxCompleted: 2,
      schedule: task => scheduled.push(task)
    })

    const request = async (code: string) => {
      const pending = cache.request({ ...INPUT, code })
      scheduled.shift()?.()

      return pending.promise
    }

    await request('alpha')
    await request('beta')
    await request('alpha') // refresh alpha's LRU position
    await request('gamma')
    await request('beta')

    expect(highlight.mock.calls.map(([input]) => input.code)).toEqual(['alpha', 'beta', 'gamma', 'beta'])
  })

  it('keys completed work by language and theme as well as code', async () => {
    const highlight = vi.fn(async () => RESULT)
    const scheduled: Array<() => void> = []
    const cache = new DiffHighlightCache({ highlight, schedule: task => scheduled.push(task) })

    const request = async (language: string, theme: string) => {
      const pending = cache.request({ ...INPUT, language, theme })

      scheduled.shift()?.()
      await pending.promise
    }

    await request('typescript', 'github-light-default')
    await request('typescript', 'github-dark-dimmed')
    await request('javascript', 'github-light-default')
    await request('typescript', 'github-light-default')

    expect(highlight).toHaveBeenCalledTimes(3)
  })

  it('deduplicates the same grammar continuation without mixing distinct continuation states', async () => {
    const firstState = ({ id: 'first' } as unknown) as GrammarState
    const secondState = ({ id: 'second' } as unknown) as GrammarState
    const highlight = vi.fn(async (_input: typeof INPUT & { grammarState?: GrammarState }) => RESULT)
    const scheduled: Array<() => void> = []
    const cache = new DiffHighlightCache({ highlight, schedule: task => scheduled.push(task) })

    const first = cache.request({ ...INPUT, grammarState: firstState })
    const duplicate = cache.request({ ...INPUT, grammarState: firstState })
    const distinct = cache.request({ ...INPUT, grammarState: secondState })

    expect(scheduled).toHaveLength(2)
    scheduled.splice(0).forEach(run => run())
    await Promise.all([first.promise, duplicate.promise, distinct.promise])

    expect(highlight).toHaveBeenCalledTimes(2)
    expect(highlight.mock.calls.map(([input]) => input.grammarState)).toEqual([firstState, secondState])
  })

  it('does not cache failures so a later request retries', async () => {
    const failure = new Error('grammar failed to load')
    const highlight = vi.fn().mockRejectedValueOnce(failure).mockResolvedValueOnce(RESULT)
    const scheduled: Array<() => void> = []
    const cache = new DiffHighlightCache({ highlight, schedule: task => scheduled.push(task) })

    const failed = cache.request(INPUT)

    scheduled.shift()?.()
    await expect(failed.promise).rejects.toBe(failure)

    const retry = cache.request(INPUT)

    scheduled.shift()?.()
    await expect(retry.promise).resolves.toBe(RESULT)
    expect(highlight).toHaveBeenCalledTimes(2)
  })

  it('skips scheduled work that no longer has an observer', async () => {
    const highlight = vi.fn(async () => RESULT)
    const scheduled: Array<() => void> = []
    const cache = new DiffHighlightCache({ highlight, schedule: task => scheduled.push(task) })

    const stale = cache.request(INPUT)

    stale.release()
    scheduled.shift()?.()

    await expect(stale.promise).rejects.toMatchObject({ name: 'AbortError' })
    expect(highlight).not.toHaveBeenCalled()
  })

  it('aborts released work after scheduling while an async import is pending', async () => {
    let finishImport!: () => void
    const importGate = new Promise<void>(resolve => (finishImport = resolve))

    const highlight = vi.fn(async (_input: typeof INPUT, signal: AbortSignal) => {
      await importGate

      if (signal.aborted) {
        const error = new Error('highlight aborted')

        error.name = 'AbortError'
        throw error
      }

      return RESULT
    })

    const scheduled: Array<() => void> = []
    const cache = new DiffHighlightCache({ highlight, schedule: task => scheduled.push(task) })
    const stale = cache.request(INPUT)

    scheduled.shift()?.()
    expect(highlight).toHaveBeenCalledOnce()
    stale.release()
    finishImport()

    await expect(stale.promise).rejects.toMatchObject({ name: 'AbortError' })

    const retry = cache.request(INPUT)

    scheduled.shift()?.()
    await expect(retry.promise).resolves.toBe(RESULT)
    expect(highlight).toHaveBeenCalledTimes(2)
  })
})
