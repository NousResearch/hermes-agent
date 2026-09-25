vi.mock('@/store/profile', async (): Promise<object> => {
  const { atom } = await import('nanostores')

  return { $activeGatewayProfile: atom<string>('default') }
})
vi.mock('@/store/session', async (): Promise<object> => {
  const { atom } = await import('nanostores')

  return { $connection: atom(null), $defaultReasoningEffort: atom<string>('') }
})

import { QueryClient, type QueryKey, QueryObserver } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'
import { deferred } from '@/test/deferred'
import type { LocalRuntimeJob } from '@/types/hermes'

import type { LocalModelsOwner } from './local-runtime-jobs'

// The BACKEND is the authority: a staged registry the poll reads from, so
// transitions arrive the way production sees them — via a poll response,
// never by mutating the cache directly.
const backend = vi.hoisted(() => ({ jobs: [] as LocalRuntimeJob[] }))

vi.mock('@/hermes', () => ({
  getLocalModelsJobs: vi.fn(async (): Promise<{ jobs: LocalRuntimeJob[] }> => ({
    jobs: structuredClone(backend.jobs)
  })),
  getLocalModelsStatus: vi.fn(async () => ({ enabled: true, update_available: false }))
}))

vi.mock('@/i18n', () => ({
  translateNow: (key: string, ...args: unknown[]) => (args.length ? `${key}:${args.join(',')}` : key)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

const { localModelsKey, localModelsOwner, watchLocalRuntimeJobs } = await import('./local-runtime-jobs')
const { getLocalModelsJobs } = await import('@/hermes')
const { $connection } = await import('@/store/session')
const { notify, notifyError } = await import('@/store/notifications')

function job(overrides: Partial<LocalRuntimeJob>): LocalRuntimeJob {
  return {
    detail: '',
    done_bytes: 0,
    error: null,
    job_id: 'j1',
    kind: 'model-download',
    model_id: 'm1',
    phase: 'downloading',
    status: 'running',
    target: 'Qwen3.6 27B',
    total_bytes: 100,
    ...overrides
  }
}

beforeEach((): void => {
  queryClient.clear()
  queryClient.setDefaultOptions({ queries: { ...queryClient.getDefaultOptions().queries, retry: false } })
  vi.clearAllMocks()
  backend.jobs = []
  queryClient.setQueryData(localModelsKey(localModelsOwner(), 'jobs'), [])
})

afterEach((): void => {
  queryClient.clear()
})

const settle = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

// Drive the poller: each advance lets the pending tick fire, the fetch read
// the staged backend snapshot, and the next tick re-arm.
// Stage-then-poll: the kick starts the loop AND the loop only re-arms while
// work is active, so each tick re-kicks (idempotent) then waits for the
// fetch to land the staged snapshot.
async function pollTick() {
  watchLocalRuntimeJobs()
  await settle(150)
  await new Promise(resolve => setTimeout(resolve, 0))
}

describe('local runtime jobs store — pause/settle contract', () => {
  it('running→error still toasts exactly once', async () => {
    backend.jobs = [job({ done_bytes: 10, job_id: 'j-err' })]
    await pollTick()

    backend.jobs = [job({ done_bytes: 10, error: 'disk full', job_id: 'j-err', status: 'error' })]
    await pollTick()

    expect(notifyError).toHaveBeenCalledTimes(1)
    expect(notify).not.toHaveBeenCalled()
  })

  it('a settle landing after the primary connection moved is bookkept silently, not thrown', async () => {
    const owner: LocalModelsOwner = localModelsOwner()
    const key: QueryKey = localModelsKey(owner, 'jobs')
    backend.jobs = [job({ done_bytes: 10 })]
    await pollTick()

    // The response was fetched for the old endpoint but its observer
    // notification fires after the connection re-homed (TanStack notifies in
    // a microtask). Published straight into the cache to hit exactly that
    // window; the poll itself would already reject the stale scope.
    ;($connection as { set: (value: unknown) => void }).set({ baseUrl: 'http://moved.example' })
    queryClient.setQueryData(key, [job({ done_bytes: 100, status: 'done' })])
    await settle(0)

    expect(notify).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()

    // Bookkeeping kept going: the job is remembered as settled, so coming
    // back to this endpoint does not re-announce it.
    ;($connection as { set: (value: unknown) => void }).set(null)
    queryClient.setQueryData(key, [job({ done_bytes: 100, status: 'done' })])
    await settle(0)
    expect(notify).not.toHaveBeenCalled()
  })

  it('publishes a pinned job response into an injected QueryClient observer', async (): Promise<void> => {
    const client: QueryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const owner: LocalModelsOwner = { connectionId: 'A', profile: 'work' }
    backend.jobs = [job({ done_bytes: 1, total_bytes: 2 })]

    const observer: QueryObserver<readonly LocalRuntimeJob[]> = new QueryObserver<readonly LocalRuntimeJob[]>(client, {
      queryKey: ['local-models', 'A', 'work', 'jobs'],
      enabled: false
    })

    const unsubscribe: () => void = observer.subscribe((): void => {})

    try {
      watchLocalRuntimeJobs(owner, client)
      await vi.waitFor((): void => expect(observer.getCurrentResult().data).toEqual(backend.jobs))
      expect(getLocalModelsJobs).toHaveBeenCalledWith(owner)
      expect(queryClient.getQueryData(localModelsKey(owner, 'jobs'))).toBeUndefined()
    } finally {
      unsubscribe()
      client.clear()
    }
  })

  it.each(['cold', 'warm'] as const)(
    'coalesces one trailing read with a %s production cache',
    async (cache: 'cold' | 'warm'): Promise<void> => {
      const owner: LocalModelsOwner = { connectionId: 'A', profile: 'work' }
      const key: QueryKey = localModelsKey(owner, 'jobs')

      if (cache === 'warm') {
        watchLocalRuntimeJobs(owner)
        await vi.waitFor((): void => expect(queryClient.getQueryData(key)).toEqual([]))
      }

      vi.mocked(getLocalModelsJobs).mockClear()
      const pending = deferred<{ jobs: LocalRuntimeJob[] }>()
      vi.mocked(getLocalModelsJobs).mockReturnValueOnce(pending.promise)
      watchLocalRuntimeJobs(owner)
      watchLocalRuntimeJobs(owner)
      watchLocalRuntimeJobs(owner)
      await Promise.resolve()
      expect(getLocalModelsJobs).toHaveBeenCalledTimes(1)
      backend.jobs = [job({ done_bytes: 40 })]
      pending.resolve({ jobs: [] })
      await vi.waitFor((): void => expect(getLocalModelsJobs).toHaveBeenCalledTimes(2))
      expect(queryClient.getQueryData<readonly LocalRuntimeJob[]>(key)?.[0]?.done_bytes).toBe(40)
    }
  )

  it('preserves the query data reference when a poll returns identical payload', async () => {
    backend.jobs = [job({ done_bytes: 40 })]
    await pollTick()
    expect(
      queryClient.getQueryData<readonly LocalRuntimeJob[]>(localModelsKey(localModelsOwner(), 'jobs')) ?? []
    ).toHaveLength(1)

    const reference =
      queryClient.getQueryData<readonly LocalRuntimeJob[]>(localModelsKey(localModelsOwner(), 'jobs')) ?? []

    await pollTick()
    await pollTick()

    // Same reference preserved — no-op polls never hand React fresh arrays.
    expect(queryClient.getQueryData<readonly LocalRuntimeJob[]>(localModelsKey(localModelsOwner(), 'jobs')) ?? []).toBe(
      reference
    )
  })

  it('structural sharing includes control flags, ranges, percent, detail and error', async () => {
    backend.jobs = [job({ done_bytes: 40, ranges: { 'model.gguf': [[0, 100]] } })]
    await pollTick()
    const first = queryClient.getQueryData<readonly LocalRuntimeJob[]>(localModelsKey(localModelsOwner(), 'jobs')) ?? []

    // Backend flips can_pause:false (e.g. a phase change) with identical bytes.
    backend.jobs = [job({ can_pause: false, done_bytes: 40, ranges: { 'model.gguf': [[0, 100]] } })]
    await pollTick()

    const second =
      queryClient.getQueryData<readonly LocalRuntimeJob[]>(localModelsKey(localModelsOwner(), 'jobs')) ?? []

    expect(second).not.toBe(first)
    expect(second[0]?.can_pause).toBe(false)

    // A ranges-only change re-publishes too.
    const before =
      queryClient.getQueryData<readonly LocalRuntimeJob[]>(localModelsKey(localModelsOwner(), 'jobs')) ?? []

    backend.jobs = [
      job({
        can_pause: false,
        done_bytes: 40,
        ranges: {
          'model.gguf': [
            [0, 100],
            [100, 200]
          ]
        }
      })
    ]
    await pollTick()
    expect(
      queryClient.getQueryData<readonly LocalRuntimeJob[]>(localModelsKey(localModelsOwner(), 'jobs')) ?? []
    ).not.toBe(before)
    expect(
      (queryClient.getQueryData<readonly LocalRuntimeJob[]>(localModelsKey(localModelsOwner(), 'jobs')) ?? [])[0]
        ?.ranges?.['model.gguf']
    ).toEqual([
      [0, 100],
      [100, 200]
    ])
  })
})
