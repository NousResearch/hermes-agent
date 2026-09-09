import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { LocalRuntimeJob } from '@/types/hermes'

// The BACKEND is the authority: a staged registry the poll reads from, so
// transitions arrive the way production sees them — via a poll response,
// never by mutating the cache directly.
const backend = vi.hoisted(() => ({ jobs: [] as LocalRuntimeJob[] }))

vi.mock('@/hermes', () => ({
  getLocalModelsJobs: vi.fn(async () => ({ jobs: backend.jobs })),
  getLocalModelsStatus: vi.fn(async () => ({ enabled: true, update_available: false }))
}))

vi.mock('@/i18n', () => ({
  translateNow: (key: string, ...args: unknown[]) => (args.length ? `${key}:${args.join(',')}` : key)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

const { $localRuntimeJobs, watchLocalRuntimeJobs } = await import('./local-runtime-jobs')
const { getLocalModelsJobs } = await import('@/hermes')
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

beforeEach(() => {
  vi.clearAllMocks()
  backend.jobs = []
  $localRuntimeJobs.set([])
})

afterEach(async () => {
  backend.jobs = []
  watchLocalRuntimeJobs()
  await vi.waitFor(() => expect($localRuntimeJobs.get()).toEqual([]))
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
  it('running→paused settles NOTHING: no error toast, job stays visible', async () => {
    backend.jobs = [job({ done_bytes: 40 })]
    await pollTick()
    expect($localRuntimeJobs.get()[0]?.status).toBe('running')

    // Backend pauses the job (status='paused', error=null).
    backend.jobs = [job({ done_bytes: 40, status: 'paused' })]
    await pollTick()
    await pollTick()

    expect(notifyError).not.toHaveBeenCalled()
    expect(notify).not.toHaveBeenCalled()

    const snapshot = $localRuntimeJobs.get()
    expect(snapshot).toHaveLength(1)
    expect(snapshot[0]?.status).toBe('paused')
  })

  it('a paused job that resumes and finishes notifies done exactly once (paused→done still notifies)', async () => {
    backend.jobs = [job({ done_bytes: 40 })]
    await pollTick()

    backend.jobs = [job({ done_bytes: 40, status: 'paused' })]
    await pollTick()

    backend.jobs = [job({ done_bytes: 60 })]
    await pollTick()

    backend.jobs = [job({ done_bytes: 100, percent: 100, status: 'done', total_bytes: 100 })]
    await pollTick()

    expect(notify).toHaveBeenCalledTimes(1)
    expect(vi.mocked(notify).mock.calls[0][0]).toMatchObject({ kind: 'success' })
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('running→error still toasts exactly once', async () => {
    backend.jobs = [job({ done_bytes: 10, job_id: 'j-err' })]
    await pollTick()

    backend.jobs = [job({ done_bytes: 10, error: 'disk full', job_id: 'j-err', status: 'error' })]
    await pollTick()

    expect(notifyError).toHaveBeenCalledTimes(1)
    expect(notify).not.toHaveBeenCalled()
  })

  it('keeps polling while a job is paused (a resume from another surface is witnessed)', async () => {
    backend.jobs = [job({ status: 'paused' })]
    await pollTick()

    const callsAfterPause = vi.mocked(getLocalModelsJobs).mock.calls.length
    expect(callsAfterPause).toBeGreaterThan(0)

    // The paused cadence is slower (3s), not dead: wait past it and the
    // poll fires again WITHOUT any new kick.
    await settle(3_400)

    expect(vi.mocked(getLocalModelsJobs).mock.calls.length).toBeGreaterThan(callsAfterPause)

    // And a backend-side resume IS picked up with no local kick at all.
    backend.jobs = [job({ done_bytes: 90 })]
    await pollTick()
    expect($localRuntimeJobs.get()[0]?.status).toBe('running')
  })

  it('coalesces refresh requests while one backend read is in flight', async () => {
    let release: (value: { jobs: LocalRuntimeJob[] }) => void = () => {}

    const pending = new Promise<{ jobs: LocalRuntimeJob[] }>(resolve => {
      release = resolve
    })

    vi.mocked(getLocalModelsJobs).mockReturnValueOnce(pending)
    watchLocalRuntimeJobs()
    watchLocalRuntimeJobs()
    watchLocalRuntimeJobs()
    expect(getLocalModelsJobs).toHaveBeenCalledTimes(1)
    backend.jobs = [job({ done_bytes: 40 })]
    release({ jobs: [] })
    await vi.waitFor(() => expect(getLocalModelsJobs).toHaveBeenCalledTimes(2))
    expect($localRuntimeJobs.get()[0]?.done_bytes).toBe(40)
  })

  it('does not re-set the atom when a poll returns the identical payload', async () => {
    backend.jobs = [job({ done_bytes: 40 })]
    await pollTick()
    expect($localRuntimeJobs.get()).toHaveLength(1)

    const reference = $localRuntimeJobs.get()
    await pollTick()
    await pollTick()

    // Same reference preserved — no-op polls never hand React fresh arrays.
    expect($localRuntimeJobs.get()).toBe(reference)
  })

  it('equality covers control flags, ranges, percent, detail and error — not just done_bytes', async () => {
    backend.jobs = [job({ done_bytes: 40, ranges: { 'model.gguf': [[0, 100]] } })]
    await pollTick()
    const first = $localRuntimeJobs.get()

    // Backend flips can_pause:false (e.g. a phase change) with identical bytes.
    backend.jobs = [job({ can_pause: false, done_bytes: 40, ranges: { 'model.gguf': [[0, 100]] } })]
    await pollTick()

    const second = $localRuntimeJobs.get()
    expect(second).not.toBe(first)
    expect(second[0]?.can_pause).toBe(false)

    // A ranges-only change re-publishes too.
    const before = $localRuntimeJobs.get()
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
    expect($localRuntimeJobs.get()).not.toBe(before)
    expect($localRuntimeJobs.get()[0]?.ranges?.['model.gguf']).toEqual([
      [0, 100],
      [100, 200]
    ])
  })
})
