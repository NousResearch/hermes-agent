import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { $activeGatewayRoute } from '@/store/gateway'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $connection } from '@/store/session'
import type { LocalRuntimeJob } from '@/types/hermes'

import {
  $localRuntimeInstallStarting,
  $localRuntimeJobs,
  startLocalRuntimeInstall,
  watchLocalRuntimeJobs
} from './local-runtime-jobs'

const notices = vi.hoisted(() => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/notifications', () => notices)
const api = vi.fn()

const job: LocalRuntimeJob = {
  job_id: 'install',
  kind: 'runtime-install',
  target: 'target',
  model_id: null,
  status: 'running',
  phase: 'download',
  detail: 'Real download',
  total_bytes: 100,
  done_bytes: 25,
  percent: 25,
  error: null
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void

  const promise = new Promise<T>((done, fail) => {
    resolve = done
    reject = fail
  })

  return { promise, resolve, reject }
}

async function flush() {
  for (let i = 0; i < 12; i++) {
    await Promise.resolve()
  }
}

let contextNumber = 0
const posts = () => api.mock.calls.filter(([request]) => request.method === 'POST')
beforeEach(() => {
  vi.useFakeTimers()
  vi.clearAllMocks()
  api.mockReset()
  setApiRequestConnection(null)
  $localModelsEnabled.set(true)
  $connection.set({ mode: 'local', baseUrl: `http://test-${++contextNumber}` } as never)
  setApiRequestProfile('default')
  $activeGatewayRoute.set('default')
  $localRuntimeJobs.set([])
  window.hermesDesktop = { api } as never
})
afterEach(async () => {
  api.mockResolvedValue({ jobs: [] })
  await vi.advanceTimersByTimeAsync(700)
  $connection.set(null)
  vi.clearAllTimers()
  vi.useRealTimers()
})
it.each(['connection', 'profile'])('retains late install acceptance after a %s round trip', async context => {
  const post = deferred<{ job_id: string }>()
  api.mockImplementation(request => (request.method === 'POST' ? post.promise : Promise.resolve({ jobs: [job] })))
  const first = startLocalRuntimeInstall()
  const original = $connection.get()

  if (context === 'connection') {
    $connection.set({ mode: 'remote' } as never)
    $connection.set(original)
  } else {
    $activeGatewayRoute.set('other')
    $activeGatewayRoute.set('default')
  }

  void startLocalRuntimeInstall()
  expect(posts()).toHaveLength(1)
  post.resolve({ job_id: 'install' })
  await first
  expect(posts()).toHaveLength(1)
  expect($localRuntimeJobs.get()).toEqual([job])
  expect($localRuntimeInstallStarting.get()).toBe(false)
})
it.each(['connection', 'profile'])('ignores late job completion and notifications after a %s switch', async context => {
  api.mockResolvedValue({ jobs: [job] })
  watchLocalRuntimeJobs()
  await flush()
  const read = deferred<{ jobs: LocalRuntimeJob[] }>()
  api.mockReturnValue(read.promise)
  await vi.advanceTimersByTimeAsync(700)

  if (context === 'connection') {
    $connection.set({ mode: 'remote' } as never)
  } else {
    $activeGatewayRoute.set('other')
  }

  read.resolve({ jobs: [{ ...job, status: 'done' }] })
  await flush()
  expect($localRuntimeJobs.get()).toEqual([])
  expect(notices.notify).not.toHaveBeenCalled()
  expect(notices.notifyError).not.toHaveBeenCalled()
  expect(vi.getTimerCount()).toBe(0)
})
it('does not let an old POST failure clear a new context start or notify it', async () => {
  const old = deferred<{ job_id: string }>()
  const current = deferred<{ job_id: string }>()
  api.mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise).mockResolvedValue({ jobs: [] })
  const first = startLocalRuntimeInstall()
  $activeGatewayRoute.set('other')
  const second = startLocalRuntimeInstall()
  old.reject(new Error('old connection failed'))
  await first
  const stillStarting = $localRuntimeInstallStarting.get()
  current.resolve({ job_id: 'new' })
  await second
  expect(stillStarting).toBe(true)
  expect(notices.notifyError).not.toHaveBeenCalled()
  expect(posts()).toHaveLength(2)
})
it.each(['connection', 'profile'])(
  'resumes accepted work on a %s round trip before permitting another install',
  async context => {
    api.mockImplementation(request =>
      Promise.resolve(request.method === 'POST' ? { job_id: job.job_id } : { jobs: [job] })
    )
    await startLocalRuntimeInstall()
    const original = $connection.get()

    if (context === 'profile') {
      $activeGatewayRoute.set('other')
      setApiRequestProfile('other')
    } else {
      $connection.set({ mode: 'remote', baseUrl: 'http://other' } as never)
    }

    const readsBefore = api.mock.calls.length
    await vi.advanceTimersByTimeAsync(1400)
    expect(api.mock.calls).toHaveLength(readsBefore)
    expect($localRuntimeJobs.get()).toEqual([])
    const discovery = deferred<{ jobs: LocalRuntimeJob[] }>()
    api.mockReturnValue(discovery.promise)

    if (context === 'profile') {
      $activeGatewayRoute.set('default')
    } else {
      $connection.set({ ...original } as never)
    }

    setApiRequestProfile('default')
    void startLocalRuntimeInstall()
    await flush()
    expect(posts()).toHaveLength(1)
    expect(api.mock.calls).toHaveLength(readsBefore + 1)
    expect(api.mock.calls.at(-1)?.[0].profile).toBe('default')
    expect($localRuntimeInstallStarting.get()).toBe(true)
    discovery.resolve({ jobs: [{ ...job, status: 'done' }] })
    await flush()
    expect($localRuntimeJobs.get()[0].status).toBe('done')
    expect(notices.notify).toHaveBeenCalledTimes(1)
    await startLocalRuntimeInstall()
    expect(posts()).toHaveLength(2)
  }
)
it.each(['error', 'done'] as const)(
  'notifies an accepted early %s exactly once after a polling outage, not history',
  async status => {
    const terminal = { ...job, job_id: 'accepted-id', status, error: status === 'error' ? 'download failed' : null }
    const jobs = [{ ...terminal, job_id: 'unrelated-history' }, terminal]
    api
      .mockResolvedValueOnce({ job_id: 'accepted-id' })
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValue({ jobs })
    await startLocalRuntimeInstall()
    expect($localRuntimeInstallStarting.get()).toBe(true)
    $activeGatewayRoute.set('other')
    setApiRequestProfile('other')
    await vi.advanceTimersByTimeAsync(1400)
    expect(api).toHaveBeenCalledTimes(2)
    $activeGatewayRoute.set('default')
    setApiRequestProfile('default')
    await flush()
    expect($localRuntimeInstallStarting.get()).toBe(false)
    const expected = status === 'error' ? notices.notifyError : notices.notify
    expect(expected).toHaveBeenCalledTimes(1)
    expect(status === 'error' ? notices.notify : notices.notifyError).not.toHaveBeenCalled()
    watchLocalRuntimeJobs()
    await flush()
    $activeGatewayRoute.set('other')
    $activeGatewayRoute.set('default')
    watchLocalRuntimeJobs()
    await flush()
    expect(expected).toHaveBeenCalledTimes(1)
  }
)
it('notifies accepted failure even when a pane read saw it before the POST response', async () => {
  const post = deferred<{ job_id: string }>()
  const failed = { ...job, job_id: 'fast-failure', status: 'error' as const, error: 'download failed' }
  api.mockImplementation(request => (request.method === 'POST' ? post.promise : Promise.resolve({ jobs: [failed] })))
  const starting = startLocalRuntimeInstall()
  watchLocalRuntimeJobs()
  await flush()
  expect(notices.notifyError).not.toHaveBeenCalled()
  post.resolve({ job_id: failed.job_id })
  await starting
  expect(notices.notifyError).toHaveBeenCalledTimes(1)
  watchLocalRuntimeJobs()
  await flush()
  expect(notices.notifyError).toHaveBeenCalledTimes(1)
})
it('isolates registry sources with the same profile and defers reads until activation settles', async () => {
  const original = $connection.get()
  setApiRequestConnection('source-a')
  $connection.set({ ...original, connectionId: 'source-a' } as never)
  api.mockImplementation(request =>
    Promise.resolve(request.method === 'POST' ? { job_id: job.job_id } : { jobs: [job] })
  )
  await startLocalRuntimeInstall()
  setApiRequestConnection('source-b')
  $connection.set({ ...original, connectionId: 'source-b' } as never)
  await flush()
  expect($localRuntimeJobs.get()).toEqual([])
  expect(api).toHaveBeenCalledTimes(2)
  setApiRequestProfile('other')
  setApiRequestConnection('source-a')
  $connection.set({ ...original, connectionId: 'source-a' } as never)
  // The REST tag is assigned by the activation callback AFTER the atoms.
  expect(api).toHaveBeenCalledTimes(2)
  setApiRequestProfile('default')
  await flush()
  expect(api.mock.calls.at(-1)?.[0]).toMatchObject({ connectionId: 'source-a', profile: 'default' })
  expect($localRuntimeJobs.get()).toEqual([job])
  await startLocalRuntimeInstall()
  expect(posts()).toHaveLength(1)
})
it('does not discover unused contexts or automatically resume with the launch flag off', async () => {
  $activeGatewayRoute.set('unused')
  $activeGatewayRoute.set('default')
  await flush()
  expect(api).not.toHaveBeenCalled()
  api.mockImplementation(request =>
    Promise.resolve(request.method === 'POST' ? { job_id: job.job_id } : { jobs: [job] })
  )
  await startLocalRuntimeInstall()
  $localModelsEnabled.set(false)
  $activeGatewayRoute.set('other')
  $activeGatewayRoute.set('default')
  await flush()
  await vi.advanceTimersByTimeAsync(1400)
  expect(api).toHaveBeenCalledTimes(2)
})
it.each(['connection', 'profile'])('reports a POST failure after returning to its owning %s', async context => {
  const post = deferred<{ job_id: string }>()
  api.mockReturnValue(post.promise)
  const starting = startLocalRuntimeInstall()
  const original = $connection.get()

  if (context === 'connection') {
    $connection.set({ mode: 'remote' } as never)
    $connection.set(original)
  } else {
    $activeGatewayRoute.set('other')
    $activeGatewayRoute.set('default')
  }

  post.reject(new Error('installation request failed'))
  await starting
  expect(notices.notifyError).toHaveBeenCalledTimes(1)
  expect($localRuntimeInstallStarting.get()).toBe(false)
  expect(posts()).toHaveLength(1)
})
it('releases a failed POST for retry', async () => {
  api
    .mockRejectedValueOnce(new Error('offline'))
    .mockImplementation(request => Promise.resolve(request.method === 'POST' ? { job_id: 'install' } : { jobs: [job] }))
  await startLocalRuntimeInstall()
  expect($localRuntimeInstallStarting.get()).toBe(false)
  expect(notices.notifyError).toHaveBeenCalledTimes(1)
  await startLocalRuntimeInstall()
  expect(posts()).toHaveLength(2)
  expect($localRuntimeJobs.get()).toEqual([job])
})
it('retries a failed post-acceptance jobs read without inventing progress or allowing another POST', async () => {
  api
    .mockResolvedValueOnce({ job_id: 'install' })
    .mockRejectedValueOnce(new Error('offline'))
    .mockResolvedValue({ jobs: [job] })
  await startLocalRuntimeInstall()
  const lockedAfterFailure = $localRuntimeInstallStarting.get()
  const snapshotAfterFailure = $localRuntimeJobs.get()
  await startLocalRuntimeInstall()
  await vi.advanceTimersByTimeAsync(700)
  expect(lockedAfterFailure).toBe(true)
  expect(snapshotAfterFailure).toEqual([])
  expect(posts()).toHaveLength(1)
  expect($localRuntimeJobs.get()).toEqual([job])
  expect($localRuntimeInstallStarting.get()).toBe(false)
})
it('keeps the shared start locked until the first authoritative jobs read finishes', async () => {
  const read = deferred<{ jobs: LocalRuntimeJob[] }>()
  api.mockImplementation(request => (request.method === 'POST' ? Promise.resolve({ job_id: 'install' }) : read.promise))
  const first = startLocalRuntimeInstall()
  await flush()
  expect($localRuntimeInstallStarting.get()).toBe(true)
  await startLocalRuntimeInstall()
  expect(posts()).toHaveLength(1)
  read.resolve({ jobs: [job] })
  await first
  expect($localRuntimeJobs.get()).toEqual([job])
  expect($localRuntimeInstallStarting.get()).toBe(false)
})
it('serializes pane polling and reads again after POST acceptance before releasing the lock', async () => {
  const post = deferred<{ job_id: string }>()
  const stale = deferred<{ jobs: LocalRuntimeJob[] }>()
  const fresh = deferred<{ jobs: LocalRuntimeJob[] }>()
  let reads = 0
  api.mockImplementation(request =>
    request.method === 'POST' ? post.promise : ++reads === 1 ? stale.promise : fresh.promise
  )
  const first = startLocalRuntimeInstall()
  watchLocalRuntimeJobs()
  post.resolve({ job_id: 'install' })
  await flush()
  watchLocalRuntimeJobs()
  expect(reads).toBe(1)
  stale.resolve({ jobs: [] })
  await flush()
  expect(reads).toBe(2)
  expect($localRuntimeInstallStarting.get()).toBe(true)
  await startLocalRuntimeInstall()
  expect(posts()).toHaveLength(1)
  fresh.resolve({ jobs: [job] })
  await first
  expect($localRuntimeJobs.get()).toEqual([job])
  expect($localRuntimeInstallStarting.get()).toBe(false)
  await vi.advanceTimersByTimeAsync(700)
  expect(reads).toBe(3)
  expect(vi.getTimerCount()).toBe(1)
})
