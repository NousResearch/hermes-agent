import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  createCronJob,
  deleteCronJob,
  getCronJob,
  getCronJobOutput,
  getCronJobOutputs,
  getCronJobRuns,
  getCronJobs,
  pauseCronJob,
  resumeCronJob,
  setApiRequestConnection,
  setApiRequestProfile,
  triggerCronJob,
  updateCronJob
} from './hermes'

// Contract: every cron helper must carry the active gateway profile, so a
// multi-profile / remote user's cron list, runs, and mutations hit the backend
// they're actually on — not the primary/default. Without it, selecting a remote
// profile still showed the local primary's jobs (the "remote cron jobs don't
// show up" bug), the counterpart to the backend-action-helper fix in
// hermes-profile-scope.test.ts.
describe('cron helpers are profile-scoped', () => {
  const api = vi.fn(async (_req: { path: string; profile?: string }) => ({}) as never)

  beforeEach(() => {
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { api }
    api.mockClear()
  })

  afterEach(() => {
    setApiRequestProfile(null)
    setApiRequestConnection(null)
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  const lastProfile = () => api.mock.calls.at(-1)?.[0].profile

  it('omits profile when none is active (single-profile users unaffected)', () => {
    void getCronJobs()
    expect(lastProfile()).toBeUndefined()
  })

  it('forwards the active profile to every cron helper', () => {
    setApiRequestProfile('coder')

    void getCronJobs()
    void getCronJob('job-1')
    void getCronJobOutputs('job-1')
    void getCronJobOutput('job-1', '2026-08-11_09-00-00')
    void getCronJobRuns('job-1')
    void createCronJob({ name: 'nightly', prompt: 'run', schedule: '0 3 * * *' } as never)
    void updateCronJob('job-1', { enabled: false } as never)
    void pauseCronJob('job-1')
    void resumeCronJob('job-1')
    void triggerCronJob('job-1')
    void deleteCronJob('job-1')

    for (const call of api.mock.calls) {
      expect(call[0].profile).toBe('coder')
    }
  })

  it('omits connectionId when the local pool serves the active gateway', () => {
    void getCronJobRuns('job-1')
    expect(api.mock.calls.at(-1)?.[0]).not.toHaveProperty('connectionId')
  })

  // Contract: with a registered gateway connection active, cron run sessions
  // live in THAT gateway's state.db — not in any local profile's. Every cron
  // helper must tag the owning connection so the main process routes the REST
  // call to the same backend the job list (and its runs) actually live on.
  // Without it, run history read a local state.db with zero cron rows and
  // every job showed "No runs yet" (#87882).
  it('forwards the active registry connection to every cron helper', () => {
    setApiRequestProfile('research')
    setApiRequestConnection('gw-tailscale')

    void getCronJobs('research')
    void getCronJob('job-1')
    void getCronJobRuns('job-1')
    void createCronJob({ name: 'nightly', prompt: 'run', schedule: '0 3 * * *' } as never)
    void updateCronJob('job-1', { enabled: false } as never)
    void pauseCronJob('job-1')
    void resumeCronJob('job-1')
    void triggerCronJob('job-1')
    void deleteCronJob('job-1')

    for (const call of api.mock.calls) {
      expect((call[0] as { connectionId?: string }).connectionId).toBe('gw-tailscale')
      expect(call[0].profile).toBe('research')
    }
  })

  it('list accepts an explicit ?profile= for endpoint-level filtering', () => {
    // profileScoped() routes the backend process; the list endpoint ALSO
    // aggregates 'all' by default, so callers pass an explicit profile to
    // filter what the endpoint returns (sidebar / cron overlay scoping).
    void getCronJobs('worker_alpha')
    expect(api.mock.calls.at(-1)?.[0].path).toBe('/api/cron/jobs?profile=worker_alpha')

    void getCronJobs('all')
    expect(api.mock.calls.at(-1)?.[0].path).toBe('/api/cron/jobs?profile=all')

    // Omitting the arg keeps the legacy unfiltered path.
    void getCronJobs()
    expect(api.mock.calls.at(-1)?.[0].path).toBe('/api/cron/jobs')
  })

  it('keeps the gateway profile separate from a run output owner profile', () => {
    setApiRequestProfile('remote_gateway')

    void getCronJobRuns('job-1', 5, 'worker_alpha')
    expect(api.mock.calls.at(-1)?.[0]).toMatchObject({
      path: '/api/cron/jobs/job-1/runs?limit=5&profile=worker_alpha',
      profile: 'remote_gateway'
    })

    void getCronJobOutputs('job-1', 5, 'worker_alpha')
    expect(api.mock.calls.at(-1)?.[0]).toMatchObject({
      path: '/api/cron/jobs/job-1/outputs?limit=5&profile=worker_alpha',
      profile: 'remote_gateway'
    })

    void getCronJobOutput('job-1', '2026-08-11_09-00-00', 'worker_alpha')
    expect(api.mock.calls.at(-1)?.[0]).toMatchObject({
      path: '/api/cron/jobs/job-1/outputs/2026-08-11_09-00-00?profile=worker_alpha',
      profile: 'remote_gateway'
    })
  })

  it('passes an explicit owner profile to every job mutation endpoint', () => {
    const payload = { name: 'nightly', prompt: 'run', schedule: '0 3 * * *' } as never

    void createCronJob(payload, 'worker_alpha')
    expect(api.mock.calls.at(-1)?.[0].path).toBe('/api/cron/jobs?profile=worker_alpha')

    void updateCronJob('shared-job', { enabled: false } as never, 'worker_alpha')
    expect(api.mock.calls.at(-1)?.[0].path).toBe('/api/cron/jobs/shared-job?profile=worker_alpha')

    void pauseCronJob('shared-job', 'worker_alpha')
    expect(api.mock.calls.at(-1)?.[0].path).toBe('/api/cron/jobs/shared-job/pause?profile=worker_alpha')

    void resumeCronJob('shared-job', 'worker_alpha')
    expect(api.mock.calls.at(-1)?.[0].path).toBe('/api/cron/jobs/shared-job/resume?profile=worker_alpha')

    void triggerCronJob('shared-job', 'worker_alpha')
    expect(api.mock.calls.at(-1)?.[0].path).toBe('/api/cron/jobs/shared-job/trigger?profile=worker_alpha')

    void deleteCronJob('shared-job', 'worker_alpha')
    expect(api.mock.calls.at(-1)?.[0].path).toBe('/api/cron/jobs/shared-job?profile=worker_alpha')
  })
})
