import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  createCronJob,
  deleteCronJob,
  getCronJob,
  getCronJobRuns,
  getCronJobs,
  pauseCronJob,
  resumeCronJob,
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
})
