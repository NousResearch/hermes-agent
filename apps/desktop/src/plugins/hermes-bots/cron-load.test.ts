/**
 * `loadRoutines` — the pane's one read, and the two things it must survive.
 *
 * 1. Profile scope (#37). cron.manage is scoped to the bot's OWN cron store
 *    through the core RPC's optional `profile` param: a bot's profile can run
 *    a separate gateway, or keep cron in ~/.hermes/profiles/<name>/cron/.
 *    Older gateways ignore the unknown param, so the `[bot:]` tag filter in
 *    selectRoutineJobs stays the fallback.
 * 2. Legacy delegated routines are paused inline before the list returns —
 *    they carry a pre-hardening prompt whose shell command was built by
 *    interpolation. A single pause RPC failing used to reject the WHOLE query:
 *    the pane showed "Could not load cronjobs" over a list that had loaded
 *    fine, and the 20s poll retried the failing pause inside a failing query
 *    forever. Each pause now swallows its own error, and the paused-state
 *    overlay only claims the jobs the gateway actually paused.
 *
 * The RPC boundary is `host.request` rather than `requestForBot`, so the real
 * routing layer (route resolution, error coercion) runs.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RoutineJob } from './types'

const request = vi.fn()

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, host: { ...sdk.host, request } }
})

const { loadRoutines } = await import('./cron')

const LEGACY_PREFIX = 'You are running the scheduled routine "'

/** The shape a pre-hardening routine was persisted with. */
function legacyJob(id: string, title: string, bot: string): RoutineJob {
  return {
    enabled: true,
    job_id: id,
    name: `[bot:${bot}] ${title}`,
    prompt_preview: `${LEGACY_PREFIX}${title}" for agent '${bot}'`,
    state: 'scheduled'
  }
}

/** Every cron.manage call, as `[action, name]`. */
function callLog() {
  return request.mock.calls.map(([, params]) => [params.action, params.name])
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('the bot\u2019s own cron store', () => {
  it('scopes the list and every inline pause to the bot profile', async () => {
    request.mockImplementation(async (_method: string, params: Record<string, unknown>) =>
      params.action === 'list' ? { jobs: [legacyJob('legacy', 'Audit', 'research')] } : { success: true }
    )

    await loadRoutines('research')

    expect(request.mock.calls.map(([method, params]) => [method, params])).toEqual([
      ['cron.manage', { action: 'list', include_disabled: true, profile: 'research' }],
      ['cron.manage', { action: 'pause', name: 'legacy', profile: 'research' }]
    ])
  })

  it('omits the scope entirely when no owner is resolved', async () => {
    request.mockResolvedValue({ jobs: [] })

    await loadRoutines(undefined)

    expect(request).toHaveBeenCalledWith('cron.manage', { action: 'list', include_disabled: true })
  })
})

describe('pausing a legacy delegated routine cannot fail the list', () => {
  it('returns the whole list and overlays only the pauses that landed', async () => {
    const jobs: RoutineJob[] = [
      legacyJob('legacy-fails', 'Audit', 'research'),
      legacyJob('legacy-pauses', 'Build', 'research'),
      {
        enabled: true,
        job_id: 'normal',
        name: '[bot:research] Report',
        prompt: 'Summarize the day',
        state: 'scheduled'
      }
    ]

    request.mockImplementation(async (_method: string, params: Record<string, unknown>) => {
      if (params.action === 'list') {
        return { jobs }
      }

      if (params.name === 'legacy-fails') {
        throw new Error('gateway rejected the pause')
      }

      return { success: true }
    })

    // Pre-fix this rejected with the pause error.
    const result = await loadRoutines('research')

    expect(result.jobs).toHaveLength(3)

    const byId = Object.fromEntries(result.jobs!.map(job => [job.job_id, job]))

    // The failed pause keeps its server state and is retried on the next poll;
    // claiming it as paused would tell the user a job is safe when it is not.
    expect(byId['legacy-fails']).toMatchObject({ enabled: true, state: 'scheduled' })
    expect(byId['legacy-pauses']).toMatchObject({ enabled: false, state: 'paused' })
    expect(byId.normal.enabled).toBe(true)

    expect(callLog()).toEqual([
      ['list', undefined],
      ['pause', 'legacy-fails'],
      ['pause', 'legacy-pauses']
    ])
  })

  it('still resolves with the list when every pause fails', async () => {
    request.mockImplementation(async (_method: string, params: Record<string, unknown>) => {
      if (params.action === 'list') {
        return { jobs: [legacyJob('only', 'Watch', 'ops')] }
      }

      throw new Error('gateway down')
    })

    const result = await loadRoutines('ops')

    expect(result.jobs![0]).toMatchObject({ enabled: true, job_id: 'only' })
  })
})

describe('the security pause is one-shot, not a poll loop', () => {
  it('pauses the persisted routine once and leaves the rest of the record intact', async () => {
    const persisted: RoutineJob = {
      ...legacyJob('legacy-job', 'Audit', 'research'),
      repeat: { completed: 1, times: 3 } as unknown as string,
      schedule: 'every 2h'
    }

    request.mockImplementation(async (_method: string, params: Record<string, unknown>) => {
      if (params.action === 'list') {
        return { jobs: [persisted] }
      }

      persisted.enabled = false
      persisted.state = 'paused'

      return { success: true }
    })

    const first = await loadRoutines('research')

    expect(first.jobs![0]).toMatchObject({
      enabled: false,
      job_id: 'legacy-job',
      repeat: { completed: 1, times: 3 },
      schedule: 'every 2h',
      state: 'paused'
    })

    request.mockClear()
    await loadRoutines('research')

    // Already paused: the next poll must not re-attempt the pause forever.
    expect(callLog()).toEqual([['list', undefined]])
  })
})
