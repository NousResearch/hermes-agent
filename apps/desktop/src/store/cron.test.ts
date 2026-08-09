import { beforeEach, describe, expect, it } from 'vitest'

import { $activeGatewayProfile, setShowAllProfiles } from '@/store/profile'
import type { CronJob } from '@/types/hermes'

import {
  $cronCache,
  cronCacheRequestForScope,
  cronJobsForScope,
  invalidateCronCache,
  updateCronCacheForScope
} from './cron'

const job = (id: string): CronJob => ({ id, name: id }) as CronJob

describe('cron profile cache', () => {
  beforeEach(() => {
    setShowAllProfiles(false)
    $activeGatewayProfile.set('default')
    invalidateCronCache()
  })

  it('keeps an old-profile fetch from replacing the visible profile', () => {
    $activeGatewayProfile.set('work')
    const workRequest = cronCacheRequestForScope('work')
    $activeGatewayProfile.set('personal')
    const personalRequest = cronCacheRequestForScope('personal')

    expect(updateCronCacheForScope(personalRequest, () => [job('personal-job')])).toBe(true)
    expect(updateCronCacheForScope(workRequest, () => [job('work-job')])).toBe(false)

    expect(cronJobsForScope('personal')).toEqual([job('personal-job')])
    expect($cronCache.get().work).toBeUndefined()
  })

  it('rejects an older same-profile fetch after a newer request starts', () => {
    $activeGatewayProfile.set('work')
    const older = cronCacheRequestForScope('work')
    const newer = cronCacheRequestForScope('work')

    expect(updateCronCacheForScope(newer, () => [job('new')])).toBe(true)
    expect(updateCronCacheForScope(older, () => [job('old')])).toBe(false)
    expect(cronJobsForScope('work')).toEqual([job('new')])
  })

  it('rejects a completion from before a same-named gateway wipe', () => {
    $activeGatewayProfile.set('work')
    const beforeWipe = cronCacheRequestForScope('work')

    invalidateCronCache()
    const afterWipe = cronCacheRequestForScope('work')
    expect(updateCronCacheForScope(afterWipe, () => [job('new-gateway')])).toBe(true)
    expect(updateCronCacheForScope(beforeWipe, () => [job('old-gateway')])).toBe(false)
    expect(cronJobsForScope('work')).toEqual([job('new-gateway')])
  })
})
