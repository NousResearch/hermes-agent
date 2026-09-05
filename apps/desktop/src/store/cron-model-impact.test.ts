import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $cronReviewRequest } from '@/store/cron'
import { $notifications, clearNotifications, dismissNotification } from '@/store/notifications'
import type { ModelAssignmentResponse } from '@/types/hermes'

const setModelAssignment = vi.fn()
const getApiRequestProfile = vi.fn<() => string | null>(() => 'default')

vi.mock('@/hermes', () => ({
  setModelAssignment: (...args: unknown[]) => setModelAssignment(...args),
  getApiRequestProfile: () => getApiRequestProfile()
}))

import {
  CRON_MODEL_IMPACT_NOTIFICATION_ID,
  invalidateCronModelImpactScope,
  setMainModelAssignment
} from '@/store/cron-model-impact'

import { deferred } from '../test/deferred'

function response(impact: ModelAssignmentResponse['cron_model_impact']): ModelAssignmentResponse {
  return {
    ok: true,
    scope: 'main',
    provider: 'nous',
    model: 'new/model',
    cron_model_impact: impact
  }
}

function positive(name = 'Morning summary'): ModelAssignmentResponse['cron_model_impact'] {
  return {
    available: true,
    guard_enabled: true,
    affected_count: 1,
    truncated: false,
    jobs: [{ id: 'job-1', name, drifted_axes: ['provider', 'model'] }]
  }
}

beforeEach(() => {
  setModelAssignment.mockReset()
  getApiRequestProfile.mockReset()
  getApiRequestProfile.mockReturnValue('default')
  clearNotifications()
  invalidateCronModelImpactScope({ clearNotification: false })
})

describe('setMainModelAssignment', () => {
  it('shows one consumer warning and routes via a read-only review action', async () => {
    setModelAssignment.mockResolvedValue(response(positive()))
    const requestCount = $cronReviewRequest.get()

    await setMainModelAssignment({ provider: 'nous', model: 'new/model' })

    expect(setModelAssignment).toHaveBeenCalledWith({
      scope: 'main',
      provider: 'nous',
      model: 'new/model'
    })
    const notification = $notifications.get().find(item => item.id === CRON_MODEL_IMPACT_NOTIFICATION_ID)
    expect(notification?.kind).toBe('warning')
    expect(notification?.title).toBe('Scheduled jobs need review')
    expect(notification?.message).toContain('1 scheduled job will be skipped')
    expect(notification?.detail).toContain('Morning summary')
    expect(notification?.action?.label).toBe('Review scheduled jobs')

    notification?.action?.onClick()
    expect($cronReviewRequest.get()).toBe(requestCount + 1)
    expect(setModelAssignment).toHaveBeenCalledTimes(1)
  })

  it('ignores malformed untrusted impact data', async () => {
    setModelAssignment.mockResolvedValue(
      response({
        available: true,
        guard_enabled: true,
        affected_count: 2,
        truncated: false,
        jobs: [{ id: 'job-1', name: 'One', drifted_axes: ['provider'] }]
      })
    )

    await setMainModelAssignment({ provider: 'nous', model: 'new/model' })

    expect($notifications.get()).toEqual([])
  })

  it('keeps an existing warning for an older backend but clears it on explicit zero impact', async () => {
    setModelAssignment.mockResolvedValueOnce(response(positive()))
    await setMainModelAssignment({ provider: 'nous', model: 'one' })
    expect($notifications.get()).toHaveLength(1)

    setModelAssignment.mockResolvedValueOnce(response(undefined))
    await setMainModelAssignment({ provider: 'nous', model: 'two' })
    expect($notifications.get()).toHaveLength(1)
    const retainedAction = $notifications.get()[0].action
    const reviewCount = $cronReviewRequest.get()
    retainedAction?.onClick()
    expect($cronReviewRequest.get()).toBe(reviewCount + 1)

    setModelAssignment.mockResolvedValueOnce(
      response({
        available: true,
        guard_enabled: true,
        affected_count: 0,
        truncated: false,
        jobs: []
      })
    )
    await setMainModelAssignment({ provider: 'nous', model: 'three' })
    expect($notifications.get()).toEqual([])
  })

  it('rejects non-persisted confirmation outcomes without changing impact state', async () => {
    setModelAssignment.mockResolvedValueOnce(response(positive()))
    await setMainModelAssignment({ provider: 'nous', model: 'one' })

    setModelAssignment.mockResolvedValueOnce({
      ok: false,
      scope: 'main',
      provider: 'openrouter',
      model: 'openai/gpt-5.5-pro',
      confirm_required: true,
      confirm_message: 'Confirm this expensive model.'
    } satisfies ModelAssignmentResponse)

    await expect(setMainModelAssignment({ provider: 'openrouter', model: 'openai/gpt-5.5-pro' })).rejects.toThrow(
      'Confirm this expensive model.'
    )
    expect($notifications.get()).toHaveLength(1)
    const action = $notifications.get()[0].action
    const reviewCount = $cronReviewRequest.get()
    action?.onClick()
    expect($cronReviewRequest.get()).toBe(reviewCount + 1)
  })

  it('publishes only the latest same-profile assignment when responses reverse', async () => {
    const first = deferred<ModelAssignmentResponse>()
    const second = deferred<ModelAssignmentResponse>()
    setModelAssignment.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)

    const firstCall = setMainModelAssignment({ provider: 'nous', model: 'first' })
    const secondCall = setMainModelAssignment({ provider: 'nous', model: 'second' })
    second.resolve(response(positive('Second job')))
    await secondCall
    first.resolve(response(positive('Stale first job')))
    await firstCall

    const notification = $notifications.get()[0]
    expect(notification.detail).toContain('Second job')
    expect(notification.detail).not.toContain('Stale first job')
  })

  it('invalidates pending responses and action closures on profile or connection changes', async () => {
    const pending = deferred<ModelAssignmentResponse>()
    setModelAssignment.mockReturnValueOnce(pending.promise)
    const call = setMainModelAssignment({ provider: 'nous', model: 'pending' })

    invalidateCronModelImpactScope()
    pending.resolve(response(positive('Stale job')))
    await call
    expect($notifications.get()).toEqual([])

    setModelAssignment.mockResolvedValueOnce(response(positive('Current job')))
    await setMainModelAssignment({ provider: 'nous', model: 'current' })
    const action = $notifications.get()[0].action
    const requestCount = $cronReviewRequest.get()
    getApiRequestProfile.mockReturnValue('other')
    invalidateCronModelImpactScope({ clearNotification: false })
    action?.onClick()
    expect($cronReviewRequest.get()).toBe(requestCount)
  })

  it('clears an obsolete warning when the drift guard is disabled', async () => {
    setModelAssignment.mockResolvedValueOnce(response(positive()))
    await setMainModelAssignment({ provider: 'nous', model: 'one' })
    expect($notifications.get()).toHaveLength(1)

    setModelAssignment.mockResolvedValueOnce(
      response({ available: true, guard_enabled: false, affected_count: 0, truncated: false, jobs: [] })
    )
    await setMainModelAssignment({ provider: 'nous', model: 'two' })

    expect($notifications.get()).toEqual([])
  })

  it('does not let a dismissed notification mutate cron configuration', async () => {
    setModelAssignment.mockResolvedValue(response(positive()))
    await setMainModelAssignment({ provider: 'nous', model: 'new/model' })

    dismissNotification(CRON_MODEL_IMPACT_NOTIFICATION_ID)

    expect($notifications.get()).toEqual([])
    expect(setModelAssignment).toHaveBeenCalledTimes(1)
  })
})
