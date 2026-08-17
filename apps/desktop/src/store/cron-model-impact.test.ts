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

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(res => {
    resolve = res
  })

  return { promise, resolve }
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

  it('surfaces a confirm prompt for guard-blocked assignments and retries with acknowledgment when accepted', async () => {
    setModelAssignment.mockResolvedValueOnce(response(positive()))
    await setMainModelAssignment({ provider: 'nous', model: 'one' })

    // First attempt trips the guard: backend answers ok:false + confirm_required
    // + confirm_message and does NOT persist. The desktop must not treat this
    // as a plain error — it must ask the user and retry with
    // confirm_expensive_model: true (mirrors the CLI's [y/N] prompt).
    const guardResponse = {
      ok: false,
      scope: 'main',
      provider: 'openrouter',
      model: 'openai/gpt-5.5-pro',
      confirm_required: true,
      confirm_message: 'Confirm this expensive model.'
    } satisfies ModelAssignmentResponse

    setModelAssignment.mockResolvedValueOnce(guardResponse)
    const pending = setMainModelAssignment({ provider: 'openrouter', model: 'openai/gpt-5.5-pro' })

    // The confirm prompt is a notification with an action; while it is open
    // the assignment must still be pending (not resolved/rejected yet).
    await vi.waitFor(() => {
      const n = $notifications.get().find(n => n.id.startsWith('model-warning-confirm-'))
      expect(n?.message).toBe('Confirm this expensive model.')
    })
    const confirmNotification = $notifications.get().find(n => n.id.startsWith('model-warning-confirm-'))

    // Accepting re-sends with the acknowledgment flag and persists.
    setModelAssignment.mockResolvedValueOnce(response(positive()))
    confirmNotification?.action?.onClick()
    await pending
    expect(setModelAssignment).toHaveBeenLastCalledWith(
      expect.objectContaining({
        provider: 'openrouter',
        model: 'openai/gpt-5.5-pro',
        scope: 'main',
        confirm_expensive_model: true
      })
    )
  })

  it('rejects (declines) guard-blocked assignments with a neutral message when the user dismisses', async () => {
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

    const pending = setMainModelAssignment({ provider: 'openrouter', model: 'openai/gpt-5.5-pro' })
    await vi.waitFor(() => {
      expect($notifications.get().some(n => n.id.startsWith('model-warning-confirm-'))).toBe(true)
    })
    const confirmNotification = $notifications.get().find(n => n.id.startsWith('model-warning-confirm-'))
    confirmNotification?.onDismiss?.()

    await expect(pending).rejects.toThrow()
    expect(setModelAssignment).not.toHaveBeenCalledWith(
      expect.objectContaining({ confirm_expensive_model: true })
    )
    // The cron impact notification from the first call is still there;
    // no new error notification with the giant warning text should exist.
    expect($notifications.get().filter(n => n.kind === 'error').length).toBe(0)
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
