import { expect, test, vi } from 'vitest'

import {
  createSpecialistDispatchQuiesce,
  SpecialistProfileQuiescedError
} from './specialist-dispatch-quiesce'

test('a lease marks exact profiles before owner teardown and release restores them', async () => {
  let quiesce!: ReturnType<typeof createSpecialistDispatchQuiesce>
  const stopProfile = vi.fn(async (profile: string) => {
    expect(quiesce.isQuiesced(profile)).toBe(true)
  })
  quiesce = createSpecialistDispatchQuiesce({ stopProfile })

  const lease = await quiesce.acquire(['explorer', 'public'])

  expect(stopProfile).toHaveBeenCalledTimes(2)
  expect(stopProfile.mock.calls.map(([profile]) => profile)).toEqual(['explorer', 'public'])
  expect(quiesce.isQuiesced('explorer')).toBe(true)
  expect(() => quiesce.assertCanStart('public')).toThrow(SpecialistProfileQuiescedError)
  // The lease tears down only its exact profiles, but must not let a different
  // renderer-owned profile reconnect and refill the fixed pool before the
  // explicit runner reserves capacity.
  expect(quiesce.isQuiesced('statistician')).toBe(false)
  expect(() => quiesce.assertCanStart('statistician')).toThrow(SpecialistProfileQuiescedError)
  expect(quiesce.release(lease.leaseId)).toEqual({ outcome: 'released', profileIds: ['explorer', 'public'] })
  expect(quiesce.isQuiesced('explorer')).toBe(false)
  expect(() => quiesce.assertCanStart('public')).not.toThrow()
  expect(() => quiesce.assertCanStart('statistician')).not.toThrow()
})

test('conflicts and teardown failure leave no stale quiesce authority', async () => {
  const stopProfile = vi.fn(async (profile: string) => {
    if (profile === 'public') {
      throw new Error('stop failed')
    }
  })
  const quiesce = createSpecialistDispatchQuiesce({ stopProfile })

  await expect(quiesce.acquire(['explorer', 'public'])).rejects.toMatchObject({ code: 'QUIESCE_STOP_FAILED' })
  expect(quiesce.isQuiesced('explorer')).toBe(false)
  expect(quiesce.isQuiesced('public')).toBe(false)

  const lease = await quiesce.acquire(['statistician'])
  await expect(quiesce.acquire(['explorer'])).rejects.toMatchObject({ code: 'QUIESCE_CONFLICT' })
  expect(quiesce.release('specialist-quiesce-missing')).toEqual({ outcome: 'rejected', reasonCode: 'UNKNOWN_QUIESCE_LEASE' })
  expect(quiesce.release(lease.leaseId)).toEqual({ outcome: 'released', profileIds: ['statistician'] })
})
