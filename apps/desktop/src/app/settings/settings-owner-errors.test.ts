import { afterEach, expect, it, vi } from 'vitest'

const notifyError = vi.hoisted(() => vi.fn())

vi.mock('@/store/notifications', () => ({ notifyError }))

import { $settingsOwner } from '@/store/settings-scope'

import { notifySettingsOwnerError } from './index'

afterEach(() => {
  vi.restoreAllMocks()
  notifyError.mockReset()
})

it.each(['Could not export config', 'Could not reset config'])('hides stale-owner rejection: %s', fallback => {
  const owner = { connectionId: 'owner-a', profile: 'default' } as NonNullable<ReturnType<typeof $settingsOwner.get>>

  const replacement = {
    connectionId: 'owner-b',
    profile: 'default'
  } as NonNullable<ReturnType<typeof $settingsOwner.get>>

  vi.spyOn($settingsOwner, 'get').mockReturnValue(replacement)

  notifySettingsOwnerError(owner, new Error('owner-a.invalid rejected the request'), fallback)

  expect(notifyError).not.toHaveBeenCalled()
})

it('reports a rejection while its owner is still current', () => {
  const owner = { connectionId: 'owner-a', profile: 'default' } as NonNullable<ReturnType<typeof $settingsOwner.get>>
  const error = new Error('request failed')

  vi.spyOn($settingsOwner, 'get').mockReturnValue(owner)

  notifySettingsOwnerError(owner, error, 'Could not export config')

  expect(notifyError).toHaveBeenCalledWith(error, 'Could not export config')
})