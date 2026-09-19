import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { GatewayRequest } from '@/app/session/hooks/use-prompt-actions/utils'
import { translateNow } from '@/i18n'
import { $notifications, dismissNotification } from '@/store/notifications'

import { ensureSetupProfile, SETUP_PROFILE } from './setup-profile'

const gateway = vi.hoisted(() => ({ connectionId: 'remote-a' as string | null }))
vi.mock('@/store/gateway', () => ({ activeGatewayConnectionId: () => gateway.connectionId }))

beforeEach(() => { gateway.connectionId = 'remote-a' })
afterEach(() => {
  for (const notice of $notifications.get()) dismissNotification(notice.id)
})

it('reports clone auth for the created guide on its captured connection without navigating', async () => {
  const location = window.location.href
  let resolve!: (value: unknown) => void
  const reply = new Promise(done => { resolve = done })
  const request = vi.fn(() => reply) as unknown as GatewayRequest
  const pending = ensureSetupProfile(request)

  expect(request).toHaveBeenCalledWith('profiles.create', expect.objectContaining({
    name: SETUP_PROFILE, clone_from: 'default', no_alias: true
  }))
  gateway.connectionId = 'remote-b'
  resolve({ name: 'canonical-guide', clone_needs_auth: ['honcho', 'openviking'] })
  await pending

  expect($notifications.get()).toEqual([expect.objectContaining({
    kind: 'info',
    message: `canonical-guide: ${translateNow('profiles.cloneNeedsAuth', 'honcho, openviking')}`,
    meta: 'remote-a'
  })])
  expect($notifications.get()[0].action).toBeUndefined()
  expect($notifications.get()[0].secondaryAction).toBeUndefined()
  expect(window.location.href).toBe(location)
})

it('keeps legacy, empty-auth and existing guide results quiet while propagating real failures', async () => {
  for (const reply of [undefined, {}, { clone_needs_auth: [] }]) {
    await ensureSetupProfile(vi.fn().mockResolvedValue(reply))
  }
  await ensureSetupProfile(vi.fn().mockRejectedValue(new Error('Profile already exists')))
  await expect(ensureSetupProfile(vi.fn().mockRejectedValue(new Error('Connection refused'))))
    .rejects.toThrow('Connection refused')
  expect($notifications.get()).toEqual([])
})
