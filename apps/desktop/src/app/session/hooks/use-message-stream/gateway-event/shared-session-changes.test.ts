import { afterEach, describe, expect, it, vi } from 'vitest'

import * as gateway from '@/store/gateway'
import { $cronChangeTick, $sessionsChangeTick } from '@/store/live-sync'

import { handleLifecycleEvent } from './lifecycle'
import type { GatewayEventContext } from './types'

function event(connectionId: string, type = 'sessions.changed'): GatewayEventContext {
  return { event: { type, connectionId, profile: 'default' }, fromActiveSource: () => false } as GatewayEventContext
}

afterEach(() => vi.restoreAllMocks())
describe('shared gateway session notifications', () => {
  it('refreshes a named-profile view for a default-tagged event on the same connection', () => {
    vi.spyOn(gateway, 'activeGatewayConnectionId').mockReturnValue('shared-host')
    const before = $sessionsChangeTick.get()
    handleLifecycleEvent(event('shared-host'))
    expect($sessionsChangeTick.get()).toBe(before + 1)
  })
  it('does not refresh the active host for another connection', () => {
    vi.spyOn(gateway, 'activeGatewayConnectionId').mockReturnValue('shared-host')
    const before = $sessionsChangeTick.get()
    handleLifecycleEvent(event('other-host'))
    expect($sessionsChangeTick.get()).toBe(before)
  })
  it('keeps profile-local events profile scoped', () => {
    vi.spyOn(gateway, 'activeGatewayConnectionId').mockReturnValue('shared-host')
    const before = $cronChangeTick.get()
    handleLifecycleEvent(event('shared-host', 'cron.changed'))
    expect($cronChangeTick.get()).toBe(before)
  })
})
