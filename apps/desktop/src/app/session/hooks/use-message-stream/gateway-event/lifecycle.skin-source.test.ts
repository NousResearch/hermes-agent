import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $backendCustomCSS, __resetBackendSkinSync } from '@/themes/backend-sync'

import { handleLifecycleEvent } from './lifecycle'
import type { GatewayEventContext } from './types'

const ready = (connectionId: string, css: string, active: boolean): GatewayEventContext => {
  const payload = { skin: { customCSS: css, name: 'default' } }

  return {
    deps: {} as GatewayEventContext['deps'],
    event: { connectionId, payload, profile: 'default', type: 'gateway.ready' } as GatewayEventContext['event'],
    explicitSid: '',
    fromActiveSource: () => active,
    isActiveEvent: false,
    occurredAt: 0,
    payload: payload as GatewayEventContext['payload'],
    scheduleConfigRefresh: vi.fn(),
    sessionId: null
  }
}

describe('gateway.ready skin per source', () => {
  beforeEach(() => __resetBackendSkinSync())

  it('an inactive source does not overwrite the active source customCSS', () => {
    handleLifecycleEvent(ready('personal', '.p{}', true))
    handleLifecycleEvent(ready('work', '.w{}', false))

    expect($backendCustomCSS.get().nous).toBe('.p{}')
  })
})
