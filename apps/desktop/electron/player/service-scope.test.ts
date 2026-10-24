import { describe, expect, it, vi } from 'vitest'

import { resetServiceScopedState, type ServiceScopedWindow } from './service-scope'

describe('provider switch cleanup', () => {
  it('closes prior-provider auth windows and forgets stored permission approvals', () => {
    const active = { isDestroyed: () => false, destroy: vi.fn() }
    const alreadyClosed = { isDestroyed: () => true, destroy: vi.fn() }
    const windows = new Set<ServiceScopedWindow>([active, alreadyClosed])
    const permissionDecisions = new Map([['mediaKeySystem:https://www.youtube.com', true]])

    resetServiceScopedState(windows, permissionDecisions)

    expect(active.destroy).toHaveBeenCalledOnce()
    expect(alreadyClosed.destroy).not.toHaveBeenCalled()
    expect(windows.size).toBe(0)
    expect(permissionDecisions.size).toBe(0)
  })
})
