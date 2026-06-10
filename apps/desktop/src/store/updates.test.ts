import { afterEach, describe, expect, it, vi } from 'vitest'

import { $notifications } from './notifications'
import { $updateApply, $updateStatus, maybeNotifyUpdateAvailable } from './updates'

vi.mock('@/i18n', () => ({
  translateNow: (key: string, ...args: unknown[]) => `${key}:${args.join(',')}`
}))

describe('maybeNotifyUpdateAvailable', () => {
  afterEach(() => {
    $notifications.set([])
    $updateStatus.set(null)
    $updateApply.set({ applying: false, stage: 'idle', message: '', percent: null, error: null, command: null, log: [] })
  })

  it('dismisses update toast when behind drops to 0', () => {
    // First: simulate an available update to create the toast
    const statusBehind = {
      behind: 5,
      branch: 'main',
      commits: [{ sha: 'abc123', message: 'test', date: '2026-01-01' }],
      supported: true,
      targetSha: 'abc123'
    } as any
    maybeNotifyUpdateAvailable(statusBehind)
    expect($notifications.get().some(n => n.id === 'desktop-update-available')).toBe(true)

    // Now: simulate the backend catching up (behind === 0)
    const statusCurrent = {
      behind: 0,
      branch: 'main',
      commits: [],
      supported: true,
      targetSha: 'abc123'
    } as any
    maybeNotifyUpdateAvailable(statusCurrent)

    // Toast should be dismissed
    expect($notifications.get().some(n => n.id === 'desktop-update-available')).toBe(false)
  })

  it('does not create a toast when behind is 0', () => {
    const status = {
      behind: 0,
      branch: 'main',
      commits: [],
      supported: true,
      targetSha: 'abc123'
    } as any
    maybeNotifyUpdateAvailable(status)
    expect($notifications.get().some(n => n.id === 'desktop-update-available')).toBe(false)
  })

  it('creates a toast when behind > 0', () => {
    const status = {
      behind: 10,
      branch: 'main',
      commits: [{ sha: 'abc123', message: 'test', date: '2026-01-01' }],
      supported: true,
      targetSha: 'abc123'
    } as any
    maybeNotifyUpdateAvailable(status)
    expect($notifications.get().some(n => n.id === 'desktop-update-available')).toBe(true)
  })

  it('does nothing when status is null', () => {
    maybeNotifyUpdateAvailable(null)
    expect($notifications.get()).toHaveLength(0)
  })

  it('does nothing when status has error', () => {
    const status = {
      behind: 5,
      branch: 'main',
      commits: [],
      error: 'check-failed',
      supported: true,
      targetSha: 'abc123'
    } as any
    maybeNotifyUpdateAvailable(status)
    expect($notifications.get()).toHaveLength(0)
  })
})
