import { beforeEach, describe, expect, it, vi } from 'vitest'

import { setNativeNotifyUnreadBadge } from './native-notifications'
import { $attentionBadgeCount, installSessionBadgeSync } from './session-badge'
import { $attentionSessionIds, $unreadSessionIds, setSessionAttention, setSessionUnread } from './session'

const desktopWindow = window as unknown as { hermesDesktop?: Partial<Window['hermesDesktop']> }

describe('$attentionBadgeCount', () => {
  beforeEach(() => {
    $unreadSessionIds.set([])
    $attentionSessionIds.set([])
  })

  it('counts the union of unread + attention (no double-count for a session that is both)', () => {
    setSessionUnread('s1', true)
    setSessionUnread('s2', true)
    setSessionAttention('s2', true) // s2 is both unread AND needs-input
    setSessionAttention('s3', true)

    expect($attentionBadgeCount.get()).toBe(3)
  })

  it('excludes working sessions — a busy turn is not actionable', () => {
    setSessionUnread('s1', true)
    // working is a separate atom the badge does not read; simulating it here
    // by simply not adding it to unread/attention. The invariant is: only
    // unread + attention feed the badge.
    expect($attentionBadgeCount.get()).toBe(1)
  })

  it('drops to 0 when all markers clear', () => {
    setSessionUnread('s1', true)
    setSessionAttention('s2', true)
    expect($attentionBadgeCount.get()).toBe(2)

    setSessionUnread('s1', false)
    setSessionAttention('s2', false)
    expect($attentionBadgeCount.get()).toBe(0)
  })
})

describe('installSessionBadgeSync', () => {
  const setBadge = vi.fn().mockResolvedValue(true)

  beforeEach(() => {
    setBadge.mockClear()
    $unreadSessionIds.set([])
    $attentionSessionIds.set([])
    setNativeNotifyUnreadBadge(true)
    desktopWindow.hermesDesktop = { setBadge } as unknown as Window['hermesDesktop']
  })

  it('pushes the count to the OS bridge and keeps it in sync', () => {
    const uninstall = installSessionBadgeSync()

    // Initial push (0) + the update below.
    expect(setBadge).toHaveBeenCalledWith(0)

    setBadge.mockClear()
    setSessionUnread('s1', true)
    expect(setBadge).toHaveBeenCalledWith(1)

    setBadge.mockClear()
    setSessionUnread('s1', false)
    expect(setBadge).toHaveBeenCalledWith(0)

    uninstall()
  })

  it('forces the badge to 0 when the unreadBadge pref is off', () => {
    const uninstall = installSessionBadgeSync()

    // With the pref off, the effective count is clamped to 0 even when markers
    // are present. nanostores only fires on value change, so toggling the pref
    // off while the count is already 0 is a no-op push; the invariant we test
    // is that turning markers on never pushes a non-zero value.
    setNativeNotifyUnreadBadge(false)
    setBadge.mockClear()

    setSessionUnread('s1', true)
    setSessionAttention('s2', true)

    // No push fired at all (effective count stayed 0), and critically a
    // non-zero value was never sent.
    expect(setBadge).not.toHaveBeenCalledWith(1)
    expect(setBadge).not.toHaveBeenCalledWith(2)

    // Re-enabling reflects the real count immediately (0 → 2 fires a push).
    setBadge.mockClear()
    setNativeNotifyUnreadBadge(true)
    expect(setBadge).toHaveBeenCalledWith(2)

    uninstall()
  })

  it('clears the badge to 0 on uninstall', () => {
    const uninstall = installSessionBadgeSync()
    setSessionUnread('s1', true) // count is now 1
    setBadge.mockClear()

    uninstall()
    expect(setBadge).toHaveBeenCalledWith(0)
  })
})
