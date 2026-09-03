import { beforeEach, describe, expect, it, vi } from 'vitest'

describe('sticky user messages store', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.resetModules()
  })

  it('defaults to enabled when no stored value exists', async () => {
    const { $stickyUserMessagesEnabled } = await import('./sticky-user-messages')
    expect($stickyUserMessagesEnabled.get()).toBe(true)
  })

  it('respects stored false value', async () => {
    localStorage.setItem('hermes.desktop.stickyUserMessages.v1', 'false')
    const { $stickyUserMessagesEnabled } = await import('./sticky-user-messages')
    expect($stickyUserMessagesEnabled.get()).toBe(false)
  })

  it('respects stored true value', async () => {
    localStorage.setItem('hermes.desktop.stickyUserMessages.v1', 'true')
    const { $stickyUserMessagesEnabled } = await import('./sticky-user-messages')
    expect($stickyUserMessagesEnabled.get()).toBe(true)
  })

  it('treats unexpected stored value as disabled (storedBoolean fallback)', async () => {
    localStorage.setItem('hermes.desktop.stickyUserMessages.v1', 'corrupted')
    const { $stickyUserMessagesEnabled } = await import('./sticky-user-messages')
    // storedBoolean returns value === 'true', so non-boolean strings are false
    expect($stickyUserMessagesEnabled.get()).toBe(false)
  })

  it('persists changes to localStorage', async () => {
    const { $stickyUserMessagesEnabled, setStickyUserMessagesEnabled } = await import('./sticky-user-messages')
    setStickyUserMessagesEnabled(false)
    expect(localStorage.getItem('hermes.desktop.stickyUserMessages.v1')).toBe('false')
    expect($stickyUserMessagesEnabled.get()).toBe(false)

    setStickyUserMessagesEnabled(true)
    expect(localStorage.getItem('hermes.desktop.stickyUserMessages.v1')).toBe('true')
    expect($stickyUserMessagesEnabled.get()).toBe(true)
  })
})
