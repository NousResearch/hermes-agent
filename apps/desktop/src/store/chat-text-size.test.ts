import { beforeEach, describe, expect, it, vi } from 'vitest'

const loadStore = async () => {
  vi.resetModules()

  return import('./chat-text-size')
}

const cssVar = (name: string) => window.document.documentElement.style.getPropertyValue(name)

describe('chat text size preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.document.documentElement.style.removeProperty('--conversation-text-font-size')
    window.document.documentElement.style.removeProperty('--conversation-caption-font-size')
    window.document.documentElement.style.removeProperty('--conversation-caption-line-height')
  })

  it('paints nothing at the default, so the styles.css rem values stand', async () => {
    const store = await loadStore()

    expect(store.$chatTextSize.get()).toBe('100')
    expect(cssVar('--conversation-text-font-size')).toBe('')
    expect(cssVar('--conversation-caption-font-size')).toBe('')
    expect(cssVar('--conversation-caption-line-height')).toBe('')
    expect(window.localStorage.getItem(store.CHAT_TEXT_SIZE_STORAGE_KEY)).toBeNull()
  })

  it('scales the conversation vars and persists the pick', async () => {
    const store = await loadStore()

    store.setChatTextSize('125')

    expect(cssVar('--conversation-text-font-size')).toBe('1.0156rem')
    expect(cssVar('--conversation-caption-font-size')).toBe('0.9375rem')
    expect(cssVar('--conversation-caption-line-height')).toBe('1.2500rem')
    expect(window.localStorage.getItem(store.CHAT_TEXT_SIZE_STORAGE_KEY)).toBe('125')
  })

  it('restores the painted override from storage on load', async () => {
    window.localStorage.setItem('hermes.desktop.chatTextSize.v1', '150')

    await loadStore()

    expect(cssVar('--conversation-text-font-size')).toBe('1.2188rem')
  })

  it('returning to the default clears the overrides and the stored key', async () => {
    window.localStorage.setItem('hermes.desktop.chatTextSize.v1', '110')
    const store = await loadStore()

    store.setChatTextSize('100')

    expect(cssVar('--conversation-text-font-size')).toBe('')
    expect(cssVar('--conversation-caption-font-size')).toBe('')
    expect(window.localStorage.getItem(store.CHAT_TEXT_SIZE_STORAGE_KEY)).toBeNull()
  })

  it('falls back to the default for an unknown stored value', async () => {
    window.localStorage.setItem('hermes.desktop.chatTextSize.v1', 'huge')

    const store = await loadStore()

    expect(store.$chatTextSize.get()).toBe('100')
    expect(cssVar('--conversation-text-font-size')).toBe('')
  })
})
