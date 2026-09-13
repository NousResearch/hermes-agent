import { beforeEach, describe, expect, it, vi } from 'vitest'

const loadStore = async () => {
  vi.resetModules()

  return import('./chat-width')
}

describe('chat width preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    document.documentElement.style.removeProperty('--composer-width')
  })

  it('defaults to full-bleed (no override) and persists changes', async () => {
    const first = await loadStore()

    expect(first.$chatWidth.get()).toBe('default')
    // Default must NOT write an override — the stylesheet value stands.
    expect(document.documentElement.style.getPropertyValue('--composer-width')).toBe('')

    first.setChatWidth('narrow')

    expect(window.localStorage.getItem('hermes.desktop.chatWidth')).toBe('narrow')
    expect(document.documentElement.style.getPropertyValue('--composer-width')).toBe('44rem')
    expect((await loadStore()).$chatWidth.get()).toBe('narrow')
  })

  it('re-applies the persisted override on load', async () => {
    window.localStorage.setItem('hermes.desktop.chatWidth', 'wide')

    await loadStore()

    expect(document.documentElement.style.getPropertyValue('--composer-width')).toBe('min(72rem, 90vw)')
  })

  it('falls back to default for an unknown stored value', async () => {
    window.localStorage.setItem('hermes.desktop.chatWidth', 'colossal')

    const store = await loadStore()

    expect(store.$chatWidth.get()).toBe('default')
    expect(document.documentElement.style.getPropertyValue('--composer-width')).toBe('')
  })

  it('clears the override when returning to default', async () => {
    const store = await loadStore()

    store.setChatWidth('wide')
    expect(document.documentElement.style.getPropertyValue('--composer-width')).toBe('min(72rem, 90vw)')

    store.setChatWidth('default')
    expect(document.documentElement.style.getPropertyValue('--composer-width')).toBe('')
  })
})
