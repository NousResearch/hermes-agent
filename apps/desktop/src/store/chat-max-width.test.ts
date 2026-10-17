// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest'

const activeProfile = vi.hoisted(() => ({ value: 'default', listeners: new Set<(profile: string) => void>() }))

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: {
    get: () => activeProfile.value,
    subscribe: (listener: (profile: string) => void) => {
      activeProfile.listeners.add(listener)

      listener(activeProfile.value)

      return () => activeProfile.listeners.delete(listener)
    }
  },
  normalizeProfileKey: (profile: string | null | undefined) => profile?.trim() || 'default'
}))

const loadStore = async () => {
  vi.resetModules()

  return import('./chat-max-width')
}

describe('chat maximum width preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    activeProfile.value = 'default'
    activeProfile.listeners.clear()
  })

  it('persists a cap for the active profile and restores each profile independently', async () => {
    const { $chatMaxWidth, setChatMaxWidth } = await loadStore()

    setChatMaxWidth('56rem')
    expect($chatMaxWidth.get()).toBe('56rem')
    expect(document.documentElement.style.getPropertyValue('--composer-width')).toBe('56rem')
    expect(window.localStorage.getItem('hermes.desktop.chat-max-width.v1')).toBe('{"default":"56rem"}')

    activeProfile.value = 'research'
    activeProfile.listeners.forEach(listener => listener('research'))
    expect($chatMaxWidth.get()).toBe('off')
    expect(document.documentElement.style.getPropertyValue('--composer-width')).toBe('100%')

    setChatMaxWidth('72rem')
    activeProfile.value = 'default'
    activeProfile.listeners.forEach(listener => listener('default'))
    expect($chatMaxWidth.get()).toBe('56rem')
  })

  it('rejects malformed stored values and unsupported picks', async () => {
    window.localStorage.setItem('hermes.desktop.chat-max-width.v1', '{"default":"999rem"}')
    const { $chatMaxWidth, setChatMaxWidth } = await loadStore()

    expect($chatMaxWidth.get()).toBe('off')
    setChatMaxWidth('999rem' as never)
    expect($chatMaxWidth.get()).toBe('off')
  })
})
