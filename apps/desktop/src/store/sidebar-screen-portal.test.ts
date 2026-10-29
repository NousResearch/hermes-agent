import { afterEach, expect, it, vi } from 'vitest'

const key = 'hermes.desktop.sidebar.screenPortal.v1'

afterEach(() => {
  localStorage.removeItem(key)
  vi.resetModules()
})

it('defaults to visible and remembers hidden and restored choices across reloads', async () => {
  localStorage.removeItem(key)
  vi.resetModules()
  const initial = await import('./sidebar-screen-portal')
  expect(initial.$sidebarScreenPortalVisible.get()).toBe(true)
  initial.setSidebarScreenPortalVisible(false)

  vi.resetModules()
  const hidden = await import('./sidebar-screen-portal')
  expect(hidden.$sidebarScreenPortalVisible.get()).toBe(false)
  hidden.setSidebarScreenPortalVisible(true)

  vi.resetModules()
  const restored = await import('./sidebar-screen-portal')
  expect(restored.$sidebarScreenPortalVisible.get()).toBe(true)
})

it('follows shared appearance preferences without echoing a removal back to storage', async () => {
  vi.resetModules()
  const { $sidebarScreenPortalVisible } = await import('./sidebar-screen-portal')

  localStorage.setItem(key, 'false')
  window.dispatchEvent(new StorageEvent('storage', { key, newValue: 'false' }))
  expect($sidebarScreenPortalVisible.get()).toBe(false)

  localStorage.removeItem(key)
  window.dispatchEvent(new StorageEvent('storage', { key, newValue: null }))
  expect($sidebarScreenPortalVisible.get()).toBe(true)
  expect(localStorage.getItem(key)).toBeNull()
})
