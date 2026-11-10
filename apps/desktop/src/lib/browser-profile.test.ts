import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'
import { browserOpensWebappWindows } from '@/test/webapp-windows'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'

const bootstrap = window as Window & {
  __HERMES_AUTH_REQUIRED__?: boolean
  __HERMES_BASE_PATH__?: string
}

beforeEach(() => {
  window.localStorage.clear()
  bootstrap.__HERMES_BASE_PATH__ = '/hermes'
  seedWebappSession('test-session')
  window.history.replaceState(null, '', '/hermes?profile=active#/session-1')
})

afterEach(() => {
  Reflect.deleteProperty(window, 'hermesDesktop')
  delete bootstrap.__HERMES_AUTH_REQUIRED__
  delete bootstrap.__HERMES_BASE_PATH__
  clearWebappSession()
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  window.localStorage.clear()
  window.history.replaceState(null, '', '/#/')
  $connection.set(null)
  vi.restoreAllMocks()
})

it('persists the default separately from the active chat and synchronizes only the same server across tabs', async () => {
  expect(installBrowserDesktopBridge()).toBe(true)
  const profile = window.hermesDesktop.profile
  const originalUrl = window.location.href
  const changed = vi.fn()
  const unsubscribe = profile.onDefaultChanged(changed)
  const route = { connectionId: 'local', profile: 'research' }

  await expect(profile.setDefault(route)).resolves.toEqual(route)
  expect(window.location.href).toBe(originalUrl)
  await expect(profile.get()).resolves.toEqual({ profile: 'active' })
  expect(changed).toHaveBeenLastCalledWith(route)
  await profile.remember('work')
  await expect(profile.getDefault()).resolves.toEqual(route)

  Reflect.deleteProperty(window, 'hermesDesktop')
  expect(installBrowserDesktopBridge()).toBe(true)
  await expect(window.hermesDesktop.profile.getDefault()).resolves.toEqual(route)
  Reflect.deleteProperty(window, 'hermesDesktop')
  bootstrap.__HERMES_BASE_PATH__ = '/another-server'
  expect(installBrowserDesktopBridge()).toBe(true)
  await expect(window.hermesDesktop.profile.getDefault()).resolves.toBeNull()

  const key = window.localStorage.key(0)!
  const updated = { ...route, profile: 'review' }
  window.localStorage.setItem(key, JSON.stringify(updated))
  const storageEvent = () => Object.assign(new Event('storage'), { key, storageArea: window.localStorage })
  window.dispatchEvent(storageEvent())
  expect(changed).toHaveBeenLastCalledWith(updated)
  unsubscribe()
  changed.mockClear()
  window.dispatchEvent(storageEvent())
  expect(changed).not.toHaveBeenCalled()
})

it('records a live profile pick in the URL without dropping the router entry state', async () => {
  const entry = { idx: 3, key: 'entry', usr: { from: 'sidebar' } }
  window.history.replaceState(entry, '', '/hermes?profile=active#/session-1')
  expect(installBrowserDesktopBridge()).toBe(true)

  await window.hermesDesktop.profile.remember('work')

  expect(new URLSearchParams(window.location.search).get('profile')).toBe('work')
  expect(window.location.hash).toBe('#/session-1')
  expect(window.history.state).toEqual(entry)
})

it('opens the requested profile on this host without carrying another session or spectator flags', async () => {
  // A gated host opens the window directly; a launch-link tab hands it a session first.
  bootstrap.__HERMES_AUTH_REQUIRED__ = true
  window.history.replaceState(null, '', '/hermes?profile=active&win=secondary&watch=1#/session-1')
  const { open } = browserOpensWebappWindows()
  expect(installBrowserDesktopBridge()).toBe(true)

  const originalUrl = window.location.href
  await window.hermesDesktop.openWindow()
  const peer = new URL(String(open.mock.calls[0][0]))
  expect(peer.searchParams.get('profile')).toBe('active')
  expect(peer.searchParams.has('watch')).toBe(false)
  expect(peer.searchParams.has('win')).toBe(false)
  expect(peer.hash).toBe('#/')
  expect(window.location.href).toBe(originalUrl)
  open.mockClear()

  await window.hermesDesktop.openWindow({ connectionId: 'local', profile: 'research' })
  const target = new URL(String(open.mock.calls[0][0]))
  expect(target.origin).toBe(window.location.origin)
  expect(target.pathname).toBe('/hermes')
  expect(target.searchParams.get('profile')).toBe('research')
  expect(target.searchParams.has('watch')).toBe(false)
  expect(target.searchParams.has('win')).toBe(false)
  expect(target.hash).toBe('#/')

  await window.hermesDesktop.openWindow({ connectionId: null, profile: 'default' })
  expect(new URL(String(open.mock.calls[1][0])).searchParams.get('profile')).toBe('default')
  open.mockClear()
  await expect(window.hermesDesktop.openWindow({ connectionId: 'another-host', profile: 'research' })).rejects.toThrow(
    'No connection with id'
  )
  await expect(
    window.hermesDesktop.profile.setDefault({ connectionId: 'another-host', profile: 'research' })
  ).rejects.toThrow('No connection with id')
  expect(open).not.toHaveBeenCalled()
})
