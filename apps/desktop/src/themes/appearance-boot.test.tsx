import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const originalBridge = Object.getOwnPropertyDescriptor(window, 'hermesDesktop')

beforeEach(() => {
  window.localStorage.clear()
  window.history.replaceState({}, '', '/')
  vi.resetModules()
})

afterEach(() => {
  cleanup()
  window.history.replaceState({}, '', '/')

  if (originalBridge) {
    Object.defineProperty(window, 'hermesDesktop', originalBridge)
  } else {
    Reflect.deleteProperty(window, 'hermesDesktop')
  }

  vi.resetModules()
})

it('first-paints an explicit peer-window route ahead of another window’s last route', async () => {
  const { skinPref, modePref } = await import('./appearance-prefs')
  skinPref.assign('default', 'catppuccin', 'local')
  modePref.assign('default', 'dark', 'local')
  skinPref.assign('research', 'ember', 'remote')
  modePref.assign('research', 'light', 'remote')
  window.localStorage.setItem(
    'hermes-desktop-active-appearance-v2',
    JSON.stringify({ connectionId: 'local', profile: 'default' })
  )
  window.history.replaceState({}, '', '/?win=secondary&connectionId=remote&profile=research')
  const fresh = await import('./context')
  expect(window.document.documentElement.dataset.hermesTheme).toBe('ember')
  expect(window.document.documentElement.dataset.hermesMode).toBe('light')
  render(
    <fresh.ThemeProvider>
      <div />
    </fresh.ThemeProvider>
  )
  expect(window.document.documentElement.dataset.hermesTheme).toBe('ember')
})

it('never borrows the local bridge skin for an unassigned remembered remote namesake', async () => {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { localSkin: { profile: 'default', skin: { name: 'catppuccin', colors: {} } } }
  })
  window.localStorage.setItem(
    'hermes-desktop-active-appearance-v2',
    JSON.stringify({ connectionId: 'remote', profile: 'default' })
  )
  const fresh = await import('./context')
  expect(window.document.documentElement.dataset.hermesTheme).toBe('nous')
  render(
    <fresh.ThemeProvider>
      <div />
    </fresh.ThemeProvider>
  )
  expect(window.document.documentElement.dataset.hermesTheme).toBe('nous')
})

it('retains the old last-profile boot hint when only legacy preferences exist', async () => {
  window.localStorage.setItem('hermes-desktop-active-profile-v1', 'work')
  window.localStorage.setItem('hermes-desktop-profile-themes-v1', JSON.stringify({ work: 'ember' }))
  window.localStorage.setItem('hermes-desktop-profile-modes-v1', JSON.stringify({ work: 'dark' }))
  const fresh = await import('./context')
  expect(window.document.documentElement.dataset.hermesTheme).toBe('ember')
  expect(window.document.documentElement.dataset.hermesMode).toBe('dark')
  render(
    <fresh.ThemeProvider>
      <div />
    </fresh.ThemeProvider>
  )
  expect(window.document.documentElement.dataset.hermesTheme).toBe('ember')
  expect(window.localStorage.getItem('hermes-desktop-active-profile-v1')).toBe('work')
})
