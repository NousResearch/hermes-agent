import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'

type BootstrapWindow = Window & { __HERMES_BASE_PATH__?: string; __HERMES_SESSION_TOKEN__?: string }
const win = window as unknown as BootstrapWindow

/** A fresh page load: the bridge is rebuilt from scratch, only storage survives. */
function reload(basePath = '') {
  Reflect.deleteProperty(win, 'hermesDesktop')
  win.__HERMES_BASE_PATH__ = basePath
  expect(installBrowserDesktopBridge()).toBe(true)

  return win.hermesDesktop!.settings
}

function onProfile(profile: string) {
  $connection.set({ ...($connection.get() ?? {}), profile } as NonNullable<ReturnType<typeof $connection.get>>)
}

beforeEach(() => {
  win.__HERMES_SESSION_TOKEN__ = 'served-token'
  localStorage.clear()
})

afterEach(() => {
  Reflect.deleteProperty(win, 'hermesDesktop')
  delete win.__HERMES_BASE_PATH__
  delete win.__HERMES_SESSION_TOKEN__
  $connection.set(null)
  localStorage.clear()
  vi.restoreAllMocks()
})

describe('browser default project folder', () => {
  it('keeps a chosen folder across reloads, per hosted gateway and profile, until cleared', async () => {
    onProfile('work')
    await expect(reload('/first').setDefaultProjectDir('/srv/repos/app')).resolves.toEqual({ dir: '/srv/repos/app' })

    expect((await reload('/first').getDefaultProjectDir()).dir).toBe('/srv/repos/app')
    // Another gateway on this origin, or another profile, browses a different machine's folders.
    expect((await reload('/second').getDefaultProjectDir()).dir).toBeNull()
    onProfile('ssh-box')
    expect((await reload('/first').getDefaultProjectDir()).dir).toBeNull()

    onProfile('work')
    await reload('/first').setDefaultProjectDir(null)
    expect((await reload('/first').getDefaultProjectDir()).dir).toBeNull()
  })

  it('rejects a choice this browser cannot keep instead of reporting it saved', async () => {
    vi.spyOn(localStorage, 'setItem').mockImplementation(() => {
      throw new Error('storage denied')
    })

    await expect(reload().setDefaultProjectDir('/srv/repos/app')).rejects.toThrow()
    expect((await reload().getDefaultProjectDir()).dir).toBeNull()
  })
})
