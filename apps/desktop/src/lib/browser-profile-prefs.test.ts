import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { NO_PROJECT_ID } from '@/app/chat/sidebar/projects/workspace-groups'
import { $defaultProfileRoute } from '@/store/default-profile'
import { $activeGatewayProfile, $newChatProfile } from '@/store/profile'
import { $projectScope, ALL_PROJECTS } from '@/store/project-scope'
import { resolveNewSessionCwd } from '@/store/projects'
import { applyConfiguredDefaultProjectDir } from '@/store/session'
import { dropTilesForProfile, migrateTilesForProfile } from '@/store/session-states'

import { createBrowserProfileBridge } from './browser-profile'
import { browserDefaultProjectDirKey, createBrowserSettings } from './browser-settings'
import { browserStartupProfileKey } from './browser-startup-profile'

type BootstrapWindow = Window & { __HERMES_BASE_PATH__?: string; __HERMES_DASHBOARD_PROFILE__?: string }

const win = window as BootstrapWindow

function host(basePath = '/hermes') {
  document.documentElement.dataset.hermesDesktopHost = 'browser'
  win.__HERMES_BASE_PATH__ = basePath
}

function unhost() {
  delete document.documentElement.dataset.hermesDesktopHost
  delete win.__HERMES_BASE_PATH__
  delete win.__HERMES_DASHBOARD_PROFILE__
}

function putFolder(profile: string, dir: string, basePath = '/hermes') {
  localStorage.setItem(browserDefaultProjectDirKey(basePath, profile), dir)
}

function putStartup(profile: string, basePath = '/hermes', connectionId: null | string = null) {
  localStorage.setItem(browserStartupProfileKey(basePath), JSON.stringify({ connectionId, profile }))
}

beforeEach(() => {
  localStorage.clear()
  unhost()
  $projectScope.set(ALL_PROJECTS)
  $newChatProfile.set(null)
  $activeGatewayProfile.set('default')
  $defaultProfileRoute.set(null)
  applyConfiguredDefaultProjectDir('/cache/alpha')
})

afterEach(() => {
  localStorage.clear()
  unhost()
  $projectScope.set(ALL_PROJECTS)
  $newChatProfile.set(null)
  $activeGatewayProfile.set('default')
  $defaultProfileRoute.set(null)
  applyConfiguredDefaultProjectDir(null)
})

describe('browser profile folder for a new chat', () => {
  it('starts a chat or tile in profile B in B folder, and stays detached when B never chose one', () => {
    expect(browserDefaultProjectDirKey('/hermes', 'beta')).toBe('hermes:browser:default-project-dir:["/hermes","beta"]')
    host()
    putFolder('alpha', '/srv/alpha')
    putFolder('beta', '/srv/beta')
    $newChatProfile.set('alpha')

    expect(resolveNewSessionCwd()).toBe('/srv/alpha')
    expect(resolveNewSessionCwd('beta')).toBe('/srv/beta')

    localStorage.removeItem(browserDefaultProjectDirKey('/hermes', 'beta'))

    expect(resolveNewSessionCwd('beta')).toBe('')
    $newChatProfile.set('beta')
    expect(resolveNewSessionCwd()).toBe('')
  })

  it('uses the active profile folder when the new-chat pin is empty', () => {
    host()
    putFolder('beta', '/srv/beta')
    $activeGatewayProfile.set('beta')

    expect(resolveNewSessionCwd()).toBe('/srv/beta')
  })

  it('keeps Home detached when the profile has a default folder', () => {
    host()
    putFolder('beta', '/srv/beta')
    $newChatProfile.set('beta')
    $projectScope.set(NO_PROJECT_ID)

    expect(resolveNewSessionCwd()).toBe('')
  })

  it('keeps the process cache for a desktop new chat, including one aimed at a profile', () => {
    putFolder('beta', '/srv/beta')
    $newChatProfile.set('beta')

    expect(resolveNewSessionCwd()).toBe('/cache/alpha')
    expect(resolveNewSessionCwd('beta')).toBe('/cache/alpha')
  })
})

describe('browser profile prefs follow rename and delete', () => {
  it('moves the folder and startup profile with a rename and leaves every other pref', () => {
    host()
    putFolder('alpha', '/srv/alpha')
    putFolder('beta', '/srv/beta')
    putFolder('alpha', '/srv/other-alpha', '/other')
    putStartup('alpha', '/hermes', 'local')
    putStartup('alpha', '/other', 'remote-box')
    $defaultProfileRoute.set({ connectionId: 'local', profile: 'alpha' })

    migrateTilesForProfile('alpha', 'renamed')

    expect(localStorage.getItem(browserDefaultProjectDirKey('/hermes', 'alpha'))).toBeNull()
    expect(localStorage.getItem(browserDefaultProjectDirKey('/hermes', 'renamed'))).toBe('/srv/alpha')
    expect(localStorage.getItem(browserDefaultProjectDirKey('/hermes', 'beta'))).toBe('/srv/beta')
    expect(localStorage.getItem(browserDefaultProjectDirKey('/other', 'alpha'))).toBe('/srv/other-alpha')
    expect(JSON.parse(localStorage.getItem(browserStartupProfileKey('/hermes'))!)).toEqual({
      connectionId: 'local',
      profile: 'renamed'
    })
    expect(JSON.parse(localStorage.getItem(browserStartupProfileKey('/other'))!)).toEqual({
      connectionId: 'remote-box',
      profile: 'alpha'
    })
    expect($defaultProfileRoute.get()).toEqual({ connectionId: 'local', profile: 'renamed' })
    expect(resolveNewSessionCwd('renamed')).toBe('/srv/alpha')
    expect(resolveNewSessionCwd('alpha')).toBe('')
  })

  // A rename never lands on a live profile, so prefs already under the new name
  // were left by a deleted profile (a delete before this fix kept them).
  it.each([
    { expected: '/srv/research', research: '/srv/research' },
    { expected: '', research: null }
  ])(
    'gives the renamed profile its own folder ($research), not what a deleted profile left',
    ({ expected, research }) => {
      host()
      putFolder('work', '/srv/deleted-work')
      putStartup('work', '/hermes', 'local')
      $defaultProfileRoute.set({ connectionId: 'local', profile: 'work' })

      if (research) {
        putFolder('research', research)
      }

      migrateTilesForProfile('research', 'work')

      expect(resolveNewSessionCwd('work')).toBe(expected)
      expect(localStorage.getItem(browserDefaultProjectDirKey('/hermes', 'research'))).toBeNull()
      expect(localStorage.getItem(browserStartupProfileKey('/hermes'))).toBeNull()
      expect($defaultProfileRoute.get()).toBeNull()
    }
  )

  it('drops a blank folder instead of carrying it to the new name', () => {
    host()
    putFolder('alpha', '   ')

    migrateTilesForProfile('alpha', 'renamed')

    expect(localStorage.getItem(browserDefaultProjectDirKey('/hermes', 'alpha'))).toBeNull()
    expect(localStorage.getItem(browserDefaultProjectDirKey('/hermes', 'renamed'))).toBeNull()
  })

  it('rewrites startup storage when this window is not currently mirroring it', () => {
    host()
    putStartup('alpha')

    migrateTilesForProfile('alpha', 'renamed')

    expect(JSON.parse(localStorage.getItem(browserStartupProfileKey('/hermes'))!).profile).toBe('renamed')
    expect($defaultProfileRoute.get()).toBeNull()
  })

  it('leaves another profile startup mirror alone while the renamed profile folder moves', () => {
    host()
    putFolder('alpha', '/srv/alpha')
    putStartup('alpha')
    $defaultProfileRoute.set({ connectionId: null, profile: 'beta' })

    migrateTilesForProfile('alpha', 'renamed')

    expect(localStorage.getItem(browserDefaultProjectDirKey('/hermes', 'renamed'))).toBe('/srv/alpha')
    expect(JSON.parse(localStorage.getItem(browserStartupProfileKey('/hermes'))!).profile).toBe('renamed')
    expect($defaultProfileRoute.get()).toEqual({ connectionId: null, profile: 'beta' })
  })

  it('drops the deleted profile folder and startup profile and leaves the others', () => {
    host()
    putFolder('alpha', '/srv/alpha')
    putFolder('beta', '/srv/beta')
    putStartup('alpha', '/hermes', 'local')
    putStartup('alpha', '/other', 'remote-box')
    $defaultProfileRoute.set({ connectionId: 'local', profile: 'alpha' })

    dropTilesForProfile('alpha')

    expect(localStorage.getItem(browserDefaultProjectDirKey('/hermes', 'alpha'))).toBeNull()
    expect(localStorage.getItem(browserDefaultProjectDirKey('/hermes', 'beta'))).toBe('/srv/beta')
    expect(localStorage.getItem(browserStartupProfileKey('/hermes'))).toBeNull()
    expect(JSON.parse(localStorage.getItem(browserStartupProfileKey('/other'))!).profile).toBe('alpha')
    expect($defaultProfileRoute.get()).toBeNull()
    expect(resolveNewSessionCwd('alpha')).toBe('')
  })

  it('leaves a stored startup route the profile bridge does not read as one', async () => {
    host()
    const bridge = createBrowserProfileBridge({ basePath: '/hermes', currentProfile: () => null, openWindow: async () => ({ ok: true }) })

    for (const stored of [
      { connectionId: 'remote-1', profile: 'work' },
      { connectionId: null, profile: ' work ' }
    ]) {
      const raw = JSON.stringify(stored)
      localStorage.setItem(browserStartupProfileKey('/hermes'), raw)
      await expect(bridge.profile.getDefault()).resolves.toBeNull()

      migrateTilesForProfile('work', 'renamed')
      expect(localStorage.getItem(browserStartupProfileKey('/hermes'))).toBe(raw)

      dropTilesForProfile('work')
      expect(localStorage.getItem(browserStartupProfileKey('/hermes'))).toBe(raw)
    }
  })

  it('leaves a startup profile that names someone else', () => {
    host()
    putFolder('alpha', '/srv/alpha')
    putStartup('beta', '/hermes', 'local')
    $defaultProfileRoute.set({ connectionId: 'local', profile: 'beta' })

    dropTilesForProfile('alpha')

    expect(localStorage.getItem(browserDefaultProjectDirKey('/hermes', 'alpha'))).toBeNull()
    expect(JSON.parse(localStorage.getItem(browserStartupProfileKey('/hermes'))!)).toEqual({
      connectionId: 'local',
      profile: 'beta'
    })
    expect($defaultProfileRoute.get()).toEqual({ connectionId: 'local', profile: 'beta' })
  })

  it('keeps folder and startup prefs when this page is not the browser host', () => {
    win.__HERMES_BASE_PATH__ = '/hermes'
    putFolder('alpha', '/srv/alpha')
    putStartup('alpha')
    $defaultProfileRoute.set({ connectionId: null, profile: 'alpha' })

    migrateTilesForProfile('alpha', 'renamed')
    dropTilesForProfile('alpha')

    expect(localStorage.getItem(browserDefaultProjectDirKey('/hermes', 'alpha'))).toBe('/srv/alpha')
    expect(localStorage.getItem(browserDefaultProjectDirKey('/hermes', 'renamed'))).toBeNull()
    expect(JSON.parse(localStorage.getItem(browserStartupProfileKey('/hermes'))!).profile).toBe('alpha')
    expect($defaultProfileRoute.get()).toEqual({ connectionId: null, profile: 'alpha' })
  })
})

// An older build keyed the folder on the tab's named profile or 'default'. A tab
// opened without `?profile=` names none, so on a gateway served as `alpha` the
// folder alpha's tab chose sits under 'default'.
describe('browser folder saved by an older build', () => {
  const legacyKey = 'hermes:browser:default-project-dir:["/hermes","default"]'

  it('moves the folder to the profile the gateway serves, once, so default does not inherit it', () => {
    host()
    win.__HERMES_DASHBOARD_PROFILE__ = 'alpha'
    localStorage.setItem(legacyKey, '/srv/alpha')

    expect(resolveNewSessionCwd('alpha')).toBe('/srv/alpha')
    expect(resolveNewSessionCwd('default')).toBe('')
    expect(localStorage.getItem(legacyKey)).toBeNull()

    // Later choices are the new build's own: default's folder stays default's.
    putFolder('default', '/srv/default')
    localStorage.removeItem(browserDefaultProjectDirKey('/hermes', 'alpha'))

    expect(resolveNewSessionCwd('alpha')).toBe('')
    expect(resolveNewSessionCwd('default')).toBe('/srv/default')
  })

  it('shows the moved folder in Settings for a tab that names no profile', async () => {
    host()
    win.__HERMES_DASHBOARD_PROFILE__ = 'alpha'
    localStorage.setItem(legacyKey, '/srv/alpha')

    const settings = createBrowserSettings({ basePath: '/hermes', currentProfile: () => null })

    await expect(settings.getDefaultProjectDir()).resolves.toMatchObject({ dir: '/srv/alpha' })
    expect(localStorage.getItem(legacyKey)).toBeNull()
  })

  it('keeps a folder the served profile already chose and leaves the old one where it was', () => {
    host()
    win.__HERMES_DASHBOARD_PROFILE__ = 'alpha'
    localStorage.setItem(legacyKey, '/srv/legacy')
    putFolder('alpha', '/srv/alpha-own')

    expect(resolveNewSessionCwd('alpha')).toBe('/srv/alpha-own')
    expect(resolveNewSessionCwd('default')).toBe('/srv/legacy')
  })

  it('leaves a gateway served as default alone, and a later launch as alpha does not take its folder', () => {
    host()
    localStorage.setItem(legacyKey, '/srv/default')

    expect(resolveNewSessionCwd('default')).toBe('/srv/default')

    win.__HERMES_DASHBOARD_PROFILE__ = 'alpha'

    expect(resolveNewSessionCwd('alpha')).toBe('')
    expect(resolveNewSessionCwd('default')).toBe('/srv/default')
  })
})
