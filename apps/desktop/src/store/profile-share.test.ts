import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopTheme } from '@/themes/types'
import type { ProfileDesktopOverlay } from '@/types/hermes'

// Keep side-effecting transitive imports inert (gateway sockets, REST).
vi.mock('@/store/gateway', async () => {
  const { atom } = await import('nanostores')

  return {
    $gateway: atom<unknown>(null),
    ensureGatewayForProfile: vi.fn(async () => undefined),
    openGatewayForProfile: vi.fn(async () => undefined)
  }
})
vi.mock('@/hermes', () => ({
  exportProfileArchive: vi.fn(async () => ({ archive: '/tmp/out.tar.gz', ok: true })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  importProfileArchive: vi.fn(async () => ({ desktop: null, name: 'imported', ok: true, path: '/tmp/p' })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ reloadOpenStarmapGraph: vi.fn(), resetStarmapGraph: vi.fn() }))

const { applyDesktopOverlay, buildDesktopOverlay, exportProfileBundle } = await import('./profile-share')
const { $profileColors, setProfileColor } = await import('./profile')
const { modePref, skinPref } = await import('@/themes/context')
const { $userThemes } = await import('@/themes/user-themes')
const { $layoutTree } = await import('@/components/pane-shell/tree/store')
const { exportProfileArchive } = await import('@/hermes')

// isValidTheme only requires background/foreground/primary at runtime; the
// static type wants the full palette, hence the cast.
const roseTheme = {
  name: 'rose-quartz',
  label: 'Rose Quartz',
  description: 'test theme',
  colors: { background: '#fff0f5', foreground: '#221122', primary: '#e91e63' }
} as unknown as DesktopTheme

beforeEach(() => {
  window.localStorage.clear()
  $userThemes.set({})
  $profileColors.set({})
})

afterEach(() => {
  vi.clearAllMocks()
})

describe('buildDesktopOverlay', () => {
  it('snapshots skin, mode, rail color, and the layout tree for the profile', () => {
    skinPref.assign('glam', 'mono')
    modePref.assign('glam', 'dark')
    setProfileColor('glam', '#e91e63')

    const overlay = buildDesktopOverlay('glam')

    expect(overlay.version).toBe(1)
    expect(overlay.skin).toBe('mono')
    expect(overlay.mode).toBe('dark')
    expect(overlay.profileColor).toBe('#e91e63')
    // Built-in skin → no bundled theme definitions.
    expect(overlay.themes).toBeUndefined()
  })

  it('bundles the full definition of a non-built-in skin', () => {
    $userThemes.set({ 'rose-quartz': roseTheme })
    skinPref.assign('glam', 'rose-quartz')

    const overlay = buildDesktopOverlay('glam')

    expect(overlay.skin).toBe('rose-quartz')
    expect(overlay.themes).toEqual({ 'rose-quartz': roseTheme })
  })
})

describe('applyDesktopOverlay', () => {
  it('installs bundled themes and assigns skin/mode/color to the new profile', () => {
    applyDesktopOverlay('glam-copy', {
      version: 1,
      skin: 'rose-quartz',
      mode: 'dark',
      themes: { 'rose-quartz': roseTheme },
      profileColor: '#e91e63'
    })

    expect($userThemes.get()['rose-quartz']).toEqual(roseTheme)
    expect(skinPref.resolve('glam-copy')).toBe('rose-quartz')
    expect(modePref.resolve('glam-copy')).toBe('dark')
    expect($profileColors.get()['glam-copy']).toBe('#e91e63')
  })

  it('ignores a skin that resolves to nothing and junk layout trees', () => {
    const before = $layoutTree.get()

    applyDesktopOverlay('glam-copy', {
      skin: 'no-such-skin',
      layoutTree: { bogus: true }
    } as ProfileDesktopOverlay)

    // Unresolvable skin → pref falls back to the default resolution.
    expect(skinPref.resolve('glam-copy')).toBe(skinPref.resolve('some-unassigned'))
    expect($layoutTree.get()).toBe(before)
  })

  it('is a no-op for a plain CLI archive (no overlay)', () => {
    expect(() => applyDesktopOverlay('glam-copy', null)).not.toThrow()
    expect(() => applyDesktopOverlay('glam-copy', undefined)).not.toThrow()
  })
})

describe('runExportProfileFlow', () => {
  // The browser Webapp bridge has no save dialog: Export must say so, not do nothing.
  it('explains itself in the browser Webapp instead of silently returning', async () => {
    const { installBrowserDesktopBridge } = await import('@/lib/browser-desktop-bridge')
    const { clearWebappSession, seedWebappSession } = await import('@/test/webapp-session')
    const { runExportProfileFlow } = await import('./profile-share')
    const { $notifications } = await import('./notifications')
    seedWebappSession('served-token')
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}')))

    try {
      expect(installBrowserDesktopBridge()).toBe(true)
      expect(window.hermesDesktop.selectSavePath).toBeUndefined()
      await expect(runExportProfileFlow('glam')).resolves.toBeNull()
      expect(exportProfileArchive).not.toHaveBeenCalled()
      expect($notifications.get().at(-1)).toMatchObject({ kind: 'error', title: 'Failed to export profile' })
      expect($notifications.get().at(-1)?.message).toContain('hermes profile export glam')
    } finally {
      clearWebappSession()
      Reflect.deleteProperty(window, 'hermesDesktop')
      document.documentElement.removeAttribute('data-hermes-desktop-host')
      vi.unstubAllGlobals()
    }
  })
})

describe('importProfileBundle', () => {
  const folderKey = (profile: string) => `hermes:browser:default-project-dir:${JSON.stringify(['/hermes', profile])}`
  const startupKey = 'hermes:webapp:default-profile:/hermes'

  function seedLeftovers() {
    window.localStorage.setItem(folderKey('imported'), '/srv/deleted-imported')
    window.localStorage.setItem(folderKey('other'), '/srv/other')
    window.localStorage.setItem(startupKey, JSON.stringify({ connectionId: null, profile: 'imported' }))
  }

  afterEach(() => {
    document.documentElement.removeAttribute('data-hermes-desktop-host')
    Reflect.deleteProperty(window, '__HERMES_BASE_PATH__')
  })

  // An import never lands on a live profile's name, so prefs stored under it were left by a deleted one.
  it('drops what a deleted profile of the same name left, only once the import succeeded', async () => {
    const { importProfileBundle } = await import('./profile-share')
    const { importProfileArchive } = await import('@/hermes')
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    Object.assign(window, { __HERMES_BASE_PATH__: '/hermes' })
    seedLeftovers()
    vi.mocked(importProfileArchive).mockRejectedValueOnce(new Error('bad archive'))

    await expect(importProfileBundle('/tmp/in.tar.gz')).rejects.toThrow('bad archive')
    expect(window.localStorage.getItem(folderKey('imported'))).toBe('/srv/deleted-imported')
    expect(window.localStorage.getItem(startupKey)).not.toBeNull()

    await expect(importProfileBundle('/tmp/in.tar.gz')).resolves.toBe('imported')
    expect(window.localStorage.getItem(folderKey('imported'))).toBeNull()
    expect(window.localStorage.getItem(startupKey)).toBeNull()
    expect(window.localStorage.getItem(folderKey('other'))).toBe('/srv/other')
  })

  it('keeps browser prefs in native Desktop', async () => {
    const { importProfileBundle } = await import('./profile-share')
    Object.assign(window, { __HERMES_BASE_PATH__: '/hermes' })
    seedLeftovers()

    await expect(importProfileBundle('/tmp/in.tar.gz')).resolves.toBe('imported')
    expect(window.localStorage.getItem(folderKey('imported'))).toBe('/srv/deleted-imported')
    expect(window.localStorage.getItem(startupKey)).not.toBeNull()
  })
})

describe('exportProfileBundle', () => {
  it('stages desktop.json into the archive through extra_files', async () => {
    skinPref.assign('glam', 'mono')

    const archive = await exportProfileBundle('glam', '/tmp/glam.tar.gz')

    expect(archive).toBe('/tmp/out.tar.gz')
    const call = vi.mocked(exportProfileArchive).mock.calls[0]
    expect(call[0]).toBe('glam')
    const overlay = JSON.parse(call[1]?.extraFiles?.['desktop.json'] ?? '{}') as ProfileDesktopOverlay
    expect(overlay.skin).toBe('mono')
    expect(call[1]?.output).toBe('/tmp/glam.tar.gz')
  })
})
