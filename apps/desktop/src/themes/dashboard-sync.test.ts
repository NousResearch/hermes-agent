import { beforeEach, describe, expect, it, vi } from 'vitest'

import { getDashboardThemes, setDashboardTheme } from '@/api/dashboard-themes'
import type { DashboardThemesResponse } from '@/types/hermes'

import { $pendingSkinApply } from './backend-sync'
import {
  DASHBOARD_SHARED_THEMES,
  ingestDashboardTheme,
  isDashboardSyncEnabled,
  publishDashboardTheme,
  refreshDashboardTheme,
  setDashboardSyncEnabled,
  toDashboardTheme,
  toDesktopSkin
} from './dashboard-sync'

vi.mock('@/api/dashboard-themes', () => ({
  getDashboardThemes: vi.fn(),
  setDashboardTheme: vi.fn()
}))

const getThemes = vi.mocked(getDashboardThemes)
const putTheme = vi.mocked(setDashboardTheme)

const profile = 'work'
const ingest = (active: null | string | undefined) => ingestDashboardTheme(active, { profile })

describe('dashboard theme sync', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.clearAllMocks()
    $pendingSkinApply.set(null)
    putTheme.mockResolvedValue({ ok: true, theme: 'midnight' })
  })

  describe('name mapping', () => {
    it('maps the four shared names in both directions', () => {
      for (const name of ['cyberpunk', 'ember', 'midnight', 'mono']) {
        expect(toDesktopSkin(name)).toBe(name)
        expect(toDashboardTheme(name)).toBe(name)
      }

      expect(DASHBOARD_SHARED_THEMES.cyberpunk).toBe('cyberpunk')
    })

    it('returns null for names only one surface knows', () => {
      for (const name of ['default', 'rose', 'slate', 'nous', undefined, '   ']) {
        expect(toDesktopSkin(name)).toBeNull()
      }

      for (const name of ['default', 'github', 'nous-alt', 'catppuccin', null]) {
        expect(toDashboardTheme(name)).toBeNull()
      }
    })

    it('trims before looking up', () => {
      expect(toDesktopSkin('  ember  ')).toBe('ember')
      expect(toDashboardTheme(' mono')).toBe('mono')
    })
  })

  describe('ingestDashboardTheme', () => {
    it('records a first-seen theme without painting', () => {
      ingest('midnight')

      expect($pendingSkinApply.get()).toBeNull()
      expect(window.localStorage.getItem('hermes-desktop-dashboard-theme-v1')).toBe('midnight')
    })

    it('paints when a shared theme name changes', () => {
      ingest('midnight')
      ingest('ember')

      expect($pendingSkinApply.get()).toBe('ember')
      expect(window.localStorage.getItem('hermes-desktop-dashboard-theme-v1')).toBe('ember')
    })

    it('ignores a repeat of the same name', () => {
      ingest('ember')
      $pendingSkinApply.set(null)
      ingest('ember')

      expect($pendingSkinApply.get()).toBeNull()
    })

    it('moves the baseline without painting for a name the desktop cannot render', () => {
      ingest('midnight')
      ingest('rose') // dashboard-only built-in

      expect($pendingSkinApply.get()).toBeNull()
      expect(window.localStorage.getItem('hermes-desktop-dashboard-theme-v1')).toBe('rose')

      ingest('ocean') // user YAML theme
      expect($pendingSkinApply.get()).toBeNull()
      expect(window.localStorage.getItem('hermes-desktop-dashboard-theme-v1')).toBe('ocean')
    })

    it('ignores empty payloads', () => {
      ingest(null)
      ingest('   ')

      expect($pendingSkinApply.get()).toBeNull()
      expect(window.localStorage.getItem('hermes-desktop-dashboard-theme-v1')).toBeNull()
    })

    it('does nothing while the profile has sync off — not even the baseline', () => {
      setDashboardSyncEnabled(profile, false)
      ingest('midnight')

      expect($pendingSkinApply.get()).toBeNull()
      expect(window.localStorage.getItem('hermes-desktop-dashboard-theme-v1')).toBeNull()
    })

    it('seeds a fresh install without painting; only a later change paints', () => {
      ingest('midnight') // no baseline yet — records it, adopts nothing

      expect($pendingSkinApply.get()).toBeNull()
      expect(window.localStorage.getItem('hermes-desktop-dashboard-theme-v1')).toBe('midnight')

      ingest('ember') // the dashboard genuinely moved — now it follows

      expect($pendingSkinApply.get()).toBe('ember')
      expect(window.localStorage.getItem('hermes-desktop-dashboard-theme-v1')).toBe('ember')
    })

    it('re-enabling sync follows the dashboard again', () => {
      ingest('ember') // baseline seeded while sync is on
      setDashboardSyncEnabled(profile, false)

      ingest('midnight') // the opt-out must suppress this observation entirely

      expect($pendingSkinApply.get()).toBeNull()
      expect(window.localStorage.getItem('hermes-desktop-dashboard-theme-v1')).toBe('ember')

      setDashboardSyncEnabled(profile, true)
      ingest('midnight') // dashboard moved while we were off → resume following it

      expect($pendingSkinApply.get()).toBe('midnight')
      expect(window.localStorage.getItem('hermes-desktop-dashboard-theme-v1')).toBe('midnight')
    })
  })

  describe('publishDashboardTheme', () => {
    it('PUTs the mapped dashboard name and records the baseline', async () => {
      await publishDashboardTheme('midnight', { profile })

      expect(putTheme).toHaveBeenCalledWith('midnight')
      expect(window.localStorage.getItem('hermes-desktop-dashboard-theme-v1')).toBe('midnight')
    })

    it('does not echo a value it just applied back to the dashboard', async () => {
      // The dashboard drove the desktop to 'midnight' (seed, then change paints).
      ingest('ember')
      ingest('midnight')

      expect($pendingSkinApply.get()).toBe('midnight')

      // Re-publishing the same skin sees a baseline that already equals the
      // mapped name — the actual echo-suppression branch inside publish.
      await publishDashboardTheme('midnight', { profile })

      expect(putTheme).not.toHaveBeenCalled()
    })

    it('never PUTs a desktop-only skin', async () => {
      await publishDashboardTheme('slate', { profile })
      await publishDashboardTheme('github', { profile })
      await publishDashboardTheme('nous', { profile })

      expect(putTheme).not.toHaveBeenCalled()
    })

    it('does nothing while the profile has sync off', async () => {
      setDashboardSyncEnabled(profile, false)

      ingest('ember')
      await publishDashboardTheme('midnight', { profile })

      expect($pendingSkinApply.get()).toBeNull()
      expect(putTheme).not.toHaveBeenCalled()
    })

    it('keeps the desktop pick when the PUT fails', async () => {
      putTheme.mockRejectedValue(new Error('backend down'))

      await expect(publishDashboardTheme('midnight', { profile })).resolves.toBeUndefined()

      expect(window.localStorage.getItem('hermes-desktop-dashboard-theme-v1')).toBeNull()
    })

    it('pins: a failed publish never causes a later refresh to revert the user pick', async () => {
      ingest('midnight') // baseline = what the server reports: 'midnight'

      putTheme.mockRejectedValue(new Error('backend down'))
      await publishDashboardTheme('ember', { profile }) // user's pick — PUT fails

      // The failed PUT must not move the baseline off the server's value...
      expect(window.localStorage.getItem('hermes-desktop-dashboard-theme-v1')).toBe('midnight')

      // ...and the next observation of the unchanged server value must be a
      // no-op, not a silent revert of the user's 'ember' paint.
      getThemes.mockResolvedValue({ themes: [], active: 'midnight' })
      await refreshDashboardTheme(profile)

      expect($pendingSkinApply.get()).toBeNull()
      expect(window.localStorage.getItem('hermes-desktop-dashboard-theme-v1')).toBe('midnight')
    })
  })

  describe('refreshDashboardTheme', () => {
    it('ingests the fetched active theme', async () => {
      getThemes.mockResolvedValue({ themes: [], active: 'ember' })

      await refreshDashboardTheme(profile) // first observation: baseline only
      expect(window.localStorage.getItem('hermes-desktop-dashboard-theme-v1')).toBe('ember')

      getThemes.mockResolvedValue({ themes: [], active: 'cyberpunk' })
      await refreshDashboardTheme(profile)

      expect($pendingSkinApply.get()).toBe('cyberpunk')
    })

    it('swallows a failed fetch', async () => {
      getThemes.mockRejectedValue(new Error('backend not up'))

      await expect(refreshDashboardTheme(profile)).resolves.toBeUndefined()
      expect($pendingSkinApply.get()).toBeNull()
    })

    it('skips the fetch entirely while sync is off', async () => {
      setDashboardSyncEnabled(profile, false)

      await refreshDashboardTheme(profile)

      expect(getThemes).not.toHaveBeenCalled()
    })

    it('collapses concurrent refreshes into a single fetch', async () => {
      // A promise we control, so both calls launch while one GET is in flight —
      // the gateway.ready + focus + visibilitychange same-tick storm.
      let resolveFetch: (value: DashboardThemesResponse) => void = () => {}
      getThemes.mockReturnValue(
        new Promise<DashboardThemesResponse>(resolve => {
          resolveFetch = resolve
        })
      )

      const first = refreshDashboardTheme(profile)
      const second = refreshDashboardTheme(profile) // same tick — must not re-fetch

      resolveFetch({ themes: [], active: 'ember' })
      await Promise.all([first, second])

      expect(getThemes).toHaveBeenCalledTimes(1)
      expect(window.localStorage.getItem('hermes-desktop-dashboard-theme-v1')).toBe('ember')
    })
  })

  describe('the per-profile toggle', () => {
    it('is enabled for a profile that never touched it', () => {
      expect(isDashboardSyncEnabled(profile)).toBe(true)
    })

    it('round-trips through disable and re-enable', () => {
      setDashboardSyncEnabled(profile, false)
      expect(isDashboardSyncEnabled(profile)).toBe(false)

      setDashboardSyncEnabled(profile, true)
      expect(isDashboardSyncEnabled(profile)).toBe(true)
    })

    it('is scoped per profile', () => {
      setDashboardSyncEnabled('work', false)

      expect(isDashboardSyncEnabled('work')).toBe(false)
      expect(isDashboardSyncEnabled('home')).toBe(true)
    })
  })
})
