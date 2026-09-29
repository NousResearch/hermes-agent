import { capabilityScoped, type ProfileScope } from './client'

export type BrowserMode = 'copy' | 'managed' | 'own'

export interface BrowserModeStatus {
  copy: {
    available: boolean
    browser?: string
    copied_at?: null | number
    google_signed_in?: boolean | null
    has_copy?: boolean
    pinned_profile?: null | string
  }
  live_profile_supported: boolean
  mode: BrowserMode
  own: { cdp_url: null | string; google_signed_in: boolean | null; has_profile: boolean; window_open: boolean }
}

const call = (profile: ProfileScope | undefined, path: string, body?: unknown) =>
  window.hermesDesktop.api<BrowserModeStatus>({ ...capabilityScoped(profile), body, method: body === undefined ? undefined : 'POST', path })

export const getBrowserStatus = (profile?: ProfileScope) => call(profile, '/api/browser/status')

export const setBrowserMode = (mode: BrowserMode, profile?: ProfileScope) => call(profile, '/api/browser/mode', { mode })

export const openBrowserSignIn = (profile?: ProfileScope) => call(profile, '/api/browser/open-signin', {})

export const importBrowserLogins = (profile?: ProfileScope) => call(profile, '/api/browser/import', {})

export const clearBrowserCopy = (profile?: ProfileScope) => call(profile, '/api/browser/clear-copy', {})

export const clearBrowserOwnProfile = (profile?: ProfileScope) => call(profile, '/api/browser/clear-own', {})
