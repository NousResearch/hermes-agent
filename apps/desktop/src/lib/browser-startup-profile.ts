import type { DesktopProfileRoute } from '@/global'

import { readJson, writeJson } from './storage'

export function browserStartupProfileKey(basePath: string): string {
  return `hermes:webapp:default-profile:${basePath || '/'}`
}

function startupRoute(basePath: string): DesktopProfileRoute | null {
  const value = readJson<Record<string, unknown>>(browserStartupProfileKey(basePath))

  if (!value || typeof value.profile !== 'string' || !value.profile.trim()) {
    return null
  }

  if (value.connectionId !== null && typeof value.connectionId !== 'string') {
    return null
  }

  return { connectionId: value.connectionId ?? null, profile: value.profile.trim() }
}

export function readBrowserStartupProfile(basePath: string): DesktopProfileRoute | null {
  return startupRoute(basePath)
}

function sameProfile(left: string, right: string): boolean {
  return left.trim() === right.trim()
}

/** The tab's startup profile follows a rename. Returns whether this page's
 * route named the old profile. Another profile's route is left alone. */
export function migrateBrowserStartupProfile(basePath: string, from: string, to: string): boolean {
  const route = startupRoute(basePath)

  if (!route || !sameProfile(route.profile, from) || sameProfile(from, to)) {
    return false
  }

  writeJson(browserStartupProfileKey(basePath), { ...route, profile: to.trim() })

  return true
}

/** A deleted profile stops being the tab's startup profile. */
export function dropBrowserStartupProfile(basePath: string, profile: string): boolean {
  const route = startupRoute(basePath)

  if (!route || !sameProfile(route.profile, profile)) {
    return false
  }

  writeJson(browserStartupProfileKey(basePath), null)

  return true
}
