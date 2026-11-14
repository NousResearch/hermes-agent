import type { DesktopProfileRoute } from '@/global'

import { requireBrowserConnection } from './browser-connection'
import { readJson, writeJson } from './storage'

export function browserStartupProfileKey(basePath: string): string {
  return `hermes:webapp:default-profile:${basePath || '/'}`
}

/** A profile route this tab can apply: a valid profile name on its one connection. The profile
 * bridge and the rename/delete migrations read the stored startup route through this alone, so
 * they agree on which stored value is a startup profile. */
export function requireProfileRoute(value: unknown): DesktopProfileRoute {
  if (!value || typeof value !== 'object') {
    throw new Error('A profile route is required.')
  }

  const { connectionId, profile } = value as Record<string, unknown>

  if (typeof profile !== 'string' || !/^[a-z0-9][a-z0-9_-]{0,63}$/.test(profile)) {
    throw new Error('Invalid profile name.')
  }

  if (connectionId !== null && typeof connectionId !== 'string') {
    throw new Error('Invalid connection id.')
  }

  requireBrowserConnection(connectionId)

  return { connectionId, profile }
}

function startupRoute(basePath: string): DesktopProfileRoute | null {
  try {
    return requireProfileRoute(readJson(browserStartupProfileKey(basePath)))
  } catch {
    return null
  }
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
