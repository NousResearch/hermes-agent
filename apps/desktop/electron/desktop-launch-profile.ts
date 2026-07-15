'use strict'

// Pure helpers for which Hermes profile the desktop primary backend launches
// as. Kept outside main.ts so the launch contract can be unit-tested without
// Electron: a profile-scoped HERMES_HOME (.../profiles/<name>) must produce
// `--profile <name>` when the desktop preference is unset.

import { serveBackendArgs } from './backend-command'

/** Mirrors hermes_cli.profiles._PROFILE_ID_RE. */
export const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

export function sanitizeProfileHint(raw: unknown): string | null {
  const candidate = String(raw || '')
    .trim()
    .toLowerCase()

  if (!candidate) {
    return null
  }

  if (candidate === 'default' || PROFILE_NAME_RE.test(candidate)) {
    return candidate
  }

  return null
}

/**
 * Prefer an environment-derived profile hint (from a profile-scoped
 * HERMES_HOME) over the desktop's stored preference. Either may be null.
 */
export function preferredDesktopLaunchProfile({
  profileHint = null,
  preference = null
}: {
  profileHint?: string | null
  preference?: string | null
} = {}): string | null {
  return profileHint || preference || null
}

export function primaryProfileKeyFromLaunch(options?: {
  profileHint?: string | null
  preference?: string | null
}): string {
  return preferredDesktopLaunchProfile(options) || 'default'
}

/**
 * Argv the primary (window) backend should spawn with for a given
 * hint + preference. Matches startHermes(): unset preference + no hint keeps
 * the legacy launch (no `--profile`); otherwise pin via `--profile`.
 */
export function primaryBackendArgsFromLaunch(options?: {
  profileHint?: string | null
  preference?: string | null
}): string[] {
  const active = preferredDesktopLaunchProfile(options)

  return serveBackendArgs(active || undefined)
}
