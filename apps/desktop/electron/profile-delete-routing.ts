'use strict'

// Profile-delete routing logic for the `hermes:api` IPC handler.
//
// When the renderer issues DELETE /api/profiles/<name>, the handler must
// tear down that profile's backend (primary window backend or pool backend)
// and then route the *next* request away from the just-deleted profile's
// pool backend -- spawning a fresh one would call ensure_hermes_home() and
// recreate the profile directory the delete just removed, leaving a zombie
// process behind (issue #52279).
//
// These helpers are pure so they can be unit-tested without Electron.

/**
 * Parse a `hermes:api` request into the profile name a DELETE targets, or
 * null when the request is not a profile-delete at all (wrong method, wrong
 * path, empty/invalid name).
 */
export function profileNameFromDeleteRequest(request) {
  if (!request || String(request.method || 'GET').toUpperCase() !== 'DELETE') {
    return null
  }

  const match = String(request.path || '').match(/^\/api\/profiles\/([^/?#]+)(?:[?#].*)?$/)

  if (!match) {
    return null
  }

  let raw = ''

  try {
    raw = decodeURIComponent(match[1])
  } catch {
    return null
  }

  const name = raw.trim()

  if (!name) {
    return null
  }

  if (name.toLowerCase() === 'default') {
    return 'default'
  }

  return name.toLowerCase()
}

export type ProfileDeleteAction = 'noop' | 'teardown-primary' | 'teardown-pool'

export interface ProfileDeleteDecision {
  action: ProfileDeleteAction
  profile: string | null
}

export interface ProfileDeleteDecisionDeps {
  isDefaultProfile: (profile: string) => boolean
  isValidProfileName: (profile: string) => boolean
  primaryProfileKey: () => string
}

/**
 * Pure decision logic for prepareProfileDeleteRequest: given the parsed
 * profile name (or null), decide which side-effecting branch the caller
 * should take and what profile name it should ultimately report as
 * torn-down. No I/O, no async -- the caller performs the actual teardown
 * based on `action`.
 */
export function decideProfileDeleteAction(
  profile: string | null,
  deps: ProfileDeleteDecisionDeps
): ProfileDeleteDecision {
  if (!profile || deps.isDefaultProfile(profile) || !deps.isValidProfileName(profile)) {
    return { action: 'noop', profile: null }
  }

  if (profile === deps.primaryProfileKey()) {
    return { action: 'teardown-primary', profile }
  }

  return { action: 'teardown-pool', profile }
}

/**
 * Route the next `hermes:api` request away from the primary/window backend
 * whenever a profile was just torn down -- otherwise ensureBackend would
 * spawn a fresh pool backend for the deleted profile, whose
 * ensure_hermes_home() recreates the directory the delete just removed.
 */
export function resolveRouteProfile(
  tornDownProfile: string | null,
  profile: string | undefined
): string | null | undefined {
  return tornDownProfile ? null : profile
}
